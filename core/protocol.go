package core

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
)

// ToolPrefs are tool-choice preferences: Choice is "", auto, none, required
// or a tool name; Calls is "", "one" or "many".
type ToolPrefs struct {
	Choice string
	Calls  string
}

// CompletionRequest is everything a protocol needs to render a chat call.
type CompletionRequest struct {
	Messages    []*Message
	Tools       *ToolSet
	ToolPrefs   ToolPrefs
	Temperature *float64
	Model       *Info
	Params      map[string]any
	Headers     map[string]string
	// Schema is the normalised structured-output schema: name, schema,
	// strict and optional description.
	Schema    map[string]any
	Thinking  *ThinkingConfig
	Citations bool
	Stream    bool
}

// Protocol renders and parses one API family's chat wire format.
type Protocol interface {
	RenderPayload(req *CompletionRequest) (map[string]any, error)
	ParseCompletionResponse(resp *Response) (*Message, error)
	CompletionURL() string
}

// Streamer is implemented by protocols that stream over SSE.
type Streamer interface {
	StreamURL() string
	// BuildChunk turns one decoded SSE data object into a chunk.
	BuildChunk(data map[string]any) (*Chunk, error)
}

// TemperatureNormalizer adjusts temperature per model (e.g. o-series = 1).
type TemperatureNormalizer interface {
	MaybeNormalizeTemperature(t *float64, model *Info) *float64
}

// MessagePreprocessor rewrites messages as they are added (auto-upload).
type MessagePreprocessor interface {
	PreprocessMessage(m *Message) (*Message, error)
}

// SyncResponder overrides the non-streaming call (Bedrock Converse).
type SyncResponder interface {
	SyncResponse(ctx context.Context, payload map[string]any, headers map[string]string) (*Message, error)
}

// StreamResponder overrides the whole streaming call (non-SSE streams).
type StreamResponder interface {
	StreamResponse(ctx context.Context, payload map[string]any, headers map[string]string, onChunk func(*Chunk) error) (*Message, error)
}

// Optional capabilities of a protocol (used by the facade functions).
type (
	ModelLister interface {
		ListModels(ctx context.Context) ([]*Info, error)
	}
	Embedder interface {
		Embed(ctx context.Context, text any, model string, dimensions *int) (*Embedding, error)
	}
	Moderator interface {
		Moderate(ctx context.Context, input any, model string) (*Moderation, error)
	}
	Painter interface {
		Paint(ctx context.Context, prompt string, opts PaintOptions) (*Image, error)
	}
	Speaker interface {
		Speak(ctx context.Context, input string, opts SpeakOptions) (*Speech, error)
	}
	Transcriber interface {
		Transcribe(ctx context.Context, audioFile string, opts TranscribeOptions) (*Transcription, error)
	}
	Animator interface {
		Animate(ctx context.Context, prompt string, model string, params map[string]any) (*Video, error)
	}
	LayoutParser interface {
		ParseLayout(ctx context.Context, file string, model string, opts map[string]any) (*LayoutParsing, error)
	}
	Decider interface {
		Decide(ctx context.Context, state any, questions *Questions, model string, opts map[string]any) (*Decision, error)
	}
)

// PaintOptions are image generation options.
type PaintOptions struct {
	Model  string
	Size   string
	With   any // reference image(s)
	Mask   any
	Params map[string]any
}

// SpeakOptions are text-to-speech options.
type SpeakOptions struct {
	Model        string
	Voice        string
	Format       string
	Instructions string
	Speed        *float64
	Params       map[string]any
	Options      map[string]any
}

// TranscribeOptions are speech-to-text options.
type TranscribeOptions struct {
	Model          string
	Language       string
	ResponseFormat string
	Prompt         string
	Temperature    *float64
	Options        map[string]any
}

// RenderRequest builds the provider payload for req (temperature
// normalised, then Params deep-merged over the protocol's payload).
func RenderRequest(proto Protocol, req *CompletionRequest) (map[string]any, error) {
	r := *req
	if tn, ok := proto.(TemperatureNormalizer); ok {
		r.Temperature = tn.MaybeNormalizeTemperature(req.Temperature, req.Model)
	}
	payload, err := proto.RenderPayload(&r)
	if err != nil {
		return nil, err
	}
	return DeepMerge(payload, req.Params), nil
}

// CompleteRequest renders, sends and parses one completion. With onChunk set
// it streams, calling onChunk per chunk, and returns the accumulated message.
func CompleteRequest(ctx context.Context, p Provider, proto Protocol, req *CompletionRequest, onChunk func(*Chunk) error) (*Message, error) {
	r := *req
	r.Stream = onChunk != nil
	payload, err := RenderRequest(proto, &r)
	if err != nil {
		return nil, err
	}
	if onChunk != nil {
		if sr, ok := proto.(StreamResponder); ok {
			return sr.StreamResponse(ctx, payload, req.Headers, onChunk)
		}
		return StreamSSE(ctx, p, proto, payload, req.Headers, onChunk)
	}
	if sr, ok := proto.(SyncResponder); ok {
		return sr.SyncResponse(ctx, payload, req.Headers)
	}
	resp, err := p.Connection().Post(ctx, proto.CompletionURL(), payload, req.Headers)
	if err != nil {
		return nil, err
	}
	return proto.ParseCompletionResponse(resp)
}

// StreamSSE runs the standard SSE streaming flow (pyllym StreamingMixin).
func StreamSSE(ctx context.Context, p Provider, proto Protocol, payload map[string]any, headers map[string]string, onChunk func(*Chunk) error) (*Message, error) {
	st, ok := proto.(Streamer)
	if !ok {
		return nil, NotSupported("streaming")
	}
	acc := NewStreamAccumulator()
	var raw strings.Builder
	sawEvent := false
	parser := NewSSEParser(func(event, data string) error {
		sawEvent = true
		if event == "error" {
			return StreamError(data)
		}
		if data == "[DONE]" {
			return nil
		}
		var parsed any
		if err := json.Unmarshal([]byte(data), &parsed); err != nil {
			slog.Debug("Failed to parse data chunk", "err", err)
			return nil
		}
		if m, ok := parsed.(map[string]any); ok {
			if _, has := m["error"]; has {
				return StreamError(data)
			}
			chunk, err := st.BuildChunk(m)
			if err != nil {
				return err
			}
			if chunk == nil {
				return nil
			}
			acc.Add(chunk)
			return onChunk(chunk)
		}
		return nil
	})
	err := p.Connection().Stream(ctx, st.StreamURL(), payload, headers, func(b []byte) error {
		if raw.Len() < 1<<20 {
			raw.Write(b)
		}
		return parser.Write(b)
	})
	if err == nil {
		err = parser.Close()
	}
	if err != nil {
		return nil, err
	}
	if !sawEvent {
		body := strings.TrimSpace(raw.String())
		if strings.HasPrefix(body, "{") && strings.Contains(body, `"error"`) {
			return nil, StreamError(body)
		}
	}
	return acc.Message(nil), nil
}

// StreamError maps an in-stream error payload to the HTTP error taxonomy.
func StreamError(data string) error {
	var parsed any
	if err := json.Unmarshal([]byte(data), &parsed); err != nil {
		parsed = map[string]any{"message": data}
	}
	message, errType := "", ""
	if m, ok := parsed.(map[string]any); ok {
		switch e := m["error"].(type) {
		case map[string]any:
			message, _ = e["message"].(string)
			errType = strings.ToLower(DigString(e, "type"))
			if errType == "" {
				if c := e["code"]; c != nil {
					errType = strings.ToLower(strings.TrimSpace(MustJSON(c)))
					errType = strings.Trim(errType, `"`)
				}
			}
		case string:
			message = e
		}
		if message == "" {
			message, _ = m["message"].(string)
		}
	}
	if message == "" {
		message = "Streaming error: " + data
	}
	kind := KindError
	switch {
	case strings.Contains(errType, "overloaded"):
		kind = KindOverloaded
	case strings.Contains(errType, "insufficient_quota"):
		kind = KindPaymentRequired
	case strings.Contains(errType, "rate_limit"):
		kind = KindRateLimit
	case strings.Contains(errType, "server_error"):
		kind = KindServer
	}
	return &Error{Kind: kind, Message: message}
}

// ProtocolFor resolves the provider's default protocol (model-less ops).
func ProtocolFor(p Provider) (Protocol, error) { return p.Protocol("", nil) }

// AsFeature returns p's default protocol as feature interface T, or
// ErrUnsupported naming the feature.
func AsFeature[T any](p Provider, feature string) (T, error) {
	var zero T
	proto, err := ProtocolFor(p)
	if err != nil {
		return zero, err
	}
	f, ok := proto.(T)
	if !ok {
		return zero, NotSupported(p.Name() + " does not support " + feature)
	}
	return f, nil
}

// IsUnsupported reports a NotSupported error.
func IsUnsupported(err error) bool { return errors.Is(err, ErrUnsupported) }
