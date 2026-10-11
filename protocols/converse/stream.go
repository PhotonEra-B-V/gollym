package converse

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"

	"github.com/PhotonEra-B-V/gollym/core"
)

// BuildChunk reads one ConverseStream event, keyed by its event type as the
// AWS SDKs yield it ({"contentBlockDelta": {...}}); the flattened shape
// ({"delta": {...}}) is accepted too. Exception events become errors.
func (c *Protocol) BuildChunk(event map[string]any) (*core.Chunk, error) {
	if isStreamErrorEvent(event) {
		return nil, streamError(event)
	}
	metaUsage := core.DigMap(event, "metadata", "usage")
	usage, _ := event["usage"].(map[string]any)
	modelID, _ := event["modelId"].(string)
	if modelID == "" {
		modelID = c.modelID()
	}
	delta := normalizedDelta(event)
	finish, _ := core.Or(core.Dig(event, "messageStop", "stopReason"), event["stopReason"]).(string)
	return &core.Chunk{
		Role:      core.RoleAssistant,
		ModelID:   modelID,
		Content:   delta["text"],
		Thinking:  core.BuildThinking(thinkingDelta(delta), thinkingSignature(event, delta)),
		ToolCalls: streamToolCalls(event, delta),
		Tokens: core.BuildTokens(
			streamInputTokens(metaUsage, usage),
			core.AsInt(core.Or(metaUsage["outputTokens"], usage["outputTokens"])),
			core.AsInt(core.Or(metaUsage["cacheReadInputTokens"], usage["cacheReadInputTokens"])),
			core.AsInt(core.Or(metaUsage["cacheWriteInputTokens"], usage["cacheWriteInputTokens"])),
			orInt(ReasoningTokens(metaUsage), ReasoningTokens(usage)),
		),
		FinishReason: finish,
	}, nil
}

func orInt(a, b *int) *int {
	if a != nil && *a != 0 {
		return a
	}
	return b
}

func streamInputTokens(metaUsage, usage map[string]any) *int {
	u := usage
	if core.Truthy(metaUsage["inputTokens"]) {
		u = metaUsage
	}
	if core.Truthy(u["inputTokens"]) {
		return InputTokens(u)
	}
	return nil
}

// normalizedDelta is contentBlockDelta.delta (or delta), decoding a JSON
// string delta.
func normalizedDelta(event map[string]any) map[string]any {
	delta := core.Or(core.Dig(event, "contentBlockDelta", "delta"), event["delta"], map[string]any{})
	switch d := delta.(type) {
	case map[string]any:
		return d
	case string:
		var m map[string]any
		if json.Unmarshal([]byte(d), &m) == nil && m != nil {
			return m
		}
	}
	return map[string]any{}
}

func thinkingDelta(delta map[string]any) string {
	rc, _ := core.Or(delta["reasoningContent"], map[string]any{}).(map[string]any)
	rt, _ := core.Or(rc["reasoningText"], map[string]any{}).(map[string]any)
	s, _ := core.Or(rt["text"], rc["text"]).(string)
	return s
}

func thinkingSignature(event, delta map[string]any) string {
	rc, _ := core.Or(delta["reasoningContent"], map[string]any{}).(map[string]any)
	rt, _ := core.Or(rc["reasoningText"], map[string]any{}).(map[string]any)
	if s, ok := core.Or(rt["signature"], rc["signature"]).(string); ok && s != "" {
		return s
	}
	start := core.DigMap(event, "contentBlockStart", "start", "reasoningContent")
	if len(start) == 0 {
		return ""
	}
	srt, _ := core.Or(start["reasoningText"], map[string]any{}).(map[string]any)
	s, _ := core.Or(srt["signature"], start["redactedContent"]).(string)
	return s
}

func streamToolCalls(event, delta map[string]any) *core.ToolCalls {
	if core.Truthy(event["contentBlockStart"]) || core.Truthy(event["start"]) {
		tu := core.DigMap(event, "contentBlockStart", "start", "toolUse")
		if len(tu) == 0 {
			tu = core.DigMap(event, "start", "toolUse")
		}
		if len(tu) == 0 {
			return nil
		}
		id, _ := tu["toolUseId"].(string)
		name, _ := tu["name"].(string)
		out := &core.ToolCalls{}
		out.Set(id, &core.ToolCall{ID: id, Name: name, Arguments: core.Or(tu["input"], map[string]any{})})
		return out
	}
	if core.Truthy(event["contentBlockDelta"]) || core.Truthy(core.Dig(event, "delta", "toolUse")) {
		input := core.Dig(delta, "toolUse", "input")
		if input == nil {
			return nil
		}
		// A fragment without id: the accumulator appends it to the latest call.
		out := &core.ToolCalls{}
		out.Set("", &core.ToolCall{Arguments: input})
		return out
	}
	return nil
}

func isStreamErrorEvent(event map[string]any) bool {
	for k := range event {
		if strings.HasSuffix(k, "Exception") {
			return true
		}
	}
	return event["type"] == "error"
}

var exceptionStatus = map[string]int{
	"throttlingException":         429,
	"validationException":         400,
	"accessDeniedException":       401,
	"unrecognizedClientException": 401,
	"serviceUnavailableException": 503,
}

// streamError maps an exception event to the HTTP error taxonomy.
func streamError(event map[string]any) error {
	if event["type"] == "error" {
		msg := core.DigString(event, "error", "message")
		if msg == "" {
			msg = "Bedrock streaming error"
		}
		return &core.Error{Kind: core.ErrorKindForStatus(500, ""), Message: msg}
	}
	key := ""
	for k := range event {
		if strings.HasSuffix(k, "Exception") {
			key = k
			break
		}
	}
	payload, _ := event[key].(map[string]any)
	msg, _ := payload["message"].(string)
	if msg == "" {
		msg = key
	}
	status, ok := exceptionStatus[key]
	if !ok {
		status = 500
	}
	return &core.Error{Kind: core.ErrorKindForStatus(status, ""), Message: msg}
}

// --- event-stream transport --------------------------------------------------

// maxFrame bounds one event-stream message (AWS caps them at 16 MB).
const maxFrame = 16 << 20

// frameReader reassembles event-stream messages from arbitrary byte chunks.
type frameReader struct {
	buf []byte
	dec *eventstream.Decoder
}

func (f *frameReader) write(b []byte, onMessage func(eventstream.Message) error) error {
	f.buf = append(f.buf, b...)
	for len(f.buf) >= 4 {
		total := binary.BigEndian.Uint32(f.buf[:4])
		if total < 16 || total > maxFrame {
			return &core.Error{Kind: core.KindError, Message: fmt.Sprintf("Malformed Bedrock event stream (frame length %d)", total)}
		}
		if uint32(len(f.buf)) < total {
			return nil
		}
		msg, err := f.dec.Decode(bytes.NewReader(f.buf[:total]), nil)
		if err != nil {
			return &core.Error{Kind: core.KindError, Message: "Malformed Bedrock event stream: " + err.Error(), Cause: err}
		}
		f.buf = f.buf[total:]
		if err := onMessage(msg); err != nil {
			return err
		}
	}
	return nil
}

func headerString(m eventstream.Message, name string) string {
	if v := m.Headers.Get(name); v != nil {
		return v.String()
	}
	return ""
}

// eventFor turns a decoded message into the event map BuildChunk reads:
// {"<event-type>": payload} or {"<exception-type>": payload}.
func eventFor(m eventstream.Message) (map[string]any, error) {
	switch headerString(m, ":message-type") {
	case "error":
		msg := headerString(m, ":error-message")
		if msg == "" {
			msg = headerString(m, ":error-code")
		}
		return map[string]any{"type": "error", "error": map[string]any{"message": msg}}, nil
	case "exception":
		var payload any
		if err := json.Unmarshal(m.Payload, &payload); err != nil {
			payload = map[string]any{"message": string(m.Payload)}
		}
		return map[string]any{headerString(m, ":exception-type"): payload}, nil
	}
	eventType := headerString(m, ":event-type")
	var payload any
	if len(m.Payload) > 0 {
		if err := json.Unmarshal(m.Payload, &payload); err != nil {
			slog.Debug("Failed to parse Bedrock event payload", "event", eventType, "err", err)
			return nil, nil
		}
	}
	if eventType == "" {
		p, _ := payload.(map[string]any)
		return p, nil
	}
	return map[string]any{eventType: core.Or(payload, map[string]any{})}, nil
}

// StreamResponse runs ConverseStream: a signed POST answered with
// application/vnd.amazon.eventstream frames, one Converse event each.
func (c *Protocol) StreamResponse(ctx context.Context, payload map[string]any, headers map[string]string, onChunk func(*core.Chunk) error) (*core.Message, error) {
	acc := core.NewStreamAccumulator()
	fr := &frameReader{dec: eventstream.NewDecoder()}
	handle := func(m eventstream.Message) error {
		event, err := eventFor(m)
		if err != nil || event == nil {
			return err
		}
		chunk, err := c.BuildChunk(event)
		if err != nil {
			return err
		}
		acc.Add(chunk)
		return onChunk(chunk)
	}
	err := c.Provider.Connection().Stream(ctx, c.StreamURL(), payload, headers, func(b []byte) error {
		return fr.write(b, handle)
	})
	if err != nil {
		return nil, err
	}
	if len(fr.buf) > 0 {
		return nil, core.WrapConnectionError(fmt.Errorf("Bedrock event stream ended mid-frame (%d bytes left)", len(fr.buf)))
	}
	return acc.Message(nil), nil
}
