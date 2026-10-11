// Package chatcompletions is the OpenAI Chat Completions wire protocol, the
// lingua franca every OpenAI-compatible provider speaks (pyllym
// protocols/chat_completions.py).
package chatcompletions

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/PhotonEra-B-V/gollym/core"
)

// EmptyParametersSchema is sent for tools without parameters.
func EmptyParametersSchema() map[string]any {
	return map[string]any{"type": "object", "properties": map[string]any{}, "required": []any{}, "additionalProperties": false, "strict": true}
}

// Protocol speaks Chat Completions for one provider.
type Protocol struct {
	Provider core.Provider
	Model    *core.Info
}

// New is the core.ProtocolFactory.
func New(p core.Provider, model *core.Info) core.Protocol {
	return &Protocol{Provider: p, Model: model}
}

func (c *Protocol) conn() *core.Connection { return c.Provider.Connection() }

// --- endpoints ---------------------------------------------------------------

func (c *Protocol) CompletionURL() string { return "chat/completions" }
func (c *Protocol) StreamURL() string     { return c.CompletionURL() }
func (c *Protocol) ModelsURL() string     { return "models" }
func (c *Protocol) EmbeddingURL() string  { return "embeddings" }
func (c *Protocol) ModerationURL() string { return "moderations" }
func (c *Protocol) ImagesURL() string     { return "images/generations" }
func (c *Protocol) SpeechURL() string     { return "audio/speech" }
func (c *Protocol) TranscriptionURL() string {
	return "audio/transcriptions"
}

// MaybeNormalizeTemperature forces 1.0 for o-series/gpt-5 and drops it for
// search models.
func (c *Protocol) MaybeNormalizeTemperature(t *float64, model *core.Info) *float64 {
	if model == nil {
		return t
	}
	return NormalizeTemperature(t, model.ID)
}

// --- chat render -------------------------------------------------------------

// RenderPayload builds the chat/completions body.
func (c *Protocol) RenderPayload(req *core.CompletionRequest) (map[string]any, error) {
	if req.Model == nil {
		return nil, errors.New("chat_completions: model is required")
	}
	msgs, err := c.FormatMessages(req.Messages)
	if err != nil {
		return nil, err
	}
	payload := map[string]any{"model": req.Model.ID, "messages": msgs, "stream": req.Stream}
	if req.Temperature != nil {
		payload["temperature"] = *req.Temperature
	}
	if req.Tools.Len() > 0 {
		var tools []any
		for _, t := range req.Tools.List() {
			tools = append(tools, ToolFor(t))
		}
		payload["tools"] = tools
		if req.ToolPrefs.Choice != "" {
			payload["tool_choice"] = BuildToolChoice(req.ToolPrefs.Choice)
		}
		if req.ToolPrefs.Calls != "" {
			payload["parallel_tool_calls"] = req.ToolPrefs.Calls == "many"
		}
	}
	if len(req.Schema) > 0 {
		payload["response_format"] = map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name": req.Schema["name"], "schema": req.Schema["schema"], "strict": req.Schema["strict"],
			},
		}
	}
	if req.Thinking != nil && req.Thinking.Effort != "" {
		payload["reasoning_effort"] = req.Thinking.Effort
	}
	if req.Stream {
		payload["stream_options"] = map[string]any{"include_usage": true}
	}
	return payload, nil
}

// --- chat parse --------------------------------------------------------------

// ParseCompletionResponse reads a chat/completions response.
func (c *Protocol) ParseCompletionResponse(resp *core.Response) (*core.Message, error) {
	msg, err := c.parseBody(resp.Body, resp)
	if err != nil {
		return nil, err
	}
	if msg == nil {
		return nil, core.NewError(core.KindError, resp, "Empty or unparseable completion response")
	}
	return msg, nil
}

func (c *Protocol) parseBody(data any, raw *core.Response) (*core.Message, error) {
	if !core.Truthy(data) {
		return nil, nil
	}
	if m, ok := data.(map[string]any); ok {
		if msg := core.DigString(m, "error", "message"); msg != "" {
			return nil, core.NewError(core.KindError, raw, msg)
		}
	}
	md := core.DigMap(data, "choices", 0, "message")
	if len(md) == 0 {
		return nil, nil
	}
	body, _ := data.(map[string]any)
	usage, _ := body["usage"].(map[string]any)
	content, blockThinking := ExtractContentAndThinking(md["content"])
	thinking := blockThinking
	if thinking == "" {
		thinking = extractThinkingText(md)
	}
	calls, err := ParseToolCalls(core.ToSafeArray(md["tool_calls"]), true)
	if err != nil {
		return nil, err
	}
	modelID, _ := body["model"].(string)
	finish, _ := core.Dig(data, "choices", 0, "finish_reason").(string)
	m := &core.Message{
		Role:         core.RoleAssistant,
		Content:      content,
		Citations:    extractCitations(md, body, content),
		Thinking:     core.BuildThinking(thinking, extractThinkingSignature(md)),
		ToolCalls:    calls,
		Tokens:       UsageTokens(usage),
		FinishReason: finish,
		ModelID:      modelID,
		Raw:          raw,
	}
	if m.Content == nil && calls.Len() > 0 {
		m.Content = ""
	}
	return m, nil
}

// UsageTokens reads Chat Completions usage (input excludes cache reads and
// writes; output is the larger of completion and total-prompt). As in
// pyllym the cache-write count is always reported (0 without usage), so the
// result is never nil.
func UsageTokens(usage map[string]any) *core.Tokens {
	cacheWrite := CacheWriteTokens(usage)
	return core.BuildTokens(InputTokens(usage), OutputTokens(usage), CacheReadTokens(usage), &cacheWrite, ThinkingTokens(usage))
}

// --- streaming ---------------------------------------------------------------

// BuildChunk reads one streamed chunk.
func (c *Protocol) BuildChunk(data map[string]any) (*core.Chunk, error) {
	usage, _ := data["usage"].(map[string]any)
	delta := core.DigMap(data, "choices", 0, "delta")
	src := delta["content"]
	if !core.Truthy(src) {
		src = core.Dig(data, "choices", 0, "message", "content")
	}
	content, blockThinking := ExtractContentAndThinking(src)
	thinking := blockThinking
	if thinking == "" {
		if s, ok := core.Or(delta["reasoning_content"], delta["reasoning"]).(string); ok {
			thinking = s
		}
	}
	sig, _ := delta["reasoning_signature"].(string)
	calls, err := ParseToolCalls(core.ToSafeArray(delta["tool_calls"]), false)
	if err != nil {
		return nil, err
	}
	modelID, _ := data["model"].(string)
	finish, _ := core.Dig(data, "choices", 0, "finish_reason").(string)
	cits := parseAnnotations(delta["annotations"], nil)
	if len(cits) == 0 {
		cits = parseRootCitations(data)
	}
	return &core.Chunk{
		Role: core.RoleAssistant, ModelID: modelID, Content: content, Citations: cits,
		Thinking: core.BuildThinking(thinking, sig), ToolCalls: calls,
		Tokens: UsageTokens(usage), FinishReason: finish,
	}, nil
}

// --- messages ----------------------------------------------------------------

// FormatMessages renders the conversation.
func (c *Protocol) FormatMessages(messages []*core.Message) ([]any, error) {
	out := make([]any, 0, len(messages))
	for _, m := range messages {
		content, err := c.formatMessageContent(m)
		if err != nil {
			return nil, err
		}
		entry := map[string]any{"role": c.FormatRole(m.Role)}
		if content != nil {
			entry["content"] = content
		}
		if calls := FormatToolCalls(m.ToolCalls); calls != nil {
			entry["tool_calls"] = calls
		}
		if m.ToolCallID != "" {
			entry["tool_call_id"] = m.ToolCallID
		}
		for k, v := range formatThinking(m) {
			entry[k] = v
		}
		out = append(out, entry)
	}
	return out, nil
}

func (c *Protocol) formatMessageContent(m *core.Message) (any, error) {
	content, err := c.FormatContent(m.Content, ContentOptions{DocumentAttachments: "pdf", Images: true, Audio: true})
	if err != nil {
		return nil, err
	}
	if content == nil && m.Role == core.RoleAssistant && m.Thinking != nil && !m.IsToolCall() {
		return "", nil
	}
	return content, nil
}

// FormatRole maps system to developer for OpenAI itself.
func (c *Protocol) FormatRole(role string) string {
	if role == core.RoleSystem && c.useDeveloperRole() {
		return "developer"
	}
	return role
}

func (c *Protocol) useDeveloperRole() bool {
	// Config values are strings: "false"/"0" mean False, as a Go caller
	// would write them; any other non-empty value is truthy.
	if v := c.Provider.Config().Get("openai_use_system_role"); v != "" {
		if b, err := strconv.ParseBool(v); err != nil || b {
			return false
		}
	}
	return c.Provider.UsesDeveloperRole()
}

// ContentOptions control which attachment kinds a provider accepts.
// DocumentAttachments is "pdf" (default), "all" or "none".
type ContentOptions struct {
	DocumentAttachments string
	Images              bool
	Audio               bool
}

// FormatContent renders message content (string, *Content, RawContent,
// decoded JSON) into Chat Completions parts.
func (c *Protocol) FormatContent(content any, o ContentOptions) (any, error) {
	switch v := content.(type) {
	case nil:
		return nil, nil
	case string:
		return v, nil
	case core.RawContent:
		if m, ok := v.Value.(map[string]any); ok {
			return core.PyJSON(m), nil
		}
		return v.Value, nil
	case map[string]any, []any:
		return core.PyJSON(v), nil
	case *core.SearchResults:
		return c.FormatContent(&v.Content, o)
	case *core.Content:
		var parts []any
		if v.Text != nil && *v.Text != "" {
			parts = append(parts, map[string]any{"type": "text", "text": *v.Text})
		}
		for _, a := range v.Attachments {
			part, err := FormatAttachment(a, o)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		}
		if parts == nil {
			parts = []any{}
		}
		return parts, nil
	}
	return content, nil
}

// FormatAttachment renders one attachment part.
func FormatAttachment(a *core.Attachment, o ContentOptions) (map[string]any, error) {
	if o.DocumentAttachments == "" {
		o.DocumentAttachments = "pdf"
	}
	if a.IsProviderFile() {
		if o.DocumentAttachments == "none" {
			return nil, &core.UnsupportedAttachmentError{Type: a.MIMEType}
		}
		return map[string]any{"type": "file", "file": map[string]any{"file_id": a.ProviderFileID()}}, nil
	}
	switch a.Type() {
	case "image":
		if !o.Images {
			return nil, &core.UnsupportedAttachmentError{Type: a.MIMEType}
		}
		u := a.URL
		if u == "" {
			s, err := a.ForLLM()
			if err != nil {
				return nil, err
			}
			u = s
		}
		return map[string]any{"type": "image_url", "image_url": map[string]any{"url": u}}, nil
	case "audio":
		if !o.Audio {
			return nil, &core.UnsupportedAttachmentError{Type: a.MIMEType}
		}
		enc, err := a.Encoded()
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": enc, "format": a.Format()}}, nil
	case "pdf", "document":
		if o.DocumentAttachments == "all" || (o.DocumentAttachments == "pdf" && a.IsPDF()) {
			data, err := a.ForLLM()
			if err != nil {
				return nil, err
			}
			return map[string]any{"type": "file", "file": map[string]any{"filename": a.Filename, "file_data": data}}, nil
		}
		return nil, &core.UnsupportedAttachmentError{Type: a.MIMEType}
	case "text":
		s, err := a.ForLLM()
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "text", "text": s}, nil
	}
	return nil, &core.UnsupportedAttachmentError{Type: a.MIMEType}
}

// --- tools -------------------------------------------------------------------

// ToolFor renders a tool definition.
func ToolFor(t core.Tool) map[string]any {
	schema := t.ParamsSchema()
	if len(schema) == 0 {
		schema = EmptyParametersSchema()
	}
	def := map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": t.Name(), "description": nilIfEmpty(t.Description()), "parameters": schema,
		},
	}
	if pp, ok := t.(core.ProviderParamser); ok && len(pp.ProviderParams()) > 0 {
		return core.DeepMerge(def, pp.ProviderParams())
	}
	return def
}

func nilIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// FormatToolCalls renders an assistant message's tool calls.
func FormatToolCalls(calls *core.ToolCalls) []any {
	if calls.Len() == 0 {
		return nil
	}
	out := make([]any, 0, calls.Len())
	for _, tc := range calls.List() {
		call := map[string]any{
			"id": tc.ID, "type": "function",
			"function": map[string]any{"name": tc.Name, "arguments": core.PyJSON(tc.Arguments)},
		}
		if tc.ThoughtSignature != "" {
			call["extra_content"] = map[string]any{"google": map[string]any{"thought_signature": tc.ThoughtSignature}}
		}
		out = append(out, call)
	}
	return out
}

// ParseToolCalls reads tool calls keyed by id; while streaming the
// arguments stay raw strings and the key is the stream index.
func ParseToolCalls(calls []any, parseArguments bool) (*core.ToolCalls, error) {
	if len(calls) == 0 {
		return nil, nil
	}
	out := &core.ToolCalls{}
	for _, raw := range calls {
		tc, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		argsRaw, _ := core.Dig(tc, "function", "arguments").(string)
		var args any = argsRaw
		if parseArguments {
			if argsRaw == "" {
				args = map[string]any{}
			} else {
				var v any
				if err := json.Unmarshal([]byte(argsRaw), &v); err != nil {
					return nil, fmt.Errorf("tool call arguments: %w", err)
				}
				args = v
			}
		}
		id, _ := tc["id"].(string)
		key := id
		// Streamed fragments after the first carry only their index, so
		// stream chunks are keyed by index; the accumulator maps it to the id.
		if !parseArguments {
			if idx := core.AsInt(tc["index"]); idx != nil {
				key = fmt.Sprint(*idx)
			}
		}
		out.Set(key, &core.ToolCall{
			ID: id, Name: core.DigString(tc, "function", "name"), Arguments: args,
			ThoughtSignature: core.DigString(tc, "extra_content", "google", "thought_signature"),
		})
	}
	return out, nil
}

// BuildToolChoice renders tool_choice.
func BuildToolChoice(choice string) any {
	switch choice {
	case "auto", "none", "required":
		return choice
	}
	return map[string]any{"type": "function", "function": map[string]any{"name": choice}}
}

// --- content/thinking extraction --------------------------------------------

var (
	thinkTagRe = regexp.MustCompile(`(?s)<think>(.*?)</think>`)
)

// ExtractContentAndThinking splits <think> tags or thinking blocks off.
func ExtractContentAndThinking(content any) (any, string) {
	switch v := content.(type) {
	case string:
		if !strings.Contains(v, "<think>") {
			return v, ""
		}
		var thinking strings.Builder
		for _, m := range thinkTagRe.FindAllStringSubmatch(v, -1) {
			thinking.WriteString(m[1])
		}
		rest := strings.TrimSpace(thinkTagRe.ReplaceAllString(v, ""))
		var c any
		if rest != "" {
			c = rest
		}
		return c, thinking.String()
	case []any:
		var text, thinking strings.Builder
		for _, b := range v {
			bm, ok := b.(map[string]any)
			if !ok {
				continue
			}
			switch bm["type"] {
			case "text":
				if s, ok := bm["text"].(string); ok {
					text.WriteString(s)
				}
			case "thinking":
				thinking.WriteString(thinkingTextFromBlock(bm))
			}
		}
		var c any
		if text.Len() > 0 {
			c = text.String()
		}
		return c, thinking.String()
	}
	return content, ""
}

func thinkingTextFromBlock(b map[string]any) string {
	switch t := b["thinking"].(type) {
	case string:
		return t
	case []any:
		var sb strings.Builder
		for _, i := range t {
			if im, ok := i.(map[string]any); ok && im["type"] == "text" {
				s, _ := im["text"].(string)
				sb.WriteString(s)
			}
		}
		return sb.String()
	}
	s, _ := b["text"].(string)
	return s
}

func extractThinkingText(md map[string]any) string {
	s, _ := core.Or(md["reasoning_content"], md["reasoning"], md["thinking"]).(string)
	return s
}

func extractThinkingSignature(md map[string]any) string {
	s, _ := core.Or(md["reasoning_signature"], md["signature"]).(string)
	return s
}

func formatThinking(m *core.Message) map[string]any {
	if m.Role != core.RoleAssistant || m.Thinking == nil {
		return nil
	}
	out := map[string]any{}
	if m.Thinking.Text != "" {
		out["reasoning"] = m.Thinking.Text
		out["reasoning_content"] = m.Thinking.Text
	}
	if m.Thinking.Signature != "" {
		out["reasoning_signature"] = m.Thinking.Signature
	}
	return out
}

func extractCitations(md, data map[string]any, content any) []core.Citation {
	if c := parseAnnotations(md["annotations"], content); len(c) > 0 {
		return c
	}
	return parseRootCitations(data)
}

func parseAnnotations(annotations any, content any) []core.Citation {
	var out []core.Citation
	for _, a := range core.ToSafeArray(annotations) {
		am, _ := a.(map[string]any)
		d, ok := am["url_citation"].(map[string]any)
		if !ok {
			continue
		}
		start, end := core.AsInt(d["start_index"]), core.AsInt(d["end_index"])
		cit := core.Citation{StartIndex: start, EndIndex: end}
		if s, ok := d["url"].(string); ok {
			cit.URL = &s
		}
		if s, ok := d["title"].(string); ok {
			cit.Title = &s
		}
		if text, ok := content.(string); ok && start != nil && end != nil {
			span := pySlice(text, *start, *end)
			cit.Text = &span
		}
		out = append(out, cit)
	}
	return out
}

func pySlice(s string, start, end int) string {
	r := []rune(s)
	n := len(r)
	clamp := func(i int) int {
		if i < 0 {
			i += n
		}
		return max(0, min(i, n))
	}
	start, end = clamp(start), clamp(end)
	if start >= end {
		return ""
	}
	return string(r[start:end])
}

func parseRootCitations(data map[string]any) []core.Citation {
	if results, ok := data["search_results"].([]any); ok && len(results) > 0 {
		var out []core.Citation
		for i, r := range results {
			rm, ok := r.(map[string]any)
			if !ok {
				continue
			}
			idx := i
			cit := core.Citation{SourceIndex: &idx}
			if s, ok := rm["url"].(string); ok {
				cit.URL = &s
			}
			if s, ok := rm["title"].(string); ok {
				cit.Title = &s
			}
			if s, ok := rm["snippet"].(string); ok {
				cit.CitedText = &s
			}
			out = append(out, cit)
		}
		return out
	}
	var out []core.Citation
	for i, u := range core.ToSafeArray(data["citations"]) {
		if s, ok := u.(string); ok {
			idx := i
			out = append(out, core.Citation{URL: &s, SourceIndex: &idx})
		}
	}
	return out
}

// --- usage -------------------------------------------------------------------

// InputTokens is prompt tokens minus cache reads and writes.
func InputTokens(u map[string]any) *int {
	if core.Truthy(u["prompt_cache_miss_tokens"]) {
		return core.AsInt(u["prompt_cache_miss_tokens"])
	}
	p := core.AsInt(u["prompt_tokens"])
	if p == nil {
		return nil
	}
	v := *p - core.Deref(CacheReadTokens(u)) - CacheWriteTokens(u)
	if v < 0 {
		v = 0
	}
	return &v
}

// OutputTokens is completion tokens, or total-prompt when larger.
func OutputTokens(u map[string]any) *int {
	c := core.AsInt(u["completion_tokens"])
	if c == nil {
		return nil
	}
	p, t := core.AsInt(u["prompt_tokens"]), core.AsInt(u["total_tokens"])
	if p != nil && t != nil {
		if gen := max(*t-*p, 0); gen > *c {
			return &gen
		}
	}
	return c
}

// CacheReadTokens are cached prompt tokens.
func CacheReadTokens(u map[string]any) *int {
	if v := core.Dig(u, "prompt_tokens_details", "cached_tokens"); core.Truthy(v) {
		return core.AsInt(v)
	}
	return core.AsInt(u["prompt_cache_hit_tokens"])
}

// CacheWriteTokens are prompt tokens written to the cache.
func CacheWriteTokens(u map[string]any) int {
	if v := core.Dig(u, "prompt_tokens_details", "cache_write_tokens"); core.Truthy(v) {
		return core.Deref(core.AsInt(v))
	}
	return 0
}

// ThinkingTokens are reasoning tokens.
func ThinkingTokens(u map[string]any) *int {
	if v := core.Dig(u, "completion_tokens_details", "reasoning_tokens"); core.Truthy(v) {
		return core.AsInt(v)
	}
	return core.AsInt(u["reasoning_tokens"])
}

// --- temperature -------------------------------------------------------------

var forcedTemperature = []*regexp.Regexp{
	regexp.MustCompile(`^o\d`),
	regexp.MustCompile(`^gpt-5(\.\d+)?(-\d{4})?$`),
	regexp.MustCompile(`^gpt-5(\.\d+)?-pro`),
}

// NormalizeTemperature forces 1.0 for models that reject other values and
// drops temperature for search models.
func NormalizeTemperature(t *float64, modelID string) *float64 {
	for _, re := range forcedTemperature {
		if re.MatchString(modelID) && t != nil && !(*t-1.0 <= 1e-9 && 1.0-*t <= 1e-9) {
			one := 1.0
			return &one
		}
	}
	if strings.Contains(modelID, "-search") {
		return nil
	}
	return t
}

// --- embeddings, moderation, models, images, speech, transcription ----------

// Embed implements core.Embedder.
func (c *Protocol) Embed(ctx context.Context, text any, model string, dimensions *int) (*core.Embedding, error) {
	payload := core.Compact(map[string]any{"model": model, "input": text, "dimensions": dimensions})
	resp, err := c.conn().Post(ctx, c.EmbeddingURL(), payload, nil)
	if err != nil {
		return nil, err
	}
	return ParseEmbeddingResponse(resp, model, text)
}

// ParseEmbeddingResponse reads an embeddings response.
func ParseEmbeddingResponse(resp *core.Response, model string, text any) (*core.Embedding, error) {
	data := resp.BodyMap()
	inputTokens := core.Deref(core.AsInt(core.Dig(data, "usage", "prompt_tokens")))
	var vectors [][]float64
	for _, d := range core.ToSafeArray(data["data"]) {
		var vec []float64
		for _, x := range core.ToSafeArray(core.Dig(d, "embedding")) {
			vec = append(vec, core.Deref(core.AsFloat(x)))
		}
		vectors = append(vectors, vec)
	}
	_, isList := text.([]string)
	if _, isAny := text.([]any); isAny {
		isList = true
	}
	var out any = vectors
	if len(vectors) == 1 && !isList {
		out = vectors[0]
	}
	return &core.Embedding{Vectors: out, Model: model, InputTokens: inputTokens}, nil
}

// Moderate implements core.Moderator.
func (c *Protocol) Moderate(ctx context.Context, input any, model string) (*core.Moderation, error) {
	resp, err := c.conn().Post(ctx, c.ModerationURL(), map[string]any{"model": model, "input": input}, nil)
	if err != nil {
		return nil, err
	}
	data := resp.BodyMap()
	if msg := core.DigString(data, "error", "message"); msg != "" {
		return nil, core.NewError(core.KindError, resp, msg)
	}
	id, _ := data["id"].(string)
	var results []map[string]any
	for _, r := range core.ToSafeArray(data["results"]) {
		if rm, ok := r.(map[string]any); ok {
			results = append(results, rm)
		}
	}
	return &core.Moderation{ID: id, Model: model, Results: results}, nil
}

// ListModels implements core.ModelLister.
func (c *Protocol) ListModels(ctx context.Context) ([]*core.Info, error) {
	resp, err := c.conn().Get(ctx, c.ModelsURL(), nil)
	if err != nil {
		return nil, err
	}
	// pyllym passes None for providers without a capabilities class.
	var caps core.Capabilities
	if c.Provider.Spec().Capabilities != nil {
		caps = c.Provider.Capabilities()
	}
	return ParseListModels(resp, c.Provider.Slug(), caps), nil
}

// ParseListModels reads a /models listing. A nil caps leaves context
// window, max tokens, capabilities and pricing empty.
func ParseListModels(resp *core.Response, slug string, caps core.Capabilities) []*core.Info {
	var out []*core.Info
	for _, raw := range core.ToSafeArray(core.Dig(resp.Body, "data")) {
		md, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		id, _ := md["id"].(string)
		capList := []any{}
		var ctxWindow, maxTokens any
		pricing := map[string]any{}
		if caps != nil {
			for _, c := range caps.CriticalCapabilitiesFor(id) {
				capList = append(capList, c)
			}
			ctxWindow, maxTokens = derefAny(caps.ContextWindowFor(id)), derefAny(caps.MaxTokensFor(id))
			pricing = caps.PricingFor(id)
		}
		out = append(out, core.NewInfo(map[string]any{
			"id": id, "name": id, "provider": slug, "created_at": md["created"],
			"context_window": ctxWindow, "max_output_tokens": maxTokens,
			"capabilities": capList, "pricing": pricing,
			"metadata": map[string]any{"object": md["object"], "owned_by": md["owned_by"]},
		}))
	}
	return out
}

func derefAny(p *int) any {
	if p == nil {
		return nil
	}
	return float64(*p)
}

// Paint implements core.Painter.
func (c *Protocol) Paint(ctx context.Context, prompt string, o core.PaintOptions) (*core.Image, error) {
	if o.With != nil || o.Mask != nil {
		return nil, &core.UnsupportedAttachmentError{Type: "image reference"}
	}
	payload := map[string]any{"model": o.Model, "prompt": prompt, "n": 1, "size": o.Size}
	for k, v := range o.Params {
		payload[k] = v
	}
	resp, err := c.conn().Post(ctx, c.ImagesURL(), payload, nil)
	if err != nil {
		return nil, err
	}
	return ParseImageResponse(resp, o.Model)
}

// ParseImageResponse reads an images response.
func ParseImageResponse(resp *core.Response, model string) (*core.Image, error) {
	data := resp.BodyMap()
	img := core.DigMap(data, "data", 0)
	if len(img) == 0 {
		return nil, core.NewError(core.KindError, nil, "Unexpected response format from image API")
	}
	usage, _ := data["usage"].(map[string]any)
	if usage == nil {
		usage = map[string]any{}
	}
	u, _ := img["url"].(string)
	rp, _ := img["revised_prompt"].(string)
	b64, _ := img["b64_json"].(string)
	return &core.Image{URL: u, MIMEType: "image/png", RevisedPrompt: rp, ModelID: model, Data: b64, Usage: usage}, nil
}

// Speak implements core.Speaker.
func (c *Protocol) Speak(ctx context.Context, input string, o core.SpeakOptions) (*core.Speech, error) {
	voice := o.Voice
	if voice == "" {
		voice = "alloy"
	}
	payload := map[string]any{"model": o.Model, "input": input, "voice": voice}
	if o.Format != "" {
		payload["response_format"] = o.Format
	}
	if o.Instructions != "" {
		payload["instructions"] = o.Instructions
	}
	if o.Speed != nil {
		payload["speed"] = *o.Speed
	}
	for k, v := range o.Params {
		payload[k] = v
	}
	resp, err := c.conn().Post(ctx, c.SpeechURL(), payload, nil)
	if err != nil {
		return nil, err
	}
	format := o.Format
	if format == "" {
		format = "mp3"
	}
	return core.NewSpeech(resp.Content, o.Model, voice, format, ""), nil
}

// Transcribe implements core.Transcriber.
func (c *Protocol) Transcribe(ctx context.Context, audioFile string, o core.TranscribeOptions) (*core.Transcription, error) {
	path := audioFile
	if strings.HasPrefix(path, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			path = filepath.Join(home, path[2:])
		}
	}
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	data := map[string]string{"model": o.Model}
	if o.Language != "" {
		data["language"] = o.Language
	}
	if o.ResponseFormat != "" {
		data["response_format"] = o.ResponseFormat
	}
	if o.Prompt != "" {
		data["prompt"] = o.Prompt
	}
	if o.Temperature != nil {
		data["temperature"] = core.PyStr(*o.Temperature) // str(float): "0.0"
	}
	payload := &core.Multipart{Data: data, Files: map[string]core.MultipartFile{"file": {Filename: filepath.Base(path), Content: content}}}
	resp, err := c.conn().Post(ctx, c.TranscriptionURL(), payload, nil)
	if err != nil {
		return nil, err
	}
	return ParseTranscription(resp, o.Model), nil
}

// ParseTranscription reads a transcription response.
func ParseTranscription(resp *core.Response, model string) *core.Transcription {
	if s, ok := resp.Body.(string); ok {
		return &core.Transcription{Text: s, Model: model}
	}
	data := resp.BodyMap()
	usage, _ := data["usage"].(map[string]any)
	text, _ := data["text"].(string)
	lang, _ := data["language"].(string)
	return &core.Transcription{
		Text: text, Model: model, Language: lang, Duration: core.AsFloat(data["duration"]),
		Segments: data["segments"], Words: data["words"],
		InputTokens:  core.AsInt(core.Or(usage["input_tokens"], usage["prompt_tokens"])),
		OutputTokens: core.AsInt(core.Or(usage["output_tokens"], usage["completion_tokens"])),
	}
}
