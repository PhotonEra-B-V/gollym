// Package anthropic is the Anthropic Messages API wire protocol (pyllym
// protocols/anthropic.py): top-level system blocks, content blocks for
// images/PDFs/text documents, tool_use/tool_result, extended and adaptive
// thinking with signatures, citations (including search_result blocks) and
// SSE streaming.
package anthropic

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/PhotonEra-B-V/gollym/core"
)

// DefaultInputSchema is sent for tools without parameters.
func DefaultInputSchema() map[string]any {
	return map[string]any{"type": "object", "properties": map[string]any{}, "required": []any{}, "additionalProperties": false, "strict": true}
}

// DefaultMaxTokens is used when the model record has no max_output_tokens.
const DefaultMaxTokens = 4096

// Protocol speaks the Messages API for one provider.
type Protocol struct {
	Provider core.Provider
	Model    *core.Info
}

// New is the core.ProtocolFactory.
func New(p core.Provider, model *core.Info) core.Protocol {
	return &Protocol{Provider: p, Model: model}
}

// --- endpoints ---------------------------------------------------------------

func (a *Protocol) CompletionURL() string { return "v1/messages" }
func (a *Protocol) StreamURL() string     { return a.CompletionURL() }
func (a *Protocol) ModelsURL() string     { return "v1/models" }

// --- render ------------------------------------------------------------------

// RenderPayload builds the v1/messages body.
func (a *Protocol) RenderPayload(req *core.CompletionRequest) (map[string]any, error) {
	if req.Model == nil {
		return nil, errors.New("anthropic: model is required")
	}
	var system, chat []*core.Message
	for _, m := range req.Messages {
		if m.Role == core.RoleSystem {
			system = append(system, m)
		} else {
			chat = append(chat, m)
		}
	}
	systemContent, err := a.buildSystemContent(system)
	if err != nil {
		return nil, err
	}
	thinkingEnabled := req.Thinking.Enabled()
	msgs := make([]any, 0, len(chat))
	for _, m := range chat {
		f, err := a.formatMessage(m, thinkingEnabled, req.Citations)
		if err != nil {
			return nil, err
		}
		msgs = append(msgs, f)
	}
	maxTokens := DefaultMaxTokens
	if mt := req.Model.MaxTokens(); mt != nil && *mt != 0 {
		maxTokens = *mt
	}
	payload := map[string]any{
		"model":      req.Model.ID,
		"messages":   msgs,
		"stream":     req.Stream,
		"max_tokens": maxTokens,
	}
	if err := addThinkingFields(payload, req.Thinking, req.Model); err != nil {
		return nil, err
	}
	if req.Tools.Len() > 0 {
		tools := make([]any, 0, req.Tools.Len())
		for _, t := range req.Tools.List() {
			tools = append(tools, FunctionFor(t))
		}
		payload["tools"] = tools
		if req.ToolPrefs.Choice != "" || req.ToolPrefs.Calls != "" {
			payload["tool_choice"] = BuildToolChoice(req.ToolPrefs)
		}
	}
	if len(systemContent) > 0 {
		payload["system"] = systemContent
	}
	if req.Temperature != nil {
		payload["temperature"] = *req.Temperature
	}
	if len(req.Schema) > 0 {
		mergeOutputConfig(payload, BuildOutputConfig(req.Schema))
	}
	return payload, nil
}

func (a *Protocol) buildSystemContent(messages []*core.Message) ([]any, error) {
	var out []any
	for _, m := range messages {
		if raw, ok := m.Content.(core.RawContent); ok {
			out = append(out, core.ToSafeArray(raw.Value)...)
			continue
		}
		f, err := a.FormatContent(m.Content, false)
		if err != nil {
			return nil, err
		}
		if l, ok := f.([]any); ok {
			out = append(out, l...)
		} else {
			out = append(out, f)
		}
	}
	return out, nil
}

// BuildOutputConfig renders structured output as output_config.format
// (the schema without its "strict" key).
func BuildOutputConfig(schema map[string]any) map[string]any {
	normalized := core.DeepCopy(schema["schema"])
	if m, ok := normalized.(map[string]any); ok {
		delete(m, "strict")
	}
	return map[string]any{"format": map[string]any{"type": "json_schema", "schema": normalized}}
}

// mergeOutputConfig is payload.setdefault("output_config", {}).update(cfg).
func mergeOutputConfig(payload, cfg map[string]any) {
	oc, ok := payload["output_config"].(map[string]any)
	if !ok {
		oc = map[string]any{}
		payload["output_config"] = oc
	}
	for k, v := range cfg {
		oc[k] = v
	}
}

// --- parse -------------------------------------------------------------------

// ParseCompletionResponse reads a v1/messages response.
func (a *Protocol) ParseCompletionResponse(resp *core.Response) (*core.Message, error) {
	data := resp.BodyMap()
	if data == nil {
		return nil, core.NewError(core.KindError, resp, "Empty or unparseable completion response")
	}
	var blocks []map[string]any
	for _, b := range core.ToSafeArray(data["content"]) {
		if bm, ok := b.(map[string]any); ok {
			blocks = append(blocks, bm)
		}
	}
	text, citations := extractTextAndCitations(blocks)
	var toolUses []any
	for _, b := range blocks {
		if b["type"] == "tool_use" {
			toolUses = append(toolUses, b)
		}
	}
	usage, _ := data["usage"].(map[string]any)
	thinkingTokens := core.AsInt(core.Or(
		core.Dig(usage, "output_tokens_details", "thinking_tokens"),
		core.Dig(usage, "output_tokens_details", "reasoning_tokens"),
		usage["thinking_tokens"],
		usage["reasoning_tokens"],
	))
	modelID, _ := data["model"].(string)
	finish, _ := data["stop_reason"].(string)
	return &core.Message{
		Role:         core.RoleAssistant,
		Content:      text,
		Citations:    citations,
		Thinking:     core.BuildThinking(extractThinkingContent(blocks), extractThinkingSignature(blocks)),
		ToolCalls:    ParseToolCalls(toolUses),
		Tokens:       core.BuildTokens(core.AsInt(usage["input_tokens"]), core.AsInt(usage["output_tokens"]), cachedTokens(data), cacheCreationTokens(data), thinkingTokens),
		FinishReason: finish,
		ModelID:      modelID,
		Raw:          resp,
	}, nil
}

// extractTextAndCitations joins text blocks; each citation spans its block
// (indices in code points, like Python's len).
func extractTextAndCitations(blocks []map[string]any) (string, []core.Citation) {
	var text strings.Builder
	n := 0
	var citations []core.Citation
	for _, b := range blocks {
		if b["type"] != "text" {
			continue
		}
		blockText := pyStr(b["text"])
		l := utf8.RuneCountInString(blockText)
		for _, c := range core.ToSafeArray(b["citations"]) {
			cm, _ := c.(map[string]any)
			start, end, t := n, n+l, blockText
			citations = append(citations, ParseCitation(cm, &t, &start, &end))
		}
		text.WriteString(blockText)
		n += l
	}
	return text.String(), citations
}

// pyStr is str(v or "") for a decoded JSON text field.
func pyStr(v any) string {
	if !core.Truthy(v) {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprint(v)
}

func extractThinkingContent(blocks []map[string]any) string {
	var sb strings.Builder
	for _, b := range blocks {
		if b["type"] == "thinking" {
			s, _ := core.Or(b["thinking"], b["text"], "").(string)
			sb.WriteString(s)
		}
	}
	return sb.String()
}

func extractThinkingSignature(blocks []map[string]any) string {
	block := firstBlock(blocks, "thinking")
	if block == nil {
		block = firstBlock(blocks, "redacted_thinking")
	}
	if block == nil {
		return ""
	}
	s, _ := core.Or(block["signature"], block["data"]).(string)
	return s
}

func firstBlock(blocks []map[string]any, typ string) map[string]any {
	for _, b := range blocks {
		if b["type"] == typ {
			return b
		}
	}
	return nil
}

// --- message formatting ------------------------------------------------------

func (a *Protocol) formatMessage(m *core.Message, thinkingEnabled, citations bool) (map[string]any, error) {
	if m.IsToolCall() {
		return a.formatToolCall(m, thinkingEnabled)
	}
	if m.IsToolResult() {
		return a.formatToolResult(m)
	}
	return a.formatBasicMessage(m, thinkingEnabled, citations)
}

func (a *Protocol) formatBasicMessage(m *core.Message, thinkingEnabled, citations bool) (map[string]any, error) {
	blocks := []any{}
	if m.Role == core.RoleAssistant && thinkingEnabled {
		if b := buildThinkingBlock(m.Thinking); b != nil {
			blocks = append(blocks, b)
		}
	}
	blocks, err := a.appendFormattedContent(blocks, m.Content, citations)
	if err != nil {
		return nil, err
	}
	return map[string]any{"role": convertRole(m.Role), "content": blocks}, nil
}

func (a *Protocol) formatToolCall(m *core.Message, thinkingEnabled bool) (map[string]any, error) {
	if raw, ok := m.Content.(core.RawContent); ok {
		var blocks []any
		if l, ok := raw.Value.([]any); ok {
			blocks = append(blocks, l...)
		} else {
			blocks = []any{raw.Value}
		}
		return map[string]any{"role": "assistant", "content": prependThinking(blocks, m, thinkingEnabled)}, nil
	}
	blocks := prependThinking([]any{}, m, thinkingEnabled)
	if m.Content != nil && m.Content != "" {
		var err error
		if blocks, err = a.appendFormattedContent(blocks, m.Content, false); err != nil {
			return nil, err
		}
	}
	for _, tc := range m.ToolCalls.List() {
		blocks = append(blocks, map[string]any{"type": "tool_use", "id": tc.ID, "name": tc.Name, "input": tc.Arguments})
	}
	return map[string]any{"role": "assistant", "content": blocks}, nil
}

func (a *Protocol) formatToolResult(m *core.Message) (map[string]any, error) {
	if raw, ok := m.Content.(core.RawContent); ok {
		return map[string]any{"role": "user", "content": raw.Value}, nil
	}
	content := m.Content
	if isEmptyContent(content) {
		content = "(no output)"
	}
	formatted, err := a.formatToolResultContent(content)
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"role": "user",
		"content": []any{map[string]any{
			"type": "tool_result", "tool_use_id": m.ToolCallID, "content": formatted,
		}},
	}, nil
}

// isEmptyContent is `content is None or len(content) == 0`.
func isEmptyContent(content any) bool {
	switch v := content.(type) {
	case nil:
		return true
	case string:
		return v == ""
	case map[string]any:
		return len(v) == 0
	case []any:
		return len(v) == 0
	}
	return false
}

func (a *Protocol) formatToolResultContent(content any) (any, error) {
	if sr, ok := content.(*core.SearchResults); ok {
		out := make([]any, 0, len(sr.Results))
		for _, r := range sr.Results {
			source := r["url"]
			if source == "" {
				source = r["title"]
			}
			out = append(out, map[string]any{
				"type":      "search_result",
				"source":    source,
				"title":     r["title"],
				"content":   []any{map[string]any{"type": "text", "text": r["text"]}},
				"citations": map[string]any{"enabled": true},
			})
		}
		return out, nil
	}
	return a.FormatContent(content, false)
}

func (a *Protocol) appendFormattedContent(blocks []any, content any, citations bool) ([]any, error) {
	f, err := a.FormatContent(content, citations)
	if err != nil {
		return nil, err
	}
	if l, ok := f.([]any); ok {
		return append(blocks, l...), nil
	}
	return append(blocks, f), nil
}

// --- media -------------------------------------------------------------------

// FormatContent renders message content as content blocks. RawContent is
// passed through as is.
func (a *Protocol) FormatContent(content any, citations bool) (any, error) {
	switch v := content.(type) {
	case core.RawContent:
		return v.Value, nil
	case map[string]any, []any:
		return []any{map[string]any{"type": "text", "text": core.PyJSON(v)}}, nil
	case *core.SearchResults:
		return a.formatRichContent(&v.Content, citations)
	case *core.Content:
		return a.formatRichContent(v, citations)
	}
	return []any{map[string]any{"type": "text", "text": content}}, nil
}

func (a *Protocol) formatRichContent(c *core.Content, citations bool) (any, error) {
	parts := []any{}
	if c.Text != nil && *c.Text != "" {
		parts = append(parts, map[string]any{"type": "text", "text": *c.Text})
	}
	for _, att := range c.Attachments {
		part, err := FormatAttachment(att, citations)
		if err != nil {
			return nil, err
		}
		parts = append(parts, part)
	}
	return parts, nil
}

// FormatAttachment renders one attachment as an image, document or text
// block.
func FormatAttachment(att *core.Attachment, citations bool) (map[string]any, error) {
	if att.IsProviderFile() {
		return formatProviderFile(att, citations), nil
	}
	switch att.Type() {
	case "image":
		return formatImage(att)
	case "pdf":
		return formatPDF(att, citations)
	case "text":
		if citations {
			return formatTextDocument(att)
		}
		s, err := att.ForLLM()
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "text", "text": s}, nil
	}
	return nil, &core.UnsupportedAttachmentError{Type: att.MIMEType}
}

func formatImage(att *core.Attachment) (map[string]any, error) {
	if att.IsURL() {
		return map[string]any{"type": "image", "source": map[string]any{"type": "url", "url": att.URL}}, nil
	}
	enc, err := att.Encoded()
	if err != nil {
		return nil, err
	}
	return map[string]any{"type": "image", "source": map[string]any{"type": "base64", "media_type": att.MIMEType, "data": enc}}, nil
}

func formatProviderFile(att *core.Attachment, citations bool) map[string]any {
	source := map[string]any{"type": "file", "file_id": att.ProviderFileID()}
	if att.IsImage() {
		return map[string]any{"type": "image", "source": source}
	}
	doc := map[string]any{"type": "document", "source": source}
	if citations {
		enableCitations(doc, att)
	}
	return doc
}

func formatPDF(att *core.Attachment, citations bool) (map[string]any, error) {
	var source map[string]any
	if att.IsURL() {
		source = map[string]any{"type": "url", "url": att.URL}
	} else {
		enc, err := att.Encoded()
		if err != nil {
			return nil, err
		}
		source = map[string]any{"type": "base64", "media_type": att.MIMEType, "data": enc}
	}
	doc := map[string]any{"type": "document", "source": source}
	if citations {
		enableCitations(doc, att)
	}
	return doc, nil
}

func formatTextDocument(att *core.Attachment) (map[string]any, error) {
	data, err := att.Content()
	if err != nil {
		return nil, err
	}
	doc := map[string]any{
		"type":   "document",
		"source": map[string]any{"type": "text", "media_type": "text/plain", "data": strings.ToValidUTF8(string(data), "�")},
	}
	enableCitations(doc, att)
	return doc, nil
}

func enableCitations(doc map[string]any, att *core.Attachment) {
	if att.Filename != "" {
		doc["title"] = att.Filename
	}
	doc["citations"] = map[string]any{"enabled": true}
}

// --- tools -------------------------------------------------------------------

// FunctionFor renders a tool declaration (provider params deep-merged in).
func FunctionFor(t core.Tool) map[string]any {
	schema := t.ParamsSchema()
	if len(schema) == 0 {
		schema = DefaultInputSchema()
	}
	var desc any
	if d := t.Description(); d != "" {
		desc = d
	}
	decl := map[string]any{"name": t.Name(), "description": desc, "input_schema": schema}
	if pp, ok := t.(core.ProviderParamser); ok && len(pp.ProviderParams()) > 0 {
		return core.DeepMerge(decl, pp.ProviderParams())
	}
	return decl
}

// ParseToolCalls reads tool_use blocks keyed by id (nil when none).
func ParseToolCalls(blocks []any) *core.ToolCalls {
	out := &core.ToolCalls{}
	for _, b := range blocks {
		bm, ok := b.(map[string]any)
		if !ok || len(bm) == 0 || bm["type"] != "tool_use" {
			continue
		}
		id, _ := bm["id"].(string)
		name, _ := bm["name"].(string)
		out.Set(id, &core.ToolCall{ID: id, Name: name, Arguments: bm["input"]})
	}
	if out.Len() == 0 {
		return nil
	}
	return out
}

// BuildToolChoice renders tool_choice: auto/none as is, required as "any",
// anything else as a specific tool; Calls sets disable_parallel_tool_use.
func BuildToolChoice(prefs core.ToolPrefs) map[string]any {
	choice := prefs.Choice
	if choice == "" {
		choice = "auto"
	}
	var typ string
	switch choice {
	case "auto", "none":
		typ = choice
	case "required":
		typ = "any"
	default:
		typ = "tool"
	}
	out := map[string]any{"type": typ}
	if typ == "tool" {
		out["name"] = choice
	}
	if typ != "none" && prefs.Calls != "" {
		out["disable_parallel_tool_use"] = prefs.Calls == "one"
	}
	return out
}

// --- thinking ----------------------------------------------------------------

func addThinkingFields(payload map[string]any, thinking *core.ThinkingConfig, model *core.Info) error {
	tp, err := BuildThinkingPayload(thinking, model)
	if err != nil || tp == nil {
		return err
	}
	if t, ok := tp["thinking"]; ok && core.Truthy(t) {
		payload["thinking"] = t
	}
	if oc, ok := tp["output_config"].(map[string]any); ok && len(oc) > 0 {
		mergeOutputConfig(payload, oc)
	}
	return nil
}

// BuildThinkingPayload renders extended thinking: a budget becomes
// {type: enabled, budget_tokens} on models with a budget_tokens option; an
// effort becomes adaptive thinking plus output_config.effort on models with
// an effort option. Effort "none" disables thinking.
func BuildThinkingPayload(thinking *core.ThinkingConfig, model *core.Info) (map[string]any, error) {
	if !thinking.Enabled() {
		return nil, nil
	}
	effort := thinking.Effort
	if effort == "none" {
		return nil, nil
	}
	if b := thinking.Budget; b != nil && *b != 0 {
		if model.ReasoningOption("budget_tokens") != nil {
			return map[string]any{"thinking": map[string]any{"type": "enabled", "budget_tokens": *b}}, nil
		}
		return nil, fmt.Errorf("Anthropic thinking budget is not supported for %s", model.ID)
	}
	if effort == "" {
		return nil, errors.New("Anthropic adaptive thinking requires an effort")
	}
	if model.ReasoningOption("effort") != nil {
		return map[string]any{"thinking": map[string]any{"type": "adaptive"}, "output_config": map[string]any{"effort": effort}}, nil
	}
	return nil, fmt.Errorf("Anthropic thinking effort is not supported for %s", model.ID)
}

func buildThinkingBlock(t *core.Thinking) map[string]any {
	if t == nil {
		return nil
	}
	if t.Text != "" {
		b := map[string]any{"type": "thinking", "thinking": t.Text}
		if t.Signature != "" {
			b["signature"] = t.Signature
		}
		return b
	}
	if t.Signature != "" {
		return map[string]any{"type": "redacted_thinking", "data": t.Signature}
	}
	return nil
}

func prependThinking(blocks []any, m *core.Message, thinkingEnabled bool) []any {
	if !thinkingEnabled {
		return blocks
	}
	if b := buildThinkingBlock(m.Thinking); b != nil {
		return append([]any{b}, blocks...)
	}
	return blocks
}

// --- streaming ---------------------------------------------------------------

// BuildChunk reads one streamed event (message_start, content_block_start,
// content_block_delta, message_delta, ...). The event name is also in the
// data's "type", so the standard SSE flow suffices.
func (a *Protocol) BuildChunk(data map[string]any) (*core.Chunk, error) {
	deltaType := core.DigString(data, "delta", "type")
	var content any
	if deltaType == "text_delta" {
		content = core.Dig(data, "delta", "text")
	}
	var thinkingText, signature string
	if deltaType == "thinking_delta" {
		thinkingText = core.DigString(data, "delta", "thinking")
	}
	if deltaType == "signature_delta" {
		signature = core.DigString(data, "delta", "signature")
	}
	modelID := core.DigString(data, "message", "model")
	finish := core.DigString(data, "delta", "stop_reason")
	tokens := core.BuildTokens(
		core.AsInt(core.Dig(data, "message", "usage", "input_tokens")),
		core.AsInt(core.Or(core.Dig(data, "message", "usage", "output_tokens"), core.Dig(data, "usage", "output_tokens"))),
		cachedTokens(data),
		cacheCreationTokens(data),
		nil,
	)
	return &core.Chunk{
		Role:         core.RoleAssistant,
		ModelID:      modelID,
		Content:      content,
		Citations:    citationsDelta(data, deltaType),
		Thinking:     core.BuildThinking(thinkingText, signature),
		Tokens:       tokens,
		ToolCalls:    toolCallsStream(data),
		FinishReason: finish,
	}, nil
}

func citationsDelta(data map[string]any, deltaType string) []core.Citation {
	if deltaType != "citations_delta" {
		return nil
	}
	c := core.DigMap(data, "delta", "citation")
	if len(c) == 0 {
		return nil
	}
	return []core.Citation{ParseCitation(c, nil, nil, nil)}
}

// toolCallsStream keys streamed tool calls by content block index so the
// accumulator can join input_json_delta fragments to their tool_use start.
func toolCallsStream(data map[string]any) *core.ToolCalls {
	index := ""
	if i := core.AsInt(data["index"]); i != nil {
		index = fmt.Sprint(*i)
	}
	if data["type"] == "content_block_delta" && core.DigString(data, "delta", "type") == "input_json_delta" {
		out := &core.ToolCalls{}
		out.Set(index, &core.ToolCall{Arguments: core.Dig(data, "delta", "partial_json")})
		return out
	}
	calls := ParseToolCalls(core.ToSafeArray(data["content_block"]))
	if data["type"] == "content_block_start" && calls != nil && index != "" {
		out := &core.ToolCalls{}
		out.Set(index, calls.List()[0])
		return out
	}
	return calls
}

// --- citations and usage -----------------------------------------------------

// ParseCitation normalises a citation (char/page location, web search or
// search_result). end_page_number is exclusive upstream, inclusive here.
func ParseCitation(data map[string]any, text *string, start, end *int) core.Citation {
	c := core.Citation{Text: text, StartIndex: start, EndIndex: end}
	if u, ok := core.Or(data["url"], data["source"]).(string); ok && (strings.HasPrefix(u, "http://") || strings.HasPrefix(u, "https://")) {
		c.URL = &u
	}
	if t, ok := core.Or(data["document_title"], data["title"]).(string); ok {
		c.Title = &t
	}
	if s, ok := data["cited_text"].(string); ok {
		c.CitedText = &s
	}
	c.SourceIndex = core.AsInt(core.Or(data["document_index"], data["search_result_index"]))
	c.StartPage = core.AsInt(data["start_page_number"])
	if ep := core.AsInt(data["end_page_number"]); ep != nil && *ep != 0 {
		v := *ep - 1
		c.EndPage = &v
	}
	return c
}

func cachedTokens(data map[string]any) *int {
	return core.AsInt(core.Or(
		core.Dig(data, "message", "usage", "cache_read_input_tokens"),
		core.Dig(data, "usage", "cache_read_input_tokens"),
	))
}

func cacheCreationTokens(data map[string]any) *int {
	direct := core.Or(
		core.Dig(data, "message", "usage", "cache_creation_input_tokens"),
		core.Dig(data, "usage", "cache_creation_input_tokens"),
	)
	if core.Truthy(direct) {
		return core.AsInt(direct)
	}
	breakdown := core.Or(core.Dig(data, "message", "usage", "cache_creation"), core.Dig(data, "usage", "cache_creation"))
	if m, ok := breakdown.(map[string]any); ok {
		sum := 0
		for _, v := range m {
			if core.Truthy(v) {
				sum += core.Deref(core.AsInt(v))
			}
		}
		return &sum
	}
	return nil
}

// --- models ------------------------------------------------------------------

// ListModels implements core.ModelLister.
func (a *Protocol) ListModels(ctx context.Context) ([]*core.Info, error) {
	resp, err := a.Provider.Connection().Get(ctx, a.ModelsURL(), nil)
	if err != nil {
		return nil, err
	}
	return ParseListModels(resp, a.Provider.Slug(), a.Provider.Capabilities()), nil
}

// ParseListModels reads a v1/models listing.
func ParseListModels(resp *core.Response, slug string, caps core.Capabilities) []*core.Info {
	var out []*core.Info
	for _, raw := range core.ToSafeArray(core.Dig(resp.Body, "data")) {
		md, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		id, _ := md["id"].(string)
		capList := []any{}
		if caps != nil {
			for _, c := range caps.CriticalCapabilitiesFor(id) {
				capList = append(capList, c)
			}
		}
		out = append(out, core.NewInfo(map[string]any{
			"id": id, "name": core.Or(md["display_name"], id), "provider": slug,
			"created_at": md["created_at"], "capabilities": capList, "metadata": map[string]any{},
		}))
	}
	return out
}

func convertRole(role string) string {
	if role == core.RoleTool || role == core.RoleUser {
		return "user"
	}
	return "assistant"
}
