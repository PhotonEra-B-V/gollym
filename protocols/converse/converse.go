// Package converse is the AWS Bedrock Converse API (pyllym
// protocols/converse.py). Requests are SigV4-signed by the bedrock
// provider's Sign hook, which the Connection applies to every request, so no
// sync_response override is needed here. Streaming (ConverseStream) uses the
// AWS event-stream binary framing and is handled by StreamResponse.
package converse

import (
	"errors"
	"log/slog"
	"path"
	"strconv"
	"strings"
	"unicode"

	"github.com/PhotonEra-B-V/gollym/core"
)

// SupportedDocumentFormats are the document formats Converse accepts.
var SupportedDocumentFormats = map[string]bool{
	"pdf": true, "csv": true, "doc": true, "docx": true, "xls": true,
	"xlsx": true, "html": true, "txt": true, "md": true,
}

// DefaultInputSchema is sent for tools without parameters.
func DefaultInputSchema() map[string]any {
	return map[string]any{"type": "object", "properties": map[string]any{}, "required": []any{}}
}

// Protocol speaks Converse for one provider and model.
type Protocol struct {
	Provider core.Provider
	Model    *core.Info
}

// New is the core.ProtocolFactory.
func New(p core.Provider, model *core.Info) core.Protocol {
	return &Protocol{Provider: p, Model: model}
}

// ReasoningEmbedded reports metadata.converse.reasoningSupported.embedded:
// such models take effort inside reasoning_config.
func ReasoningEmbedded(model *core.Info) bool {
	if model == nil {
		return false
	}
	return core.Truthy(core.Dig(model.Metadata, "converse", "reasoningSupported", "embedded"))
}

// --- endpoints ---------------------------------------------------------------

func (c *Protocol) modelID() string {
	if c.Model == nil {
		return ""
	}
	return c.Model.ID
}

func (c *Protocol) CompletionURL() string { return "/model/" + c.modelID() + "/converse" }
func (c *Protocol) StreamURL() string     { return "/model/" + c.modelID() + "/converse-stream" }

// --- render ------------------------------------------------------------------

// renderer holds per-render state (document names must be unique within one
// request).
type renderer struct {
	proto     *Protocol
	usedNames map[string]int
}

// RenderPayload builds the Converse body.
func (c *Protocol) RenderPayload(req *core.CompletionRequest) (map[string]any, error) {
	if req.Model == nil {
		return nil, errors.New("converse: model is required")
	}
	if req.Citations {
		slog.Warn("gollym does not support citations on Bedrock yet. Ignoring WithCitations.", "model", req.Model.ID)
	}
	r := &renderer{proto: c, usedNames: map[string]int{}}
	var system, chat []*core.Message
	for _, m := range req.Messages {
		if m.Role == core.RoleSystem {
			system = append(system, m)
		} else {
			chat = append(chat, m)
		}
	}
	msgs, err := r.formatMessages(chat)
	if err != nil {
		return nil, err
	}
	payload := map[string]any{"messages": msgs}
	systemBlocks, err := r.formatSystem(system)
	if err != nil {
		return nil, err
	}
	if len(systemBlocks) > 0 {
		payload["system"] = systemBlocks
	}
	inference := map[string]any{}
	if req.Temperature != nil {
		inference["temperature"] = *req.Temperature
	}
	payload["inferenceConfig"] = inference
	if tc := formatToolConfig(req.Tools, req.ToolPrefs); tc != nil {
		payload["toolConfig"] = tc
	}
	model := c.Model
	if model == nil {
		model = req.Model
	}
	if fields := reasoningFields(req.Thinking, model); fields != nil {
		payload["additionalModelRequestFields"] = fields
	}
	if oc := outputConfig(req.Schema); oc != nil {
		payload["outputConfig"] = oc
	}
	return payload, nil
}

// --- parse -------------------------------------------------------------------

// ParseCompletionResponse reads a Converse response.
func (c *Protocol) ParseCompletionResponse(resp *core.Response) (*core.Message, error) {
	msg := parseBody(resp.Body, resp)
	if msg == nil {
		return nil, core.NewError(core.KindError, resp, "Empty or unparseable completion response")
	}
	return msg, nil
}

func parseBody(data any, raw *core.Response) *core.Message {
	if !core.Truthy(data) {
		return nil
	}
	body, ok := data.(map[string]any)
	if !ok {
		return nil
	}
	blocks := blockMaps(core.Dig(body, "output", "message", "content"))
	usage, _ := body["usage"].(map[string]any)
	thinkingText, thinkingSig := parseThinking(blocks)
	calls := parseToolCalls(blocks)
	modelID, _ := body["modelId"].(string)
	finish, _ := body["stopReason"].(string)
	m := &core.Message{
		Role:         core.RoleAssistant,
		Content:      parseTextContent(blocks),
		Thinking:     core.BuildThinking(thinkingText, thinkingSig),
		ToolCalls:    calls,
		Tokens:       core.BuildTokens(InputTokens(usage), core.AsInt(usage["outputTokens"]), core.AsInt(usage["cacheReadInputTokens"]), core.AsInt(usage["cacheWriteInputTokens"]), ReasoningTokens(usage)),
		FinishReason: finish,
		ModelID:      modelID,
		Raw:          raw,
	}
	if m.Content == nil && calls.Len() > 0 {
		m.Content = ""
	}
	return m
}

func blockMaps(v any) []map[string]any {
	var out []map[string]any
	for _, b := range core.ToSafeArray(v) {
		if m, ok := b.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

func parseTextContent(blocks []map[string]any) any {
	var sb strings.Builder
	for _, b := range blocks {
		if s, ok := b["text"].(string); ok {
			sb.WriteString(s)
		}
	}
	if sb.Len() == 0 {
		return nil
	}
	return sb.String()
}

func parseThinking(blocks []map[string]any) (string, string) {
	var text strings.Builder
	var signature *string
	for _, b := range blocks {
		t, sig := parseReasoningBlock(b)
		text.WriteString(t)
		if signature == nil {
			signature = sig
		}
	}
	return text.String(), core.Deref(signature)
}

func parseReasoningBlock(b map[string]any) (string, *string) {
	rc, ok := b["reasoningContent"].(map[string]any)
	if !ok {
		return "", nil
	}
	rt, _ := rc["reasoningText"].(map[string]any)
	text, _ := rt["text"].(string)
	var signature *string
	if s, ok := rt["signature"].(string); ok {
		signature = &s
	}
	if signature == nil {
		if s, ok := rc["redactedContent"].(string); ok {
			signature = &s
		}
	}
	return text, signature
}

func parseToolCalls(blocks []map[string]any) *core.ToolCalls {
	out := &core.ToolCalls{}
	for _, b := range blocks {
		tu, _ := b["toolUse"].(map[string]any)
		if len(tu) == 0 {
			continue
		}
		id, _ := tu["toolUseId"].(string)
		name, _ := tu["name"].(string)
		out.Set(id, &core.ToolCall{ID: id, Name: name, Arguments: core.Or(tu["input"], map[string]any{})})
	}
	if out.Len() == 0 {
		return nil
	}
	return out
}

// --- message formatting ------------------------------------------------------

func (r *renderer) formatMessages(messages []*core.Message) ([]any, error) {
	rendered := []any{}
	var toolResults []any
	for _, m := range messages {
		if m.IsToolResult() {
			block, err := r.formatToolResultBlock(m)
			if err != nil {
				return nil, err
			}
			toolResults = append(toolResults, block)
			continue
		}
		if len(toolResults) > 0 {
			rendered = append(rendered, map[string]any{"role": "user", "content": toolResults})
			toolResults = nil
		}
		content, err := r.formatMessageContent(m)
		if err != nil {
			return nil, err
		}
		if len(content) > 0 {
			rendered = append(rendered, map[string]any{"role": formatRole(m.Role), "content": content})
		}
	}
	if len(toolResults) > 0 {
		rendered = append(rendered, map[string]any{"role": "user", "content": toolResults})
	}
	return rendered, nil
}

func (r *renderer) formatMessageContent(m *core.Message) ([]any, error) {
	if raw, ok := m.Content.(core.RawContent); ok {
		blocks := rawBlocks(raw.Value)
		if m.Role == core.RoleAssistant {
			return blocks, nil
		}
		// Reasoning blocks are only valid in assistant turns.
		out := []any{}
		for _, b := range blocks {
			if bm, ok := b.(map[string]any); ok {
				if _, has := bm["reasoningContent"]; !has {
					out = append(out, bm)
				}
			}
		}
		return out, nil
	}
	var blocks []any
	if m.Role == core.RoleAssistant {
		if tb := formatThinkingBlock(m.Thinking); tb != nil {
			blocks = append(blocks, tb)
		}
	}
	content, err := r.formatContent(m.Content)
	if err != nil {
		return nil, err
	}
	blocks = append(blocks, content...)
	for _, tc := range m.ToolCalls.List() {
		args := tc.Arguments
		if args == nil {
			args = map[string]any{}
		}
		blocks = append(blocks, map[string]any{"toolUse": map[string]any{"toolUseId": tc.ID, "name": tc.Name, "input": args}})
	}
	return blocks, nil
}

// rawBlocks is a RawContent value as a block list.
func rawBlocks(v any) []any {
	switch x := v.(type) {
	case []any:
		return x
	case []map[string]any:
		out := make([]any, len(x))
		for i, b := range x {
			out[i] = b
		}
		return out
	}
	return []any{v}
}

func formatRole(role string) string {
	if role == core.RoleAssistant {
		return "assistant"
	}
	return "user"
}

func (r *renderer) formatSystem(messages []*core.Message) ([]any, error) {
	var out []any
	for _, m := range messages {
		blocks, err := r.formatContent(m.Content)
		if err != nil {
			return nil, err
		}
		out = append(out, blocks...)
	}
	return out, nil
}

// --- tool results ------------------------------------------------------------

func (r *renderer) formatToolResultBlock(m *core.Message) (map[string]any, error) {
	content, err := formatToolResultContent(m.Content)
	if err != nil {
		return nil, err
	}
	return map[string]any{"toolResult": map[string]any{"toolUseId": m.ToolCallID, "content": content}}, nil
}

func formatToolResultContent(content any) ([]any, error) {
	switch v := content.(type) {
	case core.RawContent:
		return formatRawToolResult(v.Value), nil
	case map[string]any, []any:
		return []any{map[string]any{"json": v}}, nil
	case *core.SearchResults:
		return contentToolResult(&v.Content)
	case *core.Content:
		return contentToolResult(v)
	case nil:
		return []any{textToolResultBlock("")}, nil
	case string:
		return []any{textToolResultBlock(v)}, nil
	}
	return []any{textToolResultBlock(core.MustJSON(content))}, nil
}

func contentToolResult(c *core.Content) ([]any, error) {
	var blocks []any
	if c.Text != nil && *c.Text != "" {
		blocks = append(blocks, textToolResultBlock(*c.Text))
	}
	for _, a := range c.Attachments {
		s, err := a.ForLLM()
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, textToolResultBlock(s))
	}
	if len(blocks) == 0 {
		blocks = []any{textToolResultBlock("")}
	}
	return blocks, nil
}

func formatRawToolResult(value any) []any {
	var out []any
	for _, b := range rawBlocks(value) {
		bm, ok := b.(map[string]any)
		if !ok {
			continue
		}
		for _, k := range []string{"text", "json", "document", "image"} {
			if _, has := bm[k]; has {
				out = append(out, bm)
				break
			}
		}
	}
	if len(out) == 0 {
		return []any{map[string]any{"text": pyStr(value)}}
	}
	return out
}

// pyStr approximates Python's str() of a raw value.
func pyStr(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return core.PyJSON(v)
}

func textToolResultBlock(text string) map[string]any {
	if text == "" {
		text = "(no output)"
	}
	return map[string]any{"text": text}
}

// --- content and media -------------------------------------------------------

func (r *renderer) formatContent(content any) ([]any, error) {
	switch v := content.(type) {
	case nil:
		return nil, nil
	case string:
		if v == "" {
			return nil, nil
		}
		return []any{map[string]any{"text": v}}, nil
	case core.RawContent:
		return rawBlocks(v.Value), nil
	case map[string]any:
		if len(v) == 0 {
			return nil, nil
		}
		return []any{map[string]any{"text": core.PyJSON(v)}}, nil
	case []any:
		if len(v) == 0 {
			return nil, nil
		}
		return []any{map[string]any{"text": core.PyJSON(v)}}, nil
	case *core.SearchResults:
		return r.formatContentObject(&v.Content)
	case *core.Content:
		return r.formatContentObject(v)
	}
	return []any{map[string]any{"text": content}}, nil
}

func (r *renderer) formatContentObject(c *core.Content) ([]any, error) {
	var blocks []any
	if c.Text != nil && *c.Text != "" {
		blocks = append(blocks, map[string]any{"text": *c.Text})
	}
	for _, a := range c.Attachments {
		b, err := r.formatAttachment(a)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, b)
	}
	return blocks, nil
}

func (r *renderer) formatAttachment(a *core.Attachment) (map[string]any, error) {
	if a.IsProviderFile() {
		return r.formatProviderFile(a)
	}
	switch a.Type() {
	case "image":
		enc, err := a.Encoded()
		if err != nil {
			return nil, err
		}
		return map[string]any{"image": map[string]any{"format": a.Format(), "source": map[string]any{"bytes": enc}}}, nil
	case "pdf", "document":
		if !supportedDocument(a) {
			return nil, &core.UnsupportedAttachmentError{Type: a.MIMEType}
		}
		enc, err := a.Encoded()
		if err != nil {
			return nil, err
		}
		return map[string]any{"document": map[string]any{
			"format": documentFormat(a), "name": r.uniqueDocumentName(SanitizeDocumentName(a.Filename)),
			"source": map[string]any{"bytes": enc},
		}}, nil
	case "text":
		s, err := a.ForLLM()
		if err != nil {
			return nil, err
		}
		return map[string]any{"text": s}, nil
	}
	return nil, &core.UnsupportedAttachmentError{Type: a.MIMEType}
}

func (r *renderer) formatProviderFile(a *core.Attachment) (map[string]any, error) {
	if !supportedDocument(a) {
		return nil, &core.UnsupportedAttachmentError{Type: a.MIMEType}
	}
	uri := a.ProviderFileURI()
	if uri == "" {
		uri = a.ProviderFileID()
	}
	return map[string]any{"document": map[string]any{
		"format": documentFormat(a), "name": r.uniqueDocumentName(SanitizeDocumentName(a.Filename)),
		"source": map[string]any{"s3Location": map[string]any{"uri": uri}},
	}}, nil
}

// uniqueDocumentName suffixes repeated names: report, report_2, report_3.
func (r *renderer) uniqueDocumentName(base string) string {
	count := r.usedNames[base]
	r.usedNames[base] = count + 1
	if count == 0 {
		return base
	}
	return base + "_" + strconv.Itoa(count+1)
}

func documentFormat(a *core.Attachment) string {
	if ext := a.Extension(); ext != "" {
		return ext
	}
	return a.Format()
}

func supportedDocument(a *core.Attachment) bool { return SupportedDocumentFormats[documentFormat(a)] }

// SanitizeDocumentName keeps the base name without extension, replacing
// anything but letters, digits, _ and - with _ ("document" when empty).
func SanitizeDocumentName(filename string) string {
	base := ""
	if filename != "" && !strings.HasSuffix(filename, "/") {
		base = splitExt(path.Base(filename))
	}
	var sb strings.Builder
	for _, r := range base {
		if unicode.IsLetter(r) || unicode.IsNumber(r) || r == '_' || r == '-' {
			sb.WriteRune(r)
		} else {
			sb.WriteRune('_')
		}
	}
	if sb.Len() == 0 {
		return "document"
	}
	return sb.String()
}

// splitExt drops the extension like os.path.splitext (leading dots are not
// an extension).
func splitExt(name string) string {
	i := strings.LastIndex(name, ".")
	if i <= 0 || strings.Trim(name[:i], ".") == "" {
		return name
	}
	return name[:i]
}

// --- tools -------------------------------------------------------------------

func formatToolConfig(tools *core.ToolSet, prefs core.ToolPrefs) map[string]any {
	if tools.Len() == 0 {
		return nil
	}
	var specs []any
	for _, t := range tools.List() {
		specs = append(specs, ToolFor(t))
	}
	config := map[string]any{"tools": specs}
	if prefs.Choice == "" {
		return config
	}
	if choice := toolChoice(prefs.Choice); choice != nil {
		config["toolChoice"] = choice
	}
	return config
}

func toolChoice(choice string) map[string]any {
	switch choice {
	case "auto":
		return map[string]any{"auto": map[string]any{}}
	case "none":
		return nil
	case "required":
		return map[string]any{"any": map[string]any{}}
	}
	return map[string]any{"tool": map[string]any{"name": choice}}
}

// ToolFor renders a toolSpec (provider params deep-merged over it).
func ToolFor(t core.Tool) map[string]any {
	schema := t.ParamsSchema()
	if len(schema) == 0 {
		schema = DefaultInputSchema()
	}
	var desc any
	if d := t.Description(); d != "" {
		desc = d
	}
	spec := map[string]any{"toolSpec": map[string]any{
		"name": t.Name(), "description": desc, "inputSchema": map[string]any{"json": schema},
	}}
	if pp, ok := t.(core.ProviderParamser); ok && len(pp.ProviderParams()) > 0 {
		return core.DeepMerge(spec, pp.ProviderParams())
	}
	return spec
}

// --- thinking ----------------------------------------------------------------

func reasoningFields(t *core.ThinkingConfig, model *core.Info) map[string]any {
	if !t.Enabled() {
		return nil
	}
	if t.Effort != "" && t.Effort != "none" {
		if ReasoningEmbedded(model) {
			return map[string]any{"reasoning_config": map[string]any{"type": "enabled", "reasoning_effort": t.Effort}}
		}
		return map[string]any{"reasoning_effort": t.Effort}
	}
	if t.Budget == nil {
		return nil
	}
	return map[string]any{"reasoning_config": map[string]any{"type": "enabled", "budget_tokens": *t.Budget}}
}

func formatThinkingBlock(t *core.Thinking) map[string]any {
	if t == nil {
		return nil
	}
	if t.Text != "" {
		rt := map[string]any{"text": t.Text}
		if t.Signature != "" {
			rt["signature"] = t.Signature
		}
		return map[string]any{"reasoningContent": map[string]any{"reasoningText": rt}}
	}
	if t.Signature != "" {
		return map[string]any{"reasoningContent": map[string]any{"redactedContent": t.Signature}}
	}
	return nil
}

// --- structured output -------------------------------------------------------

func outputConfig(schema map[string]any) map[string]any {
	if len(schema) == 0 {
		return nil
	}
	cleaned := core.DeepCopy(schema["schema"])
	if m, ok := cleaned.(map[string]any); ok {
		delete(m, "strict")
	}
	return map[string]any{"textFormat": map[string]any{
		"type": "json_schema",
		"structure": map[string]any{"jsonSchema": map[string]any{
			"schema": core.PyJSON(cleaned), "name": schema["name"],
		}},
	}}
}

// --- usage -------------------------------------------------------------------

// InputTokens is inputTokens minus cache reads and writes (never negative).
func InputTokens(usage map[string]any) *int {
	in := core.AsInt(usage["inputTokens"])
	if in == nil {
		return nil
	}
	read := core.Deref(core.AsInt(usage["cacheReadInputTokens"]))
	write := core.Deref(core.AsInt(usage["cacheWriteInputTokens"]))
	n := max(*in-read-write, 0)
	return &n
}

// ReasoningTokens reads reasoningTokens or outputTokensDetails.reasoningTokens.
func ReasoningTokens(usage map[string]any) *int {
	return core.AsInt(core.Or(usage["reasoningTokens"], core.Dig(usage, "outputTokensDetails", "reasoningTokens")))
}
