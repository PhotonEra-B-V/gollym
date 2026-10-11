// Package gemini is the Google Gemini generateContent wire protocol (pyllym
// protocols/gemini.py). Vertex AI speaks it too, over its own
// project/location-scoped paths (see ModelPath and EmbedMethod).
package gemini

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"log/slog"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/google/uuid"

	"github.com/PhotonEra-B-V/gollym/core"
)

// Protocol speaks Gemini generateContent for one provider.
type Protocol struct {
	Provider core.Provider
	Model    *core.Info
	// ModelPath maps a model id to its resource path: "models/{id}" for the
	// Gemini API; Vertex AI uses
	// "projects/{p}/locations/{l}/publishers/google/models/{id}".
	ModelPath func(modelID string) string
	// EmbedMethod is the embedding RPC appended to the model path
	// ("batchEmbedContents"; Vertex AI uses "predict").
	EmbedMethod string
}

// New is the core.ProtocolFactory for the Gemini API.
func New(p core.Provider, model *core.Info) core.Protocol {
	return &Protocol{Provider: p, Model: model}
}

func (g *Protocol) conn() *core.Connection { return g.Provider.Connection() }

// --- endpoints ---------------------------------------------------------------

func (g *Protocol) modelPath(id string) string {
	if g.ModelPath != nil {
		return g.ModelPath(id)
	}
	return "models/" + id
}

func (g *Protocol) modelID() string {
	if g.Model == nil {
		return ""
	}
	return g.Model.ID
}

func (g *Protocol) CompletionURL() string { return g.modelPath(g.modelID()) + ":generateContent" }
func (g *Protocol) StreamURL() string {
	return g.modelPath(g.modelID()) + ":streamGenerateContent?alt=sse"
}
func (g *Protocol) ModelsURL() string { return "models" }

// EmbeddingURL is the embedding endpoint for model.
func (g *Protocol) EmbeddingURL(model string) string {
	method := g.EmbedMethod
	if method == "" {
		method = "batchEmbedContents"
	}
	return g.modelPath(model) + ":" + method
}

// --- chat render -------------------------------------------------------------

// RenderPayload builds the generateContent body.
func (g *Protocol) RenderPayload(req *core.CompletionRequest) (map[string]any, error) {
	model := req.Model
	if model == nil {
		return nil, errors.New("gemini: model is required")
	}
	if req.Citations && !model.IsCitations() {
		slog.Warn(model.ID + " does not support citations according to the model registry. " +
			"Gemini citations come from Google Search grounding: " +
			"with_params(tools=[{'google_search': {}}]).")
	}
	var rest []*core.Message
	for _, m := range req.Messages {
		if m.Role != core.RoleSystem {
			rest = append(rest, m)
		}
	}
	contents, err := g.FormatMessages(rest)
	if err != nil {
		return nil, err
	}
	genConfig := map[string]any{}
	payload := map[string]any{"contents": contents, "generationConfig": genConfig}
	if sys := SystemInstructionText(req.Messages); sys != "" {
		payload["systemInstruction"] = map[string]any{"parts": []any{map[string]any{"text": sys}}}
	}
	if req.Temperature != nil {
		genConfig["temperature"] = *req.Temperature
	}
	if len(req.Schema) > 0 {
		for k, v := range StructuredOutputConfig(req.Schema, model) {
			genConfig[k] = v
		}
	}
	if req.Thinking.Enabled() {
		genConfig["thinkingConfig"] = BuildThinkingConfig(req.Thinking)
	}
	if req.Tools.Len() > 0 {
		tools, err := FormatTools(req.Tools.List())
		if err != nil {
			return nil, err
		}
		payload["tools"] = tools
		// Gemini doesn't support controlling parallel tool calls.
		if req.ToolPrefs.Choice != "" {
			payload["toolConfig"] = BuildToolConfig(req.ToolPrefs.Choice)
		}
	}
	return payload, nil
}

// BuildThinkingConfig renders generationConfig.thinkingConfig.
func BuildThinkingConfig(t *core.ThinkingConfig) map[string]any {
	config := map[string]any{"includeThoughts": true}
	if t.Effort != "" {
		config["thinkingLevel"] = t.Effort
	}
	if t.Budget != nil {
		config["thinkingBudget"] = *t.Budget
	}
	return config
}

// --- chat parse --------------------------------------------------------------

// ParseCompletionResponse reads a generateContent response.
func (g *Protocol) ParseCompletionResponse(resp *core.Response) (*core.Message, error) {
	data, ok := resp.Body.(map[string]any)
	if !ok {
		return nil, core.NewError(core.KindError, resp, "Empty or unparseable completion response")
	}
	parts := partsOf(data)
	content := parseContent(data)
	modelID, _ := core.Or(data["modelVersion"], g.modelID()).(string)
	return &core.Message{
		Role:         core.RoleAssistant,
		Content:      content,
		Citations:    ExtractCitations(data, content),
		Thinking:     core.BuildThinking(thoughtText(parts), thoughtSignature(parts)),
		ToolCalls:    ExtractToolCalls(data),
		Tokens:       core.BuildTokens(InputTokens(data), core.Int(outputTokenSum(data)), cachedTokens(data), nil, thoughtsTokens(data)),
		FinishReason: core.DigString(data, "candidates", 0, "finishReason"),
		ModelID:      modelID,
		Raw:          resp,
	}, nil
}

func partsOf(data any) []any { return core.DigSlice(data, "candidates", 0, "content", "parts") }

func partMaps(parts []any) []map[string]any {
	out := make([]map[string]any, 0, len(parts))
	for _, p := range parts {
		if m, ok := p.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

// InputTokens is promptTokenCount minus cachedContentTokenCount (nil when
// the prompt count is missing).
func InputTokens(data map[string]any) *int {
	prompt := core.AsInt(core.Dig(data, "usageMetadata", "promptTokenCount"))
	if prompt == nil {
		return nil
	}
	cached := core.Deref(core.AsInt(core.Dig(data, "usageMetadata", "cachedContentTokenCount")))
	return core.Int(max(*prompt-cached, 0))
}

// outputTokenSum is candidatesTokenCount + thoughtsTokenCount.
func outputTokenSum(data map[string]any) int {
	candidates := core.Deref(core.AsInt(core.Dig(data, "usageMetadata", "candidatesTokenCount")))
	thoughts := core.Deref(core.AsInt(core.Dig(data, "usageMetadata", "thoughtsTokenCount")))
	return candidates + thoughts
}

func cachedTokens(data map[string]any) *int {
	return core.AsInt(core.Dig(data, "usageMetadata", "cachedContentTokenCount"))
}

func thoughtsTokens(data map[string]any) *int {
	return core.AsInt(core.Dig(data, "usageMetadata", "thoughtsTokenCount"))
}

func parseContent(data map[string]any) any {
	candidate := core.Dig(data, "candidates", 0)
	if !core.Truthy(candidate) || hasFunctionCall(candidate) {
		return ""
	}
	parts := core.DigSlice(candidate, "content", "parts")
	if len(parts) == 0 {
		return ""
	}
	var nonThought []map[string]any
	for _, p := range partMaps(parts) {
		if !core.Truthy(p["thought"]) {
			nonThought = append(nonThought, p)
		}
	}
	if len(nonThought) == 0 {
		return ""
	}
	return buildResponseContent(nonThought)
}

func hasFunctionCall(candidate any) bool {
	for _, p := range partMaps(core.DigSlice(candidate, "content", "parts")) {
		if core.Truthy(p["functionCall"]) {
			return true
		}
	}
	return false
}

func buildResponseContent(parts []map[string]any) any {
	var text strings.Builder
	var attachments []*core.Attachment
	for i, part := range parts {
		switch {
		case core.Truthy(part["text"]):
			s, _ := part["text"].(string)
			text.WriteString(s)
		case core.Truthy(part["inlineData"]):
			if a := inlineAttachment(core.DigMap(part, "inlineData"), i); a != nil {
				attachments = append(attachments, a)
			}
		case core.Truthy(part["fileData"]):
			if a := fileAttachment(core.DigMap(part, "fileData"), i); a != nil {
				attachments = append(attachments, a)
			}
		}
	}
	var joined *string
	if text.Len() > 0 {
		joined = core.Str(text.String())
	}
	if len(attachments) == 0 {
		if joined == nil {
			return nil
		}
		return *joined
	}
	return &core.Content{Text: joined, Attachments: attachments}
}

func inlineAttachment(inline map[string]any, index int) *core.Attachment {
	encoded, _ := inline["data"].(string)
	if encoded == "" {
		return nil
	}
	decoded, err := decodeBase64(encoded)
	if err != nil {
		slog.Warn("Failed to decode Gemini inline data attachment", "err", err)
		return nil
	}
	mime, _ := inline["mimeType"].(string)
	a, err := core.NewAttachment(decoded, attachmentFilename(mime, index))
	if err != nil {
		return nil
	}
	return a
}

// decodeBase64 is lenient like Python's b64decode (non-alphabet characters
// such as newlines are discarded; URL-safe and unpadded input accepted).
func decodeBase64(s string) ([]byte, error) {
	clean := strings.Map(func(r rune) rune {
		if r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '+' || r == '/' || r == '=' || r == '-' || r == '_' {
			return r
		}
		return -1
	}, s)
	if b, err := base64.StdEncoding.DecodeString(clean); err == nil {
		return b, nil
	}
	if b, err := base64.URLEncoding.DecodeString(clean); err == nil {
		return b, nil
	}
	return base64.RawStdEncoding.DecodeString(strings.TrimRight(clean, "="))
}

func fileAttachment(file map[string]any, index int) *core.Attachment {
	uri, _ := file["fileUri"].(string)
	if uri == "" {
		return nil
	}
	name, _ := file["filename"].(string)
	if name == "" {
		mime, _ := file["mimeType"].(string)
		name = attachmentFilename(mime, index)
	}
	a, err := core.NewAttachment(uri, name)
	if err != nil {
		return nil
	}
	return a
}

func attachmentFilename(mime string, index int) string {
	if mime == "" {
		return fmt.Sprintf("gemini_attachment_%d", index+1)
	}
	parts := strings.Split(mime, "/")
	ext := parts[len(parts)-1]
	switch ext {
	case "jpeg":
		ext = "jpg"
	case "plain":
		ext = "txt"
	}
	ext = strings.ReplaceAll(ext, "+", ".")
	return fmt.Sprintf("gemini_attachment_%d.%s", index+1, ext)
}

// --- thinking ----------------------------------------------------------------

func thoughtText(parts []any) string {
	var sb strings.Builder
	for _, p := range partMaps(parts) {
		if core.Truthy(p["thought"]) && core.Truthy(p["text"]) {
			s, _ := p["text"].(string)
			sb.WriteString(s)
		}
	}
	return sb.String()
}

func thoughtSignature(parts []any) string {
	for _, p := range partMaps(parts) {
		sig := core.Or(p["thoughtSignature"], p["thought_signature"],
			core.Dig(p, "functionCall", "thoughtSignature"), core.Dig(p, "functionCall", "thought_signature"))
		if s, ok := sig.(string); ok && s != "" {
			return s
		}
	}
	return ""
}

// --- grounding citations -----------------------------------------------------

// ExtractCitations reads Google Search grounding metadata. content is the
// parsed text (byte offsets are converted to character offsets against it).
func ExtractCitations(data map[string]any, content any) []core.Citation {
	metadata := core.DigMap(data, "candidates", 0, "groundingMetadata")
	if len(metadata) == 0 {
		return nil
	}
	chunks, _ := metadata["groundingChunks"].([]any)
	supports, _ := metadata["groundingSupports"].([]any)
	if len(supports) == 0 {
		return chunkCitations(chunks)
	}
	var out []core.Citation
	for _, s := range supports {
		sm, _ := s.(map[string]any)
		out = append(out, supportCitations(sm, chunks, content)...)
	}
	return out
}

func supportCitations(support map[string]any, chunks []any, content any) []core.Citation {
	segment := core.DigMap(support, "segment")
	end := core.AsInt(segment["endIndex"])
	start := core.AsInt(segment["startIndex"])
	if start == nil && end != nil {
		start = core.Int(0)
	}
	var out []core.Citation
	for _, raw := range core.ToSafeArray(support["groundingChunkIndices"]) {
		idx := core.AsInt(raw)
		if idx == nil {
			continue
		}
		var chunk any
		if *idx >= 0 && *idx < len(chunks) {
			chunk = chunks[*idx]
		}
		source := chunkSource(chunk)
		if len(source) == 0 {
			continue
		}
		cit := core.Citation{
			URL: strPtr(source["uri"]), Title: strPtr(source["title"]), Text: strPtr(segment["text"]),
			StartIndex: byteToCharIndex(content, start), EndIndex: byteToCharIndex(content, end),
			SourceIndex: core.Int(*idx),
		}
		out = append(out, cit)
	}
	return out
}

func chunkCitations(chunks []any) []core.Citation {
	var out []core.Citation
	for i, c := range chunks {
		source := chunkSource(c)
		if len(source) == 0 {
			continue
		}
		out = append(out, core.Citation{URL: strPtr(source["uri"]), Title: strPtr(source["title"]), SourceIndex: core.Int(i)})
	}
	return out
}

func chunkSource(chunk any) map[string]any {
	m, ok := chunk.(map[string]any)
	if !ok {
		return nil
	}
	src, _ := core.Or(m["web"], m["retrievedContext"]).(map[string]any)
	return src
}

func strPtr(v any) *string {
	if s, ok := v.(string); ok {
		return &s
	}
	return nil
}

// byteToCharIndex converts a UTF-8 byte offset into content to a character
// offset (a split multi-byte character is not counted).
func byteToCharIndex(content any, byteIndex *int) *int {
	s, ok := content.(string)
	if !ok || byteIndex == nil {
		return nil
	}
	b := *byteIndex
	if b < 0 {
		b += len(s)
	}
	b = max(0, min(b, len(s)))
	prefix := s[:b]
	n := 0
	for i := 0; i < len(prefix); {
		r, size := utf8.DecodeRuneInString(prefix[i:])
		if r != utf8.RuneError || size > 1 {
			n++
		}
		i += size
	}
	return &n
}

// --- messages ----------------------------------------------------------------

// SystemInstructionText joins every system message's text with blank lines.
func SystemInstructionText(messages []*core.Message) string {
	var texts []string
	for _, m := range messages {
		if m.Role != core.RoleSystem {
			continue
		}
		var c any = m.Content
		switch v := c.(type) {
		case *core.Content:
			c = v.TextOrEmpty()
		case *core.SearchResults:
			c = v.TextOrEmpty()
		case core.RawContent:
			c = v.Value
		}
		if !core.Truthy(c) {
			continue
		}
		if s, ok := c.(string); ok {
			texts = append(texts, s)
		} else {
			texts = append(texts, core.PyJSON(c))
		}
	}
	return strings.Join(texts, "\n\n")
}

// FormatRole maps assistant to "model" and tool to "user".
func FormatRole(role string) string {
	switch role {
	case core.RoleAssistant:
		return "model"
	case core.RoleTool:
		return "user"
	}
	return role
}

// FormatMessages renders contents, grouping consecutive tool results into
// one user turn and naming each result after the call it answers.
func (g *Protocol) FormatMessages(messages []*core.Message) ([]any, error) {
	out := []any{}
	names := map[string]string{}
	for i := 0; i < len(messages); {
		m := messages[i]
		if m.Role == core.RoleTool {
			var parts []any
			for i < len(messages) && messages[i].Role == core.RoleTool {
				tm := messages[i]
				name := ""
				if tm.ToolCallID != "" {
					name = names[tm.ToolCallID]
					delete(names, tm.ToolCallID)
				}
				p, err := g.formatToolResult(tm, name)
				if err != nil {
					return nil, err
				}
				parts = append(parts, p...)
				i++
			}
			out = append(out, map[string]any{"role": "user", "parts": parts})
			continue
		}
		if m.IsToolCall() {
			for _, k := range m.ToolCalls.Keys() {
				names[k] = m.ToolCalls.Get(k).Name
			}
		}
		parts, err := g.formatParts(m)
		if err != nil {
			return nil, err
		}
		out = append(out, map[string]any{"role": FormatRole(m.Role), "parts": parts})
		i++
	}
	return out, nil
}

func (g *Protocol) formatParts(m *core.Message) ([]any, error) {
	switch {
	case m.IsToolCall():
		return g.formatToolCall(m)
	case m.IsToolResult():
		return g.formatToolResult(m, "")
	}
	var parts []any
	if m.Role == core.RoleAssistant && m.Thinking != nil {
		parts = append(parts, thoughtPart(m.Thinking))
	}
	content, err := FormatContent(m.Content)
	if err != nil {
		return nil, err
	}
	return append(parts, asParts(content)...), nil
}

func thoughtPart(t *core.Thinking) map[string]any {
	part := map[string]any{"thought": true}
	if t.Text != "" {
		part["text"] = t.Text
	}
	if t.Signature != "" {
		part["thoughtSignature"] = t.Signature
	}
	return part
}

// asParts is pyllym's `x if isinstance(x, list) else [x]`.
func asParts(v any) []any {
	switch x := v.(type) {
	case []any:
		return x
	case []map[string]any:
		out := make([]any, len(x))
		for i, m := range x {
			out[i] = m
		}
		return out
	}
	return []any{v}
}

// FormatContent renders message content as Gemini parts. RawContent is sent
// as is; JSON values become their text; attachments become inline_data or
// file_data parts.
func FormatContent(content any) (any, error) {
	switch v := content.(type) {
	case core.RawContent:
		return v.Value, nil
	case map[string]any, []any:
		return []any{map[string]any{"text": core.PyJSON(v)}}, nil
	case *core.SearchResults:
		return FormatContent(&v.Content)
	case *core.Content:
		parts := []any{}
		if v.Text != nil && *v.Text != "" {
			parts = append(parts, map[string]any{"text": *v.Text})
		}
		for _, a := range v.Attachments {
			part, err := FormatAttachment(a)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		}
		return parts, nil
	}
	return []any{map[string]any{"text": content}}, nil
}

// FormatAttachment renders one attachment: text inline, documents rejected,
// provider files as file_data and everything else as base64 inline_data.
func FormatAttachment(a *core.Attachment) (map[string]any, error) {
	switch a.Type() {
	case "text":
		s, err := a.ForLLM()
		if err != nil {
			return nil, err
		}
		return map[string]any{"text": s}, nil
	case "document", "unknown":
		return nil, &core.UnsupportedAttachmentError{Type: a.MIMEType}
	}
	if a.IsProviderFile() {
		uri := a.ProviderFileURI()
		if uri == "" {
			uri = a.ProviderFileID()
		}
		return map[string]any{"file_data": map[string]any{"mime_type": a.MIMEType, "file_uri": uri}}, nil
	}
	enc, err := a.Encoded()
	if err != nil {
		return nil, err
	}
	return map[string]any{"inline_data": map[string]any{"mime_type": a.MIMEType, "data": enc}}, nil
}

// --- tools -------------------------------------------------------------------

// FormatTools renders the tools as one functionDeclarations entry.
func FormatTools(tools []core.Tool) ([]any, error) {
	if len(tools) == 0 {
		return []any{}, nil
	}
	decls := make([]any, 0, len(tools))
	for _, t := range tools {
		d, err := FunctionDeclaration(t)
		if err != nil {
			return nil, err
		}
		decls = append(decls, d)
	}
	return []any{map[string]any{"functionDeclarations": decls}}, nil
}

// FunctionDeclaration renders one tool (provider params deep-merged in).
func FunctionDeclaration(t core.Tool) (map[string]any, error) {
	var desc any
	if d := t.Description(); d != "" {
		desc = d
	}
	decl := map[string]any{"name": t.Name(), "description": desc}
	if schema := t.ParamsSchema(); len(schema) > 0 {
		params, err := ConvertToolSchema(schema)
		if err != nil {
			return nil, err
		}
		decl["parameters"] = params
	}
	if pp, ok := t.(core.ProviderParamser); ok && len(pp.ProviderParams()) > 0 {
		return core.DeepMerge(decl, pp.ProviderParams()), nil
	}
	return decl, nil
}

func (g *Protocol) formatToolCall(m *core.Message) ([]any, error) {
	var parts []any
	if m.Content != nil && !isEmptyLen(m.Content) {
		formatted, err := FormatContent(m.Content)
		if err != nil {
			return nil, err
		}
		parts = append(parts, asParts(formatted)...)
	}
	fallback := ""
	if m.Thinking != nil {
		fallback = m.Thinking.Signature
	}
	usedFallback := false
	for _, tc := range m.ToolCalls.List() {
		part := map[string]any{"functionCall": map[string]any{"name": tc.Name, "args": toolArgs(tc.Arguments)}}
		sig := tc.ThoughtSignature
		if sig == "" && fallback != "" && !usedFallback {
			sig = fallback
			usedFallback = true
		}
		if sig != "" {
			part["thoughtSignature"] = sig
		}
		parts = append(parts, part)
	}
	if parts == nil {
		parts = []any{}
	}
	return parts, nil
}

func toolArgs(v any) any {
	if v == nil {
		return map[string]any{}
	}
	return v
}

// isEmptyLen is Python's `hasattr(x, "__len__") and len(x) == 0` for the
// content types that have a length (strings and JSON containers).
func isEmptyLen(v any) bool {
	switch x := v.(type) {
	case string:
		return x == ""
	case map[string]any:
		return len(x) == 0
	case []any:
		return len(x) == 0
	}
	return false
}

func (g *Protocol) formatToolResult(m *core.Message, functionName string) ([]any, error) {
	if functionName == "" {
		functionName = m.ToolCallID
	}
	content := m.Content
	if content == nil || isEmptyLen(content) {
		content = "(no output)"
	}
	formatted, err := FormatContent(content)
	if err != nil {
		return nil, err
	}
	return []any{map[string]any{
		"functionResponse": map[string]any{
			"name":     functionName,
			"response": map[string]any{"name": functionName, "content": formatted},
		},
	}}, nil
}

// ExtractToolCalls reads functionCall parts (Gemini has no call ids, so
// each call gets a fresh UUID).
func ExtractToolCalls(data map[string]any) *core.ToolCalls {
	candidate := core.Dig(data, "candidates", 0)
	if !core.Truthy(candidate) {
		return nil
	}
	parts, ok := core.Dig(candidate, "content", "parts").([]any)
	if !ok {
		return nil
	}
	out := &core.ToolCalls{}
	for _, p := range partMaps(parts) {
		fn, ok := p["functionCall"].(map[string]any)
		if !ok || len(fn) == 0 {
			continue
		}
		id := uuid.NewString()
		sig, _ := core.Or(p["thoughtSignature"], p["thought_signature"]).(string)
		name, _ := fn["name"].(string)
		args := fn["args"]
		if !core.Truthy(args) {
			args = map[string]any{}
		}
		out.Set(id, &core.ToolCall{ID: id, Name: name, Arguments: args, ThoughtSignature: sig})
	}
	if out.Len() == 0 {
		return nil
	}
	return out
}

// BuildToolConfig renders toolConfig for a tool choice: "required" or a
// tool name force a call (mode "any"), a name also restricts to that tool.
func BuildToolConfig(choice string) map[string]any {
	specific := choice != "auto" && choice != "none" && choice != "required"
	mode := choice
	if choice == "required" || specific {
		mode = "any"
	}
	config := map[string]any{"mode": mode}
	if specific {
		config["allowedFunctionNames"] = []any{choice}
	}
	return map[string]any{"functionCallingConfig": config}
}

// --- structured output -------------------------------------------------------

// StructuredOutputConfig renders the generationConfig entries for a
// normalised schema: responseJsonSchema on Gemini >= 2.5, else the
// responseSchema dialect.
func StructuredOutputConfig(schema map[string]any, model *core.Info) map[string]any {
	config := map[string]any{"responseMimeType": "application/json"}
	if responseJSONSchemaSupported(model) {
		config["responseJsonSchema"] = buildJSONSchema(schema)
	} else {
		raw := core.Or(schema["schema"], schema)
		config["responseSchema"] = ConvertResponseSchema(raw)
	}
	return config
}

func buildJSONSchema(schema map[string]any) any {
	normalized := normalizeJSON(schema["schema"])
	if m, ok := normalized.(map[string]any); ok {
		delete(m, "strict")
	}
	return normalized
}

func responseJSONSchemaSupported(model *core.Info) bool {
	major, minor, ok := geminiVersion(model)
	return ok && (major > 2 || major == 2 && minor >= 5)
}

func geminiVersion(model *core.Info) (int, int, bool) {
	if model == nil {
		return 0, 0, false
	}
	candidates := []any{model.ID, model.Family, model.Metadata["version"], model.Metadata["description"]}
	for _, c := range candidates {
		if c == nil {
			continue
		}
		if major, minor, ok := extractVersion(pyStr(c)); ok {
			return major, minor, true
		}
	}
	return 0, 0, false
}

var versionRe = regexp.MustCompile(`(\d+)\.(\d+)|\d+`)

func extractVersion(text string) (int, int, bool) {
	m := versionRe.FindStringSubmatch(text)
	if m == nil {
		return 0, 0, false
	}
	if m[1] != "" {
		major, _ := strconv.Atoi(m[1])
		minor, _ := strconv.Atoi(m[2])
		return major, minor, true
	}
	major, _ := strconv.Atoi(m[0])
	return major, 0, true
}

// pyStr approximates Python's str() for decoded JSON scalars.
func pyStr(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case float64:
		s := strconv.FormatFloat(x, 'f', -1, 64)
		if !strings.ContainsAny(s, ".eE") {
			s += ".0"
		}
		return s
	case bool:
		if x {
			return "True"
		}
		return "False"
	case nil:
		return "None"
	}
	return fmt.Sprint(v)
}

// --- streaming ---------------------------------------------------------------

// BuildChunk reads one streamGenerateContent event.
func (g *Protocol) BuildChunk(data map[string]any) (*core.Chunk, error) {
	parts := partsOf(data)
	var content any
	var text strings.Builder
	for _, p := range partMaps(parts) {
		if !core.Truthy(p["thought"]) && core.Truthy(p["text"]) {
			s, _ := p["text"].(string)
			text.WriteString(s)
		}
	}
	if text.Len() > 0 {
		content = text.String()
	}
	var output *int
	if total := outputTokenSum(data); total > 0 {
		output = &total
	}
	modelID, _ := data["modelVersion"].(string)
	return &core.Chunk{
		Role:         core.RoleAssistant,
		ModelID:      modelID,
		Content:      content,
		Citations:    ExtractCitations(data, nil),
		Thinking:     core.BuildThinking(thoughtText(parts), thoughtSignature(parts)),
		Tokens:       core.BuildTokens(InputTokens(data), output, cachedTokens(data), nil, thoughtsTokens(data)),
		FinishReason: core.DigString(data, "candidates", 0, "finishReason"),
		ToolCalls:    ExtractToolCalls(data),
	}, nil
}

// --- embeddings --------------------------------------------------------------

// RenderEmbeddingPayload builds the batchEmbedContents body.
func RenderEmbeddingPayload(text any, model string, dimensions *int) map[string]any {
	texts, _ := textList(text)
	requests := make([]any, 0, len(texts))
	for _, t := range texts {
		requests = append(requests, core.Compact(map[string]any{
			"model":                "models/" + model,
			"content":              map[string]any{"parts": []any{map[string]any{"text": pyStr(t)}}},
			"outputDimensionality": dimensions,
		}))
	}
	return map[string]any{"requests": requests}
}

// textList returns text as a list and whether it already was one.
func textList(text any) ([]any, bool) {
	switch x := text.(type) {
	case []any:
		return x, true
	case []string:
		return core.ToSafeArray(x), true
	}
	return []any{text}, false
}

// Embed implements core.Embedder.
func (g *Protocol) Embed(ctx context.Context, text any, model string, dimensions *int) (*core.Embedding, error) {
	resp, err := g.conn().Post(ctx, g.EmbeddingURL(model), RenderEmbeddingPayload(text, model, dimensions), nil)
	if err != nil {
		return nil, err
	}
	return ParseEmbeddingResponse(resp, model, text), nil
}

// ParseEmbeddingResponse reads embeddings[].values.
func ParseEmbeddingResponse(resp *core.Response, model string, text any) *core.Embedding {
	var vectors [][]float64
	for _, e := range core.ToSafeArray(core.Dig(resp.Body, "embeddings")) {
		var vec []float64
		for _, x := range core.ToSafeArray(core.Dig(e, "values")) {
			vec = append(vec, core.Deref(core.AsFloat(x)))
		}
		vectors = append(vectors, vec)
	}
	_, isList := textList(text)
	var out any = vectors
	if len(vectors) == 1 && !isList {
		out = vectors[0]
	}
	return &core.Embedding{Vectors: out, Model: model, InputTokens: 0}
}

// --- models ------------------------------------------------------------------

// ListModels implements core.ModelLister.
func (g *Protocol) ListModels(ctx context.Context) ([]*core.Info, error) {
	resp, err := g.conn().Get(ctx, g.ModelsURL(), nil)
	if err != nil {
		return nil, err
	}
	return ParseListModels(resp, g.Provider.Slug(), g.Provider.Capabilities()), nil
}

// ParseListModels reads a models listing.
func ParseListModels(resp *core.Response, slug string, caps core.Capabilities) []*core.Info {
	var out []*core.Info
	for _, raw := range core.ToSafeArray(core.Dig(resp.Body, "models")) {
		md, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		id := strings.ReplaceAll(pyStr(md["name"]), "models/", "")
		data := map[string]any{
			"id": id, "name": core.Or(md["displayName"], id), "provider": slug, "created_at": nil,
			"context_window": md["inputTokenLimit"], "max_output_tokens": md["outputTokenLimit"],
			"capabilities": []any{}, "pricing": map[string]any{},
			"metadata": map[string]any{
				"version": md["version"], "description": md["description"],
				"supported_generation_methods": md["supportedGenerationMethods"],
			},
		}
		if caps != nil {
			data["context_window"] = core.Or(md["inputTokenLimit"], intAny(caps.ContextWindowFor(id)))
			data["max_output_tokens"] = core.Or(md["outputTokenLimit"], intAny(caps.MaxTokensFor(id)))
			capList := []any{}
			for _, c := range caps.CriticalCapabilitiesFor(id) {
				capList = append(capList, c)
			}
			data["capabilities"] = capList
			data["pricing"] = caps.PricingFor(id)
		}
		out = append(out, core.NewInfo(data))
	}
	return out
}

func intAny(p *int) any {
	if p == nil {
		return nil
	}
	return float64(*p)
}

// --- helpers -----------------------------------------------------------------

// normalizeJSON deep-copies v into plain decoded JSON (map[string]any,
// []any, float64, ...) so schemas built from typed Go values convert the
// same as decoded ones.
func normalizeJSON(v any) any {
	if v == nil {
		return nil
	}
	if isPlainJSON(v) {
		return core.DeepCopy(v)
	}
	b, err := core.MarshalPayload(v)
	if err != nil {
		return core.DeepCopy(v)
	}
	parsed := core.ParseBody("application/json", b)
	if _, ok := parsed.(string); ok {
		return core.DeepCopy(v)
	}
	return parsed
}

func isPlainJSON(v any) bool {
	switch x := v.(type) {
	case nil, string, float64, bool:
		return true
	case map[string]any:
		for _, e := range x {
			if !isPlainJSON(e) {
				return false
			}
		}
		return true
	case []any:
		for _, e := range x {
			if !isPlainJSON(e) {
				return false
			}
		}
		return true
	}
	return reflect.TypeOf(v) == nil
}
