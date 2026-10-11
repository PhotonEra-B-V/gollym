package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"
)

// ChatOptions configure NewChat.
type ChatOptions struct {
	Model    string // default: Config.DefaultModel
	Provider string
	// AssumeModelExists skips the registry lookup (needs Provider).
	AssumeModelExists bool
	Context           *Context
}

// Chat is a conversation with a model. Ask runs the agentic tool loop to
// completion; with an OnChunk callback it streams.
type Chat struct {
	Model     *Info
	Provider  Provider
	Messages  []*Message
	Tools     ToolSet
	ToolPrefs ToolPrefs
	Params    map[string]any
	Headers   map[string]string
	Schema    map[string]any

	context         *Context
	config          *Config
	temperature     *float64
	thinking        *ThinkingConfig
	citations       bool
	protocol        string
	toolConcurrency bool

	beforeMessage   []func(context.Context) error
	afterMessage    []func(context.Context, *Message) error
	beforeToolCall  []func(context.Context, *ToolCall) error
	afterToolResult []func(context.Context, any) error
}

// NewChat creates a conversation.
func NewChat(opts ChatOptions) (*Chat, error) {
	if opts.AssumeModelExists && opts.Provider == "" {
		return nil, errors.New("Provider must be specified if assume_model_exists is true")
	}
	c := &Chat{context: opts.Context, Params: map[string]any{}, Headers: map[string]string{}}
	if opts.Context != nil {
		c.config = opts.Context.Config
	} else {
		c.config = Global()
	}
	model := opts.Model
	if model == "" {
		model = c.config.DefaultModel
	}
	if err := c.setModel(model, opts.Provider, opts.AssumeModelExists); err != nil {
		return nil, err
	}
	c.toolConcurrency = c.config.ToolConcurrency
	return c, nil
}

// Config returns the chat's configuration.
func (c *Chat) Config() *Config { return c.config }

// Temperature returns the configured temperature (nil = provider default).
func (c *Chat) Temperature() *float64 { return c.temperature }

// AskOption customises one Ask.
type AskOption func(*askOpts)

type askOpts struct {
	attachments []any
	onChunk     func(*Chunk) error
}

// With attaches files: paths, URLs, []byte, readers, *Attachment or
// *UploadedFile.
func With(sources ...any) AskOption {
	return func(o *askOpts) { o.attachments = append(o.attachments, sources...) }
}

// OnChunk streams the response, calling fn for every chunk.
func OnChunk(fn func(*Chunk) error) AskOption {
	return func(o *askOpts) { o.onChunk = fn }
}

// Ask adds a user message and runs until the model is done.
// message may be a string, *Content, *SearchResults or RawContent.
func (c *Chat) Ask(ctx context.Context, message any, opts ...AskOption) (*Message, error) {
	var o askOpts
	for _, f := range opts {
		f(&o)
	}
	if err := c.AddUserMessage(message, o.attachments...); err != nil {
		return nil, err
	}
	return c.RunUntilDone(ctx, o.onChunk)
}

// Say is an alias of Ask.
func (c *Chat) Say(ctx context.Context, message any, opts ...AskOption) (*Message, error) {
	return c.Ask(ctx, message, opts...)
}

// AddUserMessage stages a user message without sending it.
func (c *Chat) AddUserMessage(message any, attachments ...any) error {
	content, err := buildContent(message, attachments)
	if err != nil {
		return err
	}
	_, err = c.AddMessage(&Message{Role: RoleUser, Content: content})
	return err
}

func buildContent(message any, attachments []any) (any, error) {
	switch m := message.(type) {
	case *Content, *SearchResults, RawContent:
		return m, nil
	case nil:
		return NewContent(nil, attachments...)
	case string:
		return NewContent(&m, attachments...)
	}
	s := fmt.Sprint(message)
	return NewContent(&s, attachments...)
}

// Send calls the model once and appends its response.
func (c *Chat) Send(ctx context.Context, onChunk func(*Chunk) error) (*Message, error) {
	if onChunk != nil {
		if err := c.runBeforeMessage(ctx); err != nil {
			return nil, err
		}
	}
	msg, err := c.providerCompletion(ctx, onChunk)
	if err != nil {
		return nil, err
	}
	if onChunk == nil {
		if err := c.runBeforeMessage(ctx); err != nil {
			return nil, err
		}
	}
	return c.finalizeCompletion(ctx, msg)
}

// ExecuteTools runs pending tool calls and appends their results.
func (c *Chat) ExecuteTools(ctx context.Context) error {
	if last := c.last(); last != nil && last.IsToolCall() {
		return c.executePendingToolCalls(ctx, last)
	}
	return nil
}

// Step makes one move: execute pending tools, or call the model.
func (c *Chat) Step(ctx context.Context, onChunk func(*Chunk) error) (*Message, error) {
	if c.IsComplete() {
		return nil, nil
	}
	if last := c.last(); last != nil && last.IsToolCall() {
		return nil, c.ExecuteTools(ctx)
	}
	return c.Send(ctx, onChunk)
}

// RunUntilDone steps until nothing is left and returns the last message.
func (c *Chat) RunUntilDone(ctx context.Context, onChunk func(*Chunk) error) (*Message, error) {
	if len(c.Messages) == 0 {
		return nil, &Error{Kind: KindError, Message: "Nothing to send: add a user message first (add_user_message or ask)"}
	}
	for !c.IsComplete() {
		if _, err := c.Step(ctx, onChunk); err != nil {
			return nil, err
		}
	}
	return c.last(), nil
}

// IsComplete reports that the model has the last word and asked no tools.
func (c *Chat) IsComplete() bool {
	last := c.last()
	if last == nil {
		return true
	}
	if last.Role == RoleUser || last.Role == RoleTool {
		return false
	}
	return !last.IsToolCall()
}

func (c *Chat) last() *Message {
	if len(c.Messages) == 0 {
		return nil
	}
	return c.Messages[len(c.Messages)-1]
}

// --- builders ---------------------------------------------------------------

// WithInstructions replaces the system instruction (or appends one).
func (c *Chat) WithInstructions(instructions string, appendInstr ...bool) *Chat {
	var system, rest []*Message
	for _, m := range c.Messages {
		if m.Role == RoleSystem {
			system = append(system, m)
		} else {
			rest = append(rest, m)
		}
	}
	if len(appendInstr) > 0 && appendInstr[0] {
		system = append(system, &Message{Role: RoleSystem, Content: instructions})
	} else if len(system) == 0 {
		system = []*Message{{Role: RoleSystem, Content: instructions}}
	} else {
		system[0].Content = instructions
		system = system[:1]
	}
	c.Messages = append(system, rest...)
	return c
}

// WithTool adds a tool.
func (c *Chat) WithTool(t Tool) *Chat {
	if t != nil {
		c.Tools.Add(t)
	}
	return c
}

// WithTools adds tools (replace clears existing ones first).
func (c *Chat) WithTools(replace bool, tools ...Tool) *Chat {
	if replace {
		c.Tools.Clear()
	}
	for _, t := range tools {
		c.WithTool(t)
	}
	return c
}

// WithToolChoice sets auto, none, required or a tool name.
func (c *Chat) WithToolChoice(choice string) error {
	valid := map[string]bool{"auto": true, "none": true, "required": true}
	for _, n := range c.Tools.Names() {
		valid[n] = true
	}
	if !valid[choice] {
		names := make([]string, 0, len(valid))
		for n := range valid {
			names = append(names, n)
		}
		sort.Strings(names)
		return fmt.Errorf("%w: Invalid tool choice: %s. Valid choices are: %s", ErrInvalidToolChoice, choice, strings.Join(names, ", "))
	}
	c.ToolPrefs.Choice = choice
	return nil
}

// WithToolCalls sets "one" or "many" (parallel tool calls).
func (c *Chat) WithToolCalls(calls string) error {
	if calls != "one" && calls != "many" {
		return fmt.Errorf("Invalid calls value: %q. Valid values are: 'many', 'one', or 1", calls)
	}
	c.ToolPrefs.Calls = calls
	return nil
}

// WithToolConcurrency runs tool calls of one response concurrently.
func (c *Chat) WithToolConcurrency(on bool) *Chat { c.toolConcurrency = on; return c }

// WithModel switches model (and provider).
func (c *Chat) WithModel(modelID, provider string, assumeExists bool) error {
	return c.setModel(modelID, provider, assumeExists)
}

func (c *Chat) setModel(modelID, provider string, assumeExists bool) error {
	model, p, err := ResolveModel(modelID, provider, assumeExists, c.config)
	if err != nil {
		return err
	}
	c.Model, c.Provider = model, p
	return nil
}

// WithTemperature sets the sampling temperature.
func (c *Chat) WithTemperature(t float64) *Chat { c.temperature = &t; return c }

// WithThinking requests reasoning by effort and/or budget.
func (c *Chat) WithThinking(effort string, budget *int) (*Chat, error) {
	if effort == "" && budget == nil {
		return c, errors.New("with_thinking requires effort or budget")
	}
	c.thinking = &ThinkingConfig{Effort: effort, Budget: budget}
	return c, nil
}

// WithCitations asks for citations where supported.
func (c *Chat) WithCitations(enabled bool) *Chat { c.citations = enabled; return c }

// WithContext rebinds the chat to a context's config.
func (c *Chat) WithContext(ctx *Context) error {
	c.context = ctx
	c.config = ctx.Config
	return c.setModel(c.Model.ID, c.Provider.Slug(), true)
}

// WithParams replaces the provider params deep-merged into every payload.
func (c *Chat) WithParams(params map[string]any) *Chat {
	if params == nil {
		params = map[string]any{}
	}
	c.Params = params
	return c
}

// WithProtocol forces a protocol of the provider.
func (c *Chat) WithProtocol(protocol string) *Chat { c.protocol = protocol; return c }

// WithHeaders replaces per-request headers.
func (c *Chat) WithHeaders(h map[string]string) *Chat {
	if h == nil {
		h = map[string]string{}
	}
	c.Headers = h
	return c
}

// SchemaProvider is anything that can describe itself as a JSON Schema.
type SchemaProvider interface{ JSONSchema() map[string]any }

// WithSchema requests structured output. schema is a JSON Schema map (bare,
// or wrapped as {name, schema, strict, description}), a SchemaProvider, or a
// struct value whose JSON Schema is derived from its fields.
func (c *Chat) WithSchema(schema any) (*Chat, error) {
	var raw map[string]any
	switch s := schema.(type) {
	case nil:
		c.Schema = nil
		return c, nil
	case map[string]any:
		raw = s
	case SchemaProvider:
		raw = s.JSONSchema()
	default:
		derived, err := SchemaFromStruct(schema)
		if err != nil {
			return c, err
		}
		raw = derived
	}
	c.Schema = NormalizeSchema(raw)
	return c, nil
}

var schemaNameRe = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// NormalizeSchema wraps a schema as {name, schema, strict, description}
// (pyllym _normalize_schema_payload).
func NormalizeSchema(raw map[string]any) map[string]any {
	if raw == nil {
		return nil
	}
	schema := DeepCopy(raw).(map[string]any)
	// schema.get("schema") or schema: an empty inner schema falls back.
	inner, _ := schema["schema"].(map[string]any)
	var def map[string]any
	if len(inner) > 0 {
		def = DeepCopy(inner).(map[string]any)
	} else {
		def = DeepCopy(schema).(map[string]any)
	}
	var strict any
	if s, ok := schema["strict"]; ok {
		strict = s
	} else if s, ok := def["strict"]; ok {
		strict = s
		delete(def, "strict")
	}
	name := "response"
	if n, ok := schema["name"]; ok && Truthy(n) {
		name = schemaNameRe.ReplaceAllString(fmt.Sprint(n), "_")
		if name == "" {
			name = "response"
		}
	}
	out := map[string]any{"name": name, "schema": def}
	if strict == nil {
		out["strict"] = true
	} else {
		out["strict"] = strict
	}
	if d, ok := schema["description"]; ok && d != nil {
		out["description"] = d
	}
	return out
}

// --- callbacks --------------------------------------------------------------

func (c *Chat) BeforeMessage(fn func(context.Context) error) *Chat {
	c.beforeMessage = append(c.beforeMessage, fn)
	return c
}
func (c *Chat) AfterMessage(fn func(context.Context, *Message) error) *Chat {
	c.afterMessage = append(c.afterMessage, fn)
	return c
}
func (c *Chat) BeforeToolCall(fn func(context.Context, *ToolCall) error) *Chat {
	c.beforeToolCall = append(c.beforeToolCall, fn)
	return c
}
func (c *Chat) AfterToolResult(fn func(context.Context, any) error) *Chat {
	c.afterToolResult = append(c.afterToolResult, fn)
	return c
}

func (c *Chat) runBeforeMessage(ctx context.Context) error {
	for _, f := range c.beforeMessage {
		if err := f(ctx); err != nil {
			return err
		}
	}
	return nil
}

func (c *Chat) runAfterMessage(ctx context.Context, m *Message) error {
	for _, f := range c.afterMessage {
		if err := f(ctx, m); err != nil {
			return err
		}
	}
	return nil
}

// --- misc -------------------------------------------------------------------

// Cost aggregates the cost of every message.
func (c *Chat) Cost() *CostAggregate {
	costs := make([]*Cost, 0, len(c.Messages))
	for _, m := range c.Messages {
		model := m.ModelInfo()
		if model == nil {
			model = c.Model
		}
		costs = append(costs, m.Cost(model))
	}
	return AggregateCosts(costs)
}

// AddMessage appends a message (validating the role and letting the
// protocol preprocess it).
func (c *Chat) AddMessage(m *Message) (*Message, error) {
	msg, err := NewMessage(*m)
	if err != nil {
		return nil, err
	}
	if c.Provider != nil {
		proto, err := c.Provider.Protocol(c.protocol, c.Model)
		if err != nil {
			return nil, err
		}
		if pp, ok := proto.(MessagePreprocessor); ok {
			if msg, err = pp.PreprocessMessage(msg); err != nil {
				return nil, err
			}
		}
	}
	c.Messages = append(c.Messages, msg)
	return msg, nil
}

// AddCompletion receives a completion produced out of band.
func (c *Chat) AddCompletion(ctx context.Context, m *Message) (*Message, error) {
	if err := c.runBeforeMessage(ctx); err != nil {
		return nil, err
	}
	return c.finalizeCompletion(ctx, m)
}

func (c *Chat) finalizeCompletion(ctx context.Context, m *Message) (*Message, error) {
	c.normalizeSchemaResponse(m)
	added, err := c.AddMessage(m)
	if err != nil {
		return nil, err
	}
	if err := c.runAfterMessage(ctx, added); err != nil {
		return nil, err
	}
	return added, nil
}

// Request is the completion request the chat would send now.
func (c *Chat) Request() *CompletionRequest {
	return &CompletionRequest{
		Messages: c.Messages, Tools: &c.Tools, ToolPrefs: c.ToolPrefs, Temperature: c.temperature,
		Model: c.Model, Params: c.Params, Headers: c.Headers, Schema: c.Schema,
		Thinking: c.thinking, Citations: c.citations,
	}
}

// Render returns the payload the next call would send.
func (c *Chat) Render() (map[string]any, error) {
	proto, err := c.Provider.Protocol(c.protocol, c.Model)
	if err != nil {
		return nil, err
	}
	return RenderRequest(proto, c.Request())
}

func (c *Chat) providerCompletion(ctx context.Context, onChunk func(*Chunk) error) (*Message, error) {
	proto, err := c.Provider.Protocol(c.protocol, c.Model)
	if err != nil {
		return nil, err
	}
	return CompleteRequest(ctx, c.Provider, proto, c.Request(), onChunk)
}

// normalizeSchemaResponse decodes a structured-output response's JSON text.
func (c *Chat) normalizeSchemaResponse(m *Message) {
	s, ok := m.Content.(string)
	if c.Schema == nil || !ok || m.IsToolCall() {
		return
	}
	var v any
	if err := json.Unmarshal([]byte(s), &v); err == nil {
		m.Content = v
	}
}

func (c *Chat) executePendingToolCalls(ctx context.Context, response *Message) error {
	calls := response.ToolCalls.List()
	if c.toolConcurrency {
		results := make([]any, len(calls))
		errs := make([]error, len(calls))
		var wg sync.WaitGroup
		for i, tc := range calls {
			wg.Add(1)
			go func() {
				defer wg.Done()
				results[i], errs[i] = c.executeToolWithCallbacks(ctx, tc)
			}()
		}
		wg.Wait()
		for i, tc := range calls {
			if errs[i] != nil {
				return errs[i]
			}
			if err := c.runBeforeMessage(ctx); err != nil {
				return err
			}
			if err := c.addToolResultMessage(ctx, tc, results[i]); err != nil {
				return err
			}
		}
	} else {
		for _, tc := range calls {
			if err := c.runBeforeMessage(ctx); err != nil {
				return err
			}
			result, err := c.executeToolWithCallbacks(ctx, tc)
			if err != nil {
				return err
			}
			if err := c.addToolResultMessage(ctx, tc, result); err != nil {
				return err
			}
		}
	}
	if ch := c.ToolPrefs.Choice; ch != "" && ch != "auto" && ch != "none" {
		c.ToolPrefs.Choice = ""
	}
	return nil
}

func (c *Chat) executeToolWithCallbacks(ctx context.Context, tc *ToolCall) (any, error) {
	for _, f := range c.beforeToolCall {
		if err := f(ctx, tc); err != nil {
			return nil, err
		}
	}
	result, err := c.executeTool(ctx, tc)
	if err != nil {
		return nil, err
	}
	for _, f := range c.afterToolResult {
		if err := f(ctx, result); err != nil {
			return nil, err
		}
	}
	return result, nil
}

func (c *Chat) executeTool(ctx context.Context, tc *ToolCall) (any, error) {
	tool, ok := c.Tools.Get(tc.Name)
	if !ok {
		return map[string]any{"error": fmt.Sprintf("Model tried to call unavailable tool `%s`. Available tools: %s.", tc.Name, PyJSON(toAnySlice(c.Tools.Names())))}, nil
	}
	return tool.Call(ctx, ToolArguments(tc))
}

func toAnySlice(s []string) []any {
	out := make([]any, len(s))
	for i, v := range s {
		out[i] = v
	}
	return out
}

func (c *Chat) addToolResultMessage(ctx context.Context, tc *ToolCall, result any) error {
	var content any
	switch r := result.(type) {
	case *Content, *SearchResults, RawContent:
		content = r
	default:
		content = toolResultString(result)
	}
	msg, err := c.AddMessage(&Message{Role: RoleTool, Content: content, ToolCallID: tc.ID})
	if err != nil {
		return err
	}
	return c.runAfterMessage(ctx, msg)
}
