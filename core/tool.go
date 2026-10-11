package core

import (
	"context"
	"fmt"
	"log/slog"
	"reflect"
	"regexp"
	"strings"
)

// Tool is a function the model may call.
type Tool interface {
	Name() string
	Description() string
	// ParamsSchema is the JSON Schema of the arguments (nil = no params).
	ParamsSchema() map[string]any
	// Call runs the tool. Returning a *Content, *SearchResults or RawContent
	// sends it as the tool result as is; anything else is stringified.
	Call(ctx context.Context, args map[string]any) (any, error)
}

// ProviderParamser adds provider-specific fields to a tool definition.
type ProviderParamser interface {
	ProviderParams() map[string]any
}

// Parameter declares one tool argument.
type Parameter struct {
	Name        string
	Type        string // string (default), integer, number, boolean, array, object
	Description string
	Required    bool
}

// Schema renders the parameter's JSON Schema.
func (p Parameter) Schema() map[string]any {
	s := map[string]any{"type": mapParamType(p.Type)}
	if p.Description != "" {
		s["description"] = p.Description
	}
	if s["type"] == "array" {
		s["items"] = map[string]any{"type": "string"}
	}
	return s
}

func mapParamType(t string) string {
	switch strings.ToLower(t) {
	case "integer", "int":
		return "integer"
	case "number", "float", "double":
		return "number"
	case "boolean", "bool":
		return "boolean"
	case "array":
		return "array"
	case "object":
		return "object"
	}
	return "string"
}

// SchemaFromParameters builds the object schema (nil when empty unless
// allowEmpty).
func SchemaFromParameters(params []Parameter, allowEmpty bool) map[string]any {
	if len(params) == 0 && !allowEmpty {
		return nil
	}
	props := map[string]any{}
	required := []any{}
	for _, p := range params {
		props[p.Name] = p.Schema()
		if p.Required {
			required = append(required, p.Name)
		}
	}
	return map[string]any{
		"type": "object", "properties": props, "required": required,
		"additionalProperties": false, "strict": true,
	}
}

// FuncTool is a Tool built from a function.
type FuncTool struct {
	ToolName    string
	Desc        string
	Params      []Parameter
	Schema      map[string]any // overrides Params when set
	Extra       map[string]any // provider params
	AcceptExtra bool           // accept unknown keyword arguments
	Fn          func(ctx context.Context, args map[string]any) (any, error)
}

// NewTool builds a FuncTool.
func NewTool(name, description string, params []Parameter, fn func(ctx context.Context, args map[string]any) (any, error)) *FuncTool {
	return &FuncTool{ToolName: name, Desc: description, Params: params, Fn: fn}
}

func (t *FuncTool) Name() string        { return t.ToolName }
func (t *FuncTool) Description() string { return t.Desc }

func (t *FuncTool) ParamsSchema() map[string]any {
	if t.Schema != nil {
		return t.Schema
	}
	return SchemaFromParameters(t.Params, len(t.Params) == 0)
}

func (t *FuncTool) ProviderParams() map[string]any { return t.Extra }

// Call validates keyword arguments against Params (missing required /
// unknown keyword), then runs Fn. Validation errors are returned to the
// model as {"error": "Invalid tool arguments: ..."}, not as Go errors.
func (t *FuncTool) Call(ctx context.Context, args map[string]any) (any, error) {
	if args == nil {
		args = map[string]any{}
	}
	if t.Schema == nil {
		if msg := validateArgs(t.Params, t.AcceptExtra, args); msg != "" {
			return map[string]any{"error": "Invalid tool arguments: " + msg}, nil
		}
	}
	slog.Debug("Tool called", "tool", t.ToolName, "args", args)
	out, err := t.Fn(ctx, args)
	slog.Debug("Tool returned", "tool", t.ToolName, "result", out)
	return out, err
}

func validateArgs(params []Parameter, acceptExtra bool, args map[string]any) string {
	for _, p := range params {
		if _, ok := args[p.Name]; p.Required && !ok {
			return "missing keyword: " + p.Name
		}
	}
	if acceptExtra {
		return ""
	}
	allowed := map[string]bool{}
	for _, p := range params {
		allowed[p.Name] = true
	}
	for _, k := range sortedKeys(args) {
		if !allowed[k] {
			return "unknown keyword: " + k
		}
	}
	return ""
}

var toolNameRe = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// ToolNameFor derives pyllym's tool name from a Go type name:
// WeatherLookupTool -> weather_lookup.
func ToolNameFor(v any) string {
	t := reflect.TypeOf(v)
	for t != nil && t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == nil {
		return ""
	}
	return ToolNameFromString(t.Name())
}

// ToolNameFromString applies pyllym's class-name rules to a name.
func ToolNameFromString(raw string) string {
	var ascii strings.Builder
	for _, r := range raw {
		if r < 128 {
			ascii.WriteRune(r)
		}
	}
	name := toolNameRe.ReplaceAllString(ascii.String(), "-")
	return strings.TrimSuffix(Underscore(name), "_tool")
}

// ToolSet is an insertion-ordered set of tools by name.
type ToolSet struct {
	names  []string
	byName map[string]Tool
}

// Add adds or replaces a tool.
func (s *ToolSet) Add(t Tool) {
	if s.byName == nil {
		s.byName = map[string]Tool{}
	}
	if _, ok := s.byName[t.Name()]; !ok {
		s.names = append(s.names, t.Name())
	}
	s.byName[t.Name()] = t
}

// Get returns a tool by name.
func (s *ToolSet) Get(name string) (Tool, bool) {
	if s == nil || s.byName == nil {
		return nil, false
	}
	t, ok := s.byName[name]
	return t, ok
}

// List returns tools in insertion order.
func (s *ToolSet) List() []Tool {
	if s == nil {
		return nil
	}
	out := make([]Tool, 0, len(s.names))
	for _, n := range s.names {
		out = append(out, s.byName[n])
	}
	return out
}

// Names returns tool names in insertion order.
func (s *ToolSet) Names() []string {
	if s == nil {
		return nil
	}
	return append([]string(nil), s.names...)
}

// Len is the number of tools.
func (s *ToolSet) Len() int {
	if s == nil {
		return 0
	}
	return len(s.names)
}

// Clear removes every tool.
func (s *ToolSet) Clear() { s.names, s.byName = nil, nil }

// ToolArguments returns a call's arguments as a map.
func ToolArguments(tc *ToolCall) map[string]any {
	switch a := tc.Arguments.(type) {
	case map[string]any:
		return a
	case nil:
		return map[string]any{}
	}
	return map[string]any{}
}

func toolResultString(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case nil:
		return "None"
	case map[string]any, []any:
		return pyRepr(x)
	}
	return fmt.Sprint(v)
}

// pyRepr approximates Python's str() of dicts/lists (single-quoted keys),
// which is what pyllym sends for non-content tool results.
func pyRepr(v any) string {
	switch x := v.(type) {
	case nil:
		return "None"
	case bool:
		if x {
			return "True"
		}
		return "False"
	case string:
		return "'" + strings.ReplaceAll(strings.ReplaceAll(x, `\`, `\\`), "'", `\'`) + "'"
	case float64:
		if x == float64(int64(x)) && x < 1e16 && x > -1e16 {
			return fmt.Sprintf("%d", int64(x))
		}
		return fmt.Sprint(x)
	case map[string]any:
		parts := make([]string, 0, len(x))
		for _, k := range sortedKeys(x) {
			parts = append(parts, pyRepr(k)+": "+pyRepr(x[k]))
		}
		return "{" + strings.Join(parts, ", ") + "}"
	case []any:
		parts := make([]string, len(x))
		for i, e := range x {
			parts[i] = pyRepr(e)
		}
		return "[" + strings.Join(parts, ", ") + "]"
	}
	return fmt.Sprint(v)
}
