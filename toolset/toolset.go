// Package toolset exposes allowlisted Go functions to the model, selected and
// described by a TOML file (pyllym's toolset module).
//
// A general chat model is bad at arithmetic and worse at statistics. The fix
// is tools: hand the model real functions and run the model -> tool -> model
// loop so the numbers come from code, not the model's guesses. This package
// reads a TOML file that lists functions by dotted path and turns each into a
// ready-to-use core.Tool:
//
//	# analysis_tools.toml
//	[[tools]]
//	path = "stats.mean"
//	description = "Arithmetic mean of a list of numbers."
//	[tools.params.data]
//	type = "array"
//	items = "number"
//	description = "The numbers to average."
//
//	[[tools]]
//	path = "stats.median"
//
// Then:
//
//	tools, err := toolset.LoadToolset("analysis_tools.toml", toolset.LoadOptions{})
//	chat.WithTools(false, tools...)
//
// # Go adaptation: a registry instead of imports
//
// pyllym imports each dotted path at runtime (importlib). Go cannot import
// code at runtime, so a dotted path names a function that Go code registered
// beforehand, typically from an init function:
//
//	func init() {
//		toolset.Register("stats.mean", Mean, toolset.Args("data"),
//			toolset.Doc("Return the arithmetic mean of data."))
//	}
//
// The first path segment plays the role of Python's top-level package: a
// path whose package has no registered function at all fails with
// *MissingToolPackageError (pyllym: "package is not installed"), which
// LoadToolset skips by default, so a toolset file can mix always-registered
// and opt-in packages. A known package without the named function is a plain
// configuration error (pyllym's AttributeError case).
//
// Python infers parameter names from the signature; Go reflection has types
// but no names, so Register takes them via Args. Types come from the Go
// signature (int -> integer, float64 -> number, []float64 -> array of number,
// map/struct -> object, ...); Default makes a parameter optional (Python's
// default values). A function of type
//
//	func(ctx context.Context, args map[string]any) (any, error)
//
// receives the raw keyword arguments (Python's **kwargs or an uninspectable
// builtin): its schema comes from the TOML params only, or is an open object.
//
// # Security model
//
// Explicit allowlist only. Only paths named in the config and registered by
// Go code are callable; there is no wildcard expansion and no model-authored
// code path. Registered functions run in-process, so a toolset is exactly as
// trusted as the code it names: review a toolset file like a list of imports.
package toolset

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"reflect"
	"sort"
	"strings"
	"sync"

	"github.com/BurntSushi/toml"

	"github.com/PhotonEra-B-V/gollym/core"
)

// MissingToolPackageError reports a toolset path whose top-level package has
// no registered functions (pyllym: the package is not installed). It matches
// core.ErrConfiguration with errors.Is.
type MissingToolPackageError struct {
	Package string
	Dotted  string
}

func (e *MissingToolPackageError) Error() string {
	return fmt.Sprintf("toolset path %s: package %s is not installed (no function registered under it). "+
		"Install it to use this tool (e.g. import a Go package that calls toolset.Register(%q, ...)).",
		pyRepr(e.Dotted), pyRepr(e.Package), e.Dotted)
}

func (e *MissingToolPackageError) Is(target error) bool { return target == core.ErrConfiguration }

func configErr(format string, args ...any) error {
	return &core.ConfigurationError{Message: fmt.Sprintf(format, args...)}
}

// RawFunc receives the model's keyword arguments unchanged.
type RawFunc = func(ctx context.Context, args map[string]any) (any, error)

var (
	ctxType   = reflect.TypeFor[context.Context]()
	errType   = reflect.TypeFor[error]()
	rawFnType = reflect.TypeFor[RawFunc]()
)

// Option configures a registration.
type Option func(*registration)

type registration struct {
	doc      string
	names    []string
	defaults map[string]any
}

// Doc sets the function's documentation; its first line is the default tool
// description (Python's docstring).
func Doc(doc string) Option { return func(r *registration) { r.doc = doc } }

// Args names the function's parameters in order (excluding a leading
// context.Context and a trailing variadic parameter).
func Args(names ...string) Option {
	return func(r *registration) { r.names = append(r.names, names...) }
}

// Default makes a parameter optional with a default value (nil = the type's
// zero value).
func Default(name string, value any) Option {
	return func(r *registration) {
		if r.defaults == nil {
			r.defaults = map[string]any{}
		}
		r.defaults[name] = value
	}
}

type param struct {
	name     string
	typ      reflect.Type
	required bool
	def      reflect.Value
}

// entry is one registered function.
type entry struct {
	path     string
	doc      string
	raw      RawFunc
	fn       reflect.Value
	hasCtx   bool
	variadic bool
	params   []param
}

// Registry maps dotted paths to registered functions.
type Registry struct {
	mu      sync.RWMutex
	entries map[string]*entry
	roots   map[string]int
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry {
	return &Registry{entries: map[string]*entry{}, roots: map[string]int{}}
}

// DefaultRegistry is the registry used by the package-level functions.
var DefaultRegistry = NewRegistry()

// Register adds fn to the default registry under a dotted path. It panics on
// an invalid registration (a programming error, like http.Handle).
func Register(path string, fn any, opts ...Option) {
	if err := DefaultRegistry.Register(path, fn, opts...); err != nil {
		panic(err)
	}
}

// ToolFromPath wraps a function of the default registry as a tool.
func ToolFromPath(dotted string, o ToolOptions) (*Tool, error) {
	return DefaultRegistry.ToolFromPath(dotted, o)
}

// LoadToolset loads a TOML toolset file against the default registry.
func LoadToolset(path string, o LoadOptions) ([]core.Tool, error) {
	return DefaultRegistry.LoadToolset(path, o)
}

// Register adds fn under a dotted path.
func (r *Registry) Register(path string, fn any, opts ...Option) error {
	if err := checkDotted(path); err != nil {
		return err
	}
	var reg registration
	for _, o := range opts {
		o(&reg)
	}
	e, err := newEntry(path, fn, reg)
	if err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.entries == nil {
		r.entries, r.roots = map[string]*entry{}, map[string]int{}
	}
	if _, dup := r.entries[path]; dup {
		return configErr("toolset path %s is already registered", pyRepr(path))
	}
	r.entries[path] = e
	r.roots[rootOf(path)]++
	return nil
}

// Paths lists the registered paths, sorted.
func (r *Registry) Paths() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, 0, len(r.entries))
	for p := range r.entries {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

func checkDotted(dotted string) error {
	if !strings.Contains(dotted, ".") {
		return configErr("toolset path %s must be a dotted 'module.attr' path", pyRepr(dotted))
	}
	return nil
}

func rootOf(dotted string) string { return strings.SplitN(dotted, ".", 2)[0] }

func newEntry(path string, fn any, reg registration) (*entry, error) {
	if fn == nil {
		return nil, configErr("toolset path %s did not resolve to a callable", pyRepr(path))
	}
	v := reflect.ValueOf(fn)
	if v.Kind() != reflect.Func || v.IsNil() {
		return nil, configErr("toolset path %s did not resolve to a callable", pyRepr(path))
	}
	e := &entry{path: path, doc: reg.doc}
	if v.Type().ConvertibleTo(rawFnType) {
		e.raw = v.Convert(rawFnType).Interface().(RawFunc)
		if len(reg.names) > 0 || len(reg.defaults) > 0 {
			return nil, configErr("toolset path %s: a raw func(ctx, args) takes its parameters from the toolset params, not Args/Default", pyRepr(path))
		}
		return e, nil
	}
	t := v.Type()
	e.fn = v
	in := 0
	if t.NumIn() > 0 && t.In(0) == ctxType {
		e.hasCtx = true
		in = 1
	}
	n := t.NumIn()
	if t.IsVariadic() {
		e.variadic = true
		n--
	}
	if len(reg.names) != n-in {
		return nil, configErr("toolset path %s: Args names %d parameters, the function has %d", pyRepr(path), len(reg.names), n-in)
	}
	seen := map[string]bool{}
	for i, name := range reg.names {
		if name == "" || seen[name] {
			return nil, configErr("toolset path %s: invalid or duplicate argument name %q", pyRepr(path), name)
		}
		seen[name] = true
		p := param{name: name, typ: t.In(in + i), required: true}
		if def, ok := reg.defaults[name]; ok {
			p.required = false
			dv, err := convert(def, p.typ)
			if err != nil {
				return nil, configErr("toolset path %s: default for %q: %v", pyRepr(path), name, err)
			}
			p.def = dv
		}
		e.params = append(e.params, p)
	}
	for name := range reg.defaults {
		if !seen[name] {
			return nil, configErr("toolset path %s: Default for unknown argument %q", pyRepr(path), name)
		}
	}
	switch t.NumOut() {
	case 0, 1:
	case 2:
		if t.Out(1) != errType {
			return nil, configErr("toolset path %s: a function with two results must return (T, error)", pyRepr(path))
		}
	default:
		return nil, configErr("toolset path %s: functions may return at most (T, error)", pyRepr(path))
	}
	return e, nil
}

// lookup resolves a dotted path (pyllym _import_callable).
func (r *Registry) lookup(dotted string) (*entry, error) {
	if err := checkDotted(dotted); err != nil {
		return nil, err
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	root := rootOf(dotted)
	if r.roots[root] == 0 {
		return nil, &MissingToolPackageError{Package: root, Dotted: dotted}
	}
	e, ok := r.entries[dotted]
	if !ok {
		i := strings.LastIndex(dotted, ".")
		return nil, configErr("toolset path %s: %s has no registered attribute %s", pyRepr(dotted), pyRepr(dotted[:i]), pyRepr(dotted[i+1:]))
	}
	return e, nil
}

// ToolOptions override what ToolFromPath infers.
type ToolOptions struct {
	Name        string
	Description string
	// Params overrides per-parameter JSON schema: name -> {type, items,
	// description, required}.
	Params map[string]any
	// ParamOrder orders params that the function signature doesn't declare
	// (their "required" order); unlisted ones follow sorted. LoadToolset
	// fills it from the file.
	ParamOrder []string
}

// Tool is a registered function exposed as a core.Tool.
type Tool struct {
	name        string
	description string
	path        string
	schema      map[string]any
	e           *entry
}

// ToolFromPath wraps the function registered at dotted as a tool. Params
// optionally override per-parameter JSON schema (type, items, description),
// needed when the signature says little or to describe a parameter.
func (r *Registry) ToolFromPath(dotted string, o ToolOptions) (*Tool, error) {
	e, err := r.lookup(dotted)
	if err != nil {
		return nil, err
	}
	name := o.Name
	if name == "" {
		name = defaultName(dotted)
	}
	desc := o.Description
	if desc == "" {
		desc = firstDocLine(e.doc)
	}
	if desc == "" {
		desc = "Call " + dotted + "."
	}
	return &Tool{name: name, description: desc, path: dotted, schema: schemaFor(e, o.Params, o.ParamOrder), e: e}, nil
}

func (t *Tool) Name() string                 { return t.name }
func (t *Tool) Description() string          { return t.description }
func (t *Tool) ParamsSchema() map[string]any { return t.schema }

// Path is the dotted path the tool was built from.
func (t *Tool) Path() string { return t.path }

// Call runs the function with the model's keyword arguments. Like pyllym's
// configured tool (which takes **kwargs) it does no schema validation; a
// call the function can't accept (missing or unknown keyword, wrong type) is
// an error, as Python's TypeError is.
func (t *Tool) Call(ctx context.Context, args map[string]any) (any, error) {
	if args == nil {
		args = map[string]any{}
	}
	slog.Debug("Tool called", "tool", t.name, "args", args)
	out, err := t.e.call(ctx, args)
	if err != nil {
		return nil, err
	}
	out = jsonable(out)
	slog.Debug("Tool returned", "tool", t.name, "result", out)
	return out, nil
}

func (e *entry) call(ctx context.Context, args map[string]any) (any, error) {
	if e.raw != nil {
		return e.raw(ctx, args)
	}
	fname := e.path[strings.LastIndex(e.path, ".")+1:]
	in := make([]reflect.Value, 0, len(e.params)+1)
	if e.hasCtx {
		in = append(in, reflect.ValueOf(ctx))
	}
	known := map[string]bool{}
	for _, p := range e.params {
		known[p.name] = true
		v, ok := args[p.name]
		if !ok {
			if p.required {
				return nil, fmt.Errorf("%s() missing 1 required argument: '%s'", fname, p.name)
			}
			in = append(in, p.def)
			continue
		}
		cv, err := convert(v, p.typ)
		if err != nil {
			return nil, fmt.Errorf("%s() argument '%s': %w", fname, p.name, err)
		}
		in = append(in, cv)
	}
	for _, k := range sortedKeys(args) {
		if !known[k] {
			return nil, fmt.Errorf("%s() got an unexpected keyword argument '%s'", fname, k)
		}
	}
	var outs []reflect.Value
	if e.variadic {
		in = append(in, reflect.MakeSlice(e.fn.Type().In(e.fn.Type().NumIn()-1), 0, 0))
		outs = e.fn.CallSlice(in)
	} else {
		outs = e.fn.Call(in)
	}
	var result any
	for _, o := range outs {
		if o.Type() == errType {
			if !o.IsNil() {
				return nil, o.Interface().(error)
			}
			continue
		}
		result = o.Interface()
	}
	return result, nil
}

// convert turns a JSON value from the model into a Go parameter value.
func convert(v any, t reflect.Type) (reflect.Value, error) {
	if v == nil {
		return reflect.Zero(t), nil
	}
	if rv := reflect.ValueOf(v); rv.Type().AssignableTo(t) {
		out := reflect.New(t).Elem()
		out.Set(rv)
		return out, nil
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return reflect.Value{}, err
	}
	ptr := reflect.New(t)
	if err := json.Unmarshal(raw, ptr.Interface()); err != nil {
		return reflect.Value{}, err
	}
	return ptr.Elem(), nil
}

// jsonable coerces composite Go results (slices, maps, structs) to plain
// JSON values so they stringify like pyllym's (pyllym _jsonable).
func jsonable(v any) any {
	switch v.(type) {
	case nil, *core.Content, *core.SearchResults, core.RawContent, map[string]any, []any, string, error:
		return v
	}
	rv := reflect.ValueOf(v)
	k := rv.Kind()
	if k == reflect.Pointer {
		if rv.IsNil() {
			return nil
		}
		k = rv.Elem().Kind()
	}
	switch k {
	case reflect.Slice, reflect.Array, reflect.Map, reflect.Struct:
		raw, err := json.Marshal(v)
		if err != nil {
			return v
		}
		var out any
		if json.Unmarshal(raw, &out) != nil {
			return v
		}
		return out
	}
	return v
}

// schemaFor builds the params object schema (pyllym _schema_for): types from
// the signature, config overrides win, and config may add params the
// signature doesn't expose.
func schemaFor(e *entry, overrides map[string]any, order []string) map[string]any {
	props := map[string]any{}
	required := []any{}
	for _, p := range e.params {
		props[p.name] = paramSchema(schemaFromType(p.typ), overrides[p.name])
		if p.required {
			required = append(required, p.name)
		}
	}
	for _, name := range orderedNames(overrides, order) {
		if _, ok := props[name]; ok {
			continue
		}
		spec := overrides[name]
		props[name] = paramSchema(map[string]any{"type": "string"}, spec)
		if m, ok := spec.(map[string]any); ok {
			if req, ok := m["required"]; !ok || core.Truthy(req) {
				required = append(required, name)
			}
		}
	}
	if len(props) == 0 {
		return map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": true}
	}
	return map[string]any{"type": "object", "properties": props, "required": required, "additionalProperties": false}
}

func orderedNames(m map[string]any, order []string) []string {
	out := make([]string, 0, len(m))
	seen := map[string]bool{}
	for _, n := range order {
		if _, ok := m[n]; ok && !seen[n] {
			seen[n] = true
			out = append(out, n)
		}
	}
	for _, n := range sortedKeys(m) {
		if !seen[n] {
			out = append(out, n)
		}
	}
	return out
}

// paramSchema applies a config override to an inferred schema (pyllym
// _param_schema).
func paramSchema(schema map[string]any, override any) map[string]any {
	o, ok := override.(map[string]any)
	if !ok {
		return schema
	}
	if t, ok := o["type"]; ok {
		schema = map[string]any{"type": t}
		if t == "array" {
			items, ok := o["items"]
			if !ok {
				items = "string"
			}
			schema["items"] = map[string]any{"type": items}
		}
	}
	if items, ok := o["items"]; ok && schema["type"] == "array" {
		schema["items"] = map[string]any{"type": items}
	}
	if d, ok := o["description"]; ok && core.Truthy(d) {
		schema["description"] = d
	}
	return schema
}

// schemaFromType maps a Go type to JSON Schema (pyllym
// _schema_from_annotation).
func schemaFromType(t reflect.Type) map[string]any {
	switch t.Kind() {
	case reflect.Pointer:
		return schemaFromType(t.Elem()) // Optional[X] advertises X
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.Slice, reflect.Array:
		if t.Elem().Kind() == reflect.Uint8 {
			return map[string]any{"type": "string"} // []byte is base64 text in JSON
		}
		return map[string]any{"type": "array", "items": schemaFromType(t.Elem())}
	case reflect.Map, reflect.Struct:
		return map[string]any{"type": "object"}
	}
	return map[string]any{"type": "string"}
}

func firstDocLine(doc string) string {
	doc = strings.TrimSpace(doc)
	if doc == "" {
		return ""
	}
	return strings.TrimSpace(strings.SplitN(doc, "\n", 2)[0])
}

// defaultName: "stats.mean" -> "stats_mean" (the last two segments).
func defaultName(dotted string) string {
	parts := strings.Split(dotted, ".")
	if len(parts) > 2 {
		parts = parts[len(parts)-2:]
	}
	return strings.ReplaceAll(strings.Join(parts, "_"), "-", "_")
}

// LoadOptions configure LoadToolset.
type LoadOptions struct {
	// Strict fails on an entry whose package isn't registered instead of
	// skipping it with a warning (pyllym skip_missing=False).
	Strict bool
}

// LoadToolset reads a TOML toolset file and returns one tool per [[tools]]
// entry. Malformed files, entries without a path, unknown paths and
// duplicate tool names are configuration errors; entries whose package has
// no registered functions are skipped with a warning unless Strict.
func (r *Registry) LoadToolset(file string, o LoadOptions) ([]core.Tool, error) {
	text, err := os.ReadFile(file)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, configErr("toolset file not found: %s", file)
		}
		return nil, configErr("could not read toolset %s: %v", file, err)
	}
	var data map[string]any
	md, err := toml.Decode(string(text), &data)
	if err != nil {
		return nil, configErr("invalid TOML in toolset %s: %v", file, err)
	}
	var entries []any
	switch v := data["tools"].(type) {
	case []map[string]any:
		for _, m := range v {
			entries = append(entries, m)
		}
	case []any:
		entries = v
	default:
		return nil, configErr("toolset %s must define an array of [[tools]] tables", file)
	}
	orders := paramOrders(md)

	var tools []core.Tool
	seen := map[string]bool{}
	for index, raw := range entries {
		entry, ok := raw.(map[string]any)
		if !ok {
			return nil, configErr("toolset %s: [[tools]] entry %d is not a table", file, index)
		}
		dotted, _ := entry["path"].(string)
		if dotted == "" {
			return nil, configErr("toolset %s: [[tools]] entry %d is missing 'path'", file, index)
		}
		opts := ToolOptions{}
		opts.Name, _ = entry["name"].(string)
		opts.Description, _ = entry["description"].(string)
		if p, ok := entry["params"]; ok {
			if opts.Params, ok = p.(map[string]any); !ok {
				return nil, configErr("toolset %s: [[tools]] entry %d: 'params' must be a table", file, index)
			}
		}
		if index < len(orders) {
			opts.ParamOrder = orders[index]
		}
		tool, err := r.ToolFromPath(dotted, opts)
		if err != nil {
			var missing *MissingToolPackageError
			if errors.As(err, &missing) && !o.Strict {
				slog.Warn(fmt.Sprintf("toolset %s: skipping %s — %v", file, pyRepr(dotted), err))
				continue
			}
			return nil, err
		}
		if seen[tool.Name()] {
			return nil, configErr("toolset %s: duplicate tool name %s (from %s); set an explicit 'name'", file, pyRepr(tool.Name()), pyRepr(dotted))
		}
		seen[tool.Name()] = true
		tools = append(tools, tool)
	}
	return tools, nil
}

// paramOrders recovers each [[tools]] entry's params in file order (Go maps
// lose it; pyllym's dicts keep it).
func paramOrders(md toml.MetaData) [][]string {
	var out [][]string
	for _, k := range md.Keys() {
		switch {
		case len(k) == 1 && k[0] == "tools":
			out = append(out, nil)
		case len(k) == 3 && k[0] == "tools" && k[1] == "params" && len(out) > 0:
			out[len(out)-1] = append(out[len(out)-1], k[2])
		}
	}
	return out
}

func sortedKeys(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// pyRepr quotes a string like Python's repr().
func pyRepr(s string) string {
	if strings.Contains(s, "'") && !strings.Contains(s, `"`) {
		return `"` + s + `"`
	}
	return "'" + strings.ReplaceAll(strings.ReplaceAll(s, `\`, `\\`), "'", `\'`) + "'"
}
