package gemini

import (
	"errors"
	"sort"
	"strings"

	"github.com/PhotonEra-B-V/gollym/core"
)

// --- tool parameters: JSON Schema -> Gemini function parameters --------------

// ConvertToolSchema converts a tool's JSON Schema (which must be an object)
// into Gemini's uppercase-typed parameter schema.
func ConvertToolSchema(schema map[string]any) (map[string]any, error) {
	if len(schema) == 0 {
		return nil, nil
	}
	s, _ := normalizeJSON(schema).(map[string]any)
	if s["type"] != "object" {
		return nil, errors.New("Gemini tool parameters must be objects")
	}
	props := map[string]any{}
	if pm, ok := core.Or(s["properties"], map[string]any{}).(map[string]any); ok {
		for k, v := range pm {
			props[k] = convertProperty(asMap(v))
		}
	}
	required := []any{}
	for _, r := range listOf(core.Or(s["required"], []any{})) {
		required = append(required, pyStr(r))
	}
	return map[string]any{"type": "OBJECT", "properties": props, "required": required}, nil
}

func convertProperty(property map[string]any) map[string]any {
	working := property
	if normalized := normalizeAnyOfSchema(property); len(normalized) > 0 {
		working = normalized
	}
	typ := paramTypeForGemini(working["type"])
	prop := map[string]any{"type": typ}
	copyAttributes(prop, property, commonAttrs)
	copyAttributes(prop, working, commonAttrs)
	switch typ {
	case "ARRAY":
		items := core.Or(working["items"], property["items"], map[string]any{"type": "string"})
		prop["items"] = convertProperty(asMap(items))
		copyAttributes(prop, working, []string{"minItems", "maxItems"})
		copyAttributes(prop, property, []string{"minItems", "maxItems"})
	case "OBJECT":
		nested := map[string]any{}
		if pm, ok := working["properties"].(map[string]any); ok {
			for k, v := range pm {
				nested[k] = convertProperty(asMap(v))
			}
		}
		prop["properties"] = nested
		if required := core.Or(working["required"], property["required"]); required != nil {
			list := []any{}
			for _, r := range listOf(required) {
				list = append(list, pyStr(r))
			}
			prop["required"] = list
		}
	}
	return prop
}

func normalizeAnyOfSchema(schema map[string]any) map[string]any {
	anyOf, ok := schema["anyOf"].([]any)
	if !ok || len(anyOf) == 0 {
		return nil
	}
	var nulls, nonNull []any
	for _, e := range anyOf {
		if strings.ToLower(pyStr(schemaType(e))) == "null" {
			nulls = append(nulls, e)
		} else {
			nonNull = append(nonNull, e)
		}
	}
	if len(nonNull) == 1 && len(nulls) > 0 {
		normalized := asMap(core.DeepCopy(nonNull[0]))
		normalized["nullable"] = true
		return normalized
	}
	if len(nonNull) > 0 {
		return asMap(core.DeepCopy(nonNull[0]))
	}
	return map[string]any{"type": "string", "nullable": true}
}

var paramTypes = map[string]string{
	"integer": "INTEGER", "number": "NUMBER", "float": "NUMBER", "double": "NUMBER",
	"boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT",
}

func paramTypeForGemini(t any) string {
	var s string
	if core.Truthy(t) {
		s = pyStr(t)
	}
	if v, ok := paramTypes[strings.ToLower(s)]; ok {
		return v
	}
	return "STRING"
}

func schemaType(s any) any {
	m, ok := s.(map[string]any)
	if !ok {
		return nil
	}
	return m["type"]
}

var commonAttrs = []string{"description", "enum", "format", "nullable", "maximum", "minimum", "multipleOf"}

var attrAliases = map[string][]string{
	"multipleOf": {"multipleOf", "multiple_of"},
	"minItems":   {"minItems", "min_items"},
	"maxItems":   {"maxItems", "max_items"},
}

func schemaValue(source map[string]any, attr string) any {
	keys, ok := attrAliases[attr]
	if !ok {
		keys = []string{attr}
	}
	for _, k := range keys {
		if v := source[k]; v != nil {
			return v
		}
	}
	return nil
}

func copyAttributes(target, source map[string]any, attrs []string) {
	for _, a := range attrs {
		if v := schemaValue(source, a); v != nil {
			target[a] = v
		}
	}
}

// --- structured output: JSON Schema -> Gemini responseSchema -----------------

// ConvertResponseSchema converts a JSON Schema into Gemini's responseSchema
// dialect: $defs are inlined (recursive refs cut off), anyOf collapses to
// the first non-null option (+ nullable), unsupported keys are dropped.
// Returns nil for an empty schema.
func ConvertResponseSchema(schema any) map[string]any {
	c := &schemaConverter{}
	raw := normalizeJSON(schema)
	if !core.Truthy(raw) {
		return nil
	}
	extracted := c.extractDefinitions(raw)
	return c.convert(extracted, map[string]bool{})
}

type schemaConverter struct {
	definitions map[string]any
}

func (c *schemaConverter) extractDefinitions(v any) any {
	switch x := v.(type) {
	case map[string]any:
		out := map[string]any{}
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			if k == "$defs" || k == "definitions" {
				c.mergeDefinitions(x[k])
			} else {
				out[k] = c.extractDefinitions(x[k])
			}
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, e := range x {
			out[i] = c.extractDefinitions(e)
		}
		return out
	}
	return v
}

func (c *schemaConverter) mergeDefinitions(raw any) {
	if !core.Truthy(raw) {
		return
	}
	extracted, ok := c.extractDefinitions(raw).(map[string]any)
	if !ok {
		return
	}
	if len(c.definitions) == 0 {
		c.definitions = extracted
	} else {
		c.definitions = core.DeepMerge(c.definitions, extracted)
	}
}

func (c *schemaConverter) convert(schema any, visited map[string]bool) map[string]any {
	s, ok := schema.(map[string]any)
	if !ok {
		return map[string]any{"type": "STRING"}
	}
	s = stripUnsupportedKeys(s)
	if core.Truthy(s["$ref"]) {
		if resolved := c.resolveReference(s, visited); resolved != nil {
			return resolved
		}
	}
	s = normalizeAnyOf(s)
	var typ string
	if core.Truthy(s["type"]) {
		typ = pyStr(s["type"])
	}
	var result map[string]any
	switch typ {
	case "object":
		result = c.buildObject(s, visited)
	case "array":
		result = c.buildArray(s, visited)
	case "number":
		result = buildScalar("NUMBER", s, "format", "minimum", "maximum", "enum", "nullable", "multipleOf")
	case "integer":
		result = buildScalar("INTEGER", s, "format", "minimum", "maximum", "enum", "nullable", "multipleOf")
	case "boolean":
		result = buildScalar("BOOLEAN", s, "nullable")
	default:
		result = buildScalar("STRING", s, "enum", "format", "nullable")
	}
	if core.Truthy(s["description"]) {
		result["description"] = s["description"]
	}
	return result
}

func stripUnsupportedKeys(s map[string]any) map[string]any {
	out := make(map[string]any, len(s))
	for k, v := range s {
		if k != "strict" && k != "additionalProperties" {
			out[k] = v
		}
	}
	return out
}

func (c *schemaConverter) resolveReference(s map[string]any, visited map[string]bool) map[string]any {
	ref := pyStr(s["$ref"])
	if ref == "" || visited[ref] {
		return nil
	}
	referenced, ok := c.lookupDefinition(ref).(map[string]any)
	if !ok {
		return nil
	}
	overrides := map[string]any{}
	for k, v := range s {
		if k != "$ref" {
			overrides[k] = v
		}
	}
	visited[ref] = true
	defer delete(visited, ref)
	return c.convert(core.DeepMerge(referenced, overrides), visited)
}

func (c *schemaConverter) lookupDefinition(ref string) any {
	var segments []string
	for _, s := range strings.Split(ref, "/") {
		if s != "" {
			segments = append(segments, s)
		}
	}
	if len(segments) == 0 {
		return nil
	}
	if segments[0] == "#" {
		segments = segments[1:]
	}
	if len(segments) > 0 && (segments[0] == "$defs" || segments[0] == "definitions") {
		segments = segments[1:]
	}
	var current any = c.definitions
	for _, seg := range segments {
		m, ok := current.(map[string]any)
		if !ok {
			return nil
		}
		current = m[seg]
	}
	if !core.Truthy(current) {
		return nil
	}
	return core.DeepCopy(current)
}

func normalizeAnyOf(s map[string]any) map[string]any {
	if !core.Truthy(s["anyOf"]) {
		return s
	}
	var nulls, nonNull []any
	for _, o := range listOf(s["anyOf"]) {
		if strings.ToLower(typeString(o)) == "null" {
			nulls = append(nulls, o)
		} else {
			nonNull = append(nonNull, o)
		}
	}
	base := map[string]any{"type": "string"}
	if len(nonNull) > 0 {
		base = map[string]any{}
		for k, v := range asMap(nonNull[0]) {
			base[k] = v
		}
	}
	if len(nulls) > 0 {
		base["nullable"] = true
	}
	out := map[string]any{}
	for k, v := range s {
		if k != "anyOf" {
			out[k] = v
		}
	}
	for k, v := range base {
		out[k] = v
	}
	return out
}

func typeString(o any) string {
	t := schemaType(o)
	if !core.Truthy(t) {
		return ""
	}
	return pyStr(t)
}

func (c *schemaConverter) buildObject(s map[string]any, visited map[string]bool) map[string]any {
	props := map[string]any{}
	if pm, ok := s["properties"].(map[string]any); ok {
		for k, v := range pm {
			props[k] = c.convert(v, visited)
		}
	}
	obj := map[string]any{"type": "OBJECT", "properties": props}
	var required []any
	seen := map[string]bool{}
	for _, r := range core.ToSafeArray(s["required"]) {
		name := pyStr(r)
		if !seen[name] {
			seen[name] = true
			required = append(required, name)
		}
	}
	if len(required) > 0 {
		obj["required"] = required
	}
	if core.Truthy(s["propertyOrdering"]) {
		obj["propertyOrdering"] = s["propertyOrdering"]
	}
	if v, ok := s["nullable"]; ok {
		obj["nullable"] = v
	}
	return obj
}

func (c *schemaConverter) buildArray(s map[string]any, visited map[string]bool) map[string]any {
	items := map[string]any{"type": "STRING"}
	if core.Truthy(s["items"]) {
		items = c.convert(s["items"], visited)
	}
	arr := map[string]any{"type": "ARRAY", "items": items}
	for _, k := range []string{"minItems", "maxItems", "nullable"} {
		if v, ok := s[k]; ok {
			arr[k] = v
		}
	}
	return arr
}

func buildScalar(typ string, s map[string]any, allowed ...string) map[string]any {
	out := map[string]any{"type": typ}
	for _, k := range allowed {
		if v, ok := s[k]; ok {
			out[k] = v
		}
	}
	return out
}

// --- helpers -----------------------------------------------------------------

func asMap(v any) map[string]any {
	if m, ok := v.(map[string]any); ok {
		return m
	}
	return map[string]any{}
}

// listOf is Python's list(x) for decoded JSON lists (scalars wrap).
func listOf(v any) []any {
	switch x := v.(type) {
	case nil:
		return nil
	case []any:
		return x
	case []string:
		return core.ToSafeArray(x)
	case string:
		out := make([]any, 0, len(x))
		for _, r := range x {
			out = append(out, string(r))
		}
		return out
	}
	return []any{v}
}
