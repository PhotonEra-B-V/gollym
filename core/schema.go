package core

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// SchemaFromStruct derives a JSON Schema from a struct value or type (Go's
// stand-in for pydantic's model_json_schema). Field names come from json
// tags; `jsonschema:"description=...,enum=a|b,minimum=1,maximum=10"` adds
// constraints. Pointer and omitempty fields are optional; the schema name is
// the struct's type name.
func SchemaFromStruct(v any) (map[string]any, error) {
	t := reflect.TypeOf(v)
	if rt, ok := v.(reflect.Type); ok {
		t = rt
	}
	for t != nil && t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == nil || t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("schema: want a struct, got %T", v)
	}
	s := typeSchema(t, map[reflect.Type]bool{})
	s["title"] = t.Name()
	return map[string]any{"name": t.Name(), "schema": s}, nil
}

var timeType = reflect.TypeOf(time.Time{})

func typeSchema(t reflect.Type, seen map[reflect.Type]bool) map[string]any {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == timeType {
		return map[string]any{"type": "string", "format": "date-time"}
	}
	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.Slice, reflect.Array:
		return map[string]any{"type": "array", "items": typeSchema(t.Elem(), seen)}
	case reflect.Map:
		return map[string]any{"type": "object", "additionalProperties": typeSchema(t.Elem(), seen)}
	case reflect.Interface:
		return map[string]any{}
	case reflect.Struct:
		if seen[t] {
			return map[string]any{"type": "object"}
		}
		seen[t] = true
		defer delete(seen, t)
		props := map[string]any{}
		required := []any{}
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if !f.IsExported() {
				continue
			}
			name, opts := f.Name, ""
			if tag, ok := f.Tag.Lookup("json"); ok {
				if tag == "-" {
					continue
				}
				parts := strings.SplitN(tag, ",", 2)
				if parts[0] != "" {
					name = parts[0]
				}
				if len(parts) > 1 {
					opts = parts[1]
				}
			}
			fs := typeSchema(f.Type, seen)
			applyJSONSchemaTag(fs, f.Tag.Get("jsonschema"))
			props[name] = fs
			if f.Type.Kind() != reflect.Pointer && !strings.Contains(opts, "omitempty") {
				required = append(required, name)
			}
		}
		return map[string]any{"type": "object", "properties": props, "required": required, "additionalProperties": false}
	}
	return map[string]any{}
}

func applyJSONSchemaTag(s map[string]any, tag string) {
	if tag == "" {
		return
	}
	for _, part := range strings.Split(tag, ",") {
		k, v, ok := strings.Cut(part, "=")
		if !ok {
			continue
		}
		switch k {
		case "description", "title", "format", "pattern":
			s[k] = v
		case "enum":
			var vals []any
			for _, e := range strings.Split(v, "|") {
				vals = append(vals, e)
			}
			s["enum"] = vals
		case "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum":
			if f, err := strconv.ParseFloat(v, 64); err == nil {
				s[k] = f
			}
		case "minItems", "maxItems", "minLength", "maxLength":
			if n, err := strconv.Atoi(v); err == nil {
				s[k] = n
			}
		}
	}
}
