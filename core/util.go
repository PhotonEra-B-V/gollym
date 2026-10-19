package core

import (
	"encoding/json"
	"maps"
	"regexp"
	"strings"
	"time"
)

// M is a JSON object as protocols build and parse it (pyllym's dict
// payloads). Payloads stay maps so provider params can be deep-merged in.
type M = map[string]any

// Dig walks nested maps and slices; int keys index slices (negative from the
// end), string keys index maps. Any miss returns nil.
func Dig(data any, keys ...any) any {
	for _, key := range keys {
		if data == nil {
			return nil
		}
		switch k := key.(type) {
		case int:
			s, ok := data.([]any)
			if !ok {
				return nil
			}
			if k < 0 {
				k += len(s)
			}
			if k < 0 || k >= len(s) {
				return nil
			}
			data = s[k]
		case string:
			m, ok := data.(map[string]any)
			if !ok {
				return nil
			}
			data = m[k]
		default:
			return nil
		}
	}
	return data
}

// DigMap is Dig asserting a map result.
func DigMap(data any, keys ...any) map[string]any {
	m, _ := Dig(data, keys...).(map[string]any)
	return m
}

// DigSlice is Dig asserting a slice result.
func DigSlice(data any, keys ...any) []any {
	s, _ := Dig(data, keys...).([]any)
	return s
}

// DigString is Dig asserting a string result ("" otherwise).
func DigString(data any, keys ...any) string {
	s, _ := Dig(data, keys...).(string)
	return s
}

// AsInt converts a decoded JSON number (float64, json.Number, int) to *int.
func AsInt(v any) *int {
	switch n := v.(type) {
	case float64:
		i := int(n)
		return &i
	case int:
		return &n
	case int64:
		i := int(n)
		return &i
	case json.Number:
		if i, err := n.Int64(); err == nil {
			x := int(i)
			return &x
		}
		if f, err := n.Float64(); err == nil {
			x := int(f)
			return &x
		}
	}
	return nil
}

// AsFloat converts a decoded JSON number to *float64.
func AsFloat(v any) *float64 {
	switch n := v.(type) {
	case float64:
		return &n
	case int:
		f := float64(n)
		return &f
	case int64:
		f := float64(n)
		return &f
	case json.Number:
		if f, err := n.Float64(); err == nil {
			return &f
		}
	}
	return nil
}

// Truthy is Python truthiness for decoded JSON values.
func Truthy(v any) bool {
	switch x := v.(type) {
	case nil:
		return false
	case bool:
		return x
	case string:
		return x != ""
	case float64:
		return x != 0
	case int:
		return x != 0
	case int64:
		return x != 0
	case json.Number:
		f, err := x.Float64()
		return err != nil || f != 0
	case []any:
		return len(x) > 0
	case map[string]any:
		return len(x) > 0
	}
	return true
}

// Or returns the first truthy value (Python's `a or b or c`), else the last.
func Or(vals ...any) any {
	for _, v := range vals {
		if Truthy(v) {
			return v
		}
	}
	if len(vals) == 0 {
		return nil
	}
	return vals[len(vals)-1]
}

// DeepMerge recursively merges overrides into a copy of original.
func DeepMerge(original, overrides map[string]any) map[string]any {
	out := maps.Clone(original)
	if out == nil {
		out = map[string]any{}
	}
	for k, ov := range overrides {
		if om, ok := out[k].(map[string]any); ok {
			if nm, ok := ov.(map[string]any); ok {
				out[k] = DeepMerge(om, nm)
				continue
			}
		}
		out[k] = ov
	}
	return out
}

// DeepCopy copies nested maps and slices of decoded JSON.
func DeepCopy(v any) any {
	switch x := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, val := range x {
			out[k] = DeepCopy(val)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, val := range x {
			out[i] = DeepCopy(val)
		}
		return out
	}
	return v
}

// Compact drops nil values (including typed nil pointers) from a map.
func Compact(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		if isNil(v) {
			continue
		}
		out[k] = v
	}
	return out
}

func isNil(v any) bool {
	if v == nil {
		return true
	}
	switch x := v.(type) {
	case *string:
		return x == nil
	case *int:
		return x == nil
	case *float64:
		return x == nil
	case *bool:
		return x == nil
	case map[string]any:
		return x == nil
	case []any:
		return x == nil
	}
	return false
}

// ToSafeArray wraps a value in a slice unless it already is one (nil -> []).
func ToSafeArray(v any) []any {
	switch x := v.(type) {
	case nil:
		return nil
	case []any:
		return x
	case []string:
		out := make([]any, len(x))
		for i, s := range x {
			out[i] = s
		}
		return out
	}
	return []any{v}
}

var (
	acronymRe1 = regexp.MustCompile(`([A-Z]+)([A-Z][a-z])`)
	acronymRe2 = regexp.MustCompile(`([a-z\d])([A-Z])`)
)

// Underscore is acronym-aware snake_casing: HTTPProxyTool -> http_proxy_tool.
func Underscore(name string) string {
	name = acronymRe1.ReplaceAllString(name, "${1}_${2}")
	name = acronymRe2.ReplaceAllString(name, "${1}_${2}")
	return strings.ToLower(name)
}

// ToTime parses ISO timestamps (pyllym utils.to_time). Like Python's
// datetime.fromisoformat it doesn't read unix seconds: those are nil. Unlike
// it, it reads models.json's "2006-01-02 15:04:05 UTC", which pyllym drops.
func ToTime(v any) *time.Time {
	switch x := v.(type) {
	case nil:
		return nil
	case time.Time:
		return &x
	case string:
		s := strings.TrimSpace(x)
		if s == "" {
			return nil
		}
		s = strings.Replace(s, "Z", "+00:00", 1)
		for _, layout := range []string{time.RFC3339Nano, "2006-01-02T15:04:05-07:00", "2006-01-02T15:04:05", "2006-01-02 15:04:05-07:00", "2006-01-02 15:04:05 UTC", "2006-01-02 15:04:05", "2006-01-02"} {
			if t, err := time.Parse(layout, s); err == nil {
				return &t
			}
		}
	}
	return nil
}

var (
	isoDay   = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`)
	isoMonth = regexp.MustCompile(`^\d{4}-\d{2}$`)
	isoYear  = regexp.MustCompile(`^\d{4}$`)
)

// ParseISODatePrefix parses YYYY, YYYY-MM or YYYY-MM-DD (missing parts = 1).
func ParseISODatePrefix(v any) *time.Time {
	if t, ok := v.(time.Time); ok {
		d := time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
		return &d
	}
	s, ok := v.(string)
	if !ok {
		return nil
	}
	s = strings.TrimSpace(s)
	var layout string
	switch {
	case isoDay.MatchString(s):
		layout = "2006-01-02"
	case isoMonth.MatchString(s):
		layout = "2006-01"
	case isoYear.MatchString(s):
		layout = "2006"
	default:
		return nil
	}
	t, err := time.Parse(layout, s)
	if err != nil {
		return nil
	}
	return &t
}

// MustJSON marshals v without HTML escaping, as Python's json.dumps output
// is compared against (separators are Go's compact form).
func MustJSON(v any) string {
	var b strings.Builder
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return ""
	}
	return strings.TrimSuffix(b.String(), "\n")
}

// PyJSON renders v like Python's json.dumps defaults (", " and ": "
// separators, non-ASCII escaped as \uXXXX). Used where pyllym serialises
// values into text the model reads (tool arguments, raw content).
func PyJSON(v any) string {
	compact := MustJSON(v)
	var b strings.Builder
	inString := false
	escaped := false
	for _, r := range compact {
		if inString {
			switch {
			case escaped:
				escaped = false
				b.WriteRune(r)
			case r == '\\':
				escaped = true
				b.WriteRune(r)
			case r == '"':
				inString = false
				b.WriteRune(r)
			case r > 0x7e:
				writeUnicodeEscape(&b, r)
			default:
				b.WriteRune(r)
			}
			continue
		}
		switch r {
		case '"':
			inString = true
			b.WriteRune(r)
		case ',':
			b.WriteString(", ")
		case ':':
			b.WriteString(": ")
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

func writeUnicodeEscape(b *strings.Builder, r rune) {
	const hex = "0123456789abcdef"
	write := func(u uint16) {
		b.WriteString(`\u`)
		b.WriteByte(hex[u>>12&0xf])
		b.WriteByte(hex[u>>8&0xf])
		b.WriteByte(hex[u>>4&0xf])
		b.WriteByte(hex[u&0xf])
	}
	if r > 0xffff {
		r -= 0x10000
		write(uint16(0xd800 + (r>>10)&0x3ff))
		write(uint16(0xdc00 + r&0x3ff))
		return
	}
	write(uint16(r))
}
