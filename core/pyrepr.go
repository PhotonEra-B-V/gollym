package core

import (
	"fmt"
	"math"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// PyStr renders v like Python's str(): strings as is, everything else as
// PyRepr. pyllym sends str(result) for tool results that aren't content, and
// error messages fall back to str(body), so the model and logs see Python's
// spelling ({'error': '...'}, True, None, 2.0).
func PyStr(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return PyRepr(v)
}

// PyRepr renders v like Python's repr() of the equivalent value: dicts with
// quoted keys, True/False/None, floats with a decimal point. Map keys are
// sorted (Go maps have no insertion order). Structs have no Python
// counterpart and render as JSON, which is what a model can read.
func PyRepr(v any) string {
	if v == nil {
		return "None"
	}
	switch x := v.(type) {
	case string:
		return pyQuote(x)
	case bool:
		if x {
			return "True"
		}
		return "False"
	case float64:
		return pyFloat(x)
	case float32:
		return pyFloat(float64(x))
	case error:
		return x.Error()
	case fmt.Stringer:
		return x.String()
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Pointer, reflect.Interface:
		if rv.IsNil() {
			return "None"
		}
		return PyRepr(rv.Elem().Interface())
	case reflect.Map:
		if rv.IsNil() {
			return "{}"
		}
		parts := make([]string, 0, rv.Len())
		for _, k := range rv.MapKeys() {
			parts = append(parts, PyRepr(k.Interface())+": "+PyRepr(rv.MapIndex(k).Interface()))
		}
		sort.Strings(parts)
		return "{" + strings.Join(parts, ", ") + "}"
	case reflect.Slice, reflect.Array:
		if rv.Kind() == reflect.Slice && rv.Type().Elem().Kind() == reflect.Uint8 {
			return "b" + pyQuote(string(rv.Bytes()))
		}
		parts := make([]string, rv.Len())
		for i := range parts {
			parts[i] = PyRepr(rv.Index(i).Interface())
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case reflect.String:
		return pyQuote(rv.String())
	case reflect.Bool:
		return PyRepr(rv.Bool())
	case reflect.Float32, reflect.Float64:
		return pyFloat(rv.Float())
	case reflect.Struct:
		return MustJSON(v)
	}
	return fmt.Sprint(v)
}

// pyFloat is Python's float repr: shortest round-trip digits, always with a
// decimal point or exponent ("2.0", "0.1", "1e-05", "1e+16").
func pyFloat(f float64) string {
	switch {
	case math.IsNaN(f):
		return "nan"
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	}
	abs := math.Abs(f)
	if abs != 0 && (abs < 1e-4 || abs >= 1e16) {
		return strconv.FormatFloat(f, 'e', -1, 64)
	}
	s := strconv.FormatFloat(f, 'f', -1, 64)
	if !strings.ContainsAny(s, ".") {
		s += ".0"
	}
	return s
}

// pyQuote is Python's str repr: single quotes unless the text has a single
// quote and no double quote; control and unprintable characters escaped.
func pyQuote(s string) string {
	quote := byte('\'')
	if strings.ContainsRune(s, '\'') && !strings.ContainsRune(s, '"') {
		quote = '"'
	}
	var b strings.Builder
	b.WriteByte(quote)
	for _, r := range s {
		switch {
		case r == '\\':
			b.WriteString(`\\`)
		case r == rune(quote):
			b.WriteByte('\\')
			b.WriteRune(r)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r == ' ' || unicode.IsPrint(r):
			b.WriteRune(r)
		case r < 0x100:
			fmt.Fprintf(&b, `\x%02x`, r)
		case r < 0x10000:
			fmt.Fprintf(&b, `\u%04x`, r)
		default:
			fmt.Fprintf(&b, `\U%08x`, r)
		}
	}
	b.WriteByte(quote)
	return b.String()
}
