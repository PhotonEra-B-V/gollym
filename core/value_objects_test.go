package core_test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/PhotonEra-B-V/gollym/core"
)

// Ports tests/test_value_objects.py.

func TestTokensBuildNilWhenAllEmpty(t *testing.T) {
	if core.BuildTokens(nil, nil, nil, nil, nil) != nil {
		t.Fatal("want nil")
	}
	tk := core.BuildTokens(core.Int(10), core.Int(5), nil, nil, nil)
	if !ptrEq(tk.Input, 10) || tk.CacheRead() != nil {
		t.Fatalf("tokens %+v", tk)
	}
	if got := tk.Map(); !reflect.DeepEqual(got, map[string]any{"input_tokens": 10, "output_tokens": 5}) {
		t.Fatalf("map %v", got)
	}
}

func TestThinkingBuildAndRedaction(t *testing.T) {
	if core.BuildThinking("", "") != nil {
		t.Fatal("want nil")
	}
	repr := core.BuildThinking("reasoning", "secret-abc123").String()
	if !strings.Contains(repr, "[REDACTED]") || strings.Contains(repr, "secret-abc123") {
		t.Fatalf("repr %q", repr)
	}
	if repr != "Thinking(text='reasoning', signature='[REDACTED]')" {
		t.Fatalf("repr %q", repr)
	}
}

func TestCitationEqualityAndRoundtrip(t *testing.T) {
	a := core.Citation{URL: core.Str("https://x"), Title: core.Str("t"), StartIndex: core.Int(0), EndIndex: core.Int(3)}
	b := core.CitationFromMap(a.Map())
	if !a.Equal(b) {
		t.Fatalf("%+v != %+v", a, b)
	}
	if _, ok := a.Map()["url"]; !ok {
		t.Fatalf("map %v", a.Map())
	}
}

func TestToolCallMapCompacts(t *testing.T) {
	tc := core.ToolCall{ID: "1", Name: "weather", Arguments: map[string]any{"city": "Paris"}}
	want := map[string]any{"id": "1", "name": "weather", "arguments": map[string]any{"city": "Paris"}}
	if got := tc.Map(); !reflect.DeepEqual(got, want) {
		t.Fatalf("map %v", got)
	}
}

func TestCostComputation(t *testing.T) {
	cost := core.NewCost(core.BuildTokens(core.Int(1_000_000), core.Int(1_000_000), nil, nil, nil), pricedModel())
	if !ptrEq(cost.Input(), 1.0) || !ptrEq(cost.Output(), 2.0) || !ptrEq(cost.Total(), 3.0) {
		t.Fatalf("cost %v", cost.Map())
	}
}
