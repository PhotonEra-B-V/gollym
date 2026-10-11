package core_test

import (
	"encoding/json"
	"net/http"
	"os"
	"reflect"
	"testing"

	"github.com/PhotonEra-B-V/gollym/core"
)

// Parity with pyllym itself: testdata/pyllym_golden.json holds outputs of
// pyllym 1.16.0 helpers (testdata/golden.py regenerates it); each case runs
// the Go counterpart on the same input.

func coreGolden(t *testing.T) map[string]json.RawMessage {
	t.Helper()
	b, err := os.ReadFile("testdata/pyllym_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g map[string]json.RawMessage
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func decode(t *testing.T, raw json.RawMessage, v any) {
	t.Helper()
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatal(err)
	}
}

// asJSON normalises a value through JSON for comparison.
func asJSON(v any) any {
	b, _ := json.Marshal(v)
	var out any
	_ = json.Unmarshal(b, &out)
	return out
}

func nilIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

var pyErrorKinds = map[string]core.ErrorKind{
	"Error": core.KindError, "BadRequestError": core.KindBadRequest, "UnauthorizedError": core.KindUnauthorized,
	"PaymentRequiredError": core.KindPaymentRequired, "ForbiddenError": core.KindForbidden,
	"ContextLengthExceededError": core.KindContextLengthExceeded, "RateLimitError": core.KindRateLimit,
	"ServerError": core.KindServer, "ServiceUnavailableError": core.KindServiceUnavailable, "OverloadedError": core.KindOverloaded,
}

func TestParityParseErrorBody(t *testing.T) {
	var cases [][2]any
	decode(t, coreGolden(t)["parse_error_body"], &cases)
	for _, c := range cases {
		want, _ := c[1].(string)
		if got := core.ParseErrorBody(c[0]); got != want {
			t.Errorf("ParseErrorBody(%v) = %q, want %q", c[0], got, want)
		}
	}
}

func TestParityStreamErrors(t *testing.T) {
	var cases [][3]string
	decode(t, coreGolden(t)["stream_errors"], &cases)
	for _, c := range cases {
		e := core.StreamError(c[0]).(*core.Error)
		if e.Kind != pyErrorKinds[c[1]] || e.Message != c[2] {
			t.Errorf("StreamError(%s) = %s %q, want %s %q", c[0], e.Kind, e.Message, c[1], c[2])
		}
	}
}

func TestParityAttachmentTypes(t *testing.T) {
	var cases [][5]any
	decode(t, coreGolden(t)["attachments"], &cases)
	for _, c := range cases {
		a := core.MustAttachment([]byte{}, c[0].(string))
		got := [5]any{c[0], a.MIMEType, a.Type(), a.Format(), nilIfEmpty(a.Extension())}
		if !reflect.DeepEqual(got, c) {
			t.Errorf("attachment %v, pyllym %v", got, c)
		}
	}
}

func TestParityURLFilenames(t *testing.T) {
	var cases [][2]string
	decode(t, coreGolden(t)["url_filenames"], &cases)
	for _, c := range cases {
		if got := core.MustAttachment(c[0], "").Filename; got != c[1] {
			t.Errorf("%s: filename %q, want %q", c[0], got, c[1])
		}
	}
}

func TestParityNames(t *testing.T) {
	g := coreGolden(t)
	var defaults, underscores, tools [][2]string
	decode(t, g["default_names"], &defaults)
	decode(t, g["underscore"], &underscores)
	decode(t, g["tool_names"], &tools)
	for _, c := range defaults {
		if got := core.DefaultInfo(c[0], "openai").Name; got != c[1] {
			t.Errorf("DefaultInfo(%q).Name = %q, want %q", c[0], got, c[1])
		}
	}
	for _, c := range underscores {
		if got := core.Underscore(c[0]); got != c[1] {
			t.Errorf("Underscore(%q) = %q, want %q", c[0], got, c[1])
		}
	}
	for _, c := range tools {
		if got := core.ToolNameFromString(c[0]); got != c[1] {
			t.Errorf("tool name of %q = %q, want %q", c[0], got, c[1])
		}
	}
}

func TestParityCosts(t *testing.T) {
	g := coreGolden(t)
	priced := core.NewInfo(map[string]any{"id": "p", "provider": "openai", "pricing": map[string]any{
		"text_tokens": map[string]any{"standard": map[string]any{"input_per_million": 2.0, "output_per_million": 8.0,
			"cache_read_input_per_million": 0.5, "cache_write_input_per_million": 2.5, "reasoning_output_per_million": 10.0}},
		"images": map[string]any{"standard": map[string]any{"input_per_million": 5.0, "output_per_million": 40.0}}}})
	sameReasoning := core.NewInfo(map[string]any{"id": "q", "provider": "openai", "pricing": map[string]any{
		"text_tokens": map[string]any{"standard": map[string]any{"input_per_million": 1.0, "output_per_million": 4.0, "reasoning_output_per_million": 4.0}}}})
	unpriced := core.NewInfo(map[string]any{"id": "u", "provider": "openai"})
	i := core.Int
	costs := map[string]*core.Cost{
		"full":            core.NewCost(core.BuildTokens(i(1000), i(2000), i(300), i(40), i(500)), priced),
		"same_reasoning":  core.NewCost(core.BuildTokens(i(1000), i(2000), nil, nil, i(500)), sameReasoning),
		"zero_tokens":     core.NewCost(core.BuildTokens(i(0), i(0), nil, nil, nil), unpriced),
		"unpriced":        core.NewCost(core.BuildTokens(i(10), nil, nil, nil, nil), unpriced),
		"no_tokens":       core.NewCost(nil, priced),
		"cache_only_zero": core.NewCost(core.BuildTokens(nil, nil, nil, i(0), nil), unpriced),
		"image": {Tokens: core.BuildTokens(i(100), i(1000), nil, nil, nil), Model: priced, Category: "images",
			InputDetails: map[string]any{"text_tokens": 40, "image_tokens": 60}},
		"image_no_details": {Tokens: core.BuildTokens(i(100), i(1000), nil, nil, nil), Model: priced, Category: "images"},
	}
	var want map[string]map[string]float64
	decode(t, g["costs"], &want)
	for name, c := range costs {
		if got := c.Map(); !reflect.DeepEqual(asJSON(got), asJSON(want[name])) {
			t.Errorf("%s: %v, pyllym %v", name, got, want[name])
		}
	}

	var missing struct {
		Dict         map[string]float64 `json:"dict"`
		InputMissing bool               `json:"input_missing"`
	}
	decode(t, g["aggregate_missing"], &missing)
	agg := core.AggregateCosts([]*core.Cost{costs["full"], costs["unpriced"], nil, costs["no_tokens"]})
	if got := aggMap(agg); !reflect.DeepEqual(asJSON(got), asJSON(missing.Dict)) || agg.IsMissing("input") != missing.InputMissing {
		t.Errorf("aggregate %v missing=%v, pyllym %v", got, agg.IsMissing("input"), missing)
	}
	var full map[string]float64
	decode(t, g["aggregate_full"], &full)
	if got := aggMap(core.AggregateCosts([]*core.Cost{costs["full"], costs["full"]})); !reflect.DeepEqual(asJSON(got), asJSON(full)) {
		t.Errorf("aggregate %v, pyllym %v", got, full)
	}
}

func aggMap(a *core.CostAggregate) map[string]float64 {
	out := map[string]float64{}
	for k, f := range map[string]func() *float64{"input": a.Input, "output": a.Output, "cache_read": a.CacheRead,
		"cache_write": a.CacheWrite, "thinking": a.Thinking, "total": a.Total} {
		if v := f(); v != nil {
			out[k] = *v
		}
	}
	return out
}

func TestParityThinkTags(t *testing.T) {
	var cases [][3]any
	decode(t, coreGolden(t)["think_tags"], &cases)
	for _, c := range cases {
		acc := core.NewStreamAccumulator()
		for _, s := range c[0].([]any) {
			acc.Add(&core.Chunk{Role: "assistant", Content: s.(string)})
		}
		m := acc.Message(nil)
		var thinking any
		if m.Thinking != nil {
			thinking = m.Thinking.Text
		}
		if m.Content != c[1] || thinking != c[2] {
			t.Errorf("%v: content %v thinking %v, pyllym %v %v", c[0], m.Content, thinking, c[1], c[2])
		}
	}
}

func TestParityRetryAfter(t *testing.T) {
	var cases [][2]any
	decode(t, coreGolden(t)["retry_after"], &cases)
	for _, c := range cases {
		h := http.Header{}
		for k, v := range c[0].(map[string]any) {
			h.Set(k, v.(string))
		}
		var got any
		if d, ok := core.RetryAfter(h); ok {
			got = d.Seconds()
		}
		if got != c[1] {
			t.Errorf("%v: %v, pyllym %v", c[0], got, c[1])
		}
	}
}

func TestParityErrorForStatus(t *testing.T) {
	var cases [][3]any
	decode(t, coreGolden(t)["error_for_status"], &cases)
	for _, c := range cases {
		msg, _ := c[1].(string)
		if got := core.ErrorKindForStatus(int(c[0].(float64)), msg); got != pyErrorKinds[c[2].(string)] {
			t.Errorf("%v %q: %s, pyllym %s", c[0], msg, got, c[2])
		}
	}
}

func TestParityNormalizeSchema(t *testing.T) {
	var cases [][2]map[string]any
	decode(t, coreGolden(t)["schemas"], &cases)
	for _, c := range cases {
		if got := core.NormalizeSchema(c[0]); !reflect.DeepEqual(asJSON(got), asJSON(c[1])) {
			t.Errorf("NormalizeSchema(%v) = %v, pyllym %v", c[0], got, c[1])
		}
	}
}
