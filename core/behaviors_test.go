package core_test

import (
	"context"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
)

// Ports tests/test_behaviors.py (the Agent row lives in agent/, the
// Anthropic and Gemini render rows with their protocols).

// --- structured output -------------------------------------------------------

type Recipe struct {
	Title string   `json:"title"`
	Steps []string `json:"steps"`
}

func TestStructuredOutputParsesJSONAndSendsSchema(t *testing.T) {
	chat := newChat(t, newCtx(t, nil), core.ChatOptions{Model: "gpt-4o"})
	if _, err := chat.WithSchema(Recipe{}); err != nil {
		t.Fatal(err)
	}
	srv, _ := mockHTTP(t, chat, testserver.Reply{Body: openaiChat(`{"title": "Toast", "steps": ["toast bread"]}`, nil)})
	msg, err := chat.Ask(context.Background(), "recipe?")
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{"title": "Toast", "steps": []any{"toast bread"}}
	if !reflect.DeepEqual(msg.Content, want) {
		t.Fatalf("content %#v", msg.Content)
	}
	if !strings.Contains(string(srv.Last(t).Body), `"json_schema"`) {
		t.Fatalf("body %s", srv.Last(t).Body)
	}
}

// --- cost --------------------------------------------------------------------

func pricedModel() *core.Info {
	return core.NewInfo(map[string]any{
		"id": "demo", "provider": "openai",
		"pricing": map[string]any{"text_tokens": map[string]any{"standard": map[string]any{"input_per_million": 1.0, "output_per_million": 2.0}}},
	})
}

func TestCostAggregateSumsMessages(t *testing.T) {
	one := core.NewCost(core.BuildTokens(core.Int(1_000_000), core.Int(1_000_000), nil, nil, nil), pricedModel())
	agg := core.AggregateCosts([]*core.Cost{one, one})
	if !ptrEq(agg.Input(), 2.0) || !ptrEq(agg.Total(), 6.0) {
		t.Fatalf("input %v total %v", agg.Input(), agg.Total())
	}
}

func TestCostTotalNilWithoutPricing(t *testing.T) {
	model := core.NewInfo(map[string]any{"id": "unpriced", "provider": "openai"})
	cost := core.NewCost(core.BuildTokens(core.Int(100), core.Int(100), nil, nil, nil), model)
	if cost.Total() != nil {
		t.Fatalf("total %v", *cost.Total())
	}
}

// --- message -----------------------------------------------------------------

func TestMessagePredicatesAndMap(t *testing.T) {
	calls := &core.ToolCalls{}
	calls.Set("c1", &core.ToolCall{ID: "c1", Name: "x", Arguments: map[string]any{}})
	call := core.MustMessage(core.Message{Role: "assistant", ToolCalls: calls})
	if !call.IsToolCall() || call.Content != "" {
		t.Fatalf("tool call message %+v", call)
	}
	result := core.MustMessage(core.Message{Role: "tool", Content: "done", ToolCallID: "c1"})
	if !result.IsToolResult() {
		t.Fatal("not a tool result")
	}
	plain := core.MustMessage(core.Message{Role: "assistant", Content: "hi", FinishReason: "stop"})
	if !plain.IsStopped() {
		t.Fatal("not stopped")
	}
	if d := plain.Map(); d["role"] != "assistant" || d["content"] != "hi" || d["finish_reason"] != "stop" {
		t.Fatalf("map %v", d)
	}
}

// --- stream accumulator ------------------------------------------------------

func TestStreamAccumulatorSplitsThinkTags(t *testing.T) {
	acc := core.NewStreamAccumulator()
	acc.Add(&core.Chunk{Role: "assistant", Content: "Hello "})
	acc.Add(&core.Chunk{Role: "assistant", Content: "<think>because</think>world"})
	msg := acc.Message(nil)
	if msg.Content != "Hello world" || msg.Thinking == nil || msg.Thinking.Text != "because" {
		t.Fatalf("content %v thinking %v", msg.Content, msg.Thinking)
	}
}

func TestStreamAccumulatorAssemblesToolCallFragments(t *testing.T) {
	frag := func(id, name, args string) *core.Chunk {
		tc := &core.ToolCalls{}
		tc.Set("0", &core.ToolCall{ID: id, Name: name, Arguments: args})
		return &core.Chunk{Role: "assistant", ToolCalls: tc}
	}
	acc := core.NewStreamAccumulator()
	acc.Add(frag("c1", "weather", ""))
	acc.Add(frag("", "", `{"city":`))
	acc.Add(frag("", "", `"Paris"}`))
	msg := acc.Message(nil)
	if !msg.IsToolCall() {
		t.Fatal("not a tool call")
	}
	tc := msg.ToolCalls.List()[0]
	if tc.Name != "weather" || !reflect.DeepEqual(tc.Arguments, map[string]any{"city": "Paris"}) {
		t.Fatalf("tool call %+v", tc)
	}
}

// --- utils -------------------------------------------------------------------

func TestUtilsToSafeArrayAndDates(t *testing.T) {
	if got := core.ToSafeArray(nil); len(got) != 0 {
		t.Fatalf("nil -> %v", got)
	}
	if got := core.ToSafeArray("x"); !reflect.DeepEqual(got, []any{"x"}) {
		t.Fatalf("x -> %v", got)
	}
	if got := core.ToSafeArray([]any{"a", "b"}); !reflect.DeepEqual(got, []any{"a", "b"}) {
		t.Fatalf("list -> %v", got)
	}
	for in, want := range map[string]string{"2025": "2025-01-01", "2025-06": "2025-06-01"} {
		d := core.ParseISODatePrefix(in)
		if d == nil || d.Format(time.DateOnly) != want {
			t.Fatalf("%s -> %v", in, d)
		}
	}
}

// --- render payloads (no network) --------------------------------------------

func TestRenderOpenAIIncludesMediaParts(t *testing.T) {
	chat := newChat(t, newCtx(t, nil), core.ChatOptions{Model: "gpt-4o"})
	if err := chat.AddUserMessage("look", "https://example.com/cat.png"); err != nil {
		t.Fatal(err)
	}
	payload, err := chat.Render()
	if err != nil {
		t.Fatal(err)
	}
	content := payload["messages"].([]any)[0].(map[string]any)["content"].([]any)
	found := false
	for _, p := range content {
		if p.(map[string]any)["type"] == "image_url" {
			found = true
		}
	}
	if !found {
		t.Fatalf("content %v", content)
	}
}
