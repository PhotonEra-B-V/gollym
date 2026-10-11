package core_test

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
)

// Ports tests/test_chat.py (OpenAI rows; the Anthropic and Gemini rows live
// with their protocols).

func weatherTool() core.Tool {
	return core.NewTool(core.ToolNameFromString("WeatherTool"), "Get weather",
		[]core.Parameter{{Name: "city", Required: true}},
		func(_ context.Context, args map[string]any) (any, error) {
			return fmt.Sprintf("Sunny in %v", args["city"]), nil
		})
}

func roles(chat *core.Chat) []string {
	var out []string
	for _, m := range chat.Messages {
		out = append(out, m.Role)
	}
	return out
}

func TestChatOpenAI(t *testing.T) {
	chat := newChat(t, newCtx(t, nil), core.ChatOptions{Model: "gpt-4o"})
	_, rd := mockHTTP(t, chat, testserver.Reply{Body: map[string]any{
		"id": "x", "model": "gpt-4o",
		"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}},
		"usage":   map[string]any{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
	}})
	msg, err := chat.Ask(context.Background(), "Hello")
	if err != nil {
		t.Fatal(err)
	}
	if msg.Content != "Hi!" || !ptrEq(msg.InputTokens(), 10) || !ptrEq(msg.OutputTokens(), 5) {
		t.Fatalf("msg %+v tokens %+v", msg, msg.Tokens)
	}
	if rd.URLs[0] != openAIURL {
		t.Fatalf("url %s", rd.URLs[0])
	}
}

func TestChatToolLoop(t *testing.T) {
	chat := newChat(t, newCtx(t, nil), core.ChatOptions{Model: "gpt-4o"}).WithTool(weatherTool())
	srv, _ := mockHTTP(t, chat,
		testserver.Reply{Body: map[string]any{
			"model": "gpt-4o",
			"choices": []any{map[string]any{
				"message": map[string]any{"role": "assistant", "content": nil, "tool_calls": []any{map[string]any{
					"id": "c1", "type": "function", "function": map[string]any{"name": "weather", "arguments": `{"city": "Paris"}`},
				}}},
				"finish_reason": "tool_calls",
			}},
			"usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 1},
		}},
		testserver.Reply{Body: map[string]any{
			"model":   "gpt-4o",
			"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": "It is sunny in Paris."}, "finish_reason": "stop"}},
			"usage":   map[string]any{"prompt_tokens": 1, "completion_tokens": 1},
		}},
	)
	msg, err := chat.Ask(context.Background(), "Weather in Paris?")
	if err != nil {
		t.Fatal(err)
	}
	if msg.Content != "It is sunny in Paris." {
		t.Fatalf("content %v", msg.Content)
	}
	if got := roles(chat); !reflect.DeepEqual(got, []string{"user", "assistant", "tool", "assistant"}) {
		t.Fatalf("roles %v", got)
	}
	if chat.Messages[2].Content != "Sunny in Paris" || chat.Messages[2].ToolCallID != "c1" {
		t.Fatalf("tool result %+v", chat.Messages[2])
	}
	if len(srv.Requests()) != 2 {
		t.Fatalf("%d requests", len(srv.Requests()))
	}
}

func TestChatStreaming(t *testing.T) {
	chat := newChat(t, newCtx(t, nil), core.ChatOptions{Model: "gpt-4o"})
	mockHTTP(t, chat, sse(
		`data: {"model":"gpt-4o","choices":[{"delta":{"content":"Hel"}}]}`+"\n\n"+
			`data: {"model":"gpt-4o","choices":[{"delta":{"content":"lo!"}}]}`+"\n\n"+
			`data: {"model":"gpt-4o","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}`+"\n\n"+
			"data: [DONE]\n\n"))
	var chunks []string
	_, err := chat.Ask(context.Background(), "Hi", core.OnChunk(func(c *core.Chunk) error {
		if s, ok := c.Content.(string); ok {
			chunks = append(chunks, s)
		}
		return nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(chunks, "") != "Hello!" {
		t.Fatalf("chunks %q", chunks)
	}
	last := chat.Messages[len(chat.Messages)-1]
	if last.Content != "Hello!" || !ptrEq(last.InputTokens(), 1) || !ptrEq(last.OutputTokens(), 2) {
		t.Fatalf("last %+v", last)
	}
}

// Callback order and tool-result strings, captured from pyllym 1.16.0 with
// the same conversation: results that aren't content go to the model as
// Python's str() (dicts with quoted keys, True/None, 2.0).
func TestChatCallbacksAndToolResultStrings(t *testing.T) {
	chat := newChat(t, newCtx(t, nil), core.ChatOptions{Model: "gpt-4o"})
	chat.WithTool(core.NewTool("weather", "w", []core.Parameter{{Name: "city", Required: true}},
		func(context.Context, map[string]any) (any, error) {
			return map[string]any{"a_list": []any{1, "a'b"}, "b_none": nil, "c_ok": true, "d_temp": 2.0}, nil
		}))
	call := func(id, name, args string) any {
		return map[string]any{"id": id, "type": "function", "function": map[string]any{"name": name, "arguments": args}}
	}
	srv, _ := mockHTTP(t, chat,
		testserver.Reply{Body: openaiChat(nil, []any{call("c1", "weather", `{"city": "x"}`), call("c2", "nope", "{}"), call("c3", "weather", "")})},
		testserver.Reply{Body: openaiChat("done", nil)})
	var log []string
	chat.BeforeMessage(func(context.Context) error { log = append(log, "before"); return nil })
	chat.AfterMessage(func(_ context.Context, m *core.Message) error { log = append(log, "after:"+m.Role); return nil })
	chat.BeforeToolCall(func(_ context.Context, tc *core.ToolCall) error {
		log = append(log, "before_tool:"+tc.Name)
		return nil
	})
	chat.AfterToolResult(func(context.Context, any) error { log = append(log, "after_tool"); return nil })
	if _, err := chat.Ask(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	want := []string{"before", "after:assistant", "before", "before_tool:weather", "after_tool", "after:tool", "before",
		"before_tool:nope", "after_tool", "after:tool", "before", "before_tool:weather", "after_tool", "after:tool", "before", "after:assistant"}
	if !reflect.DeepEqual(log, want) {
		t.Fatalf("callbacks %v", log)
	}
	var results []any
	for _, m := range chat.Messages {
		if m.Role == "tool" {
			results = append(results, m.Content)
		}
	}
	wantResults := []any{
		`{'a_list': [1, "a'b"], 'b_none': None, 'c_ok': True, 'd_temp': 2.0}`,
		"{'error': 'Model tried to call unavailable tool `nope`. Available tools: [\"weather\"].'}",
		"{'error': 'Invalid tool arguments: missing keyword: city'}",
	}
	if !reflect.DeepEqual(results, wantResults) {
		t.Fatalf("tool results %q", results)
	}
	// The tool results reach the model in the next request.
	msgs := srv.Requests()[1].JSON(t)["messages"].([]any)
	if got := msgs[len(msgs)-1].(map[string]any)["content"]; got != wantResults[2] {
		t.Fatalf("last tool message %v", got)
	}
}

// Real OpenAI streams send the call id once; later argument fragments carry
// only their index. (pyllym 1.16.0 keys fragments by id and fails on these;
// gollym keys them by index.)
func TestChatStreamingToolCallFragments(t *testing.T) {
	chat := newChat(t, newCtx(t, nil), core.ChatOptions{Model: "gpt-4o"}).WithTool(weatherTool())
	srv, _ := mockHTTP(t, chat,
		sse(`data: {"model":"gpt-4o","choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"weather","arguments":""}},{"index":1,"id":"call_b","type":"function","function":{"name":"weather","arguments":""}}]}}]}`+"\n\n"+
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}},{"index":1,"function":{"arguments":"{\"city\":\"Rome\"}"}}]}}]}`+"\n\n"+
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Oslo\"}"}}]}}]}`+"\n\n"+
			`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`+"\n\ndata: [DONE]\n\n"),
		sse(`data: {"model":"gpt-4o","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`+"\n\ndata: [DONE]\n\n"))
	msg, err := chat.Ask(context.Background(), "weather?", core.OnChunk(func(*core.Chunk) error { return nil }))
	if err != nil {
		t.Fatal(err)
	}
	if msg.Content != "ok" {
		t.Fatalf("content %v", msg.Content)
	}
	if got := roles(chat); !reflect.DeepEqual(got, []string{"user", "assistant", "tool", "tool", "assistant"}) {
		t.Fatalf("roles %v", got)
	}
	if chat.Messages[2].Content != "Sunny in Oslo" || chat.Messages[3].Content != "Sunny in Rome" || chat.Messages[3].ToolCallID != "call_b" {
		t.Fatalf("tool results %v %v", chat.Messages[2].Content, chat.Messages[3].Content)
	}
	asst := srv.Requests()[1].JSON(t)["messages"].([]any)[1].(map[string]any)
	fn := asst["tool_calls"].([]any)[0].(map[string]any)["function"].(map[string]any)
	if fn["arguments"] != `{"city": "Oslo"}` {
		t.Fatalf("replayed arguments %v", fn["arguments"])
	}
}

func TestChatToolConcurrencyKeepsCallOrder(t *testing.T) {
	chat := newChat(t, newCtx(t, nil), core.ChatOptions{Model: "gpt-4o"}).WithTool(weatherTool()).WithToolConcurrency(true)
	call := func(id, city string) any {
		return map[string]any{"id": id, "type": "function", "function": map[string]any{"name": "weather", "arguments": `{"city": "` + city + `"}`}}
	}
	mockHTTP(t, chat, testserver.Reply{Body: openaiChat(nil, []any{call("a", "A"), call("b", "B"), call("c", "C")})},
		testserver.Reply{Body: openaiChat("done", nil)})
	if _, err := chat.Ask(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, m := range chat.Messages[2:5] {
		got = append(got, m.ToolCallID+"="+m.Text())
	}
	if !reflect.DeepEqual(got, []string{"a=Sunny in A", "b=Sunny in B", "c=Sunny in C"}) {
		t.Fatalf("results %v", got)
	}
}
