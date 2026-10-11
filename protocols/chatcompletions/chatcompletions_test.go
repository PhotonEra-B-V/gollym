package chatcompletions_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
)

func completion(content string) map[string]any {
	return map[string]any{
		"model":   "glm-5.3",
		"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": content}, "finish_reason": "stop"}},
		"usage":   map[string]any{"prompt_tokens": 150, "completion_tokens": 40, "total_tokens": 190, "prompt_tokens_details": map[string]any{"cached_tokens": 20}},
	}
}

func zhipuChat(t *testing.T, base string, retries int) *gollym.Chat {
	t.Helper()
	ctx := gollym.NewContext(func(c *gollym.Config) {
		c.Set("zhipu_api_key", "test-key")
		c.Set("zhipu_api_base", base)
		c.MaxRetries = retries
		c.RetryInterval = time.Millisecond
		c.RequestTimeout = 5 * time.Second
	})
	chat, err := ctx.NewChat(gollym.ChatOptions{Model: "glm-5.3", Provider: "zhipu", AssumeModelExists: true})
	if err != nil {
		t.Fatal(err)
	}
	return chat
}

// Rekryon's commercial call: system + user, temperature 0.1, max_tokens and
// JSON mode via params.
func TestRekryonStyleJSONCall(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: completion(`{"overall_score": 7}`)})
	chat := zhipuChat(t, srv.URL, 2)
	chat.WithTemperature(0.1).WithParams(map[string]any{"max_tokens": 4096, "response_format": map[string]any{"type": "json_object"}})
	chat.WithInstructions("You are a recruiter.")
	msg, err := chat.Ask(context.Background(), "Score this candidate")
	if err != nil {
		t.Fatal(err)
	}
	req := srv.Last(t)
	if req.Path != "/chat/completions" || req.Headers.Get("Authorization") != "Bearer test-key" {
		t.Fatalf("path %s auth %q", req.Path, req.Headers.Get("Authorization"))
	}
	body := req.JSON(t)
	if body["model"] != "glm-5.3" || body["temperature"] != 0.1 || body["max_tokens"] != 4096.0 || body["stream"] != false {
		t.Fatalf("payload %v", body)
	}
	if body["response_format"].(map[string]any)["type"] != "json_object" {
		t.Fatalf("response_format %v", body["response_format"])
	}
	msgs := body["messages"].([]any)
	if len(msgs) != 2 || msgs[0].(map[string]any)["role"] != "system" || msgs[1].(map[string]any)["content"] != "Score this candidate" {
		t.Fatalf("messages %v", msgs)
	}
	if msg.Text() != `{"overall_score": 7}` {
		t.Fatalf("content %q", msg.Text())
	}
	if *msg.InputTokens() != 130 || *msg.OutputTokens() != 40 || *msg.CachedTokens() != 20 {
		t.Fatalf("tokens in=%d out=%d cached=%d", *msg.InputTokens(), *msg.OutputTokens(), *msg.CachedTokens())
	}
}

func TestRetriesServerErrorsThenSucceeds(t *testing.T) {
	srv := testserver.New(t,
		testserver.Reply{Status: 503, Body: map[string]any{"error": map[string]any{"message": "busy"}}},
		testserver.Reply{Status: 500, Body: "oops"},
		testserver.Reply{Body: completion("ok")},
	)
	msg, err := zhipuChat(t, srv.URL, 2).Ask(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	if msg.Text() != "ok" || len(srv.Requests()) != 3 {
		t.Fatalf("text %q after %d requests", msg.Text(), len(srv.Requests()))
	}
}

func TestGivesUpAfterMaxRetries(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Status: 500, Body: map[string]any{"error": map[string]any{"message": "down"}}})
	_, err := zhipuChat(t, srv.URL, 2).Ask(context.Background(), "hi")
	if !errors.Is(err, gollym.ErrServer) || err.Error() != "down" {
		t.Fatalf("err %v", err)
	}
	if n := len(srv.Requests()); n != 3 {
		t.Fatalf("%d requests, want 3", n)
	}
}

func TestClientErrorsAreNotRetried(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Status: 401, Body: map[string]any{"error": map[string]any{"message": "bad key"}}})
	_, err := zhipuChat(t, srv.URL, 2).Ask(context.Background(), "hi")
	if !errors.Is(err, gollym.ErrUnauthorized) || len(srv.Requests()) != 1 {
		t.Fatalf("err %v after %d requests", err, len(srv.Requests()))
	}
}

func TestQuota429IsPaymentRequiredAndNotRetried(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Status: 429, Body: map[string]any{"error": map[string]any{"message": "Insufficient balance or no resource package. Please recharge."}}})
	_, err := zhipuChat(t, srv.URL, 2).Ask(context.Background(), "hi")
	if !errors.Is(err, gollym.ErrPaymentRequired) || len(srv.Requests()) != 1 {
		t.Fatalf("err %v after %d requests", err, len(srv.Requests()))
	}
}

func TestConnectionErrorsAreRetried(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Hang: true}, testserver.Reply{Body: completion("ok")})
	msg, err := zhipuChat(t, srv.URL, 2).Ask(context.Background(), "hi")
	if err != nil || msg.Text() != "ok" {
		t.Fatalf("msg %v err %v", msg, err)
	}
}

func TestThinkTagsAndReasoningFallback(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: completion("<think>hmm</think>{\"a\":1}")})
	msg, err := zhipuChat(t, srv.URL, 0).Ask(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	if msg.Text() != `{"a":1}` || msg.Thinking == nil || msg.Thinking.Text != "hmm" {
		t.Fatalf("content %q thinking %v", msg.Text(), msg.Thinking)
	}
}

func TestStreamingAccumulatesContentAndUsage(t *testing.T) {
	sse := "data: {\"model\":\"glm-5.3\",\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\n" +
		"data: [DONE]\n\n"
	srv := testserver.New(t, testserver.Reply{Headers: map[string]string{"Content-Type": "text/event-stream"}, Body: sse})
	var chunks []string
	msg, err := zhipuChat(t, srv.URL, 0).Ask(context.Background(), "hi", gollym.OnChunk(func(c *gollym.Chunk) error {
		chunks = append(chunks, c.Text())
		return nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	if msg.Text() != "Hello" || msg.FinishReason != "stop" || *msg.InputTokens() != 5 || *msg.OutputTokens() != 2 {
		t.Fatalf("msg %+v", msg)
	}
	body := srv.Last(t).JSON(t)
	if body["stream"] != true || body["stream_options"].(map[string]any)["include_usage"] != true {
		t.Fatalf("payload %v", body)
	}
	if len(chunks) != 3 {
		t.Fatalf("chunks %q", chunks)
	}
}

func TestToolLoop(t *testing.T) {
	call := map[string]any{
		"model": "glm-5.3",
		"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": nil,
			"tool_calls": []any{map[string]any{"id": "call_1", "type": "function", "function": map[string]any{"name": "weather", "arguments": `{"city":"Oulu"}`}}}},
			"finish_reason": "tool_calls"}},
	}
	srv := testserver.New(t, testserver.Reply{Body: call}, testserver.Reply{Body: completion("It is cold.")})
	chat := zhipuChat(t, srv.URL, 0)
	var got map[string]any
	chat.WithTool(gollym.NewTool("weather", "Current weather", []gollym.Parameter{{Name: "city", Required: true}},
		func(_ context.Context, args map[string]any) (any, error) { got = args; return "-20C", nil }))
	msg, err := chat.Ask(context.Background(), "Weather in Oulu?")
	if err != nil {
		t.Fatal(err)
	}
	if got["city"] != "Oulu" || msg.Text() != "It is cold." {
		t.Fatalf("args %v msg %q", got, msg.Text())
	}
	second := srv.Requests()[1].JSON(t)
	msgs := second["messages"].([]any)
	toolMsg := msgs[len(msgs)-1].(map[string]any)
	if toolMsg["role"] != "tool" || toolMsg["tool_call_id"] != "call_1" || toolMsg["content"] != "-20C" {
		t.Fatalf("tool message %v", toolMsg)
	}
	asst := msgs[len(msgs)-2].(map[string]any)
	fn := asst["tool_calls"].([]any)[0].(map[string]any)["function"].(map[string]any)
	if fn["arguments"] != `{"city": "Oulu"}` {
		t.Fatalf("arguments %v", fn["arguments"])
	}
	tools := second["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools %v", tools)
	}
}
