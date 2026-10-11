package providers_test

import (
	"context"
	"reflect"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym"
	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
)

// Ports tests/test_ollama.py: Ollama speaks Chat Completions on a local
// server, so these stub that server.

func ollamaCompletion(content string) map[string]any {
	return map[string]any{
		"model":   "llama3",
		"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": content}, "finish_reason": "stop"}},
		"usage":   map[string]any{"prompt_tokens": 10, "completion_tokens": 5},
	}
}

func ollamaChat(t *testing.T, base, key string) *gollym.Chat {
	t.Helper()
	t.Setenv("OLLAMA_API_KEY", "")
	ctx := gollym.NewContext(func(c *gollym.Config) {
		c.Set("ollama_api_base", base+"/v1")
		c.Set("ollama_api_key", key)
		c.MaxRetries = 0
		c.RetryInterval = time.Millisecond
		c.RequestTimeout = 5 * time.Second
	})
	chat, err := ctx.NewChat(gollym.ChatOptions{Model: "llama3", Provider: "ollama", AssumeModelExists: true})
	if err != nil {
		t.Fatal(err)
	}
	return chat
}

func TestOllamaIsLocal(t *testing.T) {
	spec, ok := core.LookupProvider("ollama")
	if !ok || !spec.Local || spec.DisplayName() != "Ollama" || !reflect.DeepEqual(spec.Requirements, []string{"ollama_api_base"}) {
		t.Fatalf("spec %+v", spec)
	}
}

func TestOllamaChat(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: ollamaCompletion("reply from ollama")})
	msg, err := ollamaChat(t, srv.URL, "").Ask(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	if srv.Last(t).Path != "/v1/chat/completions" || msg.Text() != "reply from ollama" || *msg.InputTokens() != 10 {
		t.Fatalf("path %s msg %q", srv.Last(t).Path, msg.Text())
	}
}

func TestOllamaNoAuthHeaderWithoutKey(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: ollamaCompletion("ok")})
	if _, err := ollamaChat(t, srv.URL, "").Ask(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	if _, ok := srv.Last(t).Headers["Authorization"]; ok {
		t.Fatal("unexpected Authorization header")
	}
}

func TestOllamaBearerHeaderWithKey(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: ollamaCompletion("ok")})
	if _, err := ollamaChat(t, srv.URL, "secret-token").Ask(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	if got := srv.Last(t).Headers.Get("Authorization"); got != "Bearer secret-token" {
		t.Fatalf("auth %q", got)
	}
}

func TestOllamaToolLoop(t *testing.T) {
	call := map[string]any{
		"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": nil,
			"tool_calls": []any{map[string]any{"id": "call_1", "type": "function", "function": map[string]any{"name": "weather", "arguments": `{"city": "Rome"}`}}}},
			"finish_reason": "tool_calls"}},
	}
	srv := testserver.New(t, testserver.Reply{Body: call}, testserver.Reply{Body: ollamaCompletion("Rome is sunny.")})
	chat := ollamaChat(t, srv.URL, "")
	chat.WithTool(gollym.NewTool("weather", "weather", []gollym.Parameter{{Name: "city", Required: true}},
		func(_ context.Context, args map[string]any) (any, error) {
			return "Sunny in " + args["city"].(string), nil
		}))
	msg, err := chat.Ask(context.Background(), "weather in Rome?")
	if err != nil {
		t.Fatal(err)
	}
	var roles []string
	for _, m := range chat.Messages {
		roles = append(roles, m.Role)
	}
	if msg.Text() != "Rome is sunny." || !reflect.DeepEqual(roles, []string{"user", "assistant", "tool", "assistant"}) {
		t.Fatalf("msg %q roles %v", msg.Text(), roles)
	}
}

func TestOllamaStreaming(t *testing.T) {
	sse := "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"
	srv := testserver.New(t, testserver.Reply{Headers: map[string]string{"Content-Type": "text/event-stream"}, Body: sse})
	chat := ollamaChat(t, srv.URL, "")
	var text string
	_, err := chat.Ask(context.Background(), "hi", gollym.OnChunk(func(c *gollym.Chunk) error { text += c.Text(); return nil }))
	if err != nil {
		t.Fatal(err)
	}
	if text != "Hello!" || chat.Messages[len(chat.Messages)-1].Text() != "Hello!" {
		t.Fatalf("streamed %q last %q", text, chat.Messages[len(chat.Messages)-1].Text())
	}
}
