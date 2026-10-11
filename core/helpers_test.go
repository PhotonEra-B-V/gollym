package core_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"sync"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
	_ "github.com/PhotonEra-B-V/gollym/providers" // register built-in providers
)

// Shared fixtures for the ported pyllym core tests (tests/conftest.py and
// tests/factories.py).

const openAIURL = "https://api.openai.com/v1/chat/completions"

// testKeys are conftest's dummy credentials.
var testKeys = map[string]string{
	"openai_api_key":    "sk-test",
	"anthropic_api_key": "sk-ant-test",
	"gemini_api_key":    "gm-test",
	"deepseek_api_key":  "ds-test",
	"mistral_api_key":   "ms-test",
	"zhipu_api_key":     "zp-test",
}

// newCtx is a context with conftest's keys and no env leakage, plus fn.
func newCtx(t *testing.T, fn func(*core.Config)) *core.Context {
	t.Helper()
	for k := range testKeys {
		t.Setenv(upper(k), "")
	}
	t.Setenv("OPENAI_API_BASE", "")
	return core.NewContext(func(c *core.Config) {
		for k, v := range testKeys {
			c.Set(k, v)
		}
		if fn != nil {
			fn(c)
		}
	})
}

func upper(s string) string {
	b := []byte(s)
	for i, c := range b {
		if c >= 'a' && c <= 'z' {
			b[i] = c - 32
		}
	}
	return string(b)
}

// redirect is aioresponses: requests keep their real URL (recorded in
// URLs) but are served by the stub server.
type redirect struct {
	target *url.URL
	mu     sync.Mutex
	URLs   []string
}

func (r *redirect) RoundTrip(req *http.Request) (*http.Response, error) {
	r.mu.Lock()
	r.URLs = append(r.URLs, req.URL.String())
	r.mu.Unlock()
	out := req.Clone(req.Context())
	out.URL.Scheme, out.URL.Host = r.target.Scheme, r.target.Host
	out.Host = r.target.Host
	return http.DefaultTransport.RoundTrip(out)
}

// mockHTTP routes chat's requests to a stub serving replies.
func mockHTTP(t *testing.T, chat *core.Chat, replies ...testserver.Reply) (*testserver.Server, *redirect) {
	t.Helper()
	srv := testserver.New(t, replies...)
	u, _ := url.Parse(srv.URL)
	rd := &redirect{target: u}
	chat.Provider.Connection().Client = &http.Client{Transport: rd}
	return srv, rd
}

// newChat creates a chat in ctx (model gpt-4o by default).
func newChat(t *testing.T, ctx *core.Context, opts core.ChatOptions) *core.Chat {
	t.Helper()
	if opts.Model == "" {
		opts.Model = "gpt-4o"
	}
	chat, err := ctx.NewChat(opts)
	if err != nil {
		t.Fatal(err)
	}
	return chat
}

// sleeps records backoff waits instead of sleeping (test_retry's fixture).
func sleeps(chat *core.Chat) *[]time.Duration {
	var rec []time.Duration
	var mu sync.Mutex
	chat.Provider.Connection().Sleep = func(_ context.Context, d time.Duration) error {
		mu.Lock()
		rec = append(rec, d)
		mu.Unlock()
		return nil
	}
	return &rec
}

// openaiChat is factories.openai_chat.
func openaiChat(content any, toolCalls []any) map[string]any {
	msg := map[string]any{"role": "assistant", "content": content}
	finish := "stop"
	if len(toolCalls) > 0 {
		msg["content"] = nil
		msg["tool_calls"] = toolCalls
		finish = "tool_calls"
	}
	return map[string]any{
		"id": "cmpl-test", "model": "gpt-4o",
		"choices": []any{map[string]any{"message": msg, "finish_reason": finish}},
		"usage":   map[string]any{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
	}
}

// openaiToolCall is factories.openai_tool_call.
func openaiToolCall(name string, args map[string]any) []any {
	b, _ := json.Marshal(args)
	return []any{map[string]any{"id": "call_1", "type": "function", "function": map[string]any{"name": name, "arguments": string(b)}}}
}

func sse(body string) testserver.Reply {
	return testserver.Reply{Headers: map[string]string{"Content-Type": "text/event-stream"}, Body: body}
}

func ptrEq[T comparable](p *T, v T) bool { return p != nil && *p == v }
