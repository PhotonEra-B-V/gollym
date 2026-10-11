package providers_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
	_ "github.com/PhotonEra-B-V/gollym/providers"
)

// Ports the OpenAI / OpenAI-compatible rows of tests/test_provider_matrix.py
// (plus vLLM's plain chat rows). Requests keep their real URL, so the
// default bases are checked, but are served by a stub.

// matrixBases is provider slug -> chat/completions base URL.
var matrixBases = map[string]string{
	"openai":      "https://api.openai.com/v1",
	"deepseek":    "https://api.deepseek.com",
	"mistral":     "https://api.mistral.ai/v1",
	"xai":         "https://api.x.ai/v1",
	"perplexity":  "https://api.perplexity.ai",
	"openrouter":  "https://openrouter.ai/api/v1",
	"qwen":        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
	"zhipu":       "https://api.z.ai/api/paas/v4",
	"moonshot":    "https://api.moonshot.ai/v1",
	"minimax":     "https://api.minimax.io/v1",
	"nvidia":      "https://integrate.api.nvidia.com/v1",
	"cerebras":    "https://api.cerebras.ai/v1",
	"huggingface": "https://router.huggingface.co/v1",
	"doubao":      "https://ark.cn-beijing.volces.com/api/v3",
	"ernie":       "https://qianfan.baidubce.com/v2",
	"databricks":  "https://ws.cloud.databricks.com/serving-endpoints",
}

func matrixSlugs() []string {
	out := make([]string, 0, len(matrixBases))
	for s := range matrixBases {
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}

// matrixRedirect records each request's real URL and serves it from target.
type matrixRedirect struct {
	target *url.URL
	mu     sync.Mutex
	urls   []string
}

func (r *matrixRedirect) RoundTrip(req *http.Request) (*http.Response, error) {
	r.mu.Lock()
	r.urls = append(r.urls, req.URL.String())
	r.mu.Unlock()
	out := req.Clone(req.Context())
	out.URL.Scheme, out.URL.Host, out.Host = r.target.Scheme, r.target.Host, r.target.Host
	return http.DefaultTransport.RoundTrip(out)
}

// matrixContext is conftest's dummy credentials (env cleared).
func matrixContext(t *testing.T, fn func(*core.Config)) *core.Context {
	t.Helper()
	for slug := range matrixBases {
		t.Setenv(strings.ToUpper(slug)+"_API_KEY", "")
		t.Setenv(strings.ToUpper(slug)+"_API_BASE", "")
	}
	t.Setenv("OPENAI_USE_SYSTEM_ROLE", "")
	t.Setenv("VLLM_API_KEY", "")
	t.Setenv("VLLM_API_BASE", "")
	return core.NewContext(func(c *core.Config) {
		for slug := range matrixBases {
			c.Set(slug+"_api_key", slug+"-test")
		}
		c.Set("databricks_api_base", matrixBases["databricks"])
		if fn != nil {
			fn(c)
		}
	})
}

func matrixChat(t *testing.T, ctx *core.Context, provider string, replies ...testserver.Reply) (*core.Chat, *testserver.Server, *matrixRedirect) {
	t.Helper()
	chat, err := ctx.NewChat(core.ChatOptions{Model: "some-model", Provider: provider, AssumeModelExists: true})
	if err != nil {
		t.Fatal(err)
	}
	srv := testserver.New(t, replies...)
	u, _ := url.Parse(srv.URL)
	rd := &matrixRedirect{target: u}
	chat.Provider.Connection().Client = &http.Client{Transport: rd}
	return chat, srv, rd
}

// matrixCompletion is factories.openai_chat.
func matrixCompletion(content string, toolCalls []any) map[string]any {
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

func matrixToolCall(name string, args map[string]any) []any {
	b, _ := json.Marshal(args)
	return []any{map[string]any{"id": "call_1", "type": "function", "function": map[string]any{"name": name, "arguments": string(b)}}}
}

func TestMatrixOpenAICompatibleChat(t *testing.T) {
	for _, slug := range matrixSlugs() {
		t.Run(slug, func(t *testing.T) {
			chat, srv, rd := matrixChat(t, matrixContext(t, nil), slug, testserver.Reply{Body: matrixCompletion("reply from "+slug, nil)})
			msg, err := chat.Ask(context.Background(), "hi")
			if err != nil {
				t.Fatal(err)
			}
			if len(rd.urls) != 1 || rd.urls[0] != matrixBases[slug]+"/chat/completions" {
				t.Fatalf("urls %v", rd.urls)
			}
			if auth := srv.Last(t).Headers.Get("Authorization"); auth != "Bearer "+slug+"-test" {
				t.Fatalf("auth %q", auth)
			}
			if msg.Content != "reply from "+slug || msg.InputTokens() == nil || *msg.InputTokens() != 10 {
				t.Fatalf("msg %+v", msg)
			}
		})
	}
}

func TestMatrixOpenAICompatibleToolLoop(t *testing.T) {
	for _, slug := range matrixSlugs() {
		t.Run(slug, func(t *testing.T) {
			chat, _, _ := matrixChat(t, matrixContext(t, nil), slug,
				testserver.Reply{Body: matrixCompletion("", matrixToolCall("weather", map[string]any{"city": "Rome"}))},
				testserver.Reply{Body: matrixCompletion("Rome is sunny.", nil)})
			chat.WithTool(core.NewTool("weather", "weather", []core.Parameter{{Name: "city", Required: true}},
				func(_ context.Context, args map[string]any) (any, error) {
					return fmt.Sprintf("Sunny in %v", args["city"]), nil
				}))
			msg, err := chat.Ask(context.Background(), "weather in Rome?")
			if err != nil {
				t.Fatal(err)
			}
			var roles []string
			for _, m := range chat.Messages {
				roles = append(roles, m.Role)
			}
			if msg.Content != "Rome is sunny." || !reflect.DeepEqual(roles, []string{"user", "assistant", "tool", "assistant"}) {
				t.Fatalf("content %v roles %v", msg.Content, roles)
			}
		})
	}
}

func TestMatrixVLLMLocalChat(t *testing.T) {
	base := "http://localhost:8000/v1"
	ctx := matrixContext(t, func(c *core.Config) { c.Set("vllm_api_base", base) })
	chat, err := ctx.NewChat(core.ChatOptions{Model: "meta-llama/Llama-3.1-8B-Instruct", Provider: "vllm"})
	if err != nil {
		t.Fatal(err)
	}
	srv := testserver.New(t, testserver.Reply{Body: matrixCompletion("reply from vllm", nil)})
	u, _ := url.Parse(srv.URL)
	rd := &matrixRedirect{target: u}
	chat.Provider.Connection().Client = &http.Client{Transport: rd}
	msg, err := chat.Ask(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	if msg.Content != "reply from vllm" || rd.urls[0] != base+"/chat/completions" {
		t.Fatalf("content %v urls %v", msg.Content, rd.urls)
	}
	// No api key configured -> no Authorization header sent.
	if _, ok := srv.Last(t).Headers["Authorization"]; ok {
		t.Fatalf("headers %v", srv.Last(t).Headers)
	}
}

func firstRole(t *testing.T, srv *testserver.Server) string {
	t.Helper()
	return srv.Last(t).JSON(t)["messages"].([]any)[0].(map[string]any)["role"].(string)
}

// Only the OpenAI API itself gets "developer"; everyone else keeps "system".
func TestMatrixSystemRolePerProvider(t *testing.T) {
	for _, slug := range matrixSlugs() {
		t.Run(slug, func(t *testing.T) {
			want := "system"
			if slug == "openai" {
				want = "developer"
			}
			chat, srv, _ := matrixChat(t, matrixContext(t, nil), slug, testserver.Reply{Body: matrixCompletion("ok", nil)})
			chat.WithInstructions("be terse")
			if _, err := chat.Ask(context.Background(), "hi"); err != nil {
				t.Fatal(err)
			}
			if got := firstRole(t, srv); got != want {
				t.Fatalf("role %q, want %q", got, want)
			}
		})
	}
}

func TestMatrixOpenAIUseSystemRoleOverride(t *testing.T) {
	ctx := matrixContext(t, func(c *core.Config) { c.Set("openai_use_system_role", "true") })
	chat, srv, _ := matrixChat(t, ctx, "openai", testserver.Reply{Body: matrixCompletion("ok", nil)})
	chat.WithInstructions("be terse")
	if _, err := chat.Ask(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	if got := firstRole(t, srv); got != "system" {
		t.Fatalf("role %q", got)
	}
}

// Local providers with their own provider specs also keep "system".
func TestMatrixVLLMSystemRole(t *testing.T) {
	ctx := matrixContext(t, func(c *core.Config) { c.Set("vllm_api_base", "http://localhost:8000/v1") })
	chat, srv, _ := matrixChat(t, ctx, "vllm", testserver.Reply{Body: matrixCompletion("ok", nil)})
	chat.WithInstructions("be terse")
	if _, err := chat.Ask(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	if got := firstRole(t, srv); got != "system" {
		t.Fatalf("role %q", got)
	}
}
