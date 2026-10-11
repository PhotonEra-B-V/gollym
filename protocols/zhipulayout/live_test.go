//go:build live

package zhipulayout_test

import (
	"context"
	"os"
	"testing"

	"github.com/PhotonEra-B-V/gollym"
)

// Live smoke tests against z.ai: go test -tags live ./protocols/zhipulayout/

func liveZhipu(t *testing.T) *gollym.Context {
	t.Helper()
	key := os.Getenv("ZHIPU_API_KEY")
	if key == "" {
		t.Skip("ZHIPU_API_KEY not set")
	}
	return gollym.NewContext(func(c *gollym.Config) { c.Set("zhipu_api_key", key) })
}

func TestLiveZhipuJSONChat(t *testing.T) {
	chat, err := liveZhipu(t).NewChat(gollym.ChatOptions{Model: "glm-5.3", Provider: "zhipu", AssumeModelExists: true})
	if err != nil {
		t.Fatal(err)
	}
	chat.WithParams(map[string]any{"max_tokens": 256, "response_format": map[string]any{"type": "json_object"}})
	msg, err := chat.Ask(context.Background(), `Reply with the JSON object {"ok": true} and nothing else.`)
	if err != nil {
		t.Fatal(err)
	}
	if msg.Text() == "" {
		t.Fatal("empty reply")
	}
	t.Logf("reply %q tokens in=%v out=%v", msg.Text(), msg.InputTokens(), msg.OutputTokens())
}

func TestLiveParseLayout(t *testing.T) {
	res, err := gollym.ParseLayout(context.Background(), "https://cdn.bigmodel.cn/static/logo/introduction.png", gollym.LayoutRequest{
		ModelOptions: gollym.ModelOptions{Provider: "zhipu", Context: liveZhipu(t)},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("model %s elements %d text %.200q", res.Model, len(res.Elements), res.Text)
}
