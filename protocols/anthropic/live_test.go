//go:build live

package anthropic_test

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym"
)

// liveChat talks to api.anthropic.com (ANTHROPIC_LIVE_MODEL overrides the
// model):
//
//	ANTHROPIC_API_KEY=... go test -tags live ./protocols/anthropic/ -run Live
func liveChat(t *testing.T) *gollym.Chat {
	t.Helper()
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}
	model := os.Getenv("ANTHROPIC_LIVE_MODEL")
	if model == "" {
		model = "claude-haiku-4-5"
	}
	ctx := gollym.NewContext(func(c *gollym.Config) {
		c.Set("anthropic_api_key", key)
		c.RequestTimeout = 60 * time.Second
	})
	chat, err := ctx.NewChat(gollym.ChatOptions{Model: model})
	if err != nil {
		t.Fatal(err)
	}
	return chat.WithParams(map[string]any{"max_tokens": 256})
}

func TestLiveAsk(t *testing.T) {
	chat := liveChat(t)
	m, err := chat.Ask(context.Background(), "Reply with the single word: pong")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.ToLower(m.Text()), "pong") || m.InputTokens() == nil || m.OutputTokens() == nil {
		t.Fatalf("message %q tokens %+v", m.Text(), m.Tokens)
	}
}

func TestLiveStreamAndTool(t *testing.T) {
	chat := liveChat(t)
	var called bool
	chat.WithTool(gollym.NewTool("weather", "Current weather for a city", []gollym.Parameter{{Name: "city", Required: true}},
		func(_ context.Context, args map[string]any) (any, error) {
			called = true
			return "-20C and snowing", nil
		}))
	chunks := 0
	m, err := chat.Ask(context.Background(), "What's the weather in Oulu? Use the tool.", gollym.OnChunk(func(*gollym.Chunk) error {
		chunks++
		return nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	if !called || chunks == 0 || m.Text() == "" {
		t.Fatalf("called %v chunks %d text %q", called, chunks, m.Text())
	}
}
