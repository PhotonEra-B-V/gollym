//go:build live

package gemini_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym"
)

// TestLiveGemini makes one real call shaped like Rekryon's: run with
// `GEMINI_API_KEY=... go test -tags live ./protocols/gemini/ -run Live`.
func TestLiveGemini(t *testing.T) {
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		t.Skip("GEMINI_API_KEY not set")
	}
	model := os.Getenv("GEMINI_LIVE_MODEL")
	if model == "" {
		model = "gemini-2.5-flash"
	}
	ctx := gollym.NewContext(func(c *gollym.Config) {
		c.Set("gemini_api_key", key)
		c.RequestTimeout = 60 * time.Second
	})
	chat, err := ctx.NewChat(gollym.ChatOptions{Model: model, Provider: "gemini"})
	if err != nil {
		t.Fatal(err)
	}
	chat.WithTemperature(0.1).WithParams(map[string]any{"generationConfig": map[string]any{"maxOutputTokens": 256, "responseMimeType": "application/json"}})
	chat.WithInstructions(`Reply with JSON only: {"answer": <number>}.`)
	msg, err := chat.Ask(context.Background(), "What is 2 + 3?")
	if err != nil {
		t.Fatal(err)
	}
	if msg.Text() == "" || msg.InputTokens() == nil || *msg.InputTokens() == 0 || msg.OutputTokens() == nil || *msg.OutputTokens() == 0 {
		t.Fatalf("msg %q tokens %+v", msg.Text(), msg.Tokens)
	}
	t.Logf("%s: %s (in=%d out=%d)", msg.ModelID, msg.Text(), *msg.InputTokens(), *msg.OutputTokens())
}
