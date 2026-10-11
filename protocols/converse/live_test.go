//go:build live

package converse_test

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym/core"
)

// liveChat talks to real Bedrock with BEDROCK_API_KEY / BEDROCK_SECRET_KEY /
// BEDROCK_REGION (and optional BEDROCK_SESSION_TOKEN); BEDROCK_MODEL picks
// the model (default: Claude Haiku 4.5 via the US inference profile).
func liveChat(t *testing.T) *core.Chat {
	t.Helper()
	for _, k := range []string{"BEDROCK_API_KEY", "BEDROCK_SECRET_KEY", "BEDROCK_REGION"} {
		if os.Getenv(k) == "" {
			t.Skipf("%s not set", k)
		}
	}
	model := os.Getenv("BEDROCK_MODEL")
	if model == "" {
		model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
	}
	ctx := core.NewContext(func(c *core.Config) { c.RequestTimeout = 60 * time.Second })
	chat, err := ctx.NewChat(core.ChatOptions{Model: model, Provider: "bedrock", AssumeModelExists: true})
	if err != nil {
		t.Fatal(err)
	}
	return chat
}

func TestLiveConverse(t *testing.T) {
	msg, err := liveChat(t).Ask(context.Background(), "Reply with the single word: pong")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.ToLower(msg.Text()), "pong") || msg.InputTokens() == nil {
		t.Fatalf("message %q tokens %v", msg.Text(), msg.Tokens)
	}
}

func TestLiveConverseStream(t *testing.T) {
	chunks := 0
	msg, err := liveChat(t).Ask(context.Background(), "Count from 1 to 5.", core.OnChunk(func(*core.Chunk) error {
		chunks++
		return nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	if chunks < 2 || !strings.Contains(msg.Text(), "5") || msg.FinishReason == "" || msg.OutputTokens() == nil {
		t.Fatalf("%d chunks, message %q finish %q", chunks, msg.Text(), msg.FinishReason)
	}
}
