package core_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
)

// Ports tests/test_review_fixes.py. Python-only rows (event-loop client
// cache eviction, aclose, AttributeError on config typos) have no Go
// counterpart; the fal row lives with the fal protocol.

func TestErrorMappingGatewayAndContextLength(t *testing.T) {
	cases := []struct {
		status int
		msg    string
		want   core.ErrorKind
	}{
		{502, "", core.KindServiceUnavailable},
		{504, "", core.KindServiceUnavailable},
		{400, "prompt is too long: 210000 tokens", core.KindContextLengthExceeded},
		{400, "bad param", core.KindBadRequest},
		{413, "Input is too long", core.KindContextLengthExceeded},
		{429, "insufficient_quota", core.KindPaymentRequired},
		{418, "", core.KindBadRequest},
		{599, "", core.KindServer},
	}
	for _, c := range cases {
		if got := core.ErrorKindForStatus(c.status, c.msg); got != c.want {
			t.Errorf("%d %q: %s, want %s", c.status, c.msg, got, c.want)
		}
	}
}

func TestRunUntilDoneOnEmptyChatRaisesClearError(t *testing.T) {
	chat := newChat(t, newCtx(t, nil), core.ChatOptions{Model: "gpt-4o"})
	_, err := chat.RunUntilDone(context.Background(), nil)
	if !errors.Is(err, core.ErrAPI) || !strings.Contains(err.Error(), "Nothing to send") {
		t.Fatalf("err %v", err)
	}
}

// calls=True can't be passed in Go (calls is a string); anything but
// "one"/"many" is rejected.
func TestInvalidToolCallsRejected(t *testing.T) {
	chat := newChat(t, newCtx(t, nil), core.ChatOptions{Model: "gpt-4o"})
	if err := chat.WithToolCalls("true"); err == nil {
		t.Fatal("calls=true accepted")
	}
	if err := chat.WithToolChoice("threds"); !errors.Is(err, core.ErrInvalidToolChoice) {
		t.Fatalf("err %v", err)
	}
	if err := chat.WithToolCalls("one"); err != nil || chat.ToolPrefs.Calls != "one" {
		t.Fatalf("err %v prefs %+v", err, chat.ToolPrefs)
	}
}

func TestPendingThinkTagFlushedAtEndOfStream(t *testing.T) {
	acc := core.NewStreamAccumulator()
	acc.Add(&core.Chunk{Role: "assistant", Content: "answer is 5<"})
	if msg := acc.Message(nil); msg.Content != "answer is 5<" {
		t.Fatalf("content %v", msg.Content)
	}
}

func TestWithContextKeepsRegistryModelInfo(t *testing.T) {
	chat := newChat(t, newCtx(t, nil), core.ChatOptions{Model: "gpt-4o"})
	if err := chat.WithContext(newCtx(t, nil)); err != nil {
		t.Fatal(err)
	}
	if chat.Model.Pricing.TextTokens().Input() == nil {
		t.Fatal("model info replaced by a stub")
	}
}

func TestContextHasAnimateDelegate(t *testing.T) {
	var _ func(context.Context, string, core.AnimateRequest) (*core.Video, error) = (&core.Context{}).Animate
}

// pyllym collapses a {"text": ...} dict into Content; in Go text plus
// attachments is a *Content, and a map is a structured-output value that
// stays as is (persistence and schema replies rely on that).
func TestContentTextDoesNotLeakIntoAttachments(t *testing.T) {
	c, err := core.NewContent(core.Str("hello"), nil, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	msg := core.MustMessage(core.Message{Role: "user", Content: c})
	if msg.Content != "hello" {
		t.Fatalf("content %#v", msg.Content)
	}
	structured := core.MustMessage(core.Message{Role: "assistant", Content: map[string]any{"text": "hello"}})
	if _, ok := structured.Content.(map[string]any); !ok {
		t.Fatalf("content %#v", structured.Content)
	}
}

func TestConfigUnsetProviderOptionIsEmpty(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	if got := core.NewConfig().Get("openai_api_key"); got != "" {
		t.Fatalf("got %q", got)
	}
}

func TestConfigUnsetProviderOptionFallsBackToEnv(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-from-env")
	if got := core.NewConfig().Get("openai_api_key"); got != "sk-from-env" {
		t.Fatalf("got %q", got)
	}
}

func TestConfigCodeValueWinsOverEnv(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-from-env")
	cfg := core.NewConfig()
	cfg.Set("openai_api_key", "sk-from-code")
	if got := cfg.Get("openai_api_key"); got != "sk-from-code" {
		t.Fatalf("got %q", got)
	}
	// Un-setting (blank) restores the env fallback.
	cfg.Set("openai_api_key", "")
	if got := cfg.Get("openai_api_key"); got != "sk-from-env" {
		t.Fatalf("got %q", got)
	}
	cfg.Set("openai_api_key", "sk-from-code")
	cfg.Set("openai_api_key", "   ")
	if got := cfg.Get("openai_api_key"); got != "sk-from-env" {
		t.Fatalf("got %q", got)
	}
}

func TestConfigBlankEnvValueReadsAsEmpty(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "   ")
	if got := core.NewConfig().Get("openai_api_key"); got != "" {
		t.Fatalf("got %q", got)
	}
}

func TestBareJSONErrorBodyIn200StreamRaises(t *testing.T) {
	chat := newChat(t, newCtx(t, nil), core.ChatOptions{Model: "gpt-4o"})
	mockHTTP(t, chat, testserver.Reply{
		Headers: map[string]string{"Content-Type": "application/json"},
		Body:    `{"error": {"type": "overloaded_error", "message": "Overloaded"}}`,
	})
	_, err := chat.Ask(context.Background(), "hi", core.OnChunk(func(*core.Chunk) error { return nil }))
	if !errors.Is(err, core.ErrOverloaded) || err.Error() != "Overloaded" {
		t.Fatalf("err %v", err)
	}
}

// Python breaks out of the async iterator; in Go the callback returns an
// error, which stops the stream at once and comes back from Ask.
func TestStreamEarlyStopCancelsProducer(t *testing.T) {
	chat := newChat(t, newCtx(t, nil), core.ChatOptions{Model: "gpt-4o"})
	srv, _ := mockHTTP(t, chat, sse(
		`data: {"model":"gpt-4o","choices":[{"delta":{"content":"one"}}]}`+"\n\n"+
			`data: {"model":"gpt-4o","choices":[{"delta":{"content":"two"}}]}`+"\n\n"+
			`data: {"model":"gpt-4o","choices":[{"delta":{},"finish_reason":"stop"}]}`+"\n\n"+
			"data: [DONE]\n\n"))
	stop := errors.New("stop")
	var seen []string
	done := make(chan error, 1)
	go func() {
		_, err := chat.Ask(context.Background(), "hi", core.OnChunk(func(c *core.Chunk) error {
			seen = append(seen, c.Text())
			return stop
		}))
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, stop) || len(seen) != 1 || len(srv.Requests()) != 1 {
			t.Fatalf("err %v seen %v requests %d", err, seen, len(srv.Requests()))
		}
	case <-time.After(5 * time.Second):
		t.Fatal("hung")
	}
	if len(chat.Messages) != 1 {
		t.Fatalf("messages %d", len(chat.Messages))
	}
}

func TestMediaBlobHelpers(t *testing.T) {
	img := &core.Image{Data: "aGk="} // base64 "hi"
	if b, err := img.Blob(context.Background()); err != nil || string(b) != "hi" {
		t.Fatalf("image %q %v", b, err)
	}
	vid := &core.Video{Data: "aGk="}
	if b, err := vid.Blob(context.Background()); err != nil || string(b) != "hi" {
		t.Fatalf("video %q %v", b, err)
	}
}
