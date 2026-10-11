package core_test

import (
	"context"
	"errors"
	"net/http"
	"reflect"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
)

// Ports tests/test_retry.py: rate limits back off sensibly, server hints are
// honored, streams retry before their first byte, quota errors aren't
// retried.

var (
	retryOK  = testserver.Reply{Body: map[string]any{"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": "hi"}}}, "usage": map[string]any{}}}
	retrySSE = sse("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n")
)

func headers(kv ...string) http.Header {
	h := http.Header{}
	for i := 0; i < len(kv); i += 2 {
		h.Set(kv[i], kv[i+1])
	}
	return h
}

func retryAfter(t *testing.T, h http.Header) *float64 {
	t.Helper()
	d, ok := core.RetryAfter(h)
	if !ok {
		return nil
	}
	s := d.Seconds()
	return &s
}

// --- header parsing ------------------------------------------------------------

func TestRetryAfterParsesDelaySeconds(t *testing.T) {
	if got := retryAfter(t, headers("Retry-After", "7")); !ptrEq(got, 7.0) {
		t.Fatalf("got %v", got)
	}
	if got := retryAfter(t, headers("retry-after", "2.5")); !ptrEq(got, 2.5) {
		t.Fatalf("got %v", got)
	}
}

func TestRetryAfterMsWinsOverRetryAfter(t *testing.T) {
	if got := retryAfter(t, headers("retry-after-ms", "1500", "retry-after", "9")); !ptrEq(got, 1.5) {
		t.Fatalf("got %v", got)
	}
}

func TestRetryAfterParsesHTTPDate(t *testing.T) {
	when := time.Now().UTC().Add(30 * time.Second)
	got := retryAfter(t, headers("Retry-After", when.Format(http.TimeFormat)))
	if got == nil || *got < 27 || *got > 30 {
		t.Fatalf("got %v", got)
	}
}

func TestRetryAfterPastDateIsZero(t *testing.T) {
	when := time.Now().UTC().Add(-30 * time.Second)
	if got := retryAfter(t, headers("Retry-After", when.Format(http.TimeFormat))); !ptrEq(got, 0.0) {
		t.Fatalf("got %v", got)
	}
}

func TestRetryAfterReadsOpenAIResetWindows(t *testing.T) {
	// Only the exhausted window counts; both reset headers are always sent.
	h := headers(
		"x-ratelimit-remaining-requests", "0",
		"x-ratelimit-reset-requests", "250ms",
		"x-ratelimit-remaining-tokens", "39000",
		"x-ratelimit-reset-tokens", "6m0s",
	)
	if got := retryAfter(t, h); !ptrEq(got, 0.25) {
		t.Fatalf("requests window: %v", got)
	}
	h.Set("x-ratelimit-remaining-tokens", "0")
	if got := retryAfter(t, h); !ptrEq(got, 360.0) {
		t.Fatalf("both exhausted: %v", got)
	}
	// Neither window exhausted: the 429 is about something else, no hint.
	h.Set("x-ratelimit-remaining-requests", "5")
	h.Set("x-ratelimit-remaining-tokens", "5")
	if got := retryAfter(t, h); got != nil {
		t.Fatalf("none exhausted: %v", *got)
	}
	// No remaining counters at all: the sooner reset.
	if got := retryAfter(t, headers("x-ratelimit-reset-requests", "1h2m3.5s")); !ptrEq(got, 3723.5) {
		t.Fatalf("1h2m3.5s: %v", got)
	}
	if got := retryAfter(t, headers("x-ratelimit-reset-requests", "2s", "x-ratelimit-reset-tokens", "6m0s")); !ptrEq(got, 2.0) {
		t.Fatalf("sooner: %v", got)
	}
}

func TestRetryAfterIgnoresGarbage(t *testing.T) {
	for _, h := range []http.Header{{}, headers("retry-after", "soon"), headers("x-ratelimit-reset-requests", "n/a")} {
		if got := retryAfter(t, h); got != nil {
			t.Fatalf("%v -> %v", h, *got)
		}
	}
}

// --- request retry policy --------------------------------------------------------

// retryChat is test_retry's `sleeps` fixture: deterministic backoff, waits
// recorded instead of slept.
func retryChat(t *testing.T, fn func(*core.Config)) (*core.Chat, *[]time.Duration) {
	t.Helper()
	ctx := newCtx(t, func(c *core.Config) {
		c.MaxRetries = 3
		c.RetryInterval = 100 * time.Millisecond
		c.RateLimitRetryInterval = time.Second
		c.RetryIntervalRandomness = 0
		c.RetryMaxInterval = 60 * time.Second
		if fn != nil {
			fn(c)
		}
	})
	chat := newChat(t, ctx, core.ChatOptions{Model: "gpt-4o"})
	return chat, sleeps(chat)
}

func secs(f ...float64) []time.Duration {
	out := make([]time.Duration, len(f))
	for i, s := range f {
		out[i] = time.Duration(s * float64(time.Second))
	}
	return out
}

func status(code int, message string, hdr ...string) testserver.Reply {
	r := testserver.Reply{Status: code, Body: map[string]any{"error": map[string]any{"message": message}}}
	if len(hdr) > 0 {
		r.Headers = map[string]string{}
		for i := 0; i < len(hdr); i += 2 {
			r.Headers[hdr[i]] = hdr[i+1]
		}
	}
	return r
}

func sameSleeps(got *[]time.Duration, want []time.Duration) bool {
	if len(*got) == 0 && len(want) == 0 {
		return true
	}
	return reflect.DeepEqual(*got, want)
}

func TestRateLimitBacksOffFromRateLimitInterval(t *testing.T) {
	chat, slept := retryChat(t, nil)
	busy := status(429, "1302 too frequent")
	mockHTTP(t, chat, busy, busy, busy, retryOK)
	reply, err := chat.Ask(context.Background(), "hi")
	if err != nil || reply.Content != "hi" {
		t.Fatalf("reply %v err %v", reply, err)
	}
	if !sameSleeps(slept, secs(1, 2, 4)) {
		t.Fatalf("sleeps %v", *slept)
	}
}

func TestServerErrorsKeepTheShortInterval(t *testing.T) {
	chat, slept := retryChat(t, nil)
	mockHTTP(t, chat, status(503, "down"), retryOK)
	if _, err := chat.Ask(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	if !sameSleeps(slept, secs(0.1)) {
		t.Fatalf("sleeps %v", *slept)
	}
}

func TestRetryAfterHeaderOverridesBackoff(t *testing.T) {
	chat, slept := retryChat(t, nil)
	mockHTTP(t, chat, status(429, "slow down", "Retry-After", "12"), retryOK)
	if _, err := chat.Ask(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	if !sameSleeps(slept, secs(12)) {
		t.Fatalf("sleeps %v", *slept)
	}
}

func TestRetryAfterAboveCapGivesUpImmediately(t *testing.T) {
	chat, slept := retryChat(t, nil)
	srv, _ := mockHTTP(t, chat, status(429, "daily limit", "Retry-After", "3600"))
	_, err := chat.Ask(context.Background(), "hi")
	var e *core.Error
	if !errors.Is(err, core.ErrRateLimit) || !errors.As(err, &e) {
		t.Fatalf("err %v", err)
	}
	if len(*slept) != 0 || e.RetryAfter == nil || *e.RetryAfter != time.Hour || len(srv.Requests()) != 1 {
		t.Fatalf("sleeps %v retry_after %v requests %d", *slept, e.RetryAfter, len(srv.Requests()))
	}
}

func TestComputedBackoffIsCapped(t *testing.T) {
	chat, slept := retryChat(t, func(c *core.Config) { c.RetryMaxInterval = 1500 * time.Millisecond })
	busy := status(429, "busy")
	mockHTTP(t, chat, busy, busy, busy, retryOK)
	if _, err := chat.Ask(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	if !sameSleeps(slept, secs(1, 1.5, 1.5)) {
		t.Fatalf("sleeps %v", *slept)
	}
}

func TestExhaustedRetriesRaiseWithRetryAfter(t *testing.T) {
	chat, slept := retryChat(t, nil)
	srv, _ := mockHTTP(t, chat, status(429, "busy", "Retry-After", "2"))
	_, err := chat.Ask(context.Background(), "hi")
	var e *core.Error
	if !errors.As(err, &e) || e.Kind != core.KindRateLimit {
		t.Fatalf("err %v", err)
	}
	if !sameSleeps(slept, secs(2, 2, 2)) || *e.RetryAfter != 2*time.Second || len(srv.Requests()) != 4 {
		t.Fatalf("sleeps %v retry_after %v requests %d", *slept, e.RetryAfter, len(srv.Requests()))
	}
}

func TestQuotaExhaustionIsPaymentRequiredNotRetried(t *testing.T) {
	chat, slept := retryChat(t, nil)
	mockHTTP(t, chat, testserver.Reply{Status: 429, Body: map[string]any{"error": map[string]any{
		"code": "1113", "message": "Insufficient balance or no resource package. Please recharge.",
	}}})
	_, err := chat.Ask(context.Background(), "hi")
	if !errors.Is(err, core.ErrPaymentRequired) || len(*slept) != 0 {
		t.Fatalf("err %v sleeps %v", err, *slept)
	}
}

// --- streaming -------------------------------------------------------------------

func streamText(t *testing.T, chat *core.Chat) (string, error) {
	t.Helper()
	var out string
	_, err := chat.Ask(context.Background(), "hi", core.OnChunk(func(c *core.Chunk) error {
		if s, ok := c.Content.(string); ok {
			out += s
		}
		return nil
	}))
	return out, err
}

func TestStreamRetriesRateLimitBeforeFirstByte(t *testing.T) {
	chat, slept := retryChat(t, nil)
	mockHTTP(t, chat, status(429, "1305 rate limit"), retrySSE)
	text, err := streamText(t, chat)
	if err != nil || text != "hi" {
		t.Fatalf("text %q err %v", text, err)
	}
	if !sameSleeps(slept, secs(1)) {
		t.Fatalf("sleeps %v", *slept)
	}
}

func TestStreamGivesUpAfterMaxRetries(t *testing.T) {
	chat, slept := retryChat(t, nil)
	mockHTTP(t, chat, status(429, "busy"))
	if _, err := streamText(t, chat); !errors.Is(err, core.ErrRateLimit) {
		t.Fatalf("err %v", err)
	}
	if !sameSleeps(slept, secs(1, 2, 4)) {
		t.Fatalf("sleeps %v", *slept)
	}
}
