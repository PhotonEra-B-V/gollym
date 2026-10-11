package core_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/PhotonEra-B-V/gollym/core"
)

// Ports tests/test_errors.py: HTTP statuses map to error kinds.

func TestStatus401MapsToUnauthorized(t *testing.T) {
	chat := newChat(t, newCtx(t, nil), core.ChatOptions{Model: "gpt-4o"})
	mockHTTP(t, chat, status(401, "bad key"))
	_, err := chat.Ask(context.Background(), "hi")
	if !errors.Is(err, core.ErrUnauthorized) || !strings.Contains(err.Error(), "bad key") {
		t.Fatalf("err %v", err)
	}
}

func TestStatus400MapsToBadRequest(t *testing.T) {
	chat := newChat(t, newCtx(t, nil), core.ChatOptions{Model: "gpt-4o"})
	mockHTTP(t, chat, status(400, "nope"))
	if _, err := chat.Ask(context.Background(), "hi"); !errors.Is(err, core.ErrBadRequest) {
		t.Fatalf("err %v", err)
	}
}

func TestStatus429RaisesRateLimit(t *testing.T) {
	chat := newChat(t, newCtx(t, func(c *core.Config) { c.MaxRetries = 0 }), core.ChatOptions{Model: "gpt-4o"})
	srv, _ := mockHTTP(t, chat, status(429, "slow down"))
	if _, err := chat.Ask(context.Background(), "hi"); !errors.Is(err, core.ErrRateLimit) {
		t.Fatalf("err %v", err)
	}
	if len(srv.Requests()) != 1 {
		t.Fatalf("%d requests", len(srv.Requests()))
	}
}

func TestStatus500MapsToServerError(t *testing.T) {
	chat := newChat(t, newCtx(t, func(c *core.Config) { c.MaxRetries = 0 }), core.ChatOptions{Model: "gpt-4o"})
	mockHTTP(t, chat, status(500, "boom"))
	if _, err := chat.Ask(context.Background(), "hi"); !errors.Is(err, core.ErrServer) {
		t.Fatalf("err %v", err)
	}
}
