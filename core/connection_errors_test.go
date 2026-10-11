package core_test

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym/core"
)

// Ports tests/test_connection_errors.py: transport failures surface as
// ConnectionFailed *core.Error, never raw net/http errors. These use real
// sockets (a closed port, a server that never answers).

func closedPort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	return port
}

// fastFail points OpenAI at base with no retries and a short timeout.
func fastFail(t *testing.T, base string) *core.Chat {
	t.Helper()
	ctx := newCtx(t, func(c *core.Config) {
		c.MaxRetries = 0
		c.RequestTimeout = 500 * time.Millisecond
		if base != "" {
			c.Set("openai_api_base", base)
		}
	})
	return newChat(t, ctx, core.ChatOptions{Model: "gpt-4o"})
}

func assertConnectionFailed(t *testing.T, err error) *core.Error {
	t.Helper()
	var e *core.Error
	if !errors.As(err, &e) || !errors.Is(err, core.ErrConnectionFailed) || !errors.Is(err, core.ErrAPI) {
		t.Fatalf("want ConnectionFailed *core.Error, got %T %v", err, err)
	}
	return e
}

func TestClosedPortRaisesConnectionFailed(t *testing.T) {
	chat := fastFail(t, fmt.Sprintf("http://127.0.0.1:%d/v1", closedPort(t)))
	_, err := chat.Ask(context.Background(), "hi")
	if e := assertConnectionFailed(t, err); e.Cause == nil {
		t.Fatal("cause not kept")
	}
}

func TestTimingOutSocketRaisesConnectionFailed(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	defer close(done)
	defer l.Close()
	go func() {
		for {
			conn, err := l.Accept()
			if err != nil {
				return
			}
			go func() {
				defer conn.Close()
				_, _ = conn.Read(make([]byte, 1))
				select {
				case <-done:
				case <-time.After(30 * time.Second):
				}
			}()
		}
	}()
	chat := fastFail(t, "http://"+l.Addr().String()+"/v1")
	start := time.Now()
	_, err = chat.Ask(context.Background(), "hi")
	assertConnectionFailed(t, err)
	if time.Since(start) > 5*time.Second {
		t.Fatalf("took %v", time.Since(start))
	}
}

func TestStreamingClosedPortRaisesConnectionFailed(t *testing.T) {
	chat := fastFail(t, fmt.Sprintf("http://127.0.0.1:%d/v1", closedPort(t)))
	_, err := chat.Ask(context.Background(), "hi", core.OnChunk(func(*core.Chunk) error { return nil }))
	assertConnectionFailed(t, err)
}

// HTTP-level failures keep their specific error kinds.
func TestRetriesStillMapHTTPStatuses(t *testing.T) {
	chat := fastFail(t, "")
	mockHTTP(t, chat, status(503, "down"))
	_, err := chat.Ask(context.Background(), "hi")
	if !errors.Is(err, core.ErrServiceUnavailable) || !strings.Contains(err.Error(), "down") {
		t.Fatalf("err %v", err)
	}
}
