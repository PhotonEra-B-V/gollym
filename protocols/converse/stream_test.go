package converse_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"

	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
	"github.com/PhotonEra-B-V/gollym/protocols/converse"
)

// frame encodes one ConverseStream message as Bedrock sends it.
func frame(t *testing.T, messageType, typ string, payload any) []byte {
	t.Helper()
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	msg := eventstream.Message{Payload: body}
	msg.Headers.Set(":message-type", eventstream.StringValue(messageType))
	switch messageType {
	case "event":
		msg.Headers.Set(":event-type", eventstream.StringValue(typ))
	case "exception":
		msg.Headers.Set(":exception-type", eventstream.StringValue(typ))
	}
	msg.Headers.Set(":content-type", eventstream.StringValue("application/json"))
	var buf bytes.Buffer
	if err := eventstream.NewEncoder().Encode(&buf, msg); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func event(t *testing.T, typ string, payload any) []byte { return frame(t, "event", typ, payload) }

func streamBody(frames ...[]byte) []byte { return bytes.Join(frames, nil) }

func eventStreamReply(body []byte) testserver.Reply {
	return testserver.Reply{Headers: map[string]string{"Content-Type": "application/vnd.amazon.eventstream"}, Body: body}
}

type m = map[string]any

func TestStreamTextThinkingToolsAndUsage(t *testing.T) {
	body := streamBody(
		event(t, "messageStart", m{"role": "assistant", "p": "abc"}),
		event(t, "contentBlockDelta", m{"contentBlockIndex": 0, "delta": m{"reasoningContent": m{"text": "Let me "}}}),
		event(t, "contentBlockDelta", m{"contentBlockIndex": 0, "delta": m{"reasoningContent": m{"text": "think."}}}),
		event(t, "contentBlockDelta", m{"contentBlockIndex": 0, "delta": m{"reasoningContent": m{"signature": "sig-xyz"}}}),
		event(t, "contentBlockStop", m{"contentBlockIndex": 0}),
		event(t, "contentBlockDelta", m{"contentBlockIndex": 1, "delta": m{"text": "Hello"}}),
		event(t, "contentBlockDelta", m{"contentBlockIndex": 1, "delta": m{"text": " world"}}),
		event(t, "contentBlockStop", m{"contentBlockIndex": 1}),
		event(t, "contentBlockStart", m{"contentBlockIndex": 2, "start": m{"toolUse": m{"toolUseId": "tu1", "name": "weather"}}}),
		event(t, "contentBlockDelta", m{"contentBlockIndex": 2, "delta": m{"toolUse": m{"input": `{"city":`}}}),
		event(t, "contentBlockDelta", m{"contentBlockIndex": 2, "delta": m{"toolUse": m{"input": `"Paris"}`}}}),
		event(t, "contentBlockStop", m{"contentBlockIndex": 2}),
		event(t, "messageStop", m{"stopReason": "tool_use"}),
		event(t, "metadata", m{"usage": m{"inputTokens": 30, "outputTokens": 12, "totalTokens": 42, "cacheReadInputTokens": 10}, "metrics": m{"latencyMs": 100}}),
	)
	srv := testserver.New(t, eventStreamReply(body))
	chat := newChat(t, srv.URL, sonnet45)
	if err := chat.AddUserMessage("hi"); err != nil {
		t.Fatal(err)
	}
	var chunks []*core.Chunk
	var text strings.Builder
	msg, err := chat.Send(context.Background(), func(c *core.Chunk) error {
		chunks = append(chunks, c)
		if s, ok := c.Content.(string); ok {
			text.WriteString(s)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	req := srv.Last(t)
	if req.Path != "/model/"+sonnet45+"/converse-stream" || !strings.HasPrefix(req.Headers.Get("Authorization"), "AWS4-HMAC-SHA256") {
		t.Fatalf("path %s auth %q", req.Path, req.Headers.Get("Authorization"))
	}
	if len(chunks) != 14 || text.String() != "Hello world" {
		t.Fatalf("%d chunks, text %q", len(chunks), text.String())
	}
	if msg.Text() != "Hello world" || msg.FinishReason != "tool_use" || msg.ModelID != sonnet45 {
		t.Fatalf("message %+v", msg)
	}
	if msg.Thinking == nil || msg.Thinking.Text != "Let me think." || msg.Thinking.Signature != "sig-xyz" {
		t.Fatalf("thinking %+v", msg.Thinking)
	}
	if *msg.InputTokens() != 20 || *msg.OutputTokens() != 12 || *msg.CachedTokens() != 10 {
		t.Fatalf("tokens %v", msg.Tokens.Map())
	}
	tc := msg.ToolCalls.Get("tu1")
	if msg.ToolCalls.Len() != 1 || tc == nil || tc.Name != "weather" || !reflect.DeepEqual(tc.Arguments, map[string]any{"city": "Paris"}) {
		t.Fatalf("tool calls %v", msg.ToolCalls.List())
	}
	if len(chat.Messages) != 2 || chat.Messages[1] != msg {
		t.Fatal("streamed message was not added to the chat")
	}
}

// Ask with OnChunk runs the tool loop over two streamed responses.
func TestStreamToolLoop(t *testing.T) {
	srv := testserver.New(t,
		eventStreamReply(streamBody(
			event(t, "contentBlockStart", m{"start": m{"toolUse": m{"toolUseId": "tu1", "name": "weather"}}}),
			event(t, "contentBlockDelta", m{"delta": m{"toolUse": m{"input": `{"city":"Oslo"}`}}}),
			event(t, "messageStop", m{"stopReason": "tool_use"}),
		)),
		eventStreamReply(streamBody(
			event(t, "contentBlockDelta", m{"delta": m{"text": "Sunny."}}),
			event(t, "messageStop", m{"stopReason": "end_turn"}),
		)),
	)
	var calls []map[string]any
	chat := newChat(t, srv.URL, sonnet45)
	chat.WithTool(weatherTool(&calls))
	msg, err := chat.Ask(context.Background(), "Weather?", core.OnChunk(func(*core.Chunk) error { return nil }))
	if err != nil {
		t.Fatal(err)
	}
	if msg.Text() != "Sunny." || len(calls) != 1 || calls[0]["city"] != "Oslo" {
		t.Fatalf("msg %q calls %v", msg.Text(), calls)
	}
	result := core.Dig(srv.Last(t).JSON(t), "messages", 2, "content", 0, "toolResult")
	assertJSONEqual(t, result, m{"toolUseId": "tu1", "content": []any{m{"text": "sunny in Oslo"}}})
}

func TestStreamExceptions(t *testing.T) {
	cases := []struct {
		exception string
		target    error
	}{
		{"throttlingException", core.ErrRateLimit},
		{"validationException", core.ErrBadRequest},
		{"accessDeniedException", core.ErrUnauthorized},
		{"serviceUnavailableException", core.ErrServiceUnavailable},
		{"modelStreamErrorException", core.ErrServer},
	}
	for _, tc := range cases {
		srv := testserver.New(t, eventStreamReply(streamBody(
			event(t, "contentBlockDelta", m{"delta": m{"text": "partial"}}),
			frame(t, "exception", tc.exception, m{"message": "boom: " + tc.exception}),
		)))
		chat := newChat(t, srv.URL, sonnet45)
		_, err := chat.Ask(context.Background(), "hi", core.OnChunk(func(*core.Chunk) error { return nil }))
		if !errors.Is(err, tc.target) || err.Error() != "boom: "+tc.exception {
			t.Errorf("%s: err %v", tc.exception, err)
		}
		if n := len(srv.Requests()); n != 1 {
			t.Errorf("%s: mid-stream errors must not be retried (%d requests)", tc.exception, n)
		}
	}
}

func TestStreamErrorMessage(t *testing.T) {
	msg := eventstream.Message{}
	msg.Headers.Set(":message-type", eventstream.StringValue("error"))
	msg.Headers.Set(":error-code", eventstream.StringValue("InternalFailure"))
	msg.Headers.Set(":error-message", eventstream.StringValue("something broke"))
	var buf bytes.Buffer
	if err := eventstream.NewEncoder().Encode(&buf, msg); err != nil {
		t.Fatal(err)
	}
	srv := testserver.New(t, eventStreamReply(buf.Bytes()))
	_, err := newChat(t, srv.URL, sonnet45).Ask(context.Background(), "hi", core.OnChunk(func(*core.Chunk) error { return nil }))
	if !errors.Is(err, core.ErrServer) || err.Error() != "something broke" {
		t.Fatalf("err %v", err)
	}
}

func TestStreamHTTPErrorBeforeStreamIsRetried(t *testing.T) {
	srv := testserver.New(t,
		testserver.Reply{Status: 429, Body: m{"message": "Too many requests"}},
		eventStreamReply(event(t, "contentBlockDelta", m{"delta": m{"text": "ok"}})),
	)
	msg, err := newChat(t, srv.URL, sonnet45).Ask(context.Background(), "hi", core.OnChunk(func(*core.Chunk) error { return nil }))
	if err != nil || msg.Text() != "ok" || len(srv.Requests()) != 2 {
		t.Fatalf("msg %v err %v requests %d", msg, err, len(srv.Requests()))
	}
}

func TestStreamTruncatedFrameFails(t *testing.T) {
	full := event(t, "contentBlockDelta", m{"delta": m{"text": "ok"}})
	srv := testserver.New(t, eventStreamReply(full[:len(full)-3]))
	_, err := newChat(t, srv.URL, sonnet45, func(c *core.Config) { c.MaxRetries = 0 }).
		Ask(context.Background(), "hi", core.OnChunk(func(*core.Chunk) error { return nil }))
	if !errors.Is(err, core.ErrConnectionFailed) {
		t.Fatalf("err %v", err)
	}
}

// --- BuildChunk on raw events (pyllym build_chunk) ---------------------------

func TestBuildChunkShapes(t *testing.T) {
	p := &converse.Protocol{Model: &core.Info{ID: "fallback-model"}}
	chunk := func(ev map[string]any) *core.Chunk {
		t.Helper()
		c, err := p.BuildChunk(ev)
		if err != nil {
			t.Fatal(err)
		}
		return c
	}
	// Flattened delta (string JSON) and explicit model id.
	c := chunk(m{"delta": `{"text": "hi"}`, "modelId": "m1"})
	if c.Content != "hi" || c.ModelID != "m1" {
		t.Fatalf("chunk %+v", c)
	}
	// Start events without toolUse carry the signature of a reasoning block.
	c = chunk(m{"contentBlockStart": m{"start": m{"reasoningContent": m{"redactedContent": "red"}}}})
	if c.ToolCalls != nil || c.Thinking == nil || c.Thinking.Signature != "red" || c.ModelID != "fallback-model" {
		t.Fatalf("chunk %+v", c)
	}
	// Flattened start/toolUse delta, top-level usage and stopReason.
	c = chunk(m{"start": m{"toolUse": m{"toolUseId": "x", "name": "n", "input": m{"a": 1}}}})
	if c.ToolCalls.Get("x") == nil || !reflect.DeepEqual(c.ToolCalls.Get("x").Arguments, m{"a": 1}) {
		t.Fatalf("chunk %+v", c.ToolCalls.List())
	}
	c = chunk(m{"delta": m{"toolUse": m{"input": `{"a"`}}})
	if c.ToolCalls.Len() != 1 || c.ToolCalls.Get("").Arguments != `{"a"` {
		t.Fatalf("chunk %+v", c.ToolCalls.List())
	}
	c = chunk(m{"usage": m{"inputTokens": 7, "outputTokens": 3, "reasoningTokens": 2}, "stopReason": "max_tokens"})
	if *c.InputTokens() != 7 || *c.OutputTokens() != 3 || *c.ThinkingTokens() != 2 || !c.IsMaxTokens() {
		t.Fatalf("chunk %+v", c.Tokens.Map())
	}
	// Exceptions: unknown names are server errors; "type": "error" too.
	if _, err := p.BuildChunk(m{"internalServerException": m{"message": "x"}}); !errors.Is(err, core.ErrServer) || err.Error() != "x" {
		t.Fatalf("err %v", err)
	}
	if _, err := p.BuildChunk(m{"type": "error", "error": m{}}); !errors.Is(err, core.ErrServer) || err.Error() != "Bedrock streaming error" {
		t.Fatalf("err %v", err)
	}
	if _, err := p.BuildChunk(m{"throttlingException": m{}}); !errors.Is(err, core.ErrRateLimit) || err.Error() != "throttlingException" {
		t.Fatalf("err %v", err)
	}
}
