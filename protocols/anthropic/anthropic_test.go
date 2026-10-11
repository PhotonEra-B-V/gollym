package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym"
	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
	"github.com/PhotonEra-B-V/gollym/protocols/anthropic"
)

// --- helpers -----------------------------------------------------------------

func testContext(base string, retries int) *gollym.Context {
	return gollym.NewContext(func(c *gollym.Config) {
		c.Set("anthropic_api_key", "sk-ant-test")
		if base != "" {
			c.Set("anthropic_api_base", base)
		}
		c.MaxRetries = retries
		c.RetryInterval = time.Millisecond
		c.RequestTimeout = 5 * time.Second
	})
}

func claudeChat(t *testing.T, base, model string, retries int) *gollym.Chat {
	t.Helper()
	chat, err := testContext(base, retries).NewChat(gollym.ChatOptions{Model: model})
	if err != nil {
		t.Fatal(err)
	}
	return chat
}

// protocolFor builds the anthropic protocol for a registry model.
func protocolFor(t *testing.T, base, model string) *anthropic.Protocol {
	t.Helper()
	spec, ok := core.LookupProvider("anthropic")
	if !ok {
		t.Fatal("anthropic provider is not registered")
	}
	p, err := core.NewProvider(spec, testContext(base, 0).Config)
	if err != nil {
		t.Fatal(err)
	}
	info, err := gollym.Models().Find(model, "anthropic")
	if err != nil {
		t.Fatal(err)
	}
	return anthropic.New(p, info).(*anthropic.Protocol)
}

func anthropicMessage(text string, toolUse map[string]any) map[string]any {
	content := []any{map[string]any{"type": "text", "text": text}}
	stop := "end_turn"
	if toolUse != nil {
		block := map[string]any{"type": "tool_use"}
		for k, v := range toolUse {
			block[k] = v
		}
		content = append(content, block)
		stop = "tool_use"
	}
	return map[string]any{
		"id": "msg-test", "model": "claude-sonnet-4-6", "stop_reason": stop, "content": content,
		"usage": map[string]any{"input_tokens": 12, "output_tokens": 7},
	}
}

var golden map[string]any

func gold(t *testing.T, key string) any {
	t.Helper()
	if golden == nil {
		b, err := os.ReadFile("testdata/pyllym_golden.json")
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(b, &golden); err != nil {
			t.Fatal(err)
		}
	}
	v, ok := golden[key]
	if !ok {
		t.Fatalf("no golden %q", key)
	}
	return v
}

// normalize round-trips v through JSON so it compares with decoded golden.
func normalize(t *testing.T, v any) any {
	t.Helper()
	b, err := core.MarshalPayload(v)
	if err != nil {
		t.Fatal(err)
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func assertJSONEqual(t *testing.T, name string, got, want any) {
	t.Helper()
	g := normalize(t, got)
	w := normalize(t, want)
	if !reflect.DeepEqual(g, w) {
		gb, _ := json.MarshalIndent(g, "", " ")
		wb, _ := json.MarshalIndent(w, "", " ")
		t.Fatalf("%s mismatch\n--- got\n%s\n--- want (pyllym)\n%s", name, gb, wb)
	}
}

func msg(t *testing.T, m core.Message) *core.Message {
	t.Helper()
	out, err := core.NewMessage(m)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func content(t *testing.T, text string, sources ...any) *core.Content {
	t.Helper()
	c, err := core.NewContent(core.Str(text), sources...)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func toolSet(tools ...core.Tool) *core.ToolSet {
	s := &core.ToolSet{}
	for _, t := range tools {
		s.Add(t)
	}
	return s
}

var weatherTool = &core.FuncTool{
	ToolName: "weather", Desc: "Get weather",
	Schema: map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []any{"city"}},
}

var pingTool = &core.FuncTool{ToolName: "ping", Extra: map[string]any{"cache_control": map[string]any{"type": "ephemeral"}}}

// summary mirrors gen_golden.py's summary() of a parsed message.
func summary(m *core.Message) map[string]any {
	out := map[string]any{"content": m.Content, "finish_reason": nilIfEmpty(m.FinishReason), "model_id": nilIfEmpty(m.ModelID)}
	if m.Thinking != nil {
		out["thinking"] = map[string]any{"text": nilIfEmpty(m.Thinking.Text), "signature": nilIfEmpty(m.Thinking.Signature)}
	} else {
		out["thinking"] = nil
	}
	if m.ToolCalls.Len() > 0 {
		calls := map[string]any{}
		for _, k := range m.ToolCalls.Keys() {
			tc := m.ToolCalls.Get(k)
			calls[k] = map[string]any{"id": tc.ID, "name": tc.Name, "arguments": tc.Arguments}
		}
		out["tool_calls"] = calls
	} else {
		out["tool_calls"] = nil
	}
	if m.Tokens != nil {
		out["tokens"] = m.Tokens.Map()
	} else {
		out["tokens"] = nil
	}
	cits := []any{}
	for _, c := range m.Citations {
		cits = append(cits, c)
	}
	out["citations"] = cits
	return out
}

func nilIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func sse(events ...map[string]any) string {
	var b strings.Builder
	for _, e := range events {
		data, _ := json.Marshal(e)
		b.WriteString("event: " + e["type"].(string) + "\ndata: " + string(data) + "\n\n")
	}
	return b.String()
}

func goldenEvents(t *testing.T) []map[string]any {
	var out []map[string]any
	for _, e := range gold(t, "stream_events").([]any) {
		out = append(out, e.(map[string]any))
	}
	return out
}

// --- pyllym tests/test_chat.py -----------------------------------------------

func TestAnthropicChat(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: map[string]any{
		"id": "msg", "model": "claude-sonnet-4-6", "stop_reason": "end_turn",
		"content": []any{map[string]any{"type": "text", "text": "Hello from Claude"}},
		"usage":   map[string]any{"input_tokens": 12, "output_tokens": 7},
	}})
	m, err := claudeChat(t, srv.URL, "claude-sonnet-4-6", 0).Ask(context.Background(), "Hello")
	if err != nil {
		t.Fatal(err)
	}
	if m.Text() != "Hello from Claude" || *m.OutputTokens() != 7 || *m.InputTokens() != 12 || !m.IsStopped() {
		t.Fatalf("message %+v", m)
	}
	req := srv.Last(t)
	if req.Path != "/v1/messages" || req.Headers.Get("x-api-key") != "sk-ant-test" || req.Headers.Get("anthropic-version") != "2023-06-01" {
		t.Fatalf("path %s headers %v", req.Path, req.Headers)
	}
	assertJSONEqual(t, "payload", req.JSON(t), map[string]any{
		"model": "claude-sonnet-4-6", "stream": false, "max_tokens": 128000,
		"messages": []any{map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "Hello"}}}},
	})
}

func TestAnthropicToolLoop(t *testing.T) {
	srv := testserver.New(t,
		testserver.Reply{Body: anthropicMessage("checking", map[string]any{"id": "tu1", "name": "weather", "input": map[string]any{"city": "Paris"}})},
		testserver.Reply{Body: anthropicMessage("It is sunny in Paris.", nil)},
	)
	chat := claudeChat(t, srv.URL, "claude-sonnet-4-6", 0)
	chat.WithTool(gollym.NewTool("weather", "Get weather", []gollym.Parameter{{Name: "city", Required: true}},
		func(_ context.Context, args map[string]any) (any, error) {
			return "Sunny in " + args["city"].(string), nil
		}))
	m, err := chat.Ask(context.Background(), "Weather in Paris?")
	if err != nil {
		t.Fatal(err)
	}
	if m.Text() != "It is sunny in Paris." {
		t.Fatalf("content %q", m.Text())
	}
	var roles []string
	for _, cm := range chat.Messages {
		roles = append(roles, cm.Role)
	}
	if strings.Join(roles, ",") != "user,assistant,tool,assistant" {
		t.Fatalf("roles %v", roles)
	}
	second := srv.Requests()[1].JSON(t)
	msgs := second["messages"].([]any)
	assertJSONEqual(t, "history", msgs[1:], []any{
		map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "text", "text": "checking"},
			map[string]any{"type": "tool_use", "id": "tu1", "name": "weather", "input": map[string]any{"city": "Paris"}},
		}},
		map[string]any{"role": "user", "content": []any{map[string]any{
			"type": "tool_result", "tool_use_id": "tu1", "content": []any{map[string]any{"type": "text", "text": "Sunny in Paris"}},
		}}},
	})
	tool := second["tools"].([]any)[0].(map[string]any)
	if tool["name"] != "weather" || tool["description"] != "Get weather" || tool["input_schema"] == nil {
		t.Fatalf("tool %v", tool)
	}
}

// --- pyllym tests/test_behaviors.py / test_core.py ---------------------------

func TestRenderAnthropicShape(t *testing.T) {
	chat := claudeChat(t, "", "claude-sonnet-4-6", 0)
	if err := chat.AddUserMessage("hi"); err != nil {
		t.Fatal(err)
	}
	payload, err := chat.Render()
	if err != nil {
		t.Fatal(err)
	}
	if payload["model"] != "claude-sonnet-4-6" || payload["max_tokens"] != 128000 {
		t.Fatalf("payload %v", payload)
	}
	if payload["messages"].([]any)[0].(map[string]any)["role"] != "user" {
		t.Fatalf("messages %v", payload["messages"])
	}
}

func TestModelsRegistryLookup(t *testing.T) {
	m, err := gollym.Models().Find("claude-sonnet-4-6", "")
	if err != nil || m.Provider != "anthropic" {
		t.Fatalf("model %v err %v", m, err)
	}
}

// --- parity with pyllym (testdata/pyllym_golden.json, gen_golden.py) ---------

func TestGoldenRenderFull(t *testing.T) {
	p := protocolFor(t, "", "claude-sonnet-4-6")
	assistantJSON := msg(t, core.Message{Role: core.RoleAssistant, Content: "x"})
	assistantJSON.Content = map[string]any{"k": []any{1, "ä"}}
	messages := []*core.Message{
		msg(t, core.Message{Role: core.RoleSystem, Content: "You are terse."}),
		msg(t, core.Message{Role: core.RoleSystem, Content: core.RawContent{Value: []any{map[string]any{"type": "text", "text": "cached", "cache_control": map[string]any{"type": "ephemeral"}}}}}),
		msg(t, core.Message{Role: core.RoleUser, Content: "hi"}),
		msg(t, core.Message{Role: core.RoleAssistant, Content: "prev", Thinking: &core.Thinking{Text: "mull", Signature: "sig1"}}),
		msg(t, core.Message{Role: core.RoleUser, Content: content(t, "look", "testdata/cat.png", "testdata/report.pdf", "testdata/notes.txt")}),
		msg(t, core.Message{Role: core.RoleUser, Content: content(t, "remote", "https://example.com/cat.png", "https://example.com/doc.pdf")}),
		assistantJSON,
	}
	schema := core.NormalizeSchema(map[string]any{"name": "answer", "schema": map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "integer"}}, "strict": true}, "strict": true})
	payload, err := p.RenderPayload(&core.CompletionRequest{
		Messages: messages, Tools: toolSet(weatherTool, pingTool), ToolPrefs: core.ToolPrefs{Choice: "required", Calls: "one"},
		Temperature: core.Float(0.2), Model: p.Model, Schema: schema, Thinking: &core.ThinkingConfig{Effort: "high"},
		Citations: true, Stream: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, "render_full", payload, gold(t, "render_full"))

	payload, err = p.RenderPayload(&core.CompletionRequest{
		Messages: []*core.Message{msg(t, core.Message{Role: core.RoleUser, Content: content(t, "look", "testdata/notes.txt", "testdata/report.pdf")})},
		Model:    p.Model,
	})
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, "render_nocite", payload, gold(t, "render_nocite"))
}

func TestGoldenRenderHistory(t *testing.T) {
	p := protocolFor(t, "", "claude-3-7-sonnet-20250219")
	calls := func(id, name string) *core.ToolCalls {
		c := &core.ToolCalls{}
		c.Set(id, &core.ToolCall{ID: id, Name: name, Arguments: map[string]any{}})
		return c
	}
	weather := &core.ToolCalls{}
	weather.Set("tu1", &core.ToolCall{ID: "tu1", Name: "weather", Arguments: map[string]any{"city": "Paris"}})
	results, err := core.NewSearchResults(
		map[string]string{"title": "Q4", "url": "https://x.test/q4", "text": "Revenue up"},
		map[string]string{"title": "Notes", "text": "No url"},
	)
	if err != nil {
		t.Fatal(err)
	}
	toolJSON := msg(t, core.Message{Role: core.RoleTool, Content: "x", ToolCallID: "tu5"})
	toolJSON.Content = map[string]any{"ok": true}
	history := []*core.Message{
		msg(t, core.Message{Role: core.RoleUser, Content: "Weather?"}),
		msg(t, core.Message{Role: core.RoleAssistant, Content: "checking", ToolCalls: weather, Thinking: &core.Thinking{Text: "t1", Signature: "s1"}}),
		msg(t, core.Message{Role: core.RoleTool, Content: "Sunny", ToolCallID: "tu1"}),
		msg(t, core.Message{Role: core.RoleAssistant, Content: "", ToolCalls: calls("tu2", "search"), Thinking: &core.Thinking{Signature: "redacted"}}),
		msg(t, core.Message{Role: core.RoleTool, Content: results, ToolCallID: "tu2"}),
		msg(t, core.Message{Role: core.RoleAssistant, ToolCalls: calls("tu3", "noop")}),
		msg(t, core.Message{Role: core.RoleTool, Content: "", ToolCallID: "tu3"}),
		msg(t, core.Message{Role: core.RoleAssistant, Content: core.RawContent{Value: []any{map[string]any{"type": "tool_use", "id": "tu4", "name": "raw", "input": map[string]any{}}}}, ToolCalls: calls("tu4", "raw")}),
		msg(t, core.Message{Role: core.RoleTool, Content: core.RawContent{Value: []any{map[string]any{"type": "tool_result", "tool_use_id": "tu4", "content": "raw"}}}, ToolCallID: "tu4"}),
		toolJSON,
	}
	payload, err := p.RenderPayload(&core.CompletionRequest{
		Messages: history, Model: p.Model, Thinking: &core.ThinkingConfig{Budget: core.Int(2048)},
		ToolPrefs: core.ToolPrefs{Choice: "weather"}, Tools: toolSet(weatherTool),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, "render_history", payload, gold(t, "render_history"))

	payload, err = p.RenderPayload(&core.CompletionRequest{Messages: history, Model: p.Model, Tools: &core.ToolSet{}})
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, "render_history_nothinking", payload, gold(t, "render_history_nothinking"))
}

func TestGoldenToolChoice(t *testing.T) {
	want := gold(t, "tool_choices").(map[string]any)
	cases := map[string]core.ToolPrefs{
		"None|one":      {Calls: "one"},
		"auto|None":     {Choice: "auto"},
		"none|many":     {Choice: "none", Calls: "many"},
		"required|many": {Choice: "required", Calls: "many"},
		"weather|one":   {Choice: "weather", Calls: "one"},
	}
	for key, prefs := range cases {
		assertJSONEqual(t, key, anthropic.BuildToolChoice(prefs), want[key])
	}
}

func TestGoldenThinking(t *testing.T) {
	want := gold(t, "thinking_errors").(map[string]any)
	cases := map[string]*core.ThinkingConfig{
		"claude-3-haiku-20240307":    {Budget: core.Int(1024)},
		"claude-3-7-sonnet-20250219": {Effort: "high"},
		"claude-sonnet-4-6":          {Budget: core.Int(0)},
	}
	user := []*core.Message{msg(t, core.Message{Role: core.RoleUser, Content: "x"})}
	for model, cfg := range cases {
		p := protocolFor(t, "", model)
		_, err := p.RenderPayload(&core.CompletionRequest{Messages: user, Model: p.Model, Thinking: cfg})
		if err == nil || err.Error() != want[model] {
			t.Fatalf("%s: err %v, want %v", model, err, want[model])
		}
	}
	p := protocolFor(t, "", "claude-sonnet-4-6")
	payload, err := p.RenderPayload(&core.CompletionRequest{Messages: user, Model: p.Model, Thinking: &core.ThinkingConfig{Effort: "none"}})
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, "render_effort_none", payload, gold(t, "render_effort_none"))
	p = protocolFor(t, "", "claude-3-haiku-20240307")
	payload, err = p.RenderPayload(&core.CompletionRequest{Messages: user, Model: p.Model})
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, "render_haiku", payload, gold(t, "render_haiku"))
}

func TestGoldenParse(t *testing.T) {
	p := protocolFor(t, "", "claude-sonnet-4-6")
	for _, pair := range [][2]string{{"parse_body", "parse_summary"}, {"parse_body2", "parse_summary2"}} {
		m, err := p.ParseCompletionResponse(&core.Response{Status: 200, Body: gold(t, pair[0])})
		if err != nil {
			t.Fatal(err)
		}
		assertJSONEqual(t, pair[1], summary(m), gold(t, pair[1]))
	}
}

func TestGoldenStreamChunks(t *testing.T) {
	p := protocolFor(t, "", "claude-sonnet-4-6")
	acc := core.NewStreamAccumulator()
	for _, e := range goldenEvents(t) {
		c, err := p.BuildChunk(e)
		if err != nil {
			t.Fatal(err)
		}
		acc.Add(c)
	}
	assertJSONEqual(t, "stream_summary", summary(acc.Message(nil)), gold(t, "stream_summary"))
}

func TestGoldenListModels(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: gold(t, "list_models_body")})
	p := protocolFor(t, srv.URL, "claude-sonnet-4-6")
	models, err := gollym.ListModels(context.Background(), p.Provider)
	if err != nil {
		t.Fatal(err)
	}
	req := srv.Last(t)
	if req.Method != "GET" || req.Path != "/v1/models" || req.Headers.Get("x-api-key") != "sk-ant-test" {
		t.Fatalf("request %s %s", req.Method, req.Path)
	}
	var got []any
	for _, m := range models {
		got = append(got, map[string]any{
			"id": m.ID, "name": m.Name, "provider": m.Provider,
			"created_at": m.CreatedAt.UTC().Format("2006-01-02T15:04:05+00:00"), "capabilities": m.Capabilities,
		})
	}
	assertJSONEqual(t, "list_models", got, gold(t, "list_models"))
}

// --- end to end over HTTP ----------------------------------------------------

// The golden stream ends in a tool_use, so the chat runs the tool and
// streams a second, text-only turn.
func TestStreamingOverSSE(t *testing.T) {
	events := goldenEvents(t)
	final := []map[string]any{
		{"type": "message_start", "message": map[string]any{"model": "claude-sonnet-4-6", "usage": map[string]any{"input_tokens": 30, "output_tokens": 1}}},
		{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "text_delta", "text": "Cold."}},
		{"type": "message_delta", "delta": map[string]any{"stop_reason": "end_turn"}, "usage": map[string]any{"output_tokens": 3}},
	}
	stream := map[string]string{"Content-Type": "text/event-stream"}
	srv := testserver.New(t, testserver.Reply{Headers: stream, Body: sse(events...)}, testserver.Reply{Headers: stream, Body: sse(final...)})
	chat := claudeChat(t, srv.URL, "claude-sonnet-4-6", 0)
	var city any
	chat.WithTool(gollym.NewTool("weather", "Get weather", []gollym.Parameter{{Name: "city", Required: true}},
		func(_ context.Context, args map[string]any) (any, error) { city = args["city"]; return "-20C", nil }))
	chunks := 0
	var text strings.Builder
	m, err := chat.Ask(context.Background(), "hi", gollym.OnChunk(func(c *gollym.Chunk) error {
		chunks++
		if s, ok := c.Content.(string); ok {
			text.WriteString(s)
		}
		return nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	if chunks != len(events)+len(final) || text.String() != "HelloCold." || city != "Oulu" {
		t.Fatalf("%d chunks, text %q, city %v", chunks, text.String(), city)
	}
	if m.Text() != "Cold." || *m.InputTokens() != 30 || *m.OutputTokens() != 3 || !m.IsStopped() {
		t.Fatalf("final %+v", m)
	}
	assertJSONEqual(t, "stream_summary", summary(chat.Messages[1]), gold(t, "stream_summary"))
	body := srv.Last(t).JSON(t)
	if body["stream"] != true {
		t.Fatalf("payload %v", body)
	}
	// The streamed thinking and signature go back with the tool call when
	// thinking is enabled; without it they are dropped.
	asst := body["messages"].([]any)[1].(map[string]any)["content"].([]any)
	if len(asst) != 2 || asst[0].(map[string]any)["type"] != "text" || asst[1].(map[string]any)["type"] != "tool_use" {
		t.Fatalf("assistant blocks %v", asst)
	}
}

func TestStreamErrorEvent(t *testing.T) {
	body := sse(map[string]any{"type": "message_start", "message": map[string]any{"model": "claude-sonnet-4-6"}}) +
		"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"
	srv := testserver.New(t, testserver.Reply{Headers: map[string]string{"Content-Type": "text/event-stream"}, Body: body})
	_, err := claudeChat(t, srv.URL, "claude-sonnet-4-6", 0).Ask(context.Background(), "hi", gollym.OnChunk(func(*gollym.Chunk) error { return nil }))
	if !errors.Is(err, gollym.ErrOverloaded) || err.Error() != "Overloaded" {
		t.Fatalf("err %v", err)
	}
}

func TestHTTPErrors(t *testing.T) {
	errBody := func(typ, message string) map[string]any {
		return map[string]any{"type": "error", "error": map[string]any{"type": typ, "message": message}}
	}
	cases := []struct {
		status int
		body   map[string]any
		want   error
	}{
		{529, errBody("overloaded_error", "Overloaded"), gollym.ErrOverloaded},
		{401, errBody("authentication_error", "invalid x-api-key"), gollym.ErrUnauthorized},
		{429, errBody("rate_limit_error", "Number of request tokens has exceeded your per-minute rate limit"), gollym.ErrRateLimit},
		{400, errBody("invalid_request_error", "prompt is too long: 210000 tokens > 200000 maximum"), gollym.ErrContextLengthExceeded},
	}
	for _, tc := range cases {
		srv := testserver.New(t, testserver.Reply{Status: tc.status, Body: tc.body})
		_, err := claudeChat(t, srv.URL, "claude-sonnet-4-6", 0).Ask(context.Background(), "hi")
		if !errors.Is(err, tc.want) || err.Error() != tc.body["error"].(map[string]any)["message"] {
			t.Fatalf("%d: err %v", tc.status, err)
		}
	}
}

func TestOverloadedIsRetried(t *testing.T) {
	srv := testserver.New(t,
		testserver.Reply{Status: 529, Body: map[string]any{"type": "error", "error": map[string]any{"type": "overloaded_error", "message": "Overloaded"}}},
		testserver.Reply{Body: anthropicMessage("ok", nil)},
	)
	m, err := claudeChat(t, srv.URL, "claude-sonnet-4-6", 2).Ask(context.Background(), "hi")
	if err != nil || m.Text() != "ok" || len(srv.Requests()) != 2 {
		t.Fatalf("msg %v err %v requests %d", m, err, len(srv.Requests()))
	}
}

func TestStructuredOutput(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: anthropicMessage(`{"score": 7}`, nil)})
	chat := claudeChat(t, srv.URL, "claude-sonnet-4-6", 0)
	if _, err := chat.WithSchema(map[string]any{"type": "object", "properties": map[string]any{"score": map[string]any{"type": "integer"}}, "strict": true}); err != nil {
		t.Fatal(err)
	}
	if _, err := chat.WithThinking("low", nil); err != nil {
		t.Fatal(err)
	}
	m, err := chat.Ask(context.Background(), "Score")
	if err != nil {
		t.Fatal(err)
	}
	if got, ok := m.Content.(map[string]any); !ok || got["score"] != 7.0 {
		t.Fatalf("content %#v", m.Content)
	}
	assertJSONEqual(t, "output_config", srv.Last(t).JSON(t)["output_config"], map[string]any{
		"effort": "low",
		"format": map[string]any{"type": "json_schema", "schema": map[string]any{"type": "object", "properties": map[string]any{"score": map[string]any{"type": "integer"}}}},
	})
}

func TestInstructionsAndParams(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: anthropicMessage("ok", nil)})
	chat := claudeChat(t, srv.URL, "claude-sonnet-4-6", 0)
	chat.WithInstructions("Be brief.").WithTemperature(0).WithParams(map[string]any{"max_tokens": 512, "metadata": map[string]any{"user_id": "u1"}})
	if _, err := chat.Ask(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	body := srv.Last(t).JSON(t)
	assertJSONEqual(t, "payload", body, map[string]any{
		"model": "claude-sonnet-4-6", "stream": false, "max_tokens": 512, "temperature": 0, "metadata": map[string]any{"user_id": "u1"},
		"system":   []any{map[string]any{"type": "text", "text": "Be brief."}},
		"messages": []any{map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "hi"}}}},
	})
}

func TestProviderFiles(t *testing.T) {
	p := protocolFor(t, "", "claude-sonnet-4-6")
	img := &core.UploadedFile{ID: "file_img", Filename: "cat.png"}
	doc := &core.UploadedFile{ID: "file_doc", Filename: "report.pdf"}
	c := content(t, "see")
	for _, f := range []*core.UploadedFile{img, doc} {
		if err := c.AddAttachment(f, ""); err != nil {
			t.Fatal(err)
		}
	}
	payload, err := p.RenderPayload(&core.CompletionRequest{Messages: []*core.Message{msg(t, core.Message{Role: core.RoleUser, Content: c})}, Model: p.Model, Citations: true})
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, "provider files", payload["messages"].([]any)[0].(map[string]any)["content"], []any{
		map[string]any{"type": "text", "text": "see"},
		map[string]any{"type": "image", "source": map[string]any{"type": "file", "file_id": "file_img"}},
		map[string]any{"type": "document", "source": map[string]any{"type": "file", "file_id": "file_doc"}, "title": "report.pdf", "citations": map[string]any{"enabled": true}},
	})
}

func TestUnsupportedAttachment(t *testing.T) {
	p := protocolFor(t, "", "claude-sonnet-4-6")
	c := content(t, "listen")
	if err := c.AddAttachment([]byte("ID3fake"), "song.mp3"); err != nil {
		t.Fatal(err)
	}
	_, err := p.RenderPayload(&core.CompletionRequest{Messages: []*core.Message{msg(t, core.Message{Role: core.RoleUser, Content: c})}, Model: p.Model})
	if !errors.Is(err, core.ErrUnsupportedContent) {
		t.Fatalf("err %v", err)
	}
}

func TestSearchResultsToolLoop(t *testing.T) {
	srv := testserver.New(t,
		testserver.Reply{Body: anthropicMessage("", map[string]any{"id": "tu1", "name": "search", "input": map[string]any{"q": "q4"}})},
		testserver.Reply{Body: map[string]any{
			"model": "claude-sonnet-4-6", "stop_reason": "end_turn",
			"content": []any{map[string]any{"type": "text", "text": "Revenue rose.", "citations": []any{map[string]any{
				"type": "search_result_location", "source": "https://x.test/q4", "title": "Q4", "cited_text": "Revenue up", "search_result_index": 0,
			}}}},
			"usage": map[string]any{"input_tokens": 1, "output_tokens": 1},
		}},
	)
	chat := claudeChat(t, srv.URL, "claude-sonnet-4-6", 0)
	chat.WithTool(gollym.NewTool("search", "Search", []gollym.Parameter{{Name: "q", Required: true}},
		func(context.Context, map[string]any) (any, error) {
			return core.NewSearchResults(map[string]string{"title": "Q4", "url": "https://x.test/q4", "text": "Revenue up"})
		}))
	m, err := chat.Ask(context.Background(), "How was Q4?")
	if err != nil {
		t.Fatal(err)
	}
	result := srv.Requests()[1].JSON(t)["messages"].([]any)[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	assertJSONEqual(t, "tool_result", result["content"], []any{map[string]any{
		"type": "search_result", "source": "https://x.test/q4", "title": "Q4",
		"content": []any{map[string]any{"type": "text", "text": "Revenue up"}}, "citations": map[string]any{"enabled": true},
	}})
	if len(m.Citations) != 1 || *m.Citations[0].URL != "https://x.test/q4" || m.Citations[0].SourceIndex == nil || *m.Citations[0].SourceIndex != 0 || *m.Citations[0].Text != "Revenue rose." {
		t.Fatalf("citations %+v", m.Citations)
	}
}

func TestNotConfigured(t *testing.T) {
	ctx := gollym.NewContext(func(c *gollym.Config) { c.Set("anthropic_api_key", "") })
	_, err := ctx.NewChat(gollym.ChatOptions{Model: "claude-sonnet-4-6"})
	if !errors.Is(err, gollym.ErrConfiguration) || !strings.Contains(err.Error(), "anthropic_api_key") {
		t.Fatalf("err %v", err)
	}
}

func TestDefaultAPIBaseAndNoEmbeddings(t *testing.T) {
	spec, _ := core.LookupProvider("anthropic")
	p, err := core.NewProvider(spec, testContext("", 0).Config)
	if err != nil {
		t.Fatal(err)
	}
	if base, _ := p.(interface{ APIBase() (string, error) }).APIBase(); base != "https://api.anthropic.com" {
		t.Fatalf("api base %q", base)
	}
	if _, err := core.AsFeature[core.Embedder](p, "embeddings"); !errors.Is(err, gollym.ErrUnsupported) {
		t.Fatalf("err %v", err)
	}
}
