package persistence_test

import (
	"context"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/PhotonEra-B-V/gollym"
	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
	"github.com/PhotonEra-B-V/gollym/persistence"
)

// testSchema isolates these tests inside the TEST_DATABASE_URL database.
const testSchema = "gollym_persistence_test"

var schemaOnce sync.Once

// testPool connects to TEST_DATABASE_URL (skipping without it) with a
// fresh schema of its own as the search path.
func testPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	url := os.Getenv("TEST_DATABASE_URL")
	if url == "" {
		t.Skip("TEST_DATABASE_URL not set")
	}
	ctx := context.Background()
	var setupErr error
	schemaOnce.Do(func() {
		conn, err := pgx.Connect(ctx, url)
		if err != nil {
			setupErr = err
			return
		}
		defer conn.Close(ctx)
		_, setupErr = conn.Exec(ctx, "DROP SCHEMA IF EXISTS "+testSchema+" CASCADE; CREATE SCHEMA "+testSchema)
	})
	if setupErr != nil {
		t.Fatal(setupErr)
	}
	cfg, err := pgxpool.ParseConfig(url)
	if err != nil {
		t.Fatal(err)
	}
	cfg.ConnConfig.RuntimeParams["search_path"] = testSchema
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	return pool
}

// newStore creates the tables under a per-test prefix.
func newStore(t *testing.T, pool *pgxpool.Pool) *persistence.Store {
	t.Helper()
	prefix := strings.ToLower(strings.NewReplacer("/", "_", "-", "_").Replace(t.Name())) + "_"
	store, err := persistence.New(pool, persistence.WithTablePrefix(prefix))
	if err != nil {
		t.Fatal(err)
	}
	if err := store.CreateTables(context.Background()); err != nil {
		t.Fatal(err)
	}
	return store
}

func openAIContext(base string) *core.Context {
	return gollym.NewContext(func(c *gollym.Config) {
		c.Set("openai_api_key", "test-key")
		c.Set("openai_api_base", base)
		c.MaxRetries = 0
		c.RequestTimeout = 5 * time.Second
	})
}

func completion(message map[string]any, finish string, usage map[string]any) map[string]any {
	out := map[string]any{
		"model":   "gpt-4o",
		"choices": []any{map[string]any{"message": message, "finish_reason": finish}},
	}
	if usage != nil {
		out["usage"] = usage
	}
	return out
}

// test_persistence.py::test_chat_persistence_roundtrip
func TestChatPersistenceRoundtrip(t *testing.T) {
	pool := testPool(t)
	store := newStore(t, pool)
	srv := testserver.New(t, testserver.Reply{Body: completion(
		map[string]any{"role": "assistant", "content": "Persisted reply"}, "stop",
		map[string]any{"prompt_tokens": 3, "completion_tokens": 4},
	)})
	ctx := context.Background()

	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(ctx)
	session := store.WithDB(tx)
	record := &persistence.Chat{ModelID: "gpt-4o", Provider: core.Str("openai")}
	if err := session.CreateChat(ctx, record); err != nil {
		t.Fatal(err)
	}
	chat, err := session.ToChat(record, persistence.ChatOptions{AssumeModelExists: true, Context: openAIContext(srv.URL)})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := chat.Ask(ctx, "Hello there"); err != nil {
		t.Fatal(err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	rows, err := store.Messages(ctx, record.ID)
	if err != nil {
		t.Fatal(err)
	}
	var got [][2]string
	for _, r := range rows {
		got = append(got, [2]string{r.Role, core.Deref(r.Content)})
	}
	want := [][2]string{{"user", "Hello there"}, {"assistant", "Persisted reply"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("rows %v, want %v", got, want)
	}
	if core.Deref(rows[1].OutputTokens) != 4 || core.Deref(rows[1].InputTokens) != 3 {
		t.Fatalf("tokens in=%v out=%v", rows[1].InputTokens, rows[1].OutputTokens)
	}
	if core.Deref(rows[1].ModelID) != "gpt-4o" || core.Deref(rows[1].FinishReason) != "stop" {
		t.Fatalf("model %v finish %v", rows[1].ModelID, rows[1].FinishReason)
	}
}

// A tool round trip is persisted (tool calls in their own table) and a
// loaded chat continues the conversation with its history.
func TestToolCallsPersistAndReload(t *testing.T) {
	pool := testPool(t)
	store := newStore(t, pool)
	srv := testserver.New(t,
		testserver.Reply{Body: completion(map[string]any{
			"role": "assistant", "content": nil,
			"tool_calls": []any{map[string]any{
				"id": "call_1", "type": "function",
				"function": map[string]any{"name": "weather", "arguments": `{"city":"Paris"}`},
			}},
		}, "tool_calls", nil)},
		testserver.Reply{Body: completion(map[string]any{"role": "assistant", "content": "Sunny in Paris"}, "stop", nil)},
	)
	ctx := context.Background()
	cctx := openAIContext(srv.URL)
	record := &persistence.Chat{ModelID: "gpt-4o", Provider: core.Str("openai")}
	if err := store.CreateChat(ctx, record); err != nil {
		t.Fatal(err)
	}
	chat, err := store.ToChat(record, persistence.ChatOptions{AssumeModelExists: true, Context: cctx})
	if err != nil {
		t.Fatal(err)
	}
	weather := core.NewTool("weather", "Weather for a city", []core.Parameter{{Name: "city", Required: true}},
		func(_ context.Context, args map[string]any) (any, error) {
			return "sunny in " + args["city"].(string), nil
		})
	chat.WithTool(weather)
	if _, err := chat.Ask(ctx, "Weather in Paris?"); err != nil {
		t.Fatal(err)
	}

	loaded, err := store.LoadChat(ctx, record.ID)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.ModelID != "gpt-4o" || core.Deref(loaded.Provider) != "openai" {
		t.Fatalf("chat %+v", loaded)
	}
	var roles []string
	for _, m := range loaded.Messages {
		roles = append(roles, m.Role)
	}
	if !reflect.DeepEqual(roles, []string{"user", "assistant", "tool", "assistant"}) {
		t.Fatalf("roles %v", roles)
	}
	call := loaded.Messages[1]
	if len(call.ToolCalls) != 1 || call.ToolCalls[0].ToolCallID != "call_1" || call.ToolCalls[0].Name != "weather" ||
		!reflect.DeepEqual(call.ToolCalls[0].Arguments, map[string]any{"city": "Paris"}) {
		t.Fatalf("tool calls %+v", call.ToolCalls)
	}
	if core.Deref(loaded.Messages[2].ToolCallID) != "call_1" || core.Deref(loaded.Messages[2].Content) != "sunny in Paris" {
		t.Fatalf("tool result %+v", loaded.Messages[2])
	}

	// Continue the loaded conversation: history is sent and new rows added.
	srv2 := testserver.New(t, testserver.Reply{Body: completion(map[string]any{"role": "assistant", "content": "You're welcome"}, "stop", nil)})
	again, err := store.ToChat(loaded, persistence.ChatOptions{AssumeModelExists: true, Context: openAIContext(srv2.URL)})
	if err != nil {
		t.Fatal(err)
	}
	if len(again.Messages) != 4 || again.Messages[1].ToolCalls.Get("call_1") == nil {
		t.Fatalf("history %v", again.Messages)
	}
	if _, err := again.Ask(ctx, "Thanks"); err != nil {
		t.Fatal(err)
	}
	sent := srv2.Last(t).JSON(t)["messages"].([]any)
	if len(sent) != 5 {
		t.Fatalf("sent %d messages: %v", len(sent), sent)
	}
	rows, err := store.Messages(ctx, record.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 6 || core.Deref(rows[5].Content) != "You're welcome" {
		t.Fatalf("%d rows", len(rows))
	}

	if err := store.DeleteChat(ctx, record.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := store.LoadChat(ctx, record.ID); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("after delete: %v", err)
	}
	var n int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM "+strings.ToLower(t.Name())+"_tool_calls").Scan(&n); err != nil || n != 0 {
		t.Fatalf("tool calls left %d (%v)", n, err)
	}
}

// Every column survives a write/read round trip.
func TestMessageColumnsRoundtrip(t *testing.T) {
	pool := testPool(t)
	store := newStore(t, pool)
	ctx := context.Background()
	record := &persistence.Chat{ModelID: "claude-sonnet-4-5"}
	if err := store.CreateChat(ctx, record); err != nil {
		t.Fatal(err)
	}
	msg := core.MustMessage(core.Message{
		Role:         core.RoleAssistant,
		Content:      map[string]any{"answer": "42"},
		ModelID:      "claude-sonnet-4-5",
		Tokens:       core.BuildTokens(core.Int(10), core.Int(5), core.Int(2), core.Int(1), core.Int(3)),
		Thinking:     core.BuildThinking("pondering", "sig"),
		Citations:    []core.Citation{{URL: core.Str("https://example.com"), StartIndex: core.Int(0), EndIndex: core.Int(4)}},
		FinishReason: "end_turn",
	})
	rec := persistence.MessageFromLLM(msg, record.ID)
	if err := store.InsertMessage(ctx, rec); err != nil {
		t.Fatal(err)
	}
	loaded, err := store.LoadChat(ctx, record.ID)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Provider != nil {
		t.Fatalf("provider %v, want NULL", *loaded.Provider)
	}
	back, err := loaded.Messages[0].ToLLM()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(back.Content, map[string]any{"answer": "42"}) {
		t.Fatalf("content %#v", back.Content)
	}
	if !reflect.DeepEqual(back.Tokens, msg.Tokens) || *back.Thinking != *msg.Thinking || back.FinishReason != "end_turn" {
		t.Fatalf("message %+v", back)
	}
	if len(back.Citations) != 1 || !back.Citations[0].Equal(msg.Citations[0]) {
		t.Fatalf("citations %+v", back.Citations)
	}
}

func TestMessageFromLLMAndBack(t *testing.T) {
	calls := &core.ToolCalls{}
	calls.Set("t1", &core.ToolCall{ID: "t1", Name: "lookup", Arguments: map[string]any{"q": "é"}})
	msg := core.MustMessage(core.Message{Role: core.RoleAssistant, ToolCalls: calls})
	rec := persistence.MessageFromLLM(msg, 7)
	if rec.ChatID != 7 || core.Deref(rec.Content) != "" || rec.Content == nil || rec.ModelID != nil {
		t.Fatalf("record %+v", rec)
	}
	if len(rec.ToolCalls) != 1 || rec.ToolCalls[0].ToolCallID != "t1" {
		t.Fatalf("tool calls %+v", rec.ToolCalls)
	}

	// Structured content is stored as Python json.dumps text.
	structured := persistence.MessageFromLLM(core.MustMessage(core.Message{Role: "assistant", Content: map[string]any{"a": []any{1, "é"}}}), 1)
	if got := core.Deref(structured.Content); got != `{"a": [1, "`+"\\"+`u00e9"]}` {
		t.Fatalf("content %q", got)
	}
	// Text with attachments keeps only the text.
	withFile := persistence.MessageFromLLM(&core.Message{Role: "user", Content: &core.Content{Text: core.Str("see file")}}, 1)
	if core.Deref(withFile.Content) != "see file" {
		t.Fatalf("content %v", withFile.Content)
	}

	// Non-JSON text that starts with a brace stays text; empty args become {}.
	back, err := (&persistence.Message{Role: "assistant", Content: core.Str("{not json"),
		ToolCalls: []*persistence.ToolCall{{ToolCallID: "x", Name: "n"}}}).ToLLM()
	if err != nil {
		t.Fatal(err)
	}
	if back.Content != "{not json" || !reflect.DeepEqual(back.ToolCalls.Get("x").Arguments, map[string]any{}) {
		t.Fatalf("message %+v", back)
	}
	if back.Tokens != nil || back.Thinking != nil {
		t.Fatalf("tokens %v thinking %v", back.Tokens, back.Thinking)
	}
	if _, err := (&persistence.Message{Role: "robot"}).ToLLM(); err == nil {
		t.Fatal("invalid role accepted")
	}
}

func TestSchemaPrefix(t *testing.T) {
	ddl, err := persistence.Schema("app_")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"app_chats", "app_messages", "app_tool_calls", "REFERENCES app_chats (id) ON DELETE CASCADE"} {
		if !strings.Contains(ddl, want) {
			t.Fatalf("DDL lacks %q", want)
		}
	}
	if strings.Contains(ddl, "{{prefix}}") {
		t.Fatal("placeholder left in DDL")
	}
	if _, err := persistence.New(nil, persistence.WithTablePrefix("bad; DROP")); err == nil {
		t.Fatal("invalid prefix accepted")
	}
	s, err := persistence.New(nil)
	if err != nil || !strings.Contains(s.Schema(), "llm_chats") {
		t.Fatalf("default prefix: %v", err)
	}
}
