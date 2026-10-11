package riverjobs_test

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/riverqueue/river"
	"github.com/riverqueue/river/riverdriver/riverpgxv5"
	"github.com/riverqueue/river/rivermigrate"
	"github.com/riverqueue/river/rivertype"

	"github.com/PhotonEra-B-V/gollym"
	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
	gollymriver "github.com/PhotonEra-B-V/gollym/riverjobs"
)

func tasks(t *testing.T, base string) *gollymriver.Tasks {
	t.Helper()
	ts, err := gollymriver.Register(river.NewWorkers(), gollymriver.WithContext(openAIContext(base)))
	if err != nil {
		t.Fatal(err)
	}
	return ts
}

// test_celery.py::test_task_names_and_options (queues are insert options in
// River; see TestAskJobRecordsOutput).
func TestTaskNames(t *testing.T) {
	want := []string{"gollym.ask", "gollym.embed", "gollym.paint", "gollym.speak", "gollym.transcribe", "gollym.moderate"}
	if !reflect.DeepEqual(gollymriver.Kinds(), want) {
		t.Fatalf("kinds %v", gollymriver.Kinds())
	}
	args := []river.JobArgs{gollymriver.AskArgs{}, gollymriver.EmbedArgs{}, gollymriver.PaintArgs{}, gollymriver.SpeakArgs{},
		gollymriver.TranscribeArgs{}, gollymriver.ModerateArgs{}}
	for i, a := range args {
		if a.Kind() != want[i] {
			t.Fatalf("%T kind %q", a, a.Kind())
		}
	}
	workers := river.NewWorkers()
	if _, err := gollymriver.Register(workers); err != nil {
		t.Fatal(err)
	}
	if _, err := gollymriver.Register(workers); err == nil {
		t.Fatal("registering twice should fail")
	}
}

func chatReply(content string, usage map[string]any) map[string]any {
	out := map[string]any{
		"id": "x", "model": "gpt-4o",
		"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": content}, "finish_reason": "stop"}},
	}
	if usage != nil {
		out["usage"] = usage
	}
	return out
}

// test_celery.py::test_ask_task
func TestAskTask(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: chatReply("Hi!", map[string]any{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})})
	result, err := tasks(t, srv.URL).Ask(context.Background(), gollymriver.AskArgs{Prompt: "Hello", Model: "gpt-4o", Temperature: core.Float(0.2)})
	if err != nil {
		t.Fatal(err)
	}
	if result["role"] != "assistant" || result["content"] != "Hi!" || result["input_tokens"] != 10 {
		t.Fatalf("result %v", result)
	}
	if body := srv.Last(t).JSON(t); body["temperature"] != 0.2 || body["model"] != "gpt-4o" {
		t.Fatalf("payload %v", body)
	}
}

// test_celery.py::test_ask_task_with_prior_messages
func TestAskTaskWithPriorMessages(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: chatReply("Paris", nil)})
	result, err := tasks(t, srv.URL).Ask(context.Background(), gollymriver.AskArgs{
		Prompt:       "And its capital?",
		Model:        "gpt-4o",
		Instructions: core.Str("Answer tersely."),
		Messages: []map[string]any{
			{"role": "user", "content": "Pick a country"},
			{"role": "assistant", "content": "France"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result["content"] != "Paris" {
		t.Fatalf("result %v", result)
	}
	var roles []any
	for _, m := range srv.Last(t).JSON(t)["messages"].([]any) {
		roles = append(roles, m.(map[string]any)["role"])
	}
	// system -> developer on the wire
	if !reflect.DeepEqual(roles, []any{"developer", "user", "assistant", "user"}) {
		t.Fatalf("roles %v", roles)
	}
}

// test_celery.py::test_embed_task
func TestEmbedTask(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: map[string]any{
		"model": "text-embedding-3-small",
		"data":  []any{map[string]any{"embedding": []any{0.1, 0.2}}},
		"usage": map[string]any{"prompt_tokens": 3},
	}})
	result, err := tasks(t, srv.URL).Embed(context.Background(), gollymriver.EmbedArgs{Text: "hello", Model: "text-embedding-3-small"})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result["vectors"], []float64{0.1, 0.2}) || result["input_tokens"] != 3 {
		t.Fatalf("result %v", result)
	}
	if srv.Last(t).Path != "/embeddings" {
		t.Fatalf("path %s", srv.Last(t).Path)
	}
}

// test_celery.py::test_moderate_task
func TestModerateTask(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: map[string]any{
		"id": "modr-1", "model": "omni-moderation-latest",
		"results": []any{map[string]any{"flagged": false, "categories": map[string]any{}, "category_scores": map[string]any{}}},
	}})
	result, err := tasks(t, srv.URL).Moderate(context.Background(), gollymriver.ModerateArgs{Input: "some text", Model: "omni-moderation-latest"})
	if err != nil {
		t.Fatal(err)
	}
	if result["flagged"] != false || result["model"] != "omni-moderation-latest" || result["id"] != "modr-1" {
		t.Fatalf("result %v", result)
	}
}

func TestMessageFromMap(t *testing.T) {
	m, err := gollymriver.MessageFromMap(map[string]any{
		"role": "assistant", "content": "", "model_id": "gpt-4o", "input_tokens": 3.0, "finish_reason": "tool_calls",
		"tool_calls":       map[string]any{"c1": map[string]any{"id": "c1", "name": "f", "arguments": `{"x":1}`}},
		"thinking":         "hmm",
		"citations":        []any{map[string]any{"url": "https://a.example"}},
		"thinking_tokens":  nil,
		"unknown_ignored":  true,
		"tool_call_id":     nil,
		"cache_read_extra": 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if m.ModelID != "gpt-4o" || *m.InputTokens() != 3 || m.Thinking.Text != "hmm" || core.Deref(m.Citations[0].URL) != "https://a.example" {
		t.Fatalf("message %+v", m)
	}
	if !reflect.DeepEqual(m.ToolCalls.Get("c1").Arguments, map[string]any{"x": 1.0}) {
		t.Fatalf("tool call %+v", m.ToolCalls.Get("c1"))
	}
	back := gollymriver.MessageMap(m)
	if back["thinking"] != "hmm" || back["input_tokens"] != 3 || back["finish_reason"] != "tool_calls" {
		t.Fatalf("map %v", back)
	}
	if _, err := gollymriver.MessageFromMap(map[string]any{"content": "x"}); err == nil {
		t.Fatal("missing role accepted")
	}
	if _, err := gollymriver.MessageFromMap(map[string]any{"role": "robot"}); err == nil {
		t.Fatal("invalid role accepted")
	}
}

// --- River end to end (needs TEST_DATABASE_URL) -------------------------------

const riverSchema = "gollym_river_test"

func riverPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	url := os.Getenv("TEST_DATABASE_URL")
	if url == "" {
		t.Skip("TEST_DATABASE_URL not set")
	}
	ctx := context.Background()
	conn, err := pgx.Connect(ctx, url)
	if err != nil {
		t.Fatal(err)
	}
	_, err = conn.Exec(ctx, "CREATE SCHEMA IF NOT EXISTS "+riverSchema)
	conn.Close(ctx)
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := pgxpool.ParseConfig(url)
	if err != nil {
		t.Fatal(err)
	}
	cfg.ConnConfig.RuntimeParams["search_path"] = riverSchema
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	migrator, err := rivermigrate.New(riverpgxv5.New(pool), &rivermigrate.Config{Schema: riverSchema})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := migrator.Migrate(ctx, rivermigrate.DirectionUp, nil); err != nil {
		t.Fatal(err)
	}
	return pool
}

// A River client works an inserted ask job and records the task result as
// the job output; insert options (queue) play Celery's task options' role.
func TestAskJobRecordsOutput(t *testing.T) {
	pool := riverPool(t)
	srv := testserver.New(t, testserver.Reply{Body: chatReply("Hi!", map[string]any{"prompt_tokens": 10, "completion_tokens": 5})})
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	workers := river.NewWorkers()
	if _, err := gollymriver.Register(workers, gollymriver.WithContext(openAIContext(srv.URL))); err != nil {
		t.Fatal(err)
	}
	client, err := river.NewClient(riverpgxv5.New(pool), &river.Config{
		Schema:            riverSchema,
		Queues:            map[string]river.QueueConfig{"llm": {MaxWorkers: 2}},
		Workers:           workers,
		FetchPollInterval: 100 * time.Millisecond,
		TestOnly:          true,
	})
	if err != nil {
		t.Fatal(err)
	}
	events, stop := client.Subscribe(river.EventKindJobCompleted, river.EventKindJobFailed)
	defer stop()
	if err := client.Start(ctx); err != nil {
		t.Fatal(err)
	}
	defer client.Stop(context.Background())

	inserted, err := client.Insert(ctx, gollymriver.AskArgs{Prompt: "Hello", Model: "gpt-4o"}, &river.InsertOpts{Queue: "llm"})
	if err != nil {
		t.Fatal(err)
	}
	if inserted.Job.Kind != "gollym.ask" || inserted.Job.Queue != "llm" {
		t.Fatalf("kind %s queue %s", inserted.Job.Kind, inserted.Job.Queue)
	}
	var ev *river.Event
	select {
	case ev = <-events:
	case <-ctx.Done():
		t.Fatal("job not worked in time")
	}
	if ev.Kind != river.EventKindJobCompleted || ev.Job.ID != inserted.Job.ID {
		t.Fatalf("event %s for job %d: %v", ev.Kind, ev.Job.ID, ev.Job.Errors)
	}
	job, err := client.JobGet(ctx, inserted.Job.ID)
	if err != nil {
		t.Fatal(err)
	}
	var meta map[string]json.RawMessage
	if err := json.Unmarshal(job.Metadata, &meta); err != nil {
		t.Fatal(err)
	}
	var output map[string]any
	if err := json.Unmarshal(meta[rivertype.MetadataKeyOutput], &output); err != nil {
		t.Fatalf("output %s: %v", meta[rivertype.MetadataKeyOutput], err)
	}
	if output["content"] != "Hi!" || output["role"] != "assistant" || output["input_tokens"] != 10.0 {
		t.Fatalf("output %v", output)
	}
	if _, err := client.JobDelete(ctx, job.ID); err != nil {
		t.Fatal(err)
	}
}

func openAIContext(base string) *core.Context {
	return gollym.NewContext(func(c *gollym.Config) {
		c.Set("openai_api_key", "test-key")
		c.Set("openai_api_base", base)
		c.MaxRetries = 0
		c.RequestTimeout = 5 * time.Second
	})
}
