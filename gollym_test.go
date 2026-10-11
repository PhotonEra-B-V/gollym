package gollym_test

import (
	"context"
	"reflect"
	"slices"
	"testing"

	"github.com/PhotonEra-B-V/gollym"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
)

// Ports tests/test_capabilities.py: the facade's one-shot calls against an
// OpenAI stub.

func capCtx(t *testing.T, srv *testserver.Server) *gollym.Context {
	t.Helper()
	t.Setenv("OPENAI_API_KEY", "")
	return gollym.NewContext(func(c *gollym.Config) {
		c.Set("openai_api_key", "sk-test")
		c.Set("openai_api_base", srv.URL+"/v1")
	})
}

func openaiEmbedding(vectors [][]float64, tokens int) map[string]any {
	data := make([]any, len(vectors))
	for i, v := range vectors {
		data[i] = map[string]any{"embedding": v, "index": i}
	}
	return map[string]any{"data": data, "model": "text-embedding-3-small", "usage": map[string]any{"prompt_tokens": tokens}}
}

func TestEmbeddingSingleVector(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: openaiEmbedding([][]float64{{0.1, 0.2, 0.3}}, 4)})
	emb, err := gollym.Embed(context.Background(), "hello", gollym.EmbedOptions{ModelOptions: gollym.ModelOptions{Model: "text-embedding-3-small", Context: capCtx(t, srv)}})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(emb.Vectors, []float64{0.1, 0.2, 0.3}) || emb.InputTokens != 4 { // unwrapped for a single input
		t.Fatalf("emb %+v", emb)
	}
	req := srv.Last(t)
	if req.Path != "/v1/embeddings" || !reflect.DeepEqual(req.JSON(t), map[string]any{"model": "text-embedding-3-small", "input": "hello"}) {
		t.Fatalf("%s %s", req.Path, req.Body)
	}
}

func TestEmbeddingBatchVectors(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: openaiEmbedding([][]float64{{0.1}, {0.2}}, 3)})
	emb, err := gollym.Embed(context.Background(), []string{"a", "b"}, gollym.EmbedOptions{ModelOptions: gollym.ModelOptions{Model: "text-embedding-3-small", Context: capCtx(t, srv)}})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(emb.Vectors, [][]float64{{0.1}, {0.2}}) { // list input stays nested
		t.Fatalf("vectors %v", emb.Vectors)
	}
}

func TestModerationFlagged(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: map[string]any{
		"id": "modr-test", "model": "omni-moderation-latest",
		"results": []any{map[string]any{
			"flagged": true, "categories": map[string]any{"hate": true, "violence": false},
			"category_scores": map[string]any{"hate": 0.9},
		}},
	}})
	result, err := gollym.Moderate(context.Background(), "something", gollym.ModelOptions{Model: "omni-moderation-latest", Context: capCtx(t, srv)})
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsFlagged() || !slices.Contains(result.FlaggedCategories(), "hate") || slices.Contains(result.FlaggedCategories(), "violence") {
		t.Fatalf("result %+v", result)
	}
	if srv.Last(t).Path != "/v1/moderations" {
		t.Fatalf("path %s", srv.Last(t).Path)
	}
}

func TestImageGeneration(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: map[string]any{
		"created": 0, "data": []any{map[string]any{"revised_prompt": "a cat", "url": "https://cdn/cat.png"}}, "usage": map[string]any{"total_tokens": 0},
	}})
	image, err := gollym.Paint(context.Background(), "a cat", gollym.PaintRequest{ModelOptions: gollym.ModelOptions{Model: "gpt-image-1.5", Context: capCtx(t, srv)}})
	if err != nil {
		t.Fatal(err)
	}
	if image.URL != "https://cdn/cat.png" || image.RevisedPrompt != "a cat" {
		t.Fatalf("image %+v", image)
	}
	req := srv.Last(t)
	want := map[string]any{"model": "gpt-image-1.5", "prompt": "a cat", "n": 1.0, "size": "1024x1024"}
	if req.Path != "/v1/images/generations" || !reflect.DeepEqual(req.JSON(t), want) {
		t.Fatalf("%s %s", req.Path, req.Body)
	}
}

// The seeded openai_chat fixture drives the full parse stack.
func TestChatFromSeededFixture(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: map[string]any{
		"id": "cmpl-test", "model": "gpt-4o",
		"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}},
		"usage":   map[string]any{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
	}})
	chat, err := capCtx(t, srv).NewChat(gollym.ChatOptions{Model: "gpt-4o"})
	if err != nil {
		t.Fatal(err)
	}
	msg, err := chat.Ask(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	if msg.Content != "Hello!" || *msg.InputTokens() != 10 {
		t.Fatalf("msg %+v", msg)
	}
}
