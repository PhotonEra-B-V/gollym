package registry_test

import (
	"bytes"
	"context"
	"reflect"
	"testing"

	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
	"github.com/PhotonEra-B-V/gollym/registry"
)

// catalog is a trimmed models.dev api.json.
func catalog() map[string]any {
	return map[string]any{
		"anthropic": map[string]any{"id": "anthropic", "models": map[string]any{
			"claude-x": map[string]any{
				"id": "claude-x", "name": "Claude X", "family": "claude", "attachment": true,
				"reasoning": true, "tool_call": true, "structured_output": true, "temperature": false,
				"knowledge": "2026-06", "release_date": "2026-09-22", "last_updated": "2026-09-30",
				"modalities":   map[string]any{"input": []any{"text", "image", "pdf"}, "output": []any{"text"}},
				"open_weights": false, "limit": map[string]any{"context": 1000000, "output": 128000},
				"cost":        map[string]any{"input": 4, "output": 20, "cache_read": 0.2, "cache_write": 5},
				"description": "not copied",
			},
			"claude-new": map[string]any{
				"id": "claude-new", "name": "Claude New", "tool_call": true, "release_date": "2026-10-01",
				"modalities": map[string]any{"input": []any{"text"}, "output": []any{"text"}},
				"limit":      map[string]any{"context": 0, "output": 64000},
				"cost":       map[string]any{"input": 0, "output": 1, "input_audio": 3},
			},
		}},
		"openai": map[string]any{"models": map[string]any{
			"text-embedding-9": map[string]any{
				"id": "text-embedding-9", "name": "Embedding 9",
				"modalities": map[string]any{"input": []any{"text"}, "output": []any{"text"}},
				"limit":      map[string]any{"context": 8192, "output": 3072},
			},
		}},
		"google": map[string]any{"models": map[string]any{
			"gemini-z": map[string]any{
				"id": "gemini-z", "name": "Gemini Z",
				"modalities": map[string]any{"input": []any{"text"}, "output": []any{"text"}},
				"limit":      map[string]any{"context": 2000, "output": 100},
			},
		}},
		"amazon-bedrock": map[string]any{"models": map[string]any{
			"anthropic.claude-x": map[string]any{
				"id": "anthropic.claude-x", "name": "Claude X (Bedrock)",
				"modalities": map[string]any{"input": []any{"text"}, "output": []any{"text"}},
				"limit":      map[string]any{"context": 500, "output": 50},
			},
		}},
		"groq": map[string]any{"models": map[string]any{ // unmapped provider
			"llama": map[string]any{"id": "llama", "name": "Llama"},
		}},
	}
}

func fetchCatalog(t *testing.T) registry.ModelsDevCatalog {
	t.Helper()
	srv := testserver.New(t, testserver.Reply{Body: catalog()})
	cat, err := registry.FetchModelsDev(context.Background(), nil, srv.URL+"/api.json")
	if err != nil {
		t.Fatal(err)
	}
	if srv.Last(t).Path != "/api.json" {
		t.Fatalf("fetched %s", srv.Last(t).Path)
	}
	return cat
}

func find(t *testing.T, models []*core.Info, provider, id string) *core.Info {
	t.Helper()
	for _, m := range models {
		if m.Provider == provider && m.ID == id {
			return m
		}
	}
	t.Fatalf("%s:%s missing from %v", provider, id, ids(models))
	return nil
}

func TestFetchModelsDevErrors(t *testing.T) {
	down := testserver.New(t, testserver.Reply{Status: 503, Body: "down"})
	if _, err := registry.FetchModelsDev(context.Background(), nil, down.URL); err == nil {
		t.Fatal("503 should fail")
	}
	empty := testserver.New(t, testserver.Reply{Body: map[string]any{}})
	if _, err := registry.FetchModelsDev(context.Background(), nil, empty.URL); err == nil {
		t.Fatal("an empty catalogue should fail rather than look like no changes")
	}
}

func TestMergeModelsDev(t *testing.T) {
	cat := fetchCatalog(t)
	existing := []*core.Info{
		// Sourced from models.dev earlier: refreshed, keeps created_at, batch
		// tier, image pricing and its own capabilities and metadata.
		core.NewInfo(map[string]any{
			"id": "claude-x", "name": "Old", "provider": "anthropic", "created_at": "2026-09-01 00:00:00 UTC",
			"context_window": 200000, "capabilities": []any{"function_calling", "citations"},
			"pricing": map[string]any{
				"text_tokens": map[string]any{"standard": map[string]any{"input_per_million": 5}, "batch": map[string]any{"input_per_million": 2.5}},
				"images":      map[string]any{"standard": map[string]any{"input_per_million": 1}},
			},
			"metadata": map[string]any{"source": "models.dev", "status": "deprecated", "peak_cost": map[string]any{"input": 8}},
		}),
		// Provider listing (no source): enriched.
		core.NewInfo(map[string]any{"id": "text-embedding-9", "provider": "openai", "metadata": map[string]any{"owned_by": "openai"}}),
		// Hand-curated: untouched although models.dev knows it.
		core.NewInfo(map[string]any{"id": "gemini-z", "name": "Curated", "provider": "gemini", "metadata": map[string]any{"source": "manual"}}),
		// Vertex falls back to the Gemini catalogue.
		core.NewInfo(map[string]any{"id": "gemini-z", "provider": "vertexai"}),
		// Bedrock matches without its region prefix and keeps its id.
		core.NewInfo(map[string]any{"id": "eu.anthropic.claude-x", "provider": "bedrock"}),
		// Gone from models.dev: kept as is.
		core.NewInfo(map[string]any{"id": "claude-old", "name": "Old one", "provider": "anthropic", "metadata": map[string]any{"source": "models.dev"}}),
	}
	overrides := map[string]registry.Override{
		"anthropic:claude-x":   {Set: map[string]any{"context_window": 1234}},
		"anthropic:claude-new": {Set: map[string]any{"name": "Overridden"}},
	}
	models, report := registry.MergeModelsDev(existing, cat, overrides)

	// claude-new and anthropic.claude-x (bedrock) are new; groq is unmapped
	// and Vertex adds nothing from the Gemini catalogue.
	if report.Added != 2 || report.Skipped != 1 || report.Updated != 4 || report.Unchanged != 2 {
		t.Fatalf("report %+v", report)
	}
	want := []string{
		"anthropic:claude-new", "anthropic:claude-old", "anthropic:claude-x",
		"bedrock:anthropic.claude-x", "bedrock:eu.anthropic.claude-x",
		"gemini:gemini-z", "openai:text-embedding-9", "vertexai:gemini-z",
	}
	if !reflect.DeepEqual(ids(models), want) {
		t.Fatalf("merged %v", ids(models))
	}

	x := find(t, models, "anthropic", "claude-x")
	if x.Name != "Claude X" || x.Family != "claude" || *x.ContextWindow != 1234 || *x.MaxOutputTokens != 128000 {
		t.Fatalf("claude-x fields: %+v", x)
	}
	if x.CreatedAt.Format("2006-01-02") != "2026-09-01" || x.KnowledgeCutoff.Format("2006-01") != "2026-06" {
		t.Fatalf("claude-x dates: %v %v", x.CreatedAt, x.KnowledgeCutoff)
	}
	if !reflect.DeepEqual(x.Capabilities, []string{"function_calling", "structured_output", "reasoning", "vision", "citations"}) {
		t.Fatalf("claude-x caps %v", x.Capabilities)
	}
	text := x.Pricing.TextTokens()
	if *text.Input() != 4 || *text.CacheReadInput() != 0.2 || *text.CacheWriteInput() != 5 || *text.Batch.InputPerMillion() != 2.5 {
		t.Fatalf("claude-x text pricing %+v", text)
	}
	if *x.Pricing.Images().Input() != 1 {
		t.Fatal("claude-x lost image pricing")
	}
	if x.Metadata["source"] != "models.dev" || x.Metadata["provider_id"] != "anthropic" || x.Metadata["status"] != nil ||
		x.Metadata["peak_cost"] == nil || x.Metadata["description"] != nil || x.Metadata["last_updated"] != "2026-09-30" {
		t.Fatalf("claude-x metadata %v", x.Metadata)
	}

	emb := find(t, models, "openai", "text-embedding-9")
	if emb.Type() != "embedding" || emb.Metadata["owned_by"] != "openai" || emb.Metadata["source"] != "models.dev" {
		t.Fatalf("embedding %v %v", emb.Modalities, emb.Metadata)
	}
	if find(t, models, "gemini", "gemini-z") != existing[2] {
		t.Fatal("hand-curated entry changed")
	}
	if v := find(t, models, "vertexai", "gemini-z"); v.Name != "Gemini Z" || v.Metadata["provider_id"] != "google" {
		t.Fatalf("vertex %+v", v)
	}
	if b := find(t, models, "bedrock", "eu.anthropic.claude-x"); b.Name != "Claude X (Bedrock)" || *b.ContextWindow != 500 {
		t.Fatalf("bedrock %+v", b)
	}
	if find(t, models, "anthropic", "claude-old") != existing[5] {
		t.Fatal("entry missing upstream changed")
	}

	nw := find(t, models, "anthropic", "claude-new")
	if nw.Name != "Overridden" || nw.ContextWindow != nil || *nw.MaxOutputTokens != 64000 {
		t.Fatalf("claude-new %+v", nw)
	}
	if nw.Pricing.TextTokens().Input() != nil || *nw.Pricing.TextTokens().Output() != 1 || *nw.Pricing.AudioTokens().Input() != 3 {
		t.Fatalf("claude-new pricing %v", nw.Pricing.Map())
	}

	// A second merge over the result changes nothing.
	again, r2 := registry.MergeModelsDev(models, cat, overrides)
	if r2.Added != 0 || r2.Updated != 0 || len(again) != len(models) {
		t.Fatalf("not idempotent: %+v", r2)
	}
}

func TestOverridesParse(t *testing.T) {
	o, err := registry.Overrides()
	if err != nil {
		t.Fatal(err)
	}
	for key, v := range o {
		if v.Reason == "" || len(v.Set) == 0 {
			t.Errorf("%s: an override needs a reason and values", key)
		}
	}
}

// created_at keeps the registry's "... UTC" form through a save.
func TestMarshalCreatedAtFormat(t *testing.T) {
	m := core.NewInfo(map[string]any{"id": "a", "provider": "p", "created_at": "2024-10-22 00:00:00 UTC"})
	data, err := registry.Marshal([]*core.Info{m})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(data, []byte(`"created_at": "2024-10-22 00:00:00 UTC"`)) {
		t.Fatalf("created_at not kept: %s", data)
	}
}
