package registry_test

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym"
	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
	"github.com/PhotonEra-B-V/gollym/registry"
)

func info(provider, id, name string) *core.Info {
	return core.NewInfo(map[string]any{"id": id, "name": name, "provider": provider})
}

func ids(models []*core.Info) []string {
	out := make([]string, len(models))
	for i, m := range models {
		out[i] = m.Provider + ":" + m.ID
	}
	return out
}

// Provider listings replace that provider's entries; others (not asked, or
// failed) are kept; the result is sorted by provider then id.
func TestFetchMergedProviderEntriesWin(t *testing.T) {
	listing := testserver.New(t, testserver.Reply{Body: map[string]any{"data": []any{
		map[string]any{"id": "gpt-new", "object": "model", "created": 1700000000, "owned_by": "openai"},
		map[string]any{"id": "gpt-4o", "object": "model", "created": 1700000000, "owned_by": "openai"},
	}}})
	broken := testserver.New(t, testserver.Reply{Status: 500, Body: "down"})
	cfg := gollym.NewContext(func(c *gollym.Config) {
		c.Set("openai_api_key", "k")
		c.Set("openai_api_base", listing.URL)
		c.MaxRetries = 0
		c.RequestTimeout = 5 * time.Second
	}).Config
	openai, ok := core.LookupProvider("openai")
	if !ok {
		t.Fatal("openai not registered")
	}
	brokenSpec := *openai
	brokenSpec.Slug = "brokenai"
	brokenSpec.APIBase = func(core.Provider) (string, error) { return broken.URL, nil }

	existing := []*core.Info{
		info("openai", "gpt-old", "Old"),
		info("openai", "gpt-4o", "Stale name"),
		info("brokenai", "keep-me", "Kept"),
		info("anthropic", "claude-x", "Claude X"),
	}
	models, report := registry.FetchMerged(context.Background(), existing, registry.Options{
		Config: cfg, Providers: []*core.ProviderSpec{openai, &brokenSpec},
	})
	want := []string{"anthropic:claude-x", "brokenai:keep-me", "openai:gpt-4o", "openai:gpt-new"}
	if !reflect.DeepEqual(ids(models), want) {
		t.Fatalf("merged %v, want %v", ids(models), want)
	}
	if models[2].Name != "gpt-4o" || models[2].Metadata["owned_by"] != "openai" {
		t.Fatalf("provider entry did not win: %+v", models[2])
	}
	if !reflect.DeepEqual(report.Fetched, []string{"openai"}) || report.Failed["brokenai"] == nil {
		t.Fatalf("report %+v", report)
	}
	if listing.Last(t).Path != "/models" {
		t.Fatalf("listed %s", listing.Last(t).Path)
	}

	reg := core.NewRegistry(existing)
	registry.Refresh(context.Background(), reg, registry.Options{Config: cfg, Providers: []*core.ProviderSpec{openai}})
	if !reflect.DeepEqual(ids(reg.All()), want) {
		t.Fatalf("refreshed %v", ids(reg.All()))
	}
}

// An empty registry merges into the embedded models.json (pyllym's
// `instance().all() or _read_from_json()`).
func TestFetchMergedEmptyUsesEmbedded(t *testing.T) {
	models, report := registry.FetchMerged(context.Background(), nil, registry.Options{Providers: []*core.ProviderSpec{}})
	if len(models) != core.LoadRegistry("").Len() || len(report.Fetched) != 0 {
		t.Fatalf("%d models", len(models))
	}
	for i := 1; i < len(models); i++ {
		a, b := models[i-1], models[i]
		if a.Provider > b.Provider || (a.Provider == b.Provider && a.ID > b.ID) {
			t.Fatalf("not sorted at %d: %s:%s > %s:%s", i, a.Provider, a.ID, b.Provider, b.ID)
		}
	}
}

// Saving and reloading the embedded registry is lossless and stable.
func TestSaveRoundtrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "models.json")
	reg := core.LoadRegistry("")
	if err := registry.Save(reg, path); err != nil {
		t.Fatal(err)
	}
	first, _ := os.ReadFile(path)
	if !bytes.HasPrefix(first, []byte("[\n  {\n    \"id\": ")) {
		t.Fatalf("unexpected layout: %.40q", first)
	}
	reloaded := core.LoadRegistry(path)
	if reloaded.Len() != reg.Len() {
		t.Fatalf("reloaded %d of %d", reloaded.Len(), reg.Len())
	}
	second, err := registry.Marshal(reloaded.All())
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("save -> load -> save is not stable")
	}
	m, err := reloaded.Find("gpt-4o", "openai")
	if err != nil || m.ContextWindow == nil || m.InputPricePerMillion() == nil {
		t.Fatalf("gpt-4o after reload: %+v %v", m, err)
	}
	if empty, _ := registry.Marshal(nil); string(empty) != "[]" {
		t.Fatalf("empty %q", empty)
	}
}
