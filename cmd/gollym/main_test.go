package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
	"github.com/PhotonEra-B-V/gollym/registry"
)

func TestModelsList(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run(context.Background(), []string{"models", "list", "--provider", "anthropic"}, &out, &errOut); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) < 2 || !strings.HasPrefix(lines[0], "PROVIDER") {
		t.Fatalf("output %q", out.String())
	}
	for _, l := range lines[1:] {
		if !strings.HasPrefix(l, "anthropic ") {
			t.Fatalf("non-anthropic row %q", l)
		}
	}
}

func TestModelsRefreshWritesFile(t *testing.T) {
	refreshProviders = []*core.ProviderSpec{} // no network: nothing to fetch
	t.Cleanup(func() { refreshProviders = nil })
	dir := t.TempDir()
	out := filepath.Join(dir, "models.json")
	var stdout, stderr bytes.Buffer
	if code := run(context.Background(), []string{"models", "refresh", "--remote-only", "--no-models-dev", "--out", out}, &stdout, &stderr); code != 0 {
		t.Fatalf("exit %d: %s", code, stderr.String())
	}
	if core.LoadRegistry(out).Len() != core.LoadRegistry("").Len() {
		t.Fatalf("saved registry differs in size")
	}
	if !strings.Contains(stdout.String(), "saved ") {
		t.Fatalf("stdout %q", stdout.String())
	}
	// --registry without --out rewrites the same file.
	if code := run(context.Background(), []string{"models", "refresh", "--no-models-dev", "--registry", out}, &stdout, &stderr); code != 0 {
		t.Fatalf("exit %d: %s", code, stderr.String())
	}
	if _, err := os.Stat(filepath.Join(dir, "models.json")); err != nil {
		t.Fatal(err)
	}
}

func TestUnknownCommand(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run(context.Background(), []string{"nope"}, &out, &errOut); code != 2 {
		t.Fatalf("exit %d", code)
	}
	if code := run(context.Background(), []string{"models", "nope"}, &out, &errOut); code != 1 {
		t.Fatalf("exit %d", code)
	}
}

func TestModelsRefreshMergesModelsDev(t *testing.T) {
	refreshProviders = []*core.ProviderSpec{}
	dev := testserver.New(t, testserver.Reply{Body: map[string]any{"anthropic": map[string]any{"models": map[string]any{
		"claude-brand-new": map[string]any{"id": "claude-brand-new", "name": "Claude Brand New", "limit": map[string]any{"context": 1000}},
	}}}})
	modelsDevURL = dev.URL
	t.Cleanup(func() { refreshProviders = nil; modelsDevURL = registry.ModelsDevURL })
	out := filepath.Join(t.TempDir(), "models.json")
	var stdout, stderr bytes.Buffer
	if code := run(context.Background(), []string{"models", "refresh", "--remote-only", "--out", out}, &stdout, &stderr); code != 0 {
		t.Fatalf("exit %d: %s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "models.dev: 1 added") {
		t.Fatalf("stderr %q", stderr.String())
	}
	if m, err := core.LoadRegistry(out).Find("claude-brand-new", "anthropic"); err != nil || *m.ContextWindow != 1000 {
		t.Fatalf("new model: %+v %v", m, err)
	}
}
