// Package registry refreshes the model registry from the providers' model
// listings and saves it as models.json (pyllym's models.refresh and
// models.save_to_json).
package registry

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"sort"
	"sync"

	"github.com/PhotonEra-B-V/gollym/core"
)

// Options configure FetchMerged.
type Options struct {
	// Config supplies provider credentials (default: core.Global()).
	Config *core.Config
	// RemoteOnly skips local providers (Ollama, GPUStack, ...).
	RemoteOnly bool
	// Providers overrides the providers to ask (default: every configured
	// provider, in registration order).
	Providers []*core.ProviderSpec
}

// Report says which providers were listed and which failed (and so kept
// their existing entries).
type Report struct {
	Fetched []string
	Failed  map[string]error
}

// FetchMerged lists the models of every configured provider and merges them
// into existing: a provider that answered replaces all of its entries,
// providers that weren't asked or failed keep theirs. The result is sorted
// by provider, then id. An empty existing list falls back to the embedded
// registry.
func FetchMerged(ctx context.Context, existing []*core.Info, opts Options) ([]*core.Info, Report) {
	cfg := opts.Config
	if cfg == nil {
		cfg = core.Global()
	}
	if len(existing) == 0 {
		existing = core.LoadRegistry("").All()
	}
	specs := opts.Providers
	if specs == nil {
		specs = core.ConfiguredProviders(cfg, opts.RemoteOnly)
	}

	// Providers are listed concurrently; results are merged in order.
	results := make([][]*core.Info, len(specs))
	errs := make([]error, len(specs))
	var wg sync.WaitGroup
	for i, spec := range specs {
		wg.Go(func() {
			p, err := core.NewProvider(spec, cfg)
			if err == nil {
				results[i], err = core.ListModels(ctx, p)
			}
			errs[i] = err
		})
	}
	wg.Wait()

	report := Report{Failed: map[string]error{}}
	fetchedSlugs := map[string]bool{}
	var fetched []*core.Info
	for i, spec := range specs {
		if errs[i] != nil {
			report.Failed[spec.Slug] = errs[i]
			continue
		}
		fetched = append(fetched, results[i]...)
		fetchedSlugs[spec.Slug] = true
		report.Fetched = append(report.Fetched, spec.Slug)
	}

	merged := map[string]*core.Info{}
	for _, m := range existing {
		if !fetchedSlugs[m.Provider] {
			merged[m.Provider+":"+m.ID] = m
		}
	}
	for _, m := range fetched {
		merged[m.Provider+":"+m.ID] = m
	}
	out := make([]*core.Info, 0, len(merged))
	for _, m := range merged {
		out = append(out, m)
	}
	sort.Slice(out, func(a, b int) bool {
		if out[a].Provider != out[b].Provider {
			return out[a].Provider < out[b].Provider
		}
		return out[a].ID < out[b].ID
	})
	return out, report
}

// Refresh replaces reg's models with FetchMerged's result.
func Refresh(ctx context.Context, reg *core.Registry, opts Options) Report {
	models, report := FetchMerged(ctx, reg.All(), opts)
	reg.Replace(models)
	return report
}

// infoKeys is Info.to_dict's key order, kept so saved files diff cleanly
// against pyllym's.
var infoKeys = []string{
	"id", "name", "provider", "family", "created_at", "context_window", "max_output_tokens",
	"knowledge_cutoff", "modalities", "capabilities", "pricing", "metadata",
}

// Marshal renders models as models.json: a list of to_dict records,
// indented by two spaces. Nested objects have sorted keys, so the output
// is JSON-equal (not byte-equal) to pyllym's save_to_json.
func Marshal(models []*core.Info) ([]byte, error) {
	var b bytes.Buffer
	b.WriteString("[")
	for i, m := range models {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString("\n  {")
		d := m.Map()
		for k, key := range infoKeys {
			v, err := marshalIndent(d[key], "    ")
			if err != nil {
				return nil, err
			}
			if k > 0 {
				b.WriteString(",")
			}
			b.WriteString("\n    ")
			kb, _ := json.Marshal(key)
			b.Write(kb)
			b.WriteString(": ")
			b.Write(v)
		}
		b.WriteString("\n  }")
	}
	if len(models) > 0 {
		b.WriteString("\n")
	}
	b.WriteString("]")
	return b.Bytes(), nil
}

func marshalIndent(v any, prefix string) ([]byte, error) {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	enc.SetIndent(prefix, "  ")
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(b.Bytes(), []byte("\n")), nil
}

// Save writes reg to path as models.json.
func Save(reg *core.Registry, path string) error {
	data, err := Marshal(reg.All())
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}
