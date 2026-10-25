package registry

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/PhotonEra-B-V/gollym/core"
)

// ModelsDevURL is the models.dev catalogue: every provider's models with
// limits, modalities, capabilities and prices, community-maintained at
// github.com/sst/models.dev. pyllym's registry was seeded from it but its
// refresh never fetches it; gollym's does (see docs/PLAN.md).
const ModelsDevURL = "https://models.dev/api.json"

// modelsDevProviders maps models.dev provider keys to gollym slugs. Only
// these are merged; the order is the lookup order for a slug.
var modelsDevProviders = []struct{ key, slug string }{
	{"openai", "openai"},
	{"anthropic", "anthropic"},
	{"google", "gemini"},
	{"google-vertex", "vertexai"},
	{"google", "vertexai"}, // Vertex serves the Gemini catalogue too
	{"amazon-bedrock", "bedrock"},
	{"deepseek", "deepseek"},
	{"mistral", "mistral"},
	{"openrouter", "openrouter"},
	{"perplexity", "perplexity"},
	{"xai", "xai"},
}

// modelsDevMetadata are the metadata keys a models.dev record owns: a merge
// sets them from the record or removes them, so stale values (an old
// "status": "deprecated") don't linger. Other metadata keys are kept.
var modelsDevMetadata = []string{
	"source", "provider_id", "open_weights", "attachment", "temperature", "last_updated",
	"status", "interleaved", "cost", "limit", "knowledge", "reasoning_options",
}

// bedrockRegion is a cross-region inference profile prefix ("eu.", "global.").
var bedrockRegion = regexp.MustCompile(`^(?:[a-z]{2}|apac|global)\.`)

// ModelsDevCatalog is models.dev's api.json: provider key -> {"models": {id: record}}.
type ModelsDevCatalog map[string]struct {
	Models map[string]map[string]any `json:"models"`
}

// FetchModelsDev downloads the catalogue from url (ModelsDevURL when empty).
func FetchModelsDev(ctx context.Context, client *http.Client, url string) (ModelsDevCatalog, error) {
	if url == "" {
		url = ModelsDevURL
	}
	if client == nil {
		client = &http.Client{Timeout: time.Minute}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("models.dev: HTTP %d", resp.StatusCode)
	}
	var cat ModelsDevCatalog
	if err := json.Unmarshal(body, &cat); err != nil {
		return nil, fmt.Errorf("models.dev: %w", err)
	}
	if len(cat) == 0 {
		return nil, fmt.Errorf("models.dev: empty catalogue")
	}
	return cat, nil
}

// lookup finds id for slug, trying each models.dev provider that feeds it
// and, for Bedrock, the id without its region prefix.
func (c ModelsDevCatalog) lookup(slug, id string) (map[string]any, string) {
	for _, p := range modelsDevProviders {
		if p.slug != slug {
			continue
		}
		models := c[p.key].Models
		if rec := models[id]; rec != nil {
			return rec, p.key
		}
		if slug == "bedrock" {
			if rec := models[bedrockRegion.ReplaceAllString(id, "")]; rec != nil {
				return rec, p.key
			}
		}
	}
	return nil, ""
}

// ModelsDevRecord converts a models.dev model into a registry record
// (Info.to_dict's shape) for provider slug.
func ModelsDevRecord(rec map[string]any, slug, providerKey string) map[string]any {
	id, _ := rec["id"].(string)
	name, _ := rec["name"].(string)
	if name == "" {
		name = id
	}
	limit, _ := rec["limit"].(map[string]any)
	modalities, _ := rec["modalities"].(map[string]any)
	input := core.ToSafeArray(modalities["input"])
	// models.dev says embedding models output text; the registry types them
	// by an "embeddings" output (RubyLLM's normalize_embedding_modalities).
	if strings.Contains(strings.ToLower(id), "embed") {
		modalities = map[string]any{"input": modalities["input"], "output": []any{"embeddings"}}
	}

	var caps []any
	if core.Truthy(rec["tool_call"]) {
		caps = append(caps, "function_calling")
	}
	if core.Truthy(rec["structured_output"]) {
		caps = append(caps, "structured_output")
	}
	if core.Truthy(rec["reasoning"]) {
		caps = append(caps, "reasoning")
	}
	if slices.ContainsFunc(input, func(m any) bool { return m == "image" || m == "video" || m == "pdf" }) {
		caps = append(caps, "vision")
	}

	pricing := map[string]any{}
	if cost, ok := rec["cost"].(map[string]any); ok {
		text := tier(cost, map[string]string{
			"input": "input_per_million", "output": "output_per_million",
			"cache_read": "cache_read_input_per_million", "cache_write": "cache_write_input_per_million",
			"reasoning": "reasoning_output_per_million",
		})
		if len(text) > 0 {
			pricing["text_tokens"] = map[string]any{"standard": text}
		}
		audio := tier(cost, map[string]string{"input_audio": "input_per_million", "output_audio": "output_per_million"})
		if len(audio) > 0 {
			pricing["audio_tokens"] = map[string]any{"standard": audio}
		}
	}

	metadata := map[string]any{"source": "models.dev", "provider_id": providerKey}
	for _, k := range modelsDevMetadata {
		if v, ok := rec[k]; ok && v != nil {
			if _, set := metadata[k]; !set {
				metadata[k] = v
			}
		}
	}

	var created any
	if d, ok := rec["release_date"].(string); ok && d != "" {
		created = d + " 00:00:00 UTC"
	}
	return map[string]any{
		"id": id, "name": name, "provider": slug, "family": rec["family"],
		"created_at": created, "context_window": positive(limit["context"]), "max_output_tokens": positive(limit["output"]),
		"knowledge_cutoff": rec["knowledge"], "modalities": modalities, "capabilities": caps,
		"pricing": pricing, "metadata": metadata,
	}
}

// positive is v when it's a positive number; models.dev writes 0 for
// unknown limits.
func positive(v any) any {
	if n := core.AsInt(v); n != nil && *n > 0 {
		return *n
	}
	return nil
}

// tier renames the cost keys present in keys; zero prices are dropped like
// PricingTier does.
func tier(cost map[string]any, keys map[string]string) map[string]any {
	out := map[string]any{}
	for from, to := range keys {
		if f := core.AsFloat(cost[from]); f != nil && *f != 0 {
			out[to] = *f
		}
	}
	return out
}

// ModelsDevReport counts what MergeModelsDev changed.
type ModelsDevReport struct {
	Added, Updated, Unchanged, Skipped int
}

// MergeModelsDev layers the catalogue over models:
//   - entries sourced from models.dev or from a provider listing (no
//     metadata.source) take models.dev's values where it has one, keep
//     their own where it doesn't, keep created_at, and gain models.dev's
//     capabilities on top of theirs;
//   - entries with any other source ("manual", "known_models", ...) are
//     hand-curated and left alone;
//   - models.dev models missing from the registry are added for the mapped
//     providers; nothing is removed.
//
// overrides (keyed provider:id) are deep-merged last, so local
// corrections survive every refresh. The result is sorted like FetchMerged's.
func MergeModelsDev(models []*core.Info, cat ModelsDevCatalog, overrides map[string]Override) ([]*core.Info, ModelsDevReport) {
	var report ModelsDevReport
	byKey := map[string]*core.Info{}
	for _, m := range models {
		byKey[m.Provider+":"+m.ID] = m
	}
	out := make([]*core.Info, 0, len(models))
	for _, m := range models {
		key := m.Provider + ":" + m.ID
		rec, err := record(m)
		if err != nil {
			out = append(out, m)
			report.Unchanged++
			continue
		}
		if src, _ := m.Metadata["source"].(string); src == "" || src == "models.dev" {
			if dev, providerKey := cat.lookup(m.Provider, m.ID); dev != nil {
				fresh := ModelsDevRecord(dev, m.Provider, providerKey)
				fresh["id"] = m.ID // keep a Bedrock region prefix
				rec = layer(rec, fresh)
			}
		} else {
			report.Skipped++
		}
		if o, ok := overrides[key]; ok {
			rec = core.DeepMerge(rec, o.Set)
		}
		next := core.NewInfo(rec)
		if same(m, next) {
			report.Unchanged++
			next = m
		} else {
			report.Updated++
		}
		out = append(out, next)
	}

	// New models, in catalogue order for determinism before the final sort.
	for _, p := range modelsDevProviders {
		if p.key == "google" && p.slug == "vertexai" {
			continue // only Vertex's own listing adds Vertex models
		}
		for _, id := range slices.Sorted(maps.Keys(cat[p.key].Models)) {
			key := p.slug + ":" + id
			if byKey[key] != nil {
				continue
			}
			rec := ModelsDevRecord(cat[p.key].Models[id], p.slug, p.key)
			if o, ok := overrides[key]; ok {
				rec = core.DeepMerge(rec, o.Set)
			}
			m := core.NewInfo(rec)
			byKey[key] = m
			out = append(out, m)
			report.Added++
		}
	}
	sortModels(out)
	return out, report
}

// layer puts fresh (a models.dev record) over rec (an existing entry).
func layer(rec, fresh map[string]any) map[string]any {
	out := maps.Clone(rec)
	for _, k := range []string{"name", "family", "context_window", "max_output_tokens", "knowledge_cutoff"} {
		if fresh[k] != nil {
			out[k] = fresh[k]
		}
	}
	if out["created_at"] == nil {
		out["created_at"] = fresh["created_at"]
	}
	if mods, _ := fresh["modalities"].(map[string]any); len(core.ToSafeArray(mods["input"])) > 0 {
		out["modalities"] = mods
	}

	caps := core.ToSafeArray(fresh["capabilities"])
	for _, c := range core.ToSafeArray(rec["capabilities"]) {
		if !slices.Contains(caps, c) {
			caps = append(caps, c)
		}
	}
	out["capabilities"] = caps

	// models.dev prices replace the standard tiers it has; batch tiers and
	// categories it doesn't price (images) are kept.
	pricing := map[string]any{}
	if old, ok := rec["pricing"].(map[string]any); ok {
		for k, v := range old {
			pricing[k] = maps.Clone(v.(map[string]any))
		}
	}
	for cat, v := range fresh["pricing"].(map[string]any) {
		c, _ := pricing[cat].(map[string]any)
		if c == nil {
			c = map[string]any{}
		}
		c["standard"] = v.(map[string]any)["standard"]
		pricing[cat] = c
	}
	out["pricing"] = pricing

	md := map[string]any{}
	if old, ok := rec["metadata"].(map[string]any); ok {
		maps.Copy(md, old)
	}
	for _, k := range modelsDevMetadata {
		delete(md, k)
	}
	maps.Copy(md, fresh["metadata"].(map[string]any))
	out["metadata"] = md
	return out
}

// record is m as models.json stores it. Info.Map's typed values (pricing
// tiers are map[string]float64) don't read back through NewInfo, so it goes
// through JSON.
func record(m *core.Info) (map[string]any, error) {
	data, err := Marshal([]*core.Info{m})
	if err != nil {
		return nil, err
	}
	var recs []map[string]any
	if err := json.Unmarshal(data, &recs); err != nil {
		return nil, err
	}
	return recs[0], nil
}

// same reports whether two entries would be saved identically.
func same(a, b *core.Info) bool {
	x, err1 := Marshal([]*core.Info{a})
	y, err2 := Marshal([]*core.Info{b})
	return err1 == nil && err2 == nil && string(x) == string(y)
}

//go:embed overrides.json
var overridesJSON []byte

// Override is a local correction to models.dev data: Set is a partial
// registry record deep-merged after every refresh, Reason says why.
type Override struct {
	Reason string         `json:"reason"`
	Set    map[string]any `json:"set"`
}

// Overrides reads overrides.json, keyed provider:id. Add one when
// models.dev is wrong rather than editing models.json, which the next
// refresh would overwrite.
func Overrides() (map[string]Override, error) {
	var o map[string]Override
	if err := json.Unmarshal(overridesJSON, &o); err != nil {
		return nil, fmt.Errorf("registry: overrides.json: %w", err)
	}
	return o, nil
}
