package core

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"
)

//go:embed models.json
var modelsJSON []byte

//go:embed aliases.json
var aliasesJSON []byte

// Modalities are a model's input and output kinds.
type Modalities struct {
	Input  []string `json:"input"`
	Output []string `json:"output"`
}

// PricingTier holds non-zero prices for one tier (USD per million tokens).
type PricingTier map[string]float64

func (t PricingTier) get(k string) *float64 {
	if v, ok := t[k]; ok && v != 0 {
		return &v
	}
	return nil
}

func (t PricingTier) InputPerMillion() *float64        { return t.get("input_per_million") }
func (t PricingTier) OutputPerMillion() *float64       { return t.get("output_per_million") }
func (t PricingTier) CacheReadPerMillion() *float64    { return t.get("cache_read_input_per_million") }
func (t PricingTier) CacheWritePerMillion() *float64   { return t.get("cache_write_input_per_million") }
func (t PricingTier) ReasoningOutPerMillion() *float64 { return t.get("reasoning_output_per_million") }

// PricingCategory has standard and batch tiers (nil when empty).
type PricingCategory struct {
	Standard PricingTier
	Batch    PricingTier
}

func (c PricingCategory) std(f func(PricingTier) *float64) *float64 {
	if c.Standard == nil {
		return nil
	}
	return f(c.Standard)
}

func (c PricingCategory) Input() *float64 { return c.std(PricingTier.InputPerMillion) }
func (c PricingCategory) Output() *float64 {
	return c.std(PricingTier.OutputPerMillion)
}
func (c PricingCategory) CacheReadInput() *float64 { return c.std(PricingTier.CacheReadPerMillion) }
func (c PricingCategory) CacheWriteInput() *float64 {
	return c.std(PricingTier.CacheWritePerMillion)
}
func (c PricingCategory) ReasoningOutput() *float64 {
	return c.std(PricingTier.ReasoningOutPerMillion)
}

// Pricing groups a model's price categories.
type Pricing struct {
	categories map[string]PricingCategory
}

var pricingCategories = []string{"text_tokens", "images", "audio_tokens", "embeddings"}

func newPricing(data map[string]any) Pricing {
	p := Pricing{categories: map[string]PricingCategory{}}
	for _, name := range pricingCategories {
		raw, _ := data[name].(map[string]any)
		if raw == nil {
			continue
		}
		cat := PricingCategory{Standard: tierFrom(raw["standard"]), Batch: tierFrom(raw["batch"])}
		if cat.Standard == nil && cat.Batch == nil {
			continue
		}
		p.categories[name] = cat
	}
	return p
}

func tierFrom(v any) PricingTier {
	m, _ := v.(map[string]any)
	t := PricingTier{}
	for k, val := range m {
		if f := AsFloat(val); f != nil && *f != 0 {
			t[k] = *f
		}
	}
	if len(t) == 0 {
		return nil
	}
	return t
}

func (p Pricing) Category(name string) PricingCategory { return p.categories[name] }
func (p Pricing) TextTokens() PricingCategory          { return p.categories["text_tokens"] }
func (p Pricing) Images() PricingCategory              { return p.categories["images"] }
func (p Pricing) AudioTokens() PricingCategory         { return p.categories["audio_tokens"] }
func (p Pricing) Embeddings() PricingCategory          { return p.categories["embeddings"] }

// Map is the to_dict form.
func (p Pricing) Map() map[string]any {
	out := map[string]any{}
	for name, c := range p.categories {
		m := map[string]any{}
		if c.Standard != nil {
			m["standard"] = map[string]float64(c.Standard)
		}
		if c.Batch != nil {
			m["batch"] = map[string]float64(c.Batch)
		}
		out[name] = m
	}
	return out
}

// Info describes a model's capabilities, pricing and metadata.
type Info struct {
	ID               string
	Name             string
	Provider         string
	Family           string
	CreatedAt        *time.Time
	ContextWindow    *int
	MaxOutputTokens  *int
	KnowledgeCutoff  *time.Time
	Modalities       Modalities
	Capabilities     []string
	Pricing          Pricing
	Metadata         map[string]any
	ReasoningOptions []map[string]any
	raw              map[string]any
}

// NewInfo reads a registry/to_dict-shaped record.
func NewInfo(data map[string]any) *Info {
	i := &Info{raw: data}
	i.ID, _ = data["id"].(string)
	i.Name, _ = data["name"].(string)
	i.Provider, _ = data["provider"].(string)
	i.Family, _ = data["family"].(string)
	i.CreatedAt = ToTime(data["created_at"])
	i.ContextWindow = AsInt(data["context_window"])
	i.MaxOutputTokens = AsInt(data["max_output_tokens"])
	i.KnowledgeCutoff = ParseISODatePrefix(data["knowledge_cutoff"])
	if m, ok := data["modalities"].(map[string]any); ok {
		for _, x := range ToSafeArray(m["input"]) {
			i.Modalities.Input = append(i.Modalities.Input, fmt.Sprint(x))
		}
		for _, x := range ToSafeArray(m["output"]) {
			i.Modalities.Output = append(i.Modalities.Output, fmt.Sprint(x))
		}
	}
	for _, c := range ToSafeArray(data["capabilities"]) {
		i.Capabilities = append(i.Capabilities, fmt.Sprint(c))
	}
	pr, _ := data["pricing"].(map[string]any)
	i.Pricing = newPricing(pr)
	i.Metadata = map[string]any{}
	if md, ok := data["metadata"].(map[string]any); ok {
		for k, v := range md {
			i.Metadata[k] = v
		}
	}
	opts := data["reasoning_options"]
	if opts == nil {
		opts = i.Metadata["reasoning_options"]
	}
	for _, o := range ToSafeArray(opts) {
		om, ok := o.(map[string]any)
		if !ok {
			continue
		}
		entry := map[string]any{}
		for k, v := range om {
			entry[k] = v
		}
		if t, ok := entry["type"]; ok {
			entry["type"] = fmt.Sprint(t)
		}
		if vals, ok := entry["values"]; ok {
			var s []any
			for _, v := range ToSafeArray(vals) {
				s = append(s, fmt.Sprint(v))
			}
			entry["values"] = s
		}
		i.ReasoningOptions = append(i.ReasoningOptions, entry)
	}
	if len(i.ReasoningOptions) > 0 {
		ro := make([]any, len(i.ReasoningOptions))
		for k, v := range i.ReasoningOptions {
			ro[k] = v
		}
		i.Metadata["reasoning_options"] = ro
	}
	return i
}

// DefaultInfo is the assumed record for a model missing from the registry.
func DefaultInfo(modelID, provider string) *Info {
	name := strings.ReplaceAll(modelID, "-", " ")
	if name != "" {
		name = strings.ToUpper(name[:1]) + strings.ToLower(name[1:])
	}
	return NewInfo(map[string]any{
		"id": modelID, "name": name, "provider": provider,
		"capabilities": []any{"function_calling", "streaming", "vision", "structured_output"},
		"modalities":   map[string]any{"input": []any{"text", "image"}, "output": []any{"text"}},
		"metadata":     map[string]any{"warning": "Assuming model exists, capabilities may not be accurate"},
	})
}

// Supports reports a capability.
func (i *Info) Supports(capability string) bool { return slices.Contains(i.Capabilities, capability) }

func (i *Info) SupportsVision() bool            { return slices.Contains(i.Modalities.Input, "image") }
func (i *Info) SupportsVideo() bool             { return slices.Contains(i.Modalities.Input, "video") }
func (i *Info) SupportsFunctions() bool         { return i.Supports("function_calling") }
func (i *Info) IsStructuredOutput() bool        { return i.Supports("structured_output") }
func (i *Info) IsReasoning() bool               { return i.Supports("reasoning") }
func (i *Info) IsCitations() bool               { return i.Supports("citations") }
func (i *Info) IsStreaming() bool               { return i.Supports("streaming") }
func (i *Info) IsBatch() bool                   { return i.Supports("batch") }
func (i *Info) DisplayName() string             { return i.Name }
func (i *Info) Label() string                   { return i.Provider + " - " + i.Name }
func (i *Info) MaxTokens() *int                 { return i.MaxOutputTokens }
func (i *Info) InputPricePerMillion() *float64  { return i.Pricing.TextTokens().Input() }
func (i *Info) OutputPricePerMillion() *float64 { return i.Pricing.TextTokens().Output() }

// ReasoningOption returns the option of the given type.
func (i *Info) ReasoningOption(t string) map[string]any {
	for _, o := range i.ReasoningOptions {
		if o["type"] == t {
			return o
		}
	}
	return nil
}

// ReasoningOptionValues returns the allowed values of an option.
func (i *Info) ReasoningOptionValues(t string) []string {
	var out []string
	for _, v := range ToSafeArray(i.ReasoningOption(t)["values"]) {
		out = append(out, fmt.Sprint(v))
	}
	return out
}

// Type is embedding, moderation, decision, image, audio, video or chat.
func (i *Info) Type() string {
	out := i.Modalities.Output
	switch {
	case slices.Contains(out, "embeddings"):
		return "embedding"
	case slices.Contains(out, "moderation"):
		return "moderation"
	case slices.Contains(out, "decisions"):
		return "decision"
	case slices.Contains(out, "image"):
		return "image"
	case slices.Contains(out, "audio"):
		return "audio"
	case slices.Contains(out, "video"):
		return "video"
	}
	return "chat"
}

// CostFor prices tokens with this model.
func (i *Info) CostFor(t *Tokens) *Cost { return NewCost(t, i) }

// Map is the to_dict form (registry JSON shape).
func (i *Info) Map() map[string]any {
	var created, cutoff any
	if i.CreatedAt != nil {
		created = i.CreatedAt.Format("2006-01-02 15:04:05-07:00")
	}
	if i.KnowledgeCutoff != nil {
		cutoff = i.KnowledgeCutoff.Format("2006-01-02")
	}
	caps := make([]any, len(i.Capabilities))
	for k, c := range i.Capabilities {
		caps[k] = c
	}
	return map[string]any{
		"id": i.ID, "name": nilIfEmpty(i.Name), "provider": i.Provider, "family": nilIfEmpty(i.Family),
		"created_at": created, "context_window": intOrNil(i.ContextWindow), "max_output_tokens": intOrNil(i.MaxOutputTokens),
		"knowledge_cutoff": cutoff,
		"modalities":       map[string]any{"input": strSlice(i.Modalities.Input), "output": strSlice(i.Modalities.Output)},
		"capabilities":     caps, "pricing": i.Pricing.Map(), "metadata": i.Metadata,
	}
}

func nilIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func intOrNil(p *int) any {
	if p == nil {
		return nil
	}
	return *p
}

func strSlice(s []string) []any {
	out := make([]any, len(s))
	for i, v := range s {
		out[i] = v
	}
	return out
}

// ProviderPreference: first-party providers outrank resellers.
var ProviderPreference = []string{
	"openai", "anthropic", "gemini", "deepseek", "mistral", "perplexity", "xai",
	"vertexai", "bedrock", "openrouter", "azure", "ollama", "gpustack",
}

// Registry holds the model list.
type Registry struct {
	mu     sync.RWMutex
	models []*Info
}

// NewRegistry wraps a list of models.
func NewRegistry(models []*Info) *Registry { return &Registry{models: models} }

// LoadRegistry reads a models.json-shaped file (embedded data when path is
// empty). A missing or invalid file yields an empty registry, as in pyllym.
func LoadRegistry(path string) *Registry {
	data := modelsJSON
	if path != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			return NewRegistry(nil)
		}
		data = b
	}
	var raw []map[string]any
	dec := json.NewDecoder(strings.NewReader(string(data)))
	dec.UseNumber()
	if err := dec.Decode(&raw); err != nil {
		return NewRegistry(nil)
	}
	models := make([]*Info, 0, len(raw))
	for _, m := range raw {
		normaliseNumbers(m)
		models = append(models, NewInfo(m))
	}
	return NewRegistry(models)
}

// normaliseNumbers turns json.Number into float64 (whole numbers stay exact).
func normaliseNumbers(v any) any {
	switch x := v.(type) {
	case map[string]any:
		for k, val := range x {
			x[k] = normaliseNumbers(val)
		}
	case []any:
		for i, val := range x {
			x[i] = normaliseNumbers(val)
		}
	case json.Number:
		f, _ := x.Float64()
		return f
	}
	return v
}

var (
	modelsOnce sync.Once
	modelsInst *Registry
)

// Models returns the process-wide registry (loaded lazily from the embedded
// models.json, or Global().ModelRegistryFile when set).
func Models() *Registry {
	modelsOnce.Do(func() { modelsInst = LoadRegistry(Global().ModelRegistryFile) })
	return modelsInst
}

// All returns every model.
func (r *Registry) All() []*Info {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return append([]*Info(nil), r.models...)
}

// Replace swaps the model list (after a refresh).
func (r *Registry) Replace(models []*Info) {
	r.mu.Lock()
	r.models = models
	r.mu.Unlock()
}

// Len is the number of models.
func (r *Registry) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.models)
}

func (r *Registry) filter(f func(*Info) bool) *Registry {
	var out []*Info
	for _, m := range r.All() {
		if f(m) {
			out = append(out, m)
		}
	}
	return NewRegistry(out)
}

func (r *Registry) ChatModels() *Registry {
	return r.filter(func(m *Info) bool { return m.Type() == "chat" })
}
func (r *Registry) EmbeddingModels() *Registry {
	return r.filter(func(m *Info) bool {
		return m.Type() == "embedding" || slices.Contains(m.Modalities.Output, "embeddings")
	})
}
func (r *Registry) DecisionModels() *Registry {
	return r.filter(func(m *Info) bool { return m.Type() == "decision" })
}
func (r *Registry) AudioModels() *Registry {
	return r.filter(func(m *Info) bool { return m.Type() == "audio" || slices.Contains(m.Modalities.Output, "audio") })
}
func (r *Registry) ImageModels() *Registry {
	return r.filter(func(m *Info) bool { return m.Type() == "image" || slices.Contains(m.Modalities.Output, "image") })
}
func (r *Registry) ByFamily(f string) *Registry {
	return r.filter(func(m *Info) bool { return m.Family == f })
}
func (r *Registry) ByProvider(p string) *Registry {
	return r.filter(func(m *Info) bool { return m.Provider == p })
}

const refreshGuidance = "If the model exists at the provider, refresh the registry with `gollym models refresh` and persist it."

// Find looks a model up, resolving aliases. Without a provider, first-party
// providers win over resellers.
func (r *Registry) Find(modelID, provider string) (*Info, error) {
	models := r.All()
	if provider != "" {
		resolved := ResolveAlias(modelID, provider)
		for _, m := range models {
			if m.ID == resolved && m.Provider == provider {
				return m, nil
			}
		}
		for _, m := range models {
			if m.ID == modelID && m.Provider == provider {
				return m, nil
			}
		}
		return nil, &ModelNotFoundError{Message: fmt.Sprintf("Unknown model: %q for provider: %q. %s", modelID, provider, refreshGuidance)}
	}
	resolved := ResolveAlias(modelID, "")
	var candidates []*Info
	for _, m := range models {
		if m.ID == modelID || m.ID == resolved {
			candidates = append(candidates, m)
		}
	}
	sort.SliceStable(candidates, func(a, b int) bool {
		return candidates[a].ID == modelID && candidates[b].ID != modelID
	})
	if m := preferredMatch(candidates); m != nil {
		return m, nil
	}
	return nil, &ModelNotFoundError{Message: fmt.Sprintf("Unknown model: %q. %s", modelID, refreshGuidance)}
}

func preferredMatch(c []*Info) *Info {
	if len(c) == 0 {
		return nil
	}
	rank := func(m *Info) int {
		if i := slices.Index(ProviderPreference, m.Provider); i >= 0 {
			return i
		}
		return len(ProviderPreference)
	}
	best := c[0]
	for _, m := range c[1:] {
		if rank(m) < rank(best) {
			best = m
		}
	}
	return best
}

// aliases: alias -> provider -> model id. Python dict order decides the
// provider-less resolution, so the file's key order is kept.
var (
	aliasOnce  sync.Once
	aliasTable map[string][]aliasEntry
)

type aliasEntry struct{ provider, id string }

// ResolveAlias maps an alias to a provider's model id (or the id itself).
func ResolveAlias(modelID, provider string) string {
	aliasOnce.Do(loadAliases)
	entries := aliasTable[modelID]
	if len(entries) == 0 {
		return modelID
	}
	if provider != "" {
		for _, e := range entries {
			if e.provider == provider {
				return e.id
			}
		}
		return modelID
	}
	return entries[0].id
}

func loadAliases() {
	aliasTable = map[string][]aliasEntry{}
	dec := json.NewDecoder(strings.NewReader(string(aliasesJSON)))
	if _, err := dec.Token(); err != nil { // {
		return
	}
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return
		}
		alias, _ := tok.(string)
		if _, err := dec.Token(); err != nil { // {
			return
		}
		var entries []aliasEntry
		for dec.More() {
			pt, _ := dec.Token()
			var id string
			if err := dec.Decode(&id); err != nil {
				return
			}
			entries = append(entries, aliasEntry{provider: fmt.Sprint(pt), id: id})
		}
		_, _ = dec.Token() // }
		aliasTable[alias] = entries
	}
}
