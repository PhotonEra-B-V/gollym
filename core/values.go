package core

// Tokens is token usage for one response. Nil fields mean "not reported".
// Input excludes cached and cache-creation tokens, as in pyllym.
type Tokens struct {
	Input         *int
	Output        *int
	Cached        *int
	CacheCreation *int
	Thinking      *int
}

// BuildTokens returns nil when every count is nil (Tokens.build).
func BuildTokens(input, output, cached, cacheCreation, thinking *int) *Tokens {
	if input == nil && output == nil && cached == nil && cacheCreation == nil && thinking == nil {
		return nil
	}
	return &Tokens{Input: input, Output: output, Cached: cached, CacheCreation: cacheCreation, Thinking: thinking}
}

// CacheRead is an alias of Cached.
func (t *Tokens) CacheRead() *int { return t.Cached }

// CacheWrite is an alias of CacheCreation.
func (t *Tokens) CacheWrite() *int { return t.CacheCreation }

// Map is the to_dict form (nil counts omitted).
func (t *Tokens) Map() map[string]any {
	out := map[string]any{}
	put := func(k string, v *int) {
		if v != nil {
			out[k] = *v
		}
	}
	put("input_tokens", t.Input)
	put("output_tokens", t.Output)
	put("cached_tokens", t.Cached)
	put("cache_creation_tokens", t.CacheCreation)
	put("thinking_tokens", t.Thinking)
	return out
}

// Int returns a pointer to v (for building Tokens and optional fields).
func Int(v int) *int { return &v }

// Float returns a pointer to v.
func Float(v float64) *float64 { return &v }

// Str returns a pointer to v.
func Str(v string) *string { return &v }

// Deref returns *p or 0.
func Deref[T any](p *T) T {
	var zero T
	if p == nil {
		return zero
	}
	return *p
}

// Thinking is a provider's reasoning output.
type Thinking struct {
	Text      string
	Signature string
}

// BuildThinking returns nil when both parts are empty.
func BuildThinking(text, signature string) *Thinking {
	if text == "" && signature == "" {
		return nil
	}
	return &Thinking{Text: text, Signature: signature}
}

// String keeps signatures out of logs (pyllym's Thinking.__repr__).
func (t *Thinking) String() string {
	text, sig := "None", "None"
	if t.Text != "" {
		text = pyQuote(t.Text)
	}
	if t.Signature != "" {
		sig = "'[REDACTED]'"
	}
	return "Thinking(text=" + text + ", signature=" + sig + ")"
}

// ThinkingConfig requests reasoning across providers.
type ThinkingConfig struct {
	Effort string
	Budget *int
}

// Enabled reports whether any thinking option is set.
func (t *ThinkingConfig) Enabled() bool { return t != nil && (t.Effort != "" || t.Budget != nil) }

// Citation links a span of generated text to its source, normalised across
// providers.
type Citation struct {
	URL         *string `json:"url,omitempty"`
	Title       *string `json:"title,omitempty"`
	CitedText   *string `json:"cited_text,omitempty"`
	Text        *string `json:"text,omitempty"`
	StartIndex  *int    `json:"start_index,omitempty"`
	EndIndex    *int    `json:"end_index,omitempty"`
	SourceIndex *int    `json:"source_index,omitempty"`
	StartPage   *int    `json:"start_page,omitempty"`
	EndPage     *int    `json:"end_page,omitempty"`
}

// Map is the to_dict form (nil fields omitted).
func (c Citation) Map() map[string]any {
	out := map[string]any{}
	putS := func(k string, v *string) {
		if v != nil {
			out[k] = *v
		}
	}
	putI := func(k string, v *int) {
		if v != nil {
			out[k] = *v
		}
	}
	putS("url", c.URL)
	putS("title", c.Title)
	putS("cited_text", c.CitedText)
	putS("text", c.Text)
	putI("start_index", c.StartIndex)
	putI("end_index", c.EndIndex)
	putI("source_index", c.SourceIndex)
	putI("start_page", c.StartPage)
	putI("end_page", c.EndPage)
	return out
}

// CitationFromMap is Citation.from_dict: known keys are read, others ignored.
func CitationFromMap(d map[string]any) Citation {
	str := func(k string) *string {
		if s, ok := d[k].(string); ok {
			return &s
		}
		return nil
	}
	return Citation{
		URL: str("url"), Title: str("title"), CitedText: str("cited_text"), Text: str("text"),
		StartIndex: AsInt(d["start_index"]), EndIndex: AsInt(d["end_index"]),
		SourceIndex: AsInt(d["source_index"]), StartPage: AsInt(d["start_page"]), EndPage: AsInt(d["end_page"]),
	}
}

// Equal is value equality (citations are deduplicated by value).
func (c Citation) Equal(o Citation) bool {
	return eqp(c.URL, o.URL) && eqp(c.Title, o.Title) && eqp(c.CitedText, o.CitedText) &&
		eqp(c.Text, o.Text) && eqp(c.StartIndex, o.StartIndex) && eqp(c.EndIndex, o.EndIndex) &&
		eqp(c.SourceIndex, o.SourceIndex) && eqp(c.StartPage, o.StartPage) && eqp(c.EndPage, o.EndPage)
}

func eqp[T comparable](a, b *T) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

// ToolCall is a function call from a model to a Tool. Arguments is the
// decoded JSON object (map[string]any), or the raw argument string while a
// stream is still accumulating fragments.
type ToolCall struct {
	ID               string
	Name             string
	Arguments        any
	ThoughtSignature string
}

// Map is the to_dict form.
func (tc *ToolCall) Map() map[string]any {
	// Unset arguments are pyllym's default_factory dict.
	args := tc.Arguments
	if args == nil {
		args = map[string]any{}
	}
	out := map[string]any{"arguments": args}
	if tc.ID != "" {
		out["id"] = tc.ID
	}
	if tc.Name != "" {
		out["name"] = tc.Name
	}
	if tc.ThoughtSignature != "" {
		out["thought_signature"] = tc.ThoughtSignature
	}
	return out
}

// ToolCalls keeps pyllym's insertion-ordered dict of calls keyed by id (or
// stream index while streaming).
type ToolCalls struct {
	keys  []string
	calls map[string]*ToolCall
}

// Set adds or replaces a call, keeping first-insertion order.
func (t *ToolCalls) Set(key string, tc *ToolCall) {
	if t.calls == nil {
		t.calls = map[string]*ToolCall{}
	}
	if _, ok := t.calls[key]; !ok {
		t.keys = append(t.keys, key)
	}
	t.calls[key] = tc
}

// Get returns the call stored under key.
func (t *ToolCalls) Get(key string) *ToolCall {
	if t == nil || t.calls == nil {
		return nil
	}
	return t.calls[key]
}

// Keys returns the keys in insertion order.
func (t *ToolCalls) Keys() []string {
	if t == nil {
		return nil
	}
	return append([]string(nil), t.keys...)
}

// List returns the calls in insertion order.
func (t *ToolCalls) List() []*ToolCall {
	if t == nil {
		return nil
	}
	out := make([]*ToolCall, 0, len(t.keys))
	for _, k := range t.keys {
		out = append(out, t.calls[k])
	}
	return out
}

// Len is the number of calls.
func (t *ToolCalls) Len() int {
	if t == nil {
		return 0
	}
	return len(t.keys)
}
