package core

import "math"

// Cost components.
var costComponents = []string{"input", "output", "cache_read", "cache_write", "thinking"}

const perMillion = 1_000_000.0

// Cost prices token usage with a model's registry pricing (USD). Amounts are
// nil when unknown; Total is nil when any used component has no price.
type Cost struct {
	Tokens       *Tokens
	Model        *Info
	Category     string         // "text_tokens" (default) or "images"
	InputDetails map[string]any // image costs: text_tokens / image_tokens
}

// NewCost prices tokens for a text model.
func NewCost(t *Tokens, model *Info) *Cost {
	return &Cost{Tokens: t, Model: model, Category: "text_tokens"}
}

func (c *Cost) Input() *float64      { return c.amountFor("input") }
func (c *Cost) Output() *float64     { return c.amountFor("output") }
func (c *Cost) CacheRead() *float64  { return c.amountFor("cache_read") }
func (c *Cost) CacheWrite() *float64 { return c.amountFor("cache_write") }
func (c *Cost) Thinking() *float64   { return c.amountFor("thinking") }

// Total sums the components, or nil when unpriced/empty.
func (c *Cost) Total() *float64 {
	if !c.HasTokens() {
		return nil
	}
	for _, comp := range costComponents {
		if c.IsMissing(comp) {
			return nil
		}
	}
	var vals []float64
	for _, comp := range costComponents {
		if v := c.amountFor(comp); v != nil {
			vals = append(vals, *v)
		}
	}
	if len(vals) == 0 {
		return nil
	}
	sum := pySum(vals)
	return &sum
}

// pySum is Python's sum() of floats, which (since 3.12) uses Neumaier's
// compensated summation: plain addition differs in the last bits.
func pySum(vals []float64) float64 {
	var sum, c float64
	for _, x := range vals {
		t := sum + x
		if math.Abs(sum) >= math.Abs(x) {
			c += (sum - t) + x
		} else {
			c += (x - t) + sum
		}
		sum = t
	}
	if c != 0 && !math.IsInf(c, 0) && !math.IsNaN(c) {
		sum += c
	}
	return sum
}

// Map is the to_dict form.
func (c *Cost) Map() map[string]float64 {
	out := map[string]float64{}
	for _, comp := range costComponents {
		if v := c.amountFor(comp); v != nil {
			out[comp] = *v
		}
	}
	if t := c.Total(); t != nil {
		out["total"] = *t
	}
	return out
}

// HasTokens reports whether any component has a token count.
func (c *Cost) HasTokens() bool {
	for _, comp := range costComponents {
		if c.tokensFor(comp) != nil {
			return true
		}
	}
	return false
}

// IsMissing reports a component with tokens but no price.
func (c *Cost) IsMissing(comp string) bool {
	if comp == "input" && c.detailedImageInput() {
		return c.imageInputMissing()
	}
	if comp == "thinking" && !c.thinkingPricedSeparately() {
		return false
	}
	t := c.tokensFor(comp)
	return t != nil && *t > 0 && c.priceFor(comp) == nil
}

func (c *Cost) amountFor(comp string) *float64 {
	if comp == "input" && c.detailedImageInput() {
		return c.imageInputAmount()
	}
	t := c.tokensFor(comp)
	if t == nil {
		return nil
	}
	if *t == 0 {
		z := 0.0
		return &z
	}
	price := c.priceFor(comp)
	if price == nil || *price == 0 {
		return nil
	}
	v := float64(*t) * *price / perMillion
	return &v
}

func (c *Cost) tokensFor(comp string) *int {
	if c.Tokens == nil {
		return nil
	}
	switch comp {
	case "input":
		return c.Tokens.Input
	case "output":
		return c.Tokens.Output
	case "cache_read":
		return c.Tokens.Cached
	case "cache_write":
		return c.Tokens.CacheCreation
	case "thinking":
		if c.thinkingPricedSeparately() {
			return c.Tokens.Thinking
		}
	}
	return nil
}

func (c *Cost) priceFor(comp string) *float64 {
	tp := c.textPricing()
	switch comp {
	case "input":
		return tp.Input()
	case "output":
		return c.outputPricing().Output()
	case "cache_read":
		return tp.CacheReadInput()
	case "cache_write":
		return tp.CacheWriteInput()
	case "thinking":
		return tp.ReasoningOutput()
	}
	return nil
}

func (c *Cost) textPricing() PricingCategory {
	if c.Model == nil {
		return PricingCategory{}
	}
	return c.Model.Pricing.TextTokens()
}

func (c *Cost) imagePricing() PricingCategory {
	if c.Model == nil {
		return PricingCategory{}
	}
	return c.Model.Pricing.Images()
}

func (c *Cost) isImageCost() bool { return c.Category == "image" || c.Category == "images" }

func (c *Cost) outputPricing() PricingCategory {
	if img := c.imagePricing(); c.isImageCost() && img.Output() != nil {
		return img
	}
	return c.textPricing()
}

type imagePart struct {
	tokens *int
	price  *float64
}

func (c *Cost) imageInputParts() []imagePart {
	tp, ip := c.textPricing(), c.imagePricing()
	imgPrice := ip.Input()
	if imgPrice == nil {
		imgPrice = tp.Input()
	}
	return []imagePart{
		{AsInt(c.InputDetails["text_tokens"]), tp.Input()},
		{AsInt(c.InputDetails["image_tokens"]), imgPrice},
	}
}

func (c *Cost) detailedImageInput() bool {
	if !c.isImageCost() || c.InputDetails == nil {
		return false
	}
	for _, p := range c.imageInputParts() {
		if p.tokens != nil {
			return true
		}
	}
	return false
}

func (c *Cost) imageInputMissing() bool {
	for _, p := range c.imageInputParts() {
		if p.tokens != nil && *p.tokens > 0 && p.price == nil {
			return true
		}
	}
	return false
}

func (c *Cost) imageInputAmount() *float64 {
	if c.imageInputMissing() {
		return nil
	}
	total := 0.0
	for _, p := range c.imageInputParts() {
		if p.tokens == nil || *p.tokens == 0 || p.price == nil {
			continue
		}
		total += float64(*p.tokens) * *p.price / perMillion
	}
	return &total
}

func (c *Cost) thinkingPricedSeparately() bool {
	r := c.textPricing().ReasoningOutput()
	if r == nil || *r == 0 {
		return false
	}
	o := c.textPricing().Output()
	return o == nil || *r != *o
}

// CostAggregate is the sum of several costs.
type CostAggregate struct {
	amounts map[string]*float64
	missing map[string]bool
	tokens  bool
}

// AggregateCosts sums per-message costs (pyllym Cost.aggregate).
func AggregateCosts(costs []*Cost) *CostAggregate {
	var kept []*Cost
	for _, c := range costs {
		if c != nil && c.HasTokens() {
			kept = append(kept, c)
		}
	}
	agg := &CostAggregate{amounts: map[string]*float64{}, missing: map[string]bool{}, tokens: len(kept) > 0}
	for _, comp := range costComponents {
		for _, c := range kept {
			if c.IsMissing(comp) {
				agg.missing[comp] = true
			}
		}
	}
	for _, comp := range costComponents {
		var vals []float64
		for _, c := range kept {
			if v := c.amountFor(comp); v != nil {
				vals = append(vals, *v)
			}
		}
		if len(vals) > 0 && !agg.missing[comp] {
			s := pySum(vals)
			agg.amounts[comp] = &s
		}
	}
	return agg
}

func (a *CostAggregate) Input() *float64         { return a.amounts["input"] }
func (a *CostAggregate) Output() *float64        { return a.amounts["output"] }
func (a *CostAggregate) CacheRead() *float64     { return a.amounts["cache_read"] }
func (a *CostAggregate) CacheWrite() *float64    { return a.amounts["cache_write"] }
func (a *CostAggregate) Thinking() *float64      { return a.amounts["thinking"] }
func (a *CostAggregate) HasTokens() bool         { return a.tokens }
func (a *CostAggregate) IsMissing(c string) bool { return a.missing[c] }

// Total is nil when there are no tokens or any component is unpriced.
func (a *CostAggregate) Total() *float64 {
	if !a.tokens || len(a.missing) > 0 {
		return nil
	}
	var vals []float64
	for _, comp := range costComponents {
		if v := a.amounts[comp]; v != nil {
			vals = append(vals, *v)
		}
	}
	if len(vals) == 0 {
		return nil
	}
	sum := pySum(vals)
	return &sum
}
