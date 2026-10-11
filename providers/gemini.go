package providers

import (
	"regexp"

	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/protocols/gemini"
)

func geminiSpec() *core.ProviderSpec {
	return &core.ProviderSpec{
		Slug: "gemini", Name: "Gemini",
		Protocols:     map[string]core.ProtocolFactory{"gemini": gemini.New},
		ProtocolOrder: []string{"gemini"},
		Options:       []string{"gemini_api_key", "gemini_api_base"},
		Requirements:  []string{"gemini_api_key"},
		Capabilities:  GeminiCapabilities{},
		APIBase: func(p core.Provider) (string, error) {
			if v := p.Config().Get("gemini_api_base"); v != "" {
				return v, nil
			}
			return "https://generativelanguage.googleapis.com/v1beta", nil
		},
		Headers: func(p core.Provider) map[string]string {
			return map[string]string{"x-goog-api-key": p.Config().Get("gemini_api_key")}
		},
	}
}

type geminiPrice struct{ input, output, price float64 }

var geminiPrices = map[string]geminiPrice{
	"flash_2":          {input: 0.10, output: 0.40},
	"flash_lite_2":     {input: 0.075, output: 0.30},
	"flash":            {input: 0.075, output: 0.30},
	"flash_8b":         {input: 0.0375, output: 0.15},
	"pro":              {input: 1.25, output: 5.0},
	"pro_2_5":          {input: 0.12, output: 0.50},
	"gemini_embedding": {input: 0.002, output: 0.004},
	"embedding":        {input: 0.00, output: 0.00},
	"imagen":           {price: 0.03},
	"aqa":              {input: 0.00, output: 0.00},
}

func gemRe(p string) *regexp.Regexp { return regexp.MustCompile(p) }

var (
	gemCtx1M       = gemRe(`gemini-2\.5-pro-exp-03-25|gemini-2\.0-flash|gemini-2\.0-flash-lite|gemini-1\.5-flash|gemini-1\.5-flash-8b`)
	gemCtx2M       = gemRe(`gemini-1\.5-pro`)
	gemEmbedExp    = gemRe(`gemini-embedding-exp`)
	gemTextEmbed   = gemRe(`text-embedding-004|embedding-001`)
	gemAQA         = gemRe(`aqa`)
	gemImagen3     = gemRe(`imagen-3`)
	gemProExp      = gemRe(`gemini-2\.5-pro-exp-03-25`)
	gemMax8K       = gemRe(`gemini-2\.0-flash|gemini-2\.0-flash-lite|gemini-1\.5-flash|gemini-1\.5-flash-8b|gemini-1\.5-pro`)
	gemNoVision    = gemRe(`text-embedding|embedding-001|aqa`)
	gemVision      = gemRe(`gemini|flash|pro|imagen`)
	gemNoFunctions = gemRe(`text-embedding|embedding-001|aqa|flash-lite|imagen|gemini-2\.0-flash-lite`)
	gemFamilyAny   = gemRe(`gemini|pro|flash`)
	gemNoStruct    = gemRe(`text-embedding|embedding-001|aqa|imagen|gemini-2\.0-flash-lite|gemini-2\.5-pro-exp-03-25`)
)

var gemPricingFamilies = []struct {
	re     *regexp.Regexp
	family string
}{
	{gemRe(`gemini-2\.5-pro-exp-03-25`), "pro_2_5"},
	{gemRe(`gemini-2\.0-flash-lite`), "flash_lite_2"},
	{gemRe(`gemini-2\.0-flash`), "flash_2"},
	{gemRe(`gemini-1\.5-flash-8b`), "flash_8b"},
	{gemRe(`gemini-1\.5-flash`), "flash"},
	{gemRe(`gemini-1\.5-pro`), "pro"},
	{gemRe(`gemini-embedding-exp`), "gemini_embedding"},
	{gemRe(`text-embedding|embedding`), "embedding"},
	{gemRe(`imagen`), "imagen"},
	{gemRe(`aqa`), "aqa"},
}

// GeminiCapabilities are pyllym's id-based heuristics for Gemini models
// (used when ListModels returns bare ids).
type GeminiCapabilities struct{ core.DefaultCapabilities }

func (GeminiCapabilities) ContextWindowFor(id string) *int {
	switch {
	case gemCtx1M.MatchString(id):
		return core.Int(1_048_576)
	case gemCtx2M.MatchString(id):
		return core.Int(2_097_152)
	case gemEmbedExp.MatchString(id):
		return core.Int(8_192)
	case gemTextEmbed.MatchString(id):
		return core.Int(2_048)
	case gemAQA.MatchString(id):
		return core.Int(7_168)
	case gemImagen3.MatchString(id):
		return nil
	}
	return core.Int(32_768)
}

func (GeminiCapabilities) MaxTokensFor(id string) *int {
	switch {
	case gemProExp.MatchString(id):
		return core.Int(64_000)
	case gemMax8K.MatchString(id):
		return core.Int(8_192)
	case gemEmbedExp.MatchString(id):
		return nil
	case gemTextEmbed.MatchString(id):
		return core.Int(768)
	case gemImagen3.MatchString(id):
		return core.Int(4)
	}
	return core.Int(4_096)
}

func (c GeminiCapabilities) CriticalCapabilitiesFor(id string) []string {
	out := []string{}
	if c.SupportsFunctions(id) {
		out = append(out, "function_calling")
	}
	if c.SupportsStructuredOutput(id) {
		out = append(out, "structured_output")
	}
	if c.SupportsVision(id) {
		out = append(out, "vision")
	}
	return out
}

// PricingFor keeps pyllym's `input or price or default`, so zero-priced
// families fall back to the default prices exactly as pyllym does.
func (c GeminiCapabilities) PricingFor(id string) map[string]any {
	p, ok := geminiPrices[c.PricingFamily(id)]
	if !ok {
		p = geminiPrice{input: 0.075, output: 0.30}
	}
	or := func(a, b, def float64) float64 {
		if a != 0 {
			return a
		}
		if b != 0 {
			return b
		}
		return def
	}
	return map[string]any{
		"text_tokens": map[string]any{
			"standard": map[string]any{
				"input_per_million":  or(p.input, p.price, 0.075),
				"output_per_million": or(p.output, p.price, 0.30),
			},
		},
	}
}

func (GeminiCapabilities) SupportsVision(id string) bool {
	return !gemNoVision.MatchString(id) && gemVision.MatchString(id)
}

func (GeminiCapabilities) SupportsFunctions(id string) bool {
	return !gemNoFunctions.MatchString(id) && gemFamilyAny.MatchString(id)
}

func (GeminiCapabilities) SupportsStructuredOutput(id string) bool {
	return !gemNoStruct.MatchString(id) && gemFamilyAny.MatchString(id)
}

// PricingFamily maps an id to its price table key ("base" when unknown).
func (GeminiCapabilities) PricingFamily(id string) string {
	for _, f := range gemPricingFamilies {
		if f.re.MatchString(id) {
			return f.family
		}
	}
	return "base"
}
