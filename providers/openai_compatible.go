package providers

import (
	"maps"

	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/protocols/chatcompletions"
)

// compatible describes a provider speaking Chat Completions with Bearer auth
// (pyllym's OpenAICompatible). Config keys derive from the slug:
// <slug>_api_key and <slug>_api_base.
type compatible struct {
	slug, name   string
	defaultBase  string // "" = <slug>_api_base is required
	assumeModels bool
	extraHeaders map[string]string
	protocols    map[string]core.ProtocolFactory
	order        []string
	parseError   func(p core.Provider, r *core.Response) string
}

func (c compatible) spec() *core.ProviderSpec {
	protocols := c.protocols
	order := c.order
	if protocols == nil {
		protocols = map[string]core.ProtocolFactory{"chat_completions": chatcompletions.New}
		order = []string{"chat_completions"}
	}
	reqs := []string{c.slug + "_api_key"}
	if c.defaultBase == "" {
		reqs = []string{c.slug + "_api_base", c.slug + "_api_key"}
	}
	slug, base, extra := c.slug, c.defaultBase, c.extraHeaders
	return &core.ProviderSpec{
		Slug: slug, Name: c.name, Protocols: protocols, ProtocolOrder: order,
		Options:      []string{slug + "_api_key", slug + "_api_base"},
		Requirements: reqs,
		AssumeModels: c.assumeModels,
		ParseError:   c.parseError,
		APIBase: func(p core.Provider) (string, error) {
			if v := p.Config().Get(slug + "_api_base"); v != "" {
				return v, nil
			}
			if base == "" {
				return "", &core.ConfigurationError{Message: slug + "_api_base is required"}
			}
			return base, nil
		},
		Headers: func(p core.Provider) map[string]string {
			h := map[string]string{"Authorization": "Bearer " + p.Config().Get(slug+"_api_key")}
			maps.Copy(h, extra)
			return h
		},
	}
}

func chatOnly() (map[string]core.ProtocolFactory, []string) {
	return map[string]core.ProtocolFactory{"chat_completions": chatcompletions.New}, []string{"chat_completions"}
}

// bearerIfSet sends a token only when the option is configured (local
// servers started without --api-key).
func bearerIfSet(option string) func(p core.Provider) map[string]string {
	return func(p core.Provider) map[string]string {
		if k := p.Config().Get(option); k != "" {
			return map[string]string{"Authorization": "Bearer " + k}
		}
		return map[string]string{}
	}
}

func optionBase(option string) func(p core.Provider) (string, error) {
	return func(p core.Provider) (string, error) {
		if v := p.Config().Get(option); v != "" {
			return v, nil
		}
		return "", &core.ConfigurationError{Message: option + " is required"}
	}
}
