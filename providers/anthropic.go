package providers

import (
	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/protocols/anthropic"
)

// anthropicVersion is the Messages API version header pyllym sends.
const anthropicVersion = "2023-06-01"

// Anthropic Claude via the Messages API.
func anthropicSpec() *core.ProviderSpec {
	return &core.ProviderSpec{
		Slug: "anthropic", Name: "Anthropic",
		Protocols:     map[string]core.ProtocolFactory{"anthropic": anthropic.New},
		ProtocolOrder: []string{"anthropic"},
		Options:       []string{"anthropic_api_key", "anthropic_api_base"},
		Requirements:  []string{"anthropic_api_key"},
		Capabilities:  core.DefaultCapabilities{},
		APIBase: func(p core.Provider) (string, error) {
			if v := p.Config().Get("anthropic_api_base"); v != "" {
				return v, nil
			}
			return "https://api.anthropic.com", nil
		},
		Headers: func(p core.Provider) map[string]string {
			return map[string]string{
				"x-api-key":         p.Config().Get("anthropic_api_key"),
				"anthropic-version": anthropicVersion,
			}
		},
	}
}
