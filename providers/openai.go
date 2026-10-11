package providers

import "github.com/PhotonEra-B-V/gollym/core"

func openAISpec() *core.ProviderSpec {
	protocols, order := chatOnly()
	return &core.ProviderSpec{
		Slug: "openai", Name: "OpenAI", Protocols: protocols, ProtocolOrder: order,
		Options:       []string{"openai_api_key", "openai_api_base", "openai_organization_id", "openai_project_id", "openai_use_system_role"},
		Requirements:  []string{"openai_api_key"},
		DeveloperRole: true,
		Capabilities:  core.DefaultCapabilities{}, // OpenAICapabilities (the base heuristics)
		APIBase: func(p core.Provider) (string, error) {
			if v := p.Config().Get("openai_api_base"); v != "" {
				return v, nil
			}
			return "https://api.openai.com/v1", nil
		},
		Headers: func(p core.Provider) map[string]string {
			cfg := p.Config()
			h := map[string]string{"Authorization": "Bearer " + cfg.Get("openai_api_key")}
			if v := cfg.Get("openai_organization_id"); v != "" {
				h["OpenAI-Organization"] = v
			}
			if v := cfg.Get("openai_project_id"); v != "" {
				h["OpenAI-Project"] = v
			}
			return h
		},
	}
}
