package providers

import "github.com/PhotonEra-B-V/gollym/core"

func ollamaSpec() *core.ProviderSpec {
	protocols, order := chatOnly()
	return &core.ProviderSpec{
		Slug: "ollama", Name: "Ollama", Protocols: protocols, ProtocolOrder: order,
		Options: []string{"ollama_api_base", "ollama_api_key"}, Requirements: []string{"ollama_api_base"},
		Local: true, APIBase: optionBase("ollama_api_base"), Headers: bearerIfSet("ollama_api_key"),
	}
}

func gpuStackSpec() *core.ProviderSpec {
	protocols, order := chatOnly()
	return &core.ProviderSpec{
		Slug: "gpustack", Name: "GPUStack", Protocols: protocols, ProtocolOrder: order,
		Options: []string{"gpustack_api_base", "gpustack_api_key"}, Requirements: []string{"gpustack_api_base"},
		Local: true, APIBase: optionBase("gpustack_api_base"), Headers: bearerIfSet("gpustack_api_key"),
	}
}

// Azure assumes azure_api_base already points at an OpenAI v1-compatible
// endpoint (pyllym's simplified Azure).
func azureSpec() *core.ProviderSpec {
	protocols, order := chatOnly()
	return &core.ProviderSpec{
		Slug: "azure", Name: "Azure", Protocols: protocols, ProtocolOrder: order,
		Options:      []string{"azure_api_base", "azure_api_key", "azure_ai_auth_token", "azure_api_version"},
		Requirements: []string{"azure_api_base"},
		APIBase:      optionBase("azure_api_base"),
		Headers: func(p core.Provider) map[string]string {
			if k := p.Config().Get("azure_api_key"); k != "" {
				return map[string]string{"api-key": k}
			}
			return map[string]string{"Authorization": "Bearer " + p.Config().Get("azure_ai_auth_token")}
		},
	}
}
