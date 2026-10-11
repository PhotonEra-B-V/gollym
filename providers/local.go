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

// vLLM's OpenAI-compatible server. Its NLI /classify decisions protocol
// (OpenJev) replaces chat_completions here once ported.
func vllmSpec() *core.ProviderSpec {
	protocols, order := chatOnly()
	return &core.ProviderSpec{
		Slug: "vllm", Name: "VLLM", Protocols: protocols, ProtocolOrder: order,
		Options: []string{"vllm_api_base", "vllm_api_key"}, Requirements: []string{"vllm_api_base"},
		Local: true, AssumeModels: true, APIBase: optionBase("vllm_api_base"), Headers: bearerIfSet("vllm_api_key"),
	}
}

// Zhipu GLM via z.ai (https://open.bigmodel.cn/api/paas/v4 for China). Its
// glm-ocr layout_parsing protocol replaces chat_completions once ported.
func zhipuSpec() *core.ProviderSpec {
	return compatible{slug: "zhipu", name: "Zhipu", defaultBase: "https://api.z.ai/api/paas/v4", assumeModels: true}.spec()
}
