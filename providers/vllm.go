package providers

import (
	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/protocols/nliclassify"
)

// vLLM's OpenAI-compatible server at a user-supplied endpoint (e.g.
// http://localhost:8000/v1). A bearer token is sent only when vllm_api_key
// is set. An NLI classifier on /classify (e.g. OpenJev) also answers Decide.
func vllmSpec() *core.ProviderSpec {
	return &core.ProviderSpec{
		Slug: "vllm", Name: "VLLM",
		Protocols:     map[string]core.ProtocolFactory{"chat_completions": nliclassify.New},
		ProtocolOrder: []string{"chat_completions"},
		Options:       []string{"vllm_api_base", "vllm_api_key"}, Requirements: []string{"vllm_api_base"},
		// Served models are whatever the operator loaded; not in models.json.
		Local: true, AssumeModels: true, APIBase: optionBase("vllm_api_base"), Headers: bearerIfSet("vllm_api_key"),
	}
}
