package providers_test

import (
	"reflect"
	"slices"
	"testing"

	"github.com/PhotonEra-B-V/gollym/core"
)

// The OpenAI and OpenAI-compatible provider specs match pyllym's provider
// classes (dumped from pyllym 1.16.0's registry).

// matrixRegistryOrder is pyllym's registration order.
var matrixRegistryOrder = []string{"anthropic", "openai", "gemini", "deepseek", "mistral", "ollama", "openrouter", "perplexity", "xai", "gpustack", "vllm", "azure", "bedrock", "vertexai", "nvidia", "cerebras", "huggingface", "databricks", "qwen", "zhipu", "moonshot", "doubao", "ernie", "minimax", "fal", "typesafe"}

var matrixSpecs = []struct {
	slug, name         string
	options, reqs      []string
	local, assume, dev bool
	protocol           string
}{
	{slug: "openai", name: "OpenAI", options: []string{"openai_api_key", "openai_api_base", "openai_organization_id", "openai_project_id", "openai_use_system_role"}, reqs: []string{"openai_api_key"}, local: false, assume: false, dev: true, protocol: "chat_completions"},
	{slug: "deepseek", name: "DeepSeek", options: []string{"deepseek_api_key", "deepseek_api_base"}, reqs: []string{"deepseek_api_key"}, local: false, assume: true, dev: false, protocol: "chat_completions"},
	{slug: "mistral", name: "Mistral", options: []string{"mistral_api_key", "mistral_api_base"}, reqs: []string{"mistral_api_key"}, local: false, assume: false, dev: false, protocol: "chat_completions"},
	{slug: "openrouter", name: "OpenRouter", options: []string{"openrouter_api_key", "openrouter_api_base"}, reqs: []string{"openrouter_api_key"}, local: false, assume: false, dev: false, protocol: "chat_completions"},
	{slug: "perplexity", name: "Perplexity", options: []string{"perplexity_api_key", "perplexity_api_base"}, reqs: []string{"perplexity_api_key"}, local: false, assume: false, dev: false, protocol: "chat_completions"},
	{slug: "xai", name: "XAI", options: []string{"xai_api_key", "xai_api_base"}, reqs: []string{"xai_api_key"}, local: false, assume: false, dev: false, protocol: "chat_completions"},
	{slug: "nvidia", name: "NVIDIA", options: []string{"nvidia_api_key", "nvidia_api_base"}, reqs: []string{"nvidia_api_key"}, local: false, assume: true, dev: false, protocol: "chat_completions"},
	{slug: "cerebras", name: "Cerebras", options: []string{"cerebras_api_key", "cerebras_api_base"}, reqs: []string{"cerebras_api_key"}, local: false, assume: true, dev: false, protocol: "chat_completions"},
	{slug: "huggingface", name: "HuggingFace", options: []string{"huggingface_api_key", "huggingface_api_base"}, reqs: []string{"huggingface_api_key"}, local: false, assume: true, dev: false, protocol: "chat_completions"},
	{slug: "databricks", name: "Databricks", options: []string{"databricks_api_key", "databricks_api_base"}, reqs: []string{"databricks_api_base", "databricks_api_key"}, local: false, assume: true, dev: false, protocol: "chat_completions"},
	{slug: "qwen", name: "Qwen", options: []string{"qwen_api_key", "qwen_api_base"}, reqs: []string{"qwen_api_key"}, local: false, assume: true, dev: false, protocol: "chat_completions"},
	{slug: "moonshot", name: "Moonshot", options: []string{"moonshot_api_key", "moonshot_api_base"}, reqs: []string{"moonshot_api_key"}, local: false, assume: true, dev: false, protocol: "chat_completions"},
	{slug: "doubao", name: "Doubao", options: []string{"doubao_api_key", "doubao_api_base"}, reqs: []string{"doubao_api_key"}, local: false, assume: true, dev: false, protocol: "chat_completions"},
	{slug: "ernie", name: "ERNIE", options: []string{"ernie_api_key", "ernie_api_base"}, reqs: []string{"ernie_api_key"}, local: false, assume: true, dev: false, protocol: "chat_completions"},
	{slug: "minimax", name: "MiniMax", options: []string{"minimax_api_key", "minimax_api_base"}, reqs: []string{"minimax_api_key"}, local: false, assume: true, dev: false, protocol: "chat_completions"},
}

func TestMatrixProviderSpecsMatchPyllym(t *testing.T) {
	for _, want := range matrixSpecs {
		spec, ok := core.LookupProvider(want.slug)
		if !ok {
			t.Errorf("%s not registered", want.slug)
			continue
		}
		got := []any{spec.DisplayName(), spec.Options, spec.Requirements, spec.Local, spec.AssumeModels, spec.DeveloperRole, spec.ProtocolOrder[0]}
		exp := []any{want.name, want.options, want.reqs, want.local, want.assume, want.dev, want.protocol}
		if !reflect.DeepEqual(got, exp) {
			t.Errorf("%s: %v, pyllym %v", want.slug, got, exp)
		}
	}
}

// Registered providers keep pyllym's order (unported ones are skipped).
func TestMatrixRegistrationOrder(t *testing.T) {
	var want []string
	for _, s := range matrixRegistryOrder {
		if slices.Contains(core.ProviderSlugs(), s) {
			want = append(want, s)
		}
	}
	if got := core.ProviderSlugs(); !reflect.DeepEqual(got, want) {
		t.Fatalf("order %v, want %v", got, want)
	}
}
