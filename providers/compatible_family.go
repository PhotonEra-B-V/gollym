package providers

import (
	"regexp"
	"strings"

	"github.com/PhotonEra-B-V/gollym/core"
)

func deepSeekSpec() *core.ProviderSpec {
	return compatible{slug: "deepseek", name: "DeepSeek", defaultBase: "https://api.deepseek.com", assumeModels: true}.spec()
}

func mistralSpec() *core.ProviderSpec {
	return compatible{slug: "mistral", name: "Mistral", defaultBase: "https://api.mistral.ai/v1"}.spec()
}

func openRouterSpec() *core.ProviderSpec {
	return compatible{slug: "openrouter", name: "OpenRouter", defaultBase: "https://openrouter.ai/api/v1"}.spec()
}

var htmlTitleRe = regexp.MustCompile(`<title>(.+?)</title>`)
var leadingStatusRe = regexp.MustCompile(`^\d+\s+`)

func perplexitySpec() *core.ProviderSpec {
	return compatible{
		slug: "perplexity", name: "Perplexity", defaultBase: "https://api.perplexity.ai",
		extraHeaders: map[string]string{"Content-Type": "application/json"},
		// Perplexity returns HTML for auth errors.
		parseError: func(_ core.Provider, r *core.Response) string {
			if r == nil || !core.Truthy(r.Body) {
				return ""
			}
			if s, ok := r.Body.(string); ok && containsAll(s, "<html>", "<title>") {
				if m := htmlTitleRe.FindStringSubmatch(s); m != nil {
					return leadingStatusRe.ReplaceAllString(m[1], "")
				}
			}
			return core.ParseErrorBody(r.Body)
		},
	}.spec()
}

func xaiSpec() *core.ProviderSpec {
	return compatible{slug: "xai", name: "XAI", defaultBase: "https://api.x.ai/v1",
		extraHeaders: map[string]string{"Content-Type": "application/json"}}.spec()
}

func nvidiaSpec() *core.ProviderSpec {
	return compatible{slug: "nvidia", name: "NVIDIA", defaultBase: "https://integrate.api.nvidia.com/v1", assumeModels: true}.spec()
}

func cerebrasSpec() *core.ProviderSpec {
	return compatible{slug: "cerebras", name: "Cerebras", defaultBase: "https://api.cerebras.ai/v1", assumeModels: true}.spec()
}

func huggingFaceSpec() *core.ProviderSpec {
	return compatible{slug: "huggingface", name: "HuggingFace", defaultBase: "https://router.huggingface.co/v1", assumeModels: true}.spec()
}

// Databricks needs a per-workspace base URL
// (https://<workspace>.cloud.databricks.com/serving-endpoints).
func databricksSpec() *core.ProviderSpec {
	return compatible{slug: "databricks", name: "Databricks", assumeModels: true}.spec()
}

// Qwen defaults to the international DashScope endpoint; set qwen_api_base
// to https://dashscope.aliyuncs.com/compatible-mode/v1 for China.
func qwenSpec() *core.ProviderSpec {
	return compatible{slug: "qwen", name: "Qwen", defaultBase: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", assumeModels: true}.spec()
}

// Moonshot (Kimi); https://api.moonshot.cn/v1 for China.
func moonshotSpec() *core.ProviderSpec {
	return compatible{slug: "moonshot", name: "Moonshot", defaultBase: "https://api.moonshot.ai/v1", assumeModels: true}.spec()
}

func doubaoSpec() *core.ProviderSpec {
	return compatible{slug: "doubao", name: "Doubao", defaultBase: "https://ark.cn-beijing.volces.com/api/v3", assumeModels: true}.spec()
}

func ernieSpec() *core.ProviderSpec {
	return compatible{slug: "ernie", name: "ERNIE", defaultBase: "https://qianfan.baidubce.com/v2", assumeModels: true}.spec()
}

// MiniMax; https://api.minimaxi.com/v1 for China.
func miniMaxSpec() *core.ProviderSpec {
	return compatible{slug: "minimax", name: "MiniMax", defaultBase: "https://api.minimax.io/v1", assumeModels: true}.spec()
}

func containsAll(s string, subs ...string) bool {
	for _, sub := range subs {
		if !strings.Contains(s, sub) {
			return false
		}
	}
	return true
}
