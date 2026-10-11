package providers

import (
	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/protocols/zhipulayout"
)

// Zhipu GLM via the z.ai API (OpenAI-compatible). Set zhipu_api_base to
// https://open.bigmodel.cn/api/paas/v4 for the China region. Serves the GLM
// family plus glm-ocr layout parsing on z.ai's own layout_parsing endpoint.
func zhipuSpec() *core.ProviderSpec {
	return compatible{
		slug: "zhipu", name: "Zhipu", defaultBase: "https://api.z.ai/api/paas/v4", assumeModels: true,
		protocols: map[string]core.ProtocolFactory{"chat_completions": zhipulayout.New},
		order:     []string{"chat_completions"},
	}.spec()
}
