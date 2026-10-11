// Package providers registers every built-in provider with core. Import it
// (the gollym facade does) before resolving models.
package providers

import "github.com/PhotonEra-B-V/gollym/core"

// builtins in pyllym's registration order. A nil spec (a provider whose
// protocol isn't ported yet) is skipped.
var builtins = []func() *core.ProviderSpec{
	anthropicSpec,
	openAISpec,
	geminiSpec,
	deepSeekSpec,
	mistralSpec,
	ollamaSpec,
	openRouterSpec,
	perplexitySpec,
	xaiSpec,
	gpuStackSpec,
	vllmSpec,
	azureSpec,
	bedrockSpec,
	vertexAISpec,
	nvidiaSpec,
	cerebrasSpec,
	huggingFaceSpec,
	databricksSpec,
	qwenSpec,
	zhipuSpec,
	moonshotSpec,
	doubaoSpec,
	ernieSpec,
	miniMaxSpec,
	falSpec,
	typeSafeSpec,
}

func init() {
	for _, f := range builtins {
		if spec := f(); spec != nil {
			core.RegisterProvider(spec)
		}
	}
}
