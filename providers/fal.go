package providers

import (
	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/protocols/fal"
)

// fal.ai hosted image and video models, referenced by fal id (fal-ai/flux/dev,
// fal-ai/qwen-image, fal-ai/ltx-video-13b-distilled, fal-ai/wan-t2v, ...).
func falSpec() *core.ProviderSpec {
	return &core.ProviderSpec{
		Slug: "fal", Name: "Fal",
		Protocols:     map[string]core.ProtocolFactory{"fal": fal.New},
		ProtocolOrder: []string{"fal"},
		Options:       []string{"fal_api_key", "fal_api_base", "fal_queue_base"},
		Requirements:  []string{"fal_api_key"},
		AssumeModels:  true,
		APIBase: func(p core.Provider) (string, error) {
			if v := p.Config().Get("fal_api_base"); v != "" {
				return v, nil
			}
			return "https://fal.run", nil
		},
		Headers: func(p core.Provider) map[string]string {
			return map[string]string{"Authorization": "Key " + p.Config().Get("fal_api_key")}
		},
	}
}
