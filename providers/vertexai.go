package providers

import (
	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/protocols/gemini"
)

// vertexAIMissingToken is pyllym's guidance when no access token is set.
const vertexAIMissingToken = "Vertex AI requires an OAuth access token. pyllym does not mint Google " +
	"credentials; supply a pre-obtained token via " +
	"config.vertexai_access_token (e.g. from " +
	"`gcloud auth print-access-token`)."

// vertexAISpec hosts the Gemini family behind project/location-scoped
// endpoints. Token minting is not implemented: a pre-obtained
// vertexai_access_token is required (pyllym raises from its headers
// property; Headers can't fail here, so APIBase reports the
// ConfigurationError before any request is sent).
func vertexAISpec() *core.ProviderSpec {
	return &core.ProviderSpec{
		Slug: "vertexai", Name: "VertexAI",
		Protocols:     map[string]core.ProtocolFactory{"gemini": vertexAIGemini},
		ProtocolOrder: []string{"gemini"},
		Options: []string{
			"vertexai_project_id", "vertexai_location", "vertexai_access_token",
			"vertexai_service_account_key", "vertexai_api_base",
		},
		Requirements: []string{"vertexai_project_id", "vertexai_location"},
		// Only the Gemini protocol is ported; publisher-prefixed MaaS models
		// (anthropic/mistral/openai-compatible) are not supported, as in pyllym.
		APIBase: func(p core.Provider) (string, error) {
			cfg := p.Config()
			if cfg.Get("vertexai_access_token") == "" {
				return "", &core.ConfigurationError{Message: vertexAIMissingToken}
			}
			if v := cfg.Get("vertexai_api_base"); v != "" {
				return v, nil
			}
			location := cfg.Get("vertexai_location")
			if location == "global" {
				return "https://aiplatform.googleapis.com/v1beta1", nil
			}
			return "https://" + location + "-aiplatform.googleapis.com/v1beta1", nil
		},
		Headers: func(p core.Provider) map[string]string {
			if token := p.Config().Get("vertexai_access_token"); token != "" {
				return map[string]string{"Authorization": "Bearer " + token}
			}
			return map[string]string{}
		},
	}
}

// VertexAILocationPath is projects/{project}/locations/{location}.
func VertexAILocationPath(cfg *core.Config) string {
	return "projects/" + cfg.Get("vertexai_project_id") + "/locations/" + cfg.Get("vertexai_location")
}

// VertexAIModelPath is the publisher model path of a Google model.
func VertexAIModelPath(cfg *core.Config, model string) string {
	return VertexAILocationPath(cfg) + "/publishers/google/models/" + model
}

// vertexAIGemini is the Gemini protocol over Vertex AI's paths:
// {model path}:generateContent, :streamGenerateContent?alt=sse and :predict.
func vertexAIGemini(p core.Provider, model *core.Info) core.Protocol {
	cfg := p.Config()
	return &gemini.Protocol{
		Provider: p, Model: model,
		ModelPath:   func(id string) string { return VertexAIModelPath(cfg, id) },
		EmbedMethod: "predict",
	}
}
