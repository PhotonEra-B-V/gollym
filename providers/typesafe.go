package providers

import (
	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/protocols/systemone"
)

// TypeSafe AI (Jev, the "System One" decision model): typed questions
// answered with calibrated distributions via Decide; no chat. Bearer auth
// (typesafe_api_key); set typesafe_api_base for a gateway such as LiteLLM.
func typeSafeSpec() *core.ProviderSpec {
	return compatible{
		slug: "typesafe", name: "TypeSafe", defaultBase: "https://api.typesafe.ai/v1",
		// Jev versions ship faster than the registry; unknown ids pass through.
		assumeModels: true,
		protocols:    map[string]core.ProtocolFactory{"systemone": systemone.New},
		order:        []string{"systemone"},
	}.spec()
}
