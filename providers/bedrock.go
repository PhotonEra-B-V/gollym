package providers

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"

	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/protocols/converse"
)

// bedrockNow is the signing clock (fixed in tests).
var bedrockNow = time.Now

var bedrockSigner = v4.NewSigner()

// bedrockSpec is AWS Bedrock over the Converse API. Every request is SigV4
// signed with static keys from bedrock_api_key / bedrock_secret_key (and
// bedrock_session_token for temporary credentials). Model ids, including
// cross-region inference profiles (us.anthropic...), go into the URL as is.
func bedrockSpec() *core.ProviderSpec {
	return &core.ProviderSpec{
		Slug: "bedrock", Name: "Bedrock",
		Protocols:       map[string]core.ProtocolFactory{"converse": converse.New},
		ProtocolOrder:   []string{"converse"},
		DefaultProtocol: "converse",
		Options:         []string{"bedrock_api_key", "bedrock_secret_key", "bedrock_region", "bedrock_session_token", "bedrock_api_base"},
		Requirements:    []string{"bedrock_api_key", "bedrock_secret_key", "bedrock_region"},
		APIBase: func(p core.Provider) (string, error) {
			if v := p.Config().Get("bedrock_api_base"); v != "" {
				return v, nil
			}
			return "https://bedrock-runtime." + p.Config().Get("bedrock_region") + ".amazonaws.com", nil
		},
		ParseError: bedrockParseError,
		Sign:       bedrockSign,
	}
}

// bedrockParseError prefers AWS's message / Message / __type fields.
func bedrockParseError(_ core.Provider, r *core.Response) string {
	if r == nil || r.Body == nil {
		return ""
	}
	switch b := r.Body.(type) {
	case string:
		if b == "" {
			return ""
		}
	case map[string]any:
		for _, k := range []string{"message", "Message", "error", "__type"} {
			if s, ok := b[k].(string); ok && s != "" {
				return s
			}
		}
	}
	return core.ParseErrorBody(r.Body)
}

// bedrockSign adds the SigV4 Authorization, X-Amz-Date,
// X-Amz-Content-Sha256 and (with a session token) X-Amz-Security-Token
// headers. It runs per attempt, so retries carry a fresh date.
func bedrockSign(p core.Provider, req *http.Request, body []byte) error {
	cfg := p.Config()
	sum := sha256.Sum256(body)
	payloadHash := hex.EncodeToString(sum[:])
	req.Header.Set("X-Amz-Content-Sha256", payloadHash)
	creds := aws.Credentials{
		AccessKeyID:     cfg.Get("bedrock_api_key"),
		SecretAccessKey: cfg.Get("bedrock_secret_key"),
		SessionToken:    cfg.Get("bedrock_session_token"),
	}
	return bedrockSigner.SignHTTP(req.Context(), creds, req, payloadHash, "bedrock", cfg.Get("bedrock_region"), bedrockNow().UTC())
}
