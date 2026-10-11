package providers

import (
	"net/http"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym/core"
)

// Known answers from pyllym's Bedrock.sign_headers at 2026-10-11T12:34:56Z.
// A request carrying only host + x-amz-* headers must sign identically.
func TestBedrockSigV4MatchesPyllym(t *testing.T) {
	defer func(f func() time.Time) { bedrockNow = f }(bedrockNow)
	bedrockNow = func() time.Time { return time.Date(2026, 10, 11, 12, 34, 56, 0, time.UTC) }
	const (
		path = "/model/us.anthropic.claude-sonnet-4-5-20250929-v1:0/converse"
		body = `{"messages": [{"role": "user", "content": [{"text": "hi"}]}]}`
	)
	cases := []struct {
		name, token, auth string
	}{
		{"static keys", "", "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20261011/us-east-1/bedrock/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=048a30ca79aca57e95172d82b001fbe53fac3107c9a959ce91031e63ca3af7cd"},
		{"session token", "SESSIONTOKEN", "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20261011/us-east-1/bedrock/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date;x-amz-security-token, Signature=447649912674fb9ce7e3ac8ba74b94c4ce98226e3a34ed6ebdc4e9d281782aba"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := core.NewConfig()
			cfg.Set("bedrock_api_key", "AKIDEXAMPLE")
			cfg.Set("bedrock_secret_key", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")
			cfg.Set("bedrock_region", "us-east-1")
			cfg.Set("bedrock_session_token", tc.token)
			p, err := core.NewProvider(bedrockSpec(), cfg)
			if err != nil {
				t.Fatal(err)
			}
			base, _ := p.APIBase()
			if base != "https://bedrock-runtime.us-east-1.amazonaws.com" {
				t.Fatalf("api base %q", base)
			}
			req, _ := http.NewRequest(http.MethodPost, base+path, nil)
			if err := bedrockSign(p, req, []byte(body)); err != nil {
				t.Fatal(err)
			}
			if got := req.Header.Get("Authorization"); got != tc.auth {
				t.Fatalf("Authorization\n got %s\nwant %s", got, tc.auth)
			}
			if req.Header.Get("X-Amz-Date") != "20261011T123456Z" ||
				req.Header.Get("X-Amz-Content-Sha256") != "23aab397f4d1f51b8846a1cea8ead51c4766038fe8b80730645a50f05af4d41e" ||
				req.Header.Get("X-Amz-Security-Token") != tc.token {
				t.Fatalf("headers %v", req.Header)
			}
		})
	}
}

func TestBedrockParseError(t *testing.T) {
	for _, tc := range []struct {
		body any
		want string
	}{
		{nil, ""},
		{"", ""},
		{map[string]any{"message": "lower"}, "lower"},
		{map[string]any{"Message": "upper"}, "upper"},
		{map[string]any{"error": "plain"}, "plain"},
		{map[string]any{"__type": "ValidationException"}, "ValidationException"},
		{map[string]any{"error": map[string]any{"message": "nested"}}, "nested"},
		{`{"message": "from text"}`, "from text"},
		{"not json", "not json"},
	} {
		if got := bedrockParseError(nil, &core.Response{Body: tc.body}); got != tc.want {
			t.Errorf("%v: got %q want %q", tc.body, got, tc.want)
		}
	}
}
