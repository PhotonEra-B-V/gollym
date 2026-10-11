package core

import (
	"errors"
	"strings"
	"testing"
)

// Internal rows of the ported pyllym tests.

// test_review_fixes.test_shared_client_reused_across_chats: connections
// share one pooled HTTP client per proxy setting.
func TestSharedClientReusedAcrossConnections(t *testing.T) {
	spec := &ProviderSpec{Slug: "stub"}
	a, err := NewProvider(spec, NewConfig())
	if err != nil {
		t.Fatal(err)
	}
	b, _ := NewProvider(spec, NewConfig())
	if a.Connection() == b.Connection() || a.Connection().client() != b.Connection().client() {
		t.Fatal("HTTP client not shared")
	}
	cfg := NewConfig()
	cfg.HTTPProxy = "http://proxy.local:3128"
	c, _ := NewProvider(spec, cfg)
	if c.Connection().client() == a.Connection().client() {
		t.Fatal("proxy config shares the direct client")
	}
}

func TestPyRepr(t *testing.T) {
	cases := []struct {
		in   any
		want string
	}{
		{map[string]any{"error": `Available tools: ["weather"].`}, `{'error': 'Available tools: ["weather"].'}`},
		{[]any{1.0, true, nil, "it's"}, `[1.0, True, None, "it's"]`},
		{map[string]any{"a": "x\ny", "b": `both ' and "`}, `{'a': 'x\ny', 'b': 'both \' and "'}`},
		{0.1, "0.1"},
		{1e16, "1e+16"},
		{1e-5, "1e-05"},
		{42, "42"},
		{[]string{"a"}, "['a']"},
		{map[string]int{"n": 1}, "{'n': 1}"},
	}
	for _, c := range cases {
		if got := PyRepr(c.in); got != c.want {
			t.Errorf("PyRepr(%#v) = %s, want %s", c.in, got, c.want)
		}
	}
	if PyStr("plain") != "plain" || PyStr(true) != "True" || PyStr(nil) != "None" || PyStr(2.0) != "2.0" {
		t.Error("PyStr")
	}
}

type stubProtocol struct{ name string }

func (s *stubProtocol) RenderPayload(*CompletionRequest) (map[string]any, error) { return nil, nil }
func (s *stubProtocol) ParseCompletionResponse(*Response) (*Message, error)      { return nil, nil }
func (s *stubProtocol) CompletionURL() string                                    { return "" }

func stubSpec() *ProviderSpec {
	factory := func(name string) ProtocolFactory {
		return func(Provider, *Info) Protocol { return &stubProtocol{name} }
	}
	return &ProviderSpec{
		Slug:          "stubby",
		Protocols:     map[string]ProtocolFactory{"main": factory("main"), "alt": factory("alt")},
		ProtocolOrder: []string{"main", "alt"},
		ProtocolFor:   func(Provider, *Info) string { return "alt" },
	}
}

// Model-less operations use the configured or default protocol, never the
// per-model choice (pyllym's Provider._default_protocol); chat requests do
// consult it (Provider._resolve_protocol).
func TestDefaultProtocolSkipsPerModelChoice(t *testing.T) {
	cfg := NewConfig()
	p, _ := NewProvider(stubSpec(), cfg)
	if proto, _ := ProtocolFor(p); proto.(*stubProtocol).name != "main" {
		t.Fatalf("default protocol %s", proto.(*stubProtocol).name)
	}
	if proto, _ := p.Protocol("", nil); proto.(*stubProtocol).name != "alt" {
		t.Fatalf("chat protocol %s", proto.(*stubProtocol).name)
	}
	cfg.Set("stubby_protocol", "alt")
	if proto, _ := ProtocolFor(p); proto.(*stubProtocol).name != "alt" {
		t.Fatalf("configured protocol %s", proto.(*stubProtocol).name)
	}
	cfg.Set("stubby_protocol", "nope")
	_, err := ProtocolFor(p)
	if !errors.Is(err, ErrAPI) || err.Error() != "nope is not a protocol of stubby. Available: alt, main" {
		t.Fatalf("err %v", err)
	}
}

func TestUnknownProviderIsAnAPIError(t *testing.T) {
	_, err := MustLookupProvider("nope")
	if !errors.Is(err, ErrAPI) || !strings.HasPrefix(err.Error(), "Unknown provider: 'nope'. Available providers: ") {
		t.Fatalf("err %v", err)
	}
}

// add_message runs the protocol's preprocess step, so an unknown protocol
// fails there, as in pyllym.
func TestAddMessageWithUnknownProtocolFails(t *testing.T) {
	p, _ := NewProvider(stubSpec(), NewConfig())
	chat := &Chat{Provider: p, Model: DefaultInfo("m", "stubby"), config: NewConfig()}
	chat.WithProtocol("bogus")
	if err := chat.AddUserMessage("hi"); !errors.Is(err, ErrAPI) {
		t.Fatalf("err %v", err)
	}
	chat.WithProtocol("")
	if err := chat.AddUserMessage("hi"); err != nil || len(chat.Messages) != 1 {
		t.Fatalf("err %v", err)
	}
}

func TestMIMEForNameMatchesGuessType(t *testing.T) {
	cases := map[string]string{
		"x.tgz": "application/x-tar", "x.tar.gz": "application/x-tar", "x.svgz": "image/svg+xml",
		".bashrc": "", "a.TXT": "text/plain", "https://h/p/a.png?x=1": "image/png",
		"data:image/gif;base64,R0lG": "image/gif", "data:,hi": "text/plain", "x.Z": "", "a.txt.Z": "text/plain",
	}
	for name, want := range cases {
		if got := MIMEForName(name); got != want {
			t.Errorf("MIMEForName(%q) = %q, want %q", name, got, want)
		}
	}
}
