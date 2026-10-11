package core

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
)

// Capabilities are per-provider heuristics used when a model list returns
// bare ids (the packaged registry is the primary source of truth).
type Capabilities interface {
	ContextWindowFor(modelID string) *int
	MaxTokensFor(modelID string) *int
	CriticalCapabilitiesFor(modelID string) []string
	PricingFor(modelID string) map[string]any
	ModelType(modelID string) string
	ModelFamily(modelID string) string
}

// DefaultCapabilities returns conservative defaults.
type DefaultCapabilities struct{}

func (DefaultCapabilities) ContextWindowFor(string) *int            { return nil }
func (DefaultCapabilities) MaxTokensFor(string) *int                { return nil }
func (DefaultCapabilities) CriticalCapabilitiesFor(string) []string { return []string{"streaming"} }
func (DefaultCapabilities) PricingFor(string) map[string]any        { return map[string]any{} }
func (DefaultCapabilities) ModelType(string) string                 { return "chat" }
func (DefaultCapabilities) ModelFamily(string) string               { return "" }

// ProtocolFactory builds a protocol bound to a provider and (for chat) a
// model. model is nil for model-less operations (embeddings, list models).
type ProtocolFactory func(p Provider, model *Info) Protocol

// ProviderSpec declares a provider: where it lives, how it authenticates and
// which protocols it speaks. Function fields replace pyllym's overridable
// methods; nil means the default.
type ProviderSpec struct {
	Slug string
	// Name is the display name (defaults to Slug).
	Name string
	// Protocols by name; DefaultProtocol defaults to the only/first entry
	// listed in ProtocolOrder.
	Protocols       map[string]ProtocolFactory
	ProtocolOrder   []string
	DefaultProtocol string

	// Options are the provider's config keys (e.g. openai_api_key); the
	// <slug>_protocol key is added automatically.
	Options []string
	// Requirements must be set (in code or env) for the provider to work.
	Requirements []string

	Local         bool
	AssumeModels  bool
	DeveloperRole bool // send system messages as OpenAI's "developer" role

	Capabilities Capabilities

	APIBase    func(p Provider) (string, error)
	Headers    func(p Provider) map[string]string
	ParseError func(p Provider, r *Response) string
	// ProtocolFor picks a protocol per model when none is configured.
	ProtocolFor func(p Provider, model *Info) string
	// Sign signs an outgoing request (Bedrock SigV4); body is the payload.
	Sign func(p Provider, req *http.Request, body []byte) error
	// Wrap lets a provider return its own type (embedding *BaseProvider) to
	// add optional interfaces such as FileProvider.
	Wrap func(base *BaseProvider) Provider
}

// DisplayName is Name or Slug.
func (s *ProviderSpec) DisplayName() string {
	if s.Name != "" {
		return s.Name
	}
	return s.Slug
}

func (s *ProviderSpec) defaultProtocol() string {
	if s.DefaultProtocol != "" {
		return s.DefaultProtocol
	}
	if len(s.ProtocolOrder) > 0 {
		return s.ProtocolOrder[0]
	}
	for name := range s.Protocols {
		return name
	}
	return ""
}

// IsConfigured reports whether every requirement has a value.
func (s *ProviderSpec) IsConfigured(cfg *Config) bool {
	for _, r := range s.Requirements {
		if cfg.Get(r) == "" {
			return false
		}
	}
	return true
}

// Provider is a configured provider instance.
type Provider interface {
	Endpoint
	Spec() *ProviderSpec
	Slug() string
	Name() string
	Config() *Config
	Connection() *Connection
	Capabilities() Capabilities
	// UsesDeveloperRole reports whether system messages become "developer".
	UsesDeveloperRole() bool
	// ProtocolName resolves the protocol for a request: explicit name, then
	// the <slug>_protocol option, then the per-model choice, then default.
	ProtocolName(explicit string, model *Info) (string, error)
	// Protocol builds the named protocol.
	Protocol(name string, model *Info) (Protocol, error)
}

// BaseProvider implements Provider from a spec.
type BaseProvider struct {
	spec *ProviderSpec
	cfg  *Config
	conn *Connection
	self Provider
}

func (b *BaseProvider) Spec() *ProviderSpec { return b.spec }
func (b *BaseProvider) Slug() string        { return b.spec.Slug }
func (b *BaseProvider) Name() string        { return b.spec.DisplayName() }
func (b *BaseProvider) Config() *Config     { return b.cfg }
func (b *BaseProvider) Connection() *Connection {
	return b.conn
}

// Self is the outermost provider value (the Wrap result).
func (b *BaseProvider) Self() Provider { return b.self }

func (b *BaseProvider) Capabilities() Capabilities {
	if b.spec.Capabilities != nil {
		return b.spec.Capabilities
	}
	return DefaultCapabilities{}
}

func (b *BaseProvider) UsesDeveloperRole() bool { return b.spec.DeveloperRole }

func (b *BaseProvider) APIBase() (string, error) {
	if b.spec.APIBase == nil {
		return "", &ConfigurationError{Message: b.spec.Slug + " has no api_base"}
	}
	return b.spec.APIBase(b.self)
}

func (b *BaseProvider) Headers() map[string]string {
	if b.spec.Headers == nil {
		return map[string]string{}
	}
	return b.spec.Headers(b.self)
}

func (b *BaseProvider) ParseError(r *Response) string {
	if b.spec.ParseError != nil {
		return b.spec.ParseError(b.self, r)
	}
	if r == nil || r.Body == nil {
		return ""
	}
	if s, ok := r.Body.(string); ok && s == "" {
		return ""
	}
	return ParseErrorBody(r.Body)
}

// Sign implements RequestSigner when the spec signs requests.
func (b *BaseProvider) Sign(req *http.Request, body []byte) error {
	if b.spec.Sign == nil {
		return nil
	}
	return b.spec.Sign(b.self, req, body)
}

func (b *BaseProvider) ProtocolName(explicit string, model *Info) (string, error) {
	name := explicit
	if name == "" {
		name = b.cfg.Get(b.spec.Slug + "_protocol")
	}
	if name == "" && b.spec.ProtocolFor != nil {
		name = b.spec.ProtocolFor(b.self, model)
	}
	if name == "" {
		name = b.spec.defaultProtocol()
	}
	if _, ok := b.spec.Protocols[name]; !ok {
		names := make([]string, 0, len(b.spec.Protocols))
		for n := range b.spec.Protocols {
			names = append(names, n)
		}
		sort.Strings(names)
		return "", &Error{Kind: KindError, Message: fmt.Sprintf("%s is not a protocol of %s. Available: %s", name, b.spec.DisplayName(), strings.Join(names, ", "))}
	}
	return name, nil
}

// DefaultProtocolName is the configured <slug>_protocol, else the spec's
// default; unlike ProtocolName it never asks spec.ProtocolFor (pyllym's
// _default_protocol, used by embeddings, images and the other model-less
// operations).
func (b *BaseProvider) DefaultProtocolName() string {
	if name := b.cfg.Get(b.spec.Slug + "_protocol"); name != "" {
		return name
	}
	return b.spec.defaultProtocol()
}

func (b *BaseProvider) Protocol(name string, model *Info) (Protocol, error) {
	resolved, err := b.ProtocolName(name, model)
	if err != nil {
		return nil, err
	}
	return b.spec.Protocols[resolved](b.self, model), nil
}

// RequestSigner signs outgoing requests (checked by Connection).
type RequestSigner interface {
	Sign(req *http.Request, body []byte) error
}

// FileProvider manages provider-side files.
type FileProvider interface {
	UploadFile(ctx context.Context, file any, opts map[string]any) (*UploadedFile, error)
	FindFile(ctx context.Context, id string) (*UploadedFile, error)
	DownloadFile(ctx context.Context, id string) ([]byte, error)
}

var (
	registryMu sync.RWMutex
	providers  = map[string]*ProviderSpec{}
	provOrder  []string
)

// RegisterProvider adds a provider under spec.Slug.
func RegisterProvider(spec *ProviderSpec) {
	registryMu.Lock()
	defer registryMu.Unlock()
	if _, ok := providers[spec.Slug]; !ok {
		provOrder = append(provOrder, spec.Slug)
	}
	providers[spec.Slug] = spec
}

// LookupProvider returns a registered spec.
func LookupProvider(slug string) (*ProviderSpec, bool) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	s, ok := providers[slug]
	return s, ok
}

// MustLookupProvider is pyllym's resolve_bang.
func MustLookupProvider(slug string) (*ProviderSpec, error) {
	if s, ok := LookupProvider(slug); ok {
		return s, nil
	}
	return nil, &Error{Kind: KindError, Message: fmt.Sprintf("Unknown provider: %s. Available providers: %s", pyQuote(slug), strings.Join(ProviderSlugs(), ", "))}
}

// ProviderSlugs lists registered providers in registration order.
func ProviderSlugs() []string {
	registryMu.RLock()
	defer registryMu.RUnlock()
	return append([]string(nil), provOrder...)
}

// ProviderSpecs lists registered specs in registration order.
func ProviderSpecs() []*ProviderSpec {
	registryMu.RLock()
	defer registryMu.RUnlock()
	out := make([]*ProviderSpec, 0, len(provOrder))
	for _, s := range provOrder {
		out = append(out, providers[s])
	}
	return out
}

// ConfiguredProviders lists specs whose requirements are met.
func ConfiguredProviders(cfg *Config, remoteOnly bool) []*ProviderSpec {
	var out []*ProviderSpec
	for _, s := range ProviderSpecs() {
		if remoteOnly && s.Local {
			continue
		}
		if s.IsConfigured(cfg) {
			out = append(out, s)
		}
	}
	return out
}

// NewProvider instantiates a provider, failing when it isn't configured.
func NewProvider(spec *ProviderSpec, cfg *Config) (Provider, error) {
	if !spec.IsConfigured(cfg) {
		var lines []string
		for _, r := range spec.Requirements {
			if cfg.Get(r) == "" {
				lines = append(lines, fmt.Sprintf("  cfg.Set(%q, os.Getenv(%q))", r, strings.ToUpper(r)))
			}
		}
		return nil, &ConfigurationError{Message: fmt.Sprintf(
			"%s provider is not configured. Add this to your initialization:\n\ngollym.Configure(func(cfg *gollym.Config) {\n%s\n})",
			spec.DisplayName(), strings.Join(lines, "\n"))}
	}
	base := &BaseProvider{spec: spec, cfg: cfg}
	base.self = base
	if spec.Wrap != nil {
		base.self = spec.Wrap(base)
	}
	base.conn = NewConnection(base.self, cfg)
	return base.self, nil
}

// ResolveModel finds a model and instantiates its provider (models.resolve).
// Local providers and providers that assume models exist accept any id.
func ResolveModel(modelID, provider string, assumeExists bool, cfg *Config) (*Info, Provider, error) {
	if cfg == nil {
		cfg = Global()
	}
	var spec *ProviderSpec
	if provider != "" {
		spec, _ = LookupProvider(provider)
		if spec != nil && (spec.Local || spec.AssumeModels) {
			assumeExists = true
		}
	}
	var model *Info
	if assumeExists {
		if provider == "" {
			return nil, nil, fmt.Errorf("Provider must be specified if assume_exists is true")
		}
		if spec == nil {
			s, err := MustLookupProvider(provider)
			if err != nil {
				return nil, nil, err
			}
			spec = s
		}
		if m, err := Models().Find(modelID, provider); err == nil {
			model = m
		} else {
			model = DefaultInfo(modelID, spec.Slug)
		}
	} else {
		m, err := Models().Find(modelID, provider)
		if err != nil {
			return nil, nil, err
		}
		model = m
		s, err := MustLookupProvider(m.Provider)
		if err != nil {
			return nil, nil, err
		}
		spec = s
	}
	p, err := NewProvider(spec, cfg)
	if err != nil {
		return nil, nil, err
	}
	return model, p, nil
}
