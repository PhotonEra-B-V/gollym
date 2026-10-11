package core

import (
	"maps"
	"os"
	"strings"
	"sync"
	"time"
)

// Config holds global and provider-specific options (pyllym's Configuration).
//
// System options are typed fields. Provider options (openai_api_key,
// zhipu_api_base, ...) live in a string bag: reading one that was never set
// falls back to the uppercase environment variable of the same name
// (openai_api_key -> OPENAI_API_KEY), then "". Values set in code win, and
// setting "" restores the environment fallback.
type Config struct {
	DefaultModel              string
	DefaultEmbeddingModel     string
	DefaultModerationModel    string
	DefaultDecisionModel      string
	DefaultImageModel         string
	DefaultSpeechModel        string
	DefaultTranscriptionModel string
	DefaultLayoutParsingModel string
	DefaultVideoModel         string

	// ModelRegistryFile overrides the embedded models.json when set.
	ModelRegistryFile string

	RequestTimeout time.Duration
	MaxRetries     int
	// RetryInterval is the first backoff for transient failures.
	RetryInterval           time.Duration
	RetryBackoffFactor      float64
	RetryIntervalRandomness float64
	// RateLimitRetryInterval is the first backoff for 429s without a
	// Retry-After hint.
	RateLimitRetryInterval time.Duration
	// RetryMaxInterval caps a single wait; a server asking for longer is not
	// retried at all. Zero disables the cap.
	RetryMaxInterval time.Duration

	HTTPProxy            string
	ToolConcurrency      bool
	AutoUploadLargeFiles bool
	DeprecationBehavior  string
	LogStreamDebug       bool

	mu   sync.RWMutex
	opts map[string]string
}

// NewConfig returns a Config with pyllym's system defaults.
func NewConfig() *Config {
	return &Config{
		DefaultModel:              "gpt-5.4",
		DefaultEmbeddingModel:     "text-embedding-3-small",
		DefaultModerationModel:    "omni-moderation-latest",
		DefaultDecisionModel:      "jev-latest",
		DefaultImageModel:         "gpt-image-1.5",
		DefaultSpeechModel:        "gpt-4o-mini-tts",
		DefaultTranscriptionModel: "whisper-1",
		DefaultLayoutParsingModel: "glm-ocr",
		RequestTimeout:            300 * time.Second,
		MaxRetries:                3,
		RetryInterval:             100 * time.Millisecond,
		RetryBackoffFactor:        2,
		RetryIntervalRandomness:   0.5,
		RateLimitRetryInterval:    time.Second,
		RetryMaxInterval:          60 * time.Second,
		AutoUploadLargeFiles:      true,
		DeprecationBehavior:       "warn",
		LogStreamDebug:            os.Getenv("PYLLM_STREAM_DEBUG") == "true",
		opts:                      map[string]string{},
	}
}

// Get reads a provider option, falling back to its uppercase env var.
func (c *Config) Get(name string) string {
	c.mu.RLock()
	v, ok := c.opts[name]
	c.mu.RUnlock()
	if ok {
		return v
	}
	if env := os.Getenv(strings.ToUpper(name)); strings.TrimSpace(env) != "" {
		return env
	}
	return ""
}

// Set sets a provider option. Blank values restore the env fallback.
func (c *Config) Set(name, value string) *Config {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.opts == nil {
		c.opts = map[string]string{}
	}
	if strings.TrimSpace(value) == "" {
		delete(c.opts, name)
	} else {
		c.opts[name] = value
	}
	return c
}

// Copy returns an independent copy (pyllym's Configuration.copy, used by
// Context so per-call changes never leak into the global config).
func (c *Config) Copy() *Config {
	c.mu.RLock()
	defer c.mu.RUnlock()
	cp := &Config{
		DefaultModel:              c.DefaultModel,
		DefaultEmbeddingModel:     c.DefaultEmbeddingModel,
		DefaultModerationModel:    c.DefaultModerationModel,
		DefaultDecisionModel:      c.DefaultDecisionModel,
		DefaultImageModel:         c.DefaultImageModel,
		DefaultSpeechModel:        c.DefaultSpeechModel,
		DefaultTranscriptionModel: c.DefaultTranscriptionModel,
		DefaultLayoutParsingModel: c.DefaultLayoutParsingModel,
		DefaultVideoModel:         c.DefaultVideoModel,
		ModelRegistryFile:         c.ModelRegistryFile,
		RequestTimeout:            c.RequestTimeout,
		MaxRetries:                c.MaxRetries,
		RetryInterval:             c.RetryInterval,
		RetryBackoffFactor:        c.RetryBackoffFactor,
		RetryIntervalRandomness:   c.RetryIntervalRandomness,
		RateLimitRetryInterval:    c.RateLimitRetryInterval,
		RetryMaxInterval:          c.RetryMaxInterval,
		HTTPProxy:                 c.HTTPProxy,
		ToolConcurrency:           c.ToolConcurrency,
		AutoUploadLargeFiles:      c.AutoUploadLargeFiles,
		DeprecationBehavior:       c.DeprecationBehavior,
		LogStreamDebug:            c.LogStreamDebug,
		opts:                      maps.Clone(c.opts),
	}
	if cp.opts == nil {
		cp.opts = map[string]string{}
	}
	return cp
}

var secretSuffixes = []string{"_id", "_key", "_secret", "_token", "_credential_provider"}

// IsSecretOption reports whether an option name holds a credential, so it is
// redacted from dumps and logs.
func IsSecretOption(name string) bool {
	for _, s := range secretSuffixes {
		if strings.HasSuffix(name, s) {
			return true
		}
	}
	return false
}

// Options returns the provider options set in code (secrets redacted unless
// redact is false).
func (c *Config) Options(redact bool) map[string]string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := map[string]string{}
	for k, v := range c.opts {
		if redact && IsSecretOption(k) {
			continue
		}
		out[k] = v
	}
	return out
}

var (
	globalMu  sync.Mutex
	globalCfg *Config
)

// Global returns the process-wide configuration singleton.
func Global() *Config {
	globalMu.Lock()
	defer globalMu.Unlock()
	if globalCfg == nil {
		globalCfg = NewConfig()
	}
	return globalCfg
}

// Configure applies fn to the global configuration.
func Configure(fn func(*Config)) *Config {
	c := Global()
	fn(c)
	return c
}
