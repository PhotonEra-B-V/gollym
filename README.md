# gollym

One interface for every major AI provider, in Go. gollym is the Go port of
[pyllym](https://github.com/PhotonEra-B-V/pyllym) 1.16.0: the same providers,
model registry, wire formats and behaviour, with Go idioms (blocking calls
with `context.Context`, callbacks for streaming, goroutines for concurrent
tools).

Works with OpenAI, Anthropic, Google Gemini, Vertex AI, AWS Bedrock,
DeepSeek, Mistral, Ollama, vLLM, OpenRouter, Perplexity, xAI, GPUStack,
Azure, NVIDIA, Cerebras, Hugging Face, Databricks, Qwen, Zhipu GLM,
Moonshot, Doubao, ERNIE, MiniMax, fal.ai, TypeSafe and any OpenAI-compatible
API.

```go
import "github.com/PhotonEra-B-V/gollym"

gollym.Configure(func(c *gollym.Config) { c.Set("openai_api_key", os.Getenv("OPENAI_API_KEY")) })

chat, err := gollym.NewChat(gollym.ChatOptions{Model: "gpt-5.4"})
msg, err := chat.Ask(ctx, "What's the best way to learn Go?")
fmt.Println(msg.Text())
```

## Install

```bash
go get github.com/PhotonEra-B-V/gollym
```

## Features

- **Chat** with text, images, audio, PDFs and documents (`gollym.With(paths...)`)
- **Streaming** (`gollym.OnChunk(func(*gollym.Chunk) error)`)
- **Tools** (function calling), sequential or concurrent, plus an **MCP client** (`mcp`) and TOML **toolsets** (`toolset`)
- **Structured output** from a JSON Schema or a Go struct (`chat.WithSchema`)
- **Thinking / reasoning** controls (`chat.WithThinking`)
- **Decisions**: typed, calibrated answers from decision models (`gollym.Decide`)
- **Embeddings**, **image** and **video** generation, **speech**, **transcription**, **moderation**, **layout parsing**
- **Citations** normalised across providers
- A bundled **model registry** (`models.json`) with pricing and capabilities, and **cost** per message and per chat
- Retries with backoff that honour `Retry-After`; quota errors are never retried
- Optional **persistence** of chats in PostgreSQL (`persistence`) and **River** background jobs (`riverjobs`)
- **bdd** / **tdg**: TOML spec → test plan → rendered Go tests

## Configuration

Provider options are strings named `<provider>_<option>`
(`openai_api_key`, `zhipu_api_base`, ...). Unset options fall back to the
uppercase environment variable (`OPENAI_API_KEY`); values set in code win.

```go
gollym.Configure(func(c *gollym.Config) {
	c.Set("anthropic_api_key", key)
	c.RequestTimeout = 2 * time.Minute
	c.MaxRetries = 2
})

// A per-call scope with its own copy of the configuration:
ctx := gollym.NewContext(func(c *gollym.Config) { c.Set("zhipu_api_key", other) })
chat, err := ctx.NewChat(gollym.ChatOptions{Model: "glm-5.3", Provider: "zhipu", AssumeModelExists: true})
```

## Examples

```go
// Instructions, temperature and provider params deep-merged into the payload.
chat.WithInstructions("You are a careful analyst.").
	WithTemperature(0.1).
	WithParams(map[string]any{"max_tokens": 4096, "response_format": map[string]any{"type": "json_object"}})

// Structured output from a struct.
type Invoice struct {
	Number string  `json:"number"`
	Total  float64 `json:"total"`
}
chat.WithSchema(Invoice{})
msg, err := chat.Ask(ctx, "Extract the invoice.", gollym.With("invoice.pdf"))
fields := msg.Content.(map[string]any)

// A tool.
weather := gollym.NewTool("weather", "Current weather in a city",
	[]gollym.Parameter{{Name: "city", Required: true}},
	func(ctx context.Context, args map[string]any) (any, error) { return "-5°C, snow", nil })
chat.WithTool(weather)

// Streaming.
_, err = chat.Ask(ctx, "Tell me a story", gollym.OnChunk(func(c *gollym.Chunk) error {
	fmt.Print(c.Text())
	return nil
}))

// Cost so far (USD, from the registry's pricing).
if total := chat.Cost().Total(); total != nil {
	fmt.Printf("$%.4f\n", *total)
}
```

## Layout

| Package | |
|---|---|
| `gollym` | The facade: type aliases and functions re-exported from `core`; registers the built-in providers |
| `core` | Config, Chat, messages, attachments, the model registry, cost, errors, the HTTP connection, provider and protocol interfaces |
| `protocols/*` | Wire formats: `chatcompletions`, `gemini`, `anthropic`, `converse` (Bedrock), `fal`, `nliclassify`, `systemone`, `zhipulayout` |
| `providers` | One file per provider |
| `toolset`, `mcp`, `agent` | Tools from TOML, tools from MCP servers, declarative agents |
| `persistence`, `riverjobs`, `registry` | Chats in PostgreSQL, River jobs, registry refresh |
| `bdd`, `tdg` | Spec-driven test builders |
| `cmd/gollym` | `gollym models refresh`, `gollym models list` |

Architecture, as in pyllym: **provider** (where and who) → **protocol**
(wire format) → **connection** (HTTP and retries). See `CLAUDE.md` and
`docs/PLAN.md` for how the port maps to pyllym and its tests.

## Development

```bash
go build ./... && go vet ./... && test -z "$(gofmt -l .)"
go test ./...                              # no network; DB tests need TEST_DATABASE_URL
go test -tags live ./protocols/gemini/     # real calls when the provider's key is set
```

## License

MIT, like pyllym.
