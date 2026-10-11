# CLAUDE.md

**gollym** is the Go port of pyllym (`PhotonEra-B-V/malayopython_reticulatus_llm`,
`/Users/kaarlekulvik/Projects/malayopython_reticulatus_llm`, v1.16.0): one
interface for every major AI provider. Behaviour follows pyllym exactly unless
this file or `docs/PLAN.md` says otherwise. When in doubt, read the Python
source and do what it does.

## Layout

```
gollym.go              facade: type aliases + functions re-exported from core,
                       blank-imports providers so they register
core/                  everything shared: Config, Context, Chat, Message,
                       Content, Attachment, MIME, Tokens, Cost, errors,
                       Connection (HTTP + retries), SSE, StreamAccumulator,
                       Provider/Protocol interfaces, provider registry, model
                       registry (embedded models.json/aliases.json), media
                       value types, decisions, tools
protocols/<name>/      one wire format per package (chatcompletions, gemini,
                       anthropic, converse, fal, nliclassify, systemone,
                       zhipulayout)
providers/             one file per provider; providers.go holds the ordered
                       registration list (pyllym's order)
internal/testserver/   recording httptest stub (pyllym's stub_llm_server)
toolset/ mcp/ agent/ persistence/ river/ bdd/ tdg/   higher-level packages
cmd/                   CLIs (models refresh, bdd, tdg)
```

## Architecture (same as pyllym)

**Provider (where/who) → Protocol (wire format) → Connection (HTTP).**

- A provider is a `core.ProviderSpec` (data + func fields instead of
  overridden methods). `core.NewProvider(spec, cfg)` builds a
  `*core.BaseProvider`; `spec.Wrap` returns a custom type embedding it when a
  provider needs extra interfaces (`core.FileProvider`). `spec.Sign` signs
  requests (Bedrock SigV4).
- A protocol implements `core.Protocol` (RenderPayload,
  ParseCompletionResponse, CompletionURL) plus optional interfaces:
  `core.Streamer` (SSE), `core.TemperatureNormalizer`,
  `core.MessagePreprocessor`, `core.SyncResponder`/`core.StreamResponder`
  (non-standard calls), and the feature interfaces `core.Embedder`,
  `Moderator`, `Painter`, `Speaker`, `Transcriber`, `Animator`,
  `LayoutParser`, `Decider`, `ModelLister`. Protocols that extend another
  embed it (`type ZhipuLayout struct{ *chatcompletions.Protocol }`).
- Payloads are `map[string]any` (pyllym's dicts) so `Chat.WithParams` can be
  deep-merged over them. Parse with `core.Dig`, `core.DigMap`, `core.AsInt`,
  `core.Truthy`, `core.Or` (Python truthiness and `a or b`).
- Python `None` vs missing: use pointers (`*int`, `*string`) where pyllym
  distinguishes them; `""` where it doesn't.
- Errors: `*core.Error` with a `Kind`; callers match with
  `errors.Is(err, core.ErrRateLimit)`. Never panic on provider data.
- Everything that does I/O takes a `context.Context` first.

## Rules

- **Parity:** request bodies must be JSON-equal to pyllym's for the same
  inputs (key order doesn't matter), and parsed results equal. Port pyllym's
  tests for your area into Go tests; a Python test without a Go counterpart
  is a gap, list it in docs/PLAN.md.
- **No real network in tests.** Use `internal/testserver`. Live tests are
  behind `//go:build live` and skip without keys.
- Only edit files in your own area. `core/` is shared: if you need a change
  there, keep it minimal and backwards compatible, and say so in your report.
- Keep `go build ./...` green at every step: other agents build the same
  module. Run `go vet` and `gofmt -l .` before finishing.
- Dependencies are already in go.mod (toml, MCP go-sdk, AWS SDK v2 signer,
  eventstream, pgx, River). Avoid adding others; if you must, `go get` it and
  mention it.
- Comments: say what and why, briefly, like the surrounding code.

## Commands

```bash
go build ./... && go vet ./... && test -z "$(gofmt -l .)"
go test ./...
go test ./protocols/gemini/ -run TestName
```
