# gollym — plan

Status: **G1 in progress** (2026-10-11). Core, the facade, Chat Completions
and the OpenAI-compatible providers are built and tested. Everything else is
being ported in parallel.

Port of pyllym v1.16.0 (`51e5339`). Rekryon (`go-shallowflaws`) is the first
user: it replaces its Python ML service with Go and calls every LLM through
gollym (see `go-shallowflaws/docs/ML_TO_GO_PLAN.md`).

## Decisions

| Topic | Decision |
|---|---|
| Scope | All of pyllym except the `reactllm`/`reactullm` bridges. |
| Module | `github.com/PhotonEra-B-V/gollym`, public like pyllym. |
| API shape | `core` package + `gollym` facade re-exporting it. Builder methods on `Chat` as in pyllym (`WithInstructions`, `WithSchema`, `WithParams`, ...); `Ask(ctx, msg, gollym.With(files...), gollym.OnChunk(fn))`. |
| Payloads | `map[string]any`, deep-merged with params, JSON-equal to pyllym's. |
| Async | Go: blocking calls with `context.Context`; streaming via `OnChunk` callbacks; concurrent tools via goroutines. |
| Retries | pyllym 1.16.0: `max_retries+1` attempts; retry 429/5xx/529/timeouts/connection errors; Retry-After (and OpenAI reset headers) wins, capped by `RetryMaxInterval`; 429 backs off from 1 s; quota 429s are PaymentRequired and never retried. |
| MIME | Python's built-in `mimetypes` table embedded (`core/mimetypes.json`) for identical extension lookups; magic-byte sniffing for the rest. |
| Structured output | `WithSchema` takes a JSON Schema map, a `SchemaProvider`, or a struct (reflection stands in for pydantic). |
| Persistence | SQLAlchemy → a `database/sql`/pgx store with the same tables. |
| Celery | → optional River workers (`river/`). |
| bdd / tdg | Ported; they render Go tests instead of pytest. |

## Phases

| Phase | Scope | Exit |
|---|---|---|
| **G1 core** ✅ partly | core, facade, chat_completions, OpenAI-compatible providers, testserver | chat_completions tests (payload, usage, retries, streaming, tools) green |
| **G2 protocols** | gemini (+ vertexai), anthropic, converse (+ bedrock SigV4, eventstream), fal, nliclassify (vllm), systemone (typesafe), zhipulayout (zhipu) | pyllym's provider tests ported and green |
| **G3 higher level** | toolset, mcp, agent, persistence, river (Celery), models refresh CLI | pyllym's tests for each ported |
| **G4 bdd/tdg** | TOML spec → plan → rendered Go tests + brief, with the safety gate | pyllym's bdd/tdg tests ported |
| **G5 core test port** | test_chat, test_core, test_behaviors, test_provider_matrix, test_retry, test_connection_errors, test_errors, test_value_objects, test_capabilities, test_review_fixes | all green |
| **G6 release** | README, examples, v0.1.0 tag, push to GitHub (owner confirms) | Rekryon builds against the tag |

## Test map

Filled in as tests are ported: pyllym file → Go file.
