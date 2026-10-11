# gollym — plan

Status: **G1–G5 built** (2026-10-11): every pyllym module except the
`reactllm`/`reactullm` bridges is ported and tested. Open: G6 (README,
examples, tag, publish).

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
| Celery | → optional River workers (`riverjobs/`). |
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

| pyllym | Go | Not ported (why) |
|---|---|---|
| test_chat.py | core/chat_test.go | Anthropic/Gemini rows → their protocol packages |
| test_core.py | core/core_test.go | — |
| test_behaviors.py | core/behaviors_test.go | Agent row → agent/; Anthropic/Gemini render rows → their protocols |
| test_retry.py | core/retry_test.go | — |
| test_connection_errors.py | core/connection_errors_test.go | — |
| test_errors.py | core/errors_test.go | — |
| test_value_objects.py | core/value_objects_test.go | — |
| test_capabilities.py | gollym_test.go | — |
| test_review_fixes.py | core/review_fixes_test.go, core/internal_test.go | event-loop client cache / aclose / config AttributeError (Python-only); fal row → fal |
| test_provider_matrix.py (OpenAI-compatible, vLLM chat rows) | providers/openai_matrix_test.go, providers/openai_registry_test.go | other providers' rows → their packages |
| (pyllym itself, golden) | core/parity_test.go, protocols/chatcompletions/parity_test.go | regenerate with testdata/golden.py |
| test_gemini_extra.py, test_chat.py (gemini), test_behaviors.py (gemini render) | protocols/gemini/gemini_test.go | — (pyllym has no Vertex tests; Vertex covered by new tests) |
| test_chat.py (anthropic), test_behaviors.py (anthropic render), test_core.py (claude lookup) | protocols/anthropic/anthropic_test.go | — |
| test_bedrock.py | protocols/converse/converse_test.go, stream_test.go, providers/bedrock_test.go | — |
| test_zhipu_layout.py | protocols/zhipulayout/zhipulayout_test.go | — |
| test_fal.py, test_review_fixes.py (fal queue base) | protocols/fal/fal_test.go | — |
| test_decisions.py | protocols/systemone/, protocols/nliclassify/ tests | Decision.to_dict round trip (no Map yet) |
| test_ollama.py | providers/ollama_test.go | — |
| test_toolset.py | toolset/toolset_test.go | — |
| test_mcp.py | mcp/mcp_test.go | SDK-not-installed and SDK-generation probing (Python-only) |
| test_behaviors.py (agent) | agent/agent_test.go | — |
| test_persistence.py | persistence/persistence_test.go | — |
| test_celery.py | riverjobs/river_test.go | run_async (Python-only) |
| test_bdd.py | bdd/bdd_test.go, bdd/compile_test.go | module identity test (Python-only) |
| test_tdg_deps.py, test_tdg_runs.py | tdg/deps_test.go, tdg/runs_test.go | stdlib runtime fallback → go list fallback |
| test_seeder.py, seed_fixtures.py | — | provider fixture seeder: not ported yet |
| test_reactullm_*.py | — | reactllm/reactullm bridges out of scope |
