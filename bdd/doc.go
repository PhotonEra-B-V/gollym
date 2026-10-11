// Package bdd is the BDD-to-TDD builder (pyllym's pyllym.bdd): it turns a
// TOML spec into a red Go test suite that serves as the building instruction
// for an implementing agent.
//
// TOML is the single spec front end. It carries the API surface, coverable
// [[rules]] and [[edge_cases]], optional mermaid [[sequences]] (parsed by
// ParseSequence into message ids M1, M2, ...), and, in plan mode,
// hand-written [[cases]].
//
// Pipeline: parse (Parse) -> plan (deterministic TomlSpec.ToTestPlan in plan
// mode, or a structured-output call through gollym in request mode, see
// PlanFromSpec) -> safety gate (ScanPlan) -> coverage check (ValidatePlan) ->
// deterministic rendering (RenderTests) plus an implementation brief
// (RenderBrief):
//
//	results, err := bdd.Build(ctx, "features/", "tests/generated",
//		bdd.Options{Chat: gollym.ChatOptions{Model: "gpt-5.4"}})
//
// Or from the shell:
//
//	gollym-bdd features/ --out tests/generated --model gpt-5.4
//
// The LLM only ever fills the TestPlan schema; test code is rendered from
// templates, so plans are reviewable and re-runs diff cleanly. Review the plan
// and generated tests before implementation: the suite is treated as the
// specification from then on.
//
// Differences from pyllym, which renders pytest: the plan's executable
// strings are Go (given: statements; when: one single-valued expression bound
// to result; then: boolean expressions), test_name test_x renders as func
// TestX(t *testing.T), fixtures are typed helpers (FixtureDef.Type), is_async
// gives the test ctx := t.Context(), and pytest's bdd_pending marker is the
// bdd_pending build tag (run the suite with go test -tags bdd_pending).
//
// # Security model
//
// A spec's given/when/then strings, fixture bodies and types, and API imports
// are rendered into a Go test file that then gets compiled and run: every
// such string is executable Go (true in plan mode too, so a hostile .toml
// from a PR is arbitrary code execution). Before rendering, ScanPlan screens
// every one of them with go/parser + go/ast (both modes, unconditionally,
// never downgraded to a warning) and Build returns *PlanSafetyError on
// imports, file/process/network/reflection packages (os, os/exec, syscall,
// unsafe, reflect, net, ...), goroutines of unknown code, //go: or cgo
// directives, or non-Go; nothing runnable is written. This gate is defense in
// depth, not a sandbox: a determined attacker can evade a static denylist.
// The real boundary is procedural: (1) never run a freshly generated suite
// where secrets or network are reachable before a human reviews the diff;
// (2) sandbox generation and execution of untrusted specs; (3) treat .toml
// specs from untrusted contributors as needing review before generation (the
// spec text reaches the planner prompt via ToPrompt, a prompt-injection
// vector) and before execution.
package bdd
