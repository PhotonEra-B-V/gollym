package bdd

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/PhotonEra-B-V/gollym/core"
	_ "github.com/PhotonEra-B-V/gollym/providers" // the planner needs registered providers
)

// Expanding a request-mode TOML spec into a TestPlan via structured output
// (pyllym's bdd/planner.py).
//
// PlanFromSpec uses the structured-output pattern (WithInstructions /
// WithSchema / Ask): the model only ever fills TestPlan; it never writes Go
// text. Plan-mode specs skip this entirely (TomlSpec.ToTestPlan is
// deterministic, no LLM).

// SpecPlannerInstructions is the planner's system instruction.
const SpecPlannerInstructions = `You are a test architect. You receive one behavior specification (an API
surface plus rules, edge cases, and optional sequence diagrams) and produce a
test plan that will be rendered into a failing Go test file (package
testing). The suite is written BEFORE any implementation exists — it is the
specification an implementer must satisfy without editing the tests.

Rules:
- Commit to the public API surface exactly as given: package import path,
  imports, and the exact signatures. Every test must exercise only that surface.
- Write one or more test cases covering the rules, plus derived edge cases
  (boundaries, error paths) marked edge_case=true. Do not invent behavior the
  spec does not imply.
- 'given' entries, fixture bodies, 'when', and 'then' must be Go source
  against the declared API, referring to imported packages by package name
  (e.g. retry.Backoff). 'given' entries are statements. 'when' is a single
  single-valued expression whose value is assigned with ` + "`result := ...`" + `.
  'then' entries are bare boolean Go expressions (no ` + "`if`" + `, no t.Fatal),
  typically about ` + "`result`" + `. The test's *testing.T is ` + "`t`" + `.
- For calls that return an error (or several values), make the call a 'given'
  statement, e.g. ` + "`_, err := retry.Backoff(-1)`" + `, set when = "nil", and
  assert in 'then', e.g. ` + "`err != nil`" + ` or ` + "`errors.Is(err, retry.ErrNegative)`" + `.
- Set is_async=true when the act needs a context.Context: the rendered test
  declares ` + "`ctx := t.Context()`" + ` for it.
- api.imports are Go import specs, e.g. "example.com/myapp/retry" or
  retry "example.com/myapp/retry"; list every package the snippets use
  ("testing" is always present; unused imports are dropped). Never use os,
  os/exec, syscall, unsafe, reflect, net or other file/process/network
  packages, never spawn goroutines except with an inline func literal, and
  never write import statements inside snippets.
- Prefer plain values over mocks. Use fixtures only for setup shared by
  several cases. A fixture becomes a helper the test calls as
  ` + "`name := fixtureName(t)`" + `: give its Go result type in 'type'; its body
  ends with 'return'; register teardown with t.Cleanup.
- test_name values must be unique, snake_case, and start with 'test_'
  (they are rendered as TestCamelCase functions).

COVERAGE IS MECHANICALLY ENFORCED — the build fails if any id is uncovered:
- Set 'covers' on each test case to the ids it exercises. Every rule id, every
  edge-case id, and every sequence-diagram message id listed below MUST appear
  in some case's 'covers'. Never invent ids; only claim ids that were given.

When sequence diagrams are present, they are binding collaboration contracts:
- Design the API so the collaborators named in a diagram are injectable
  (struct fields, constructor or function parameters, small interfaces or
  func values) so tests can substitute recording fakes.
- Add at least one interaction-order case: build a plain ` + "`calls := []string{}`" + `
  in 'given', pass func-valued fakes that append to it, and assert in 'then'
  that the recorded order matches the diagram
  (e.g. slices.Equal(calls, []string{"query_users"})).
`

// Planner fills a TestPlan for a request-mode spec. ChatPlanner is the
// structured-output implementation; tests inject fakes.
type Planner interface {
	Plan(ctx context.Context, spec *TomlSpec, apiHint string) (*TestPlan, error)
}

// ChatPlanner plans through a fresh gollym Chat per spec.
type ChatPlanner struct {
	Options core.ChatOptions
}

// Plan implements Planner.
func (p ChatPlanner) Plan(ctx context.Context, spec *TomlSpec, apiHint string) (*TestPlan, error) {
	chat, err := core.NewChat(p.Options)
	if err != nil {
		return nil, err
	}
	return PlanFromSpec(ctx, spec, apiHint, chat)
}

// SpecPrompt is the request-mode prompt: spec.ToPrompt plus the optional API
// hint.
func SpecPrompt(spec *TomlSpec, apiHint string) string {
	prompt := spec.ToPrompt()
	if apiHint != "" {
		prompt += "\nTarget API constraints (honor these exactly):\n" + apiHint + "\n"
	}
	return prompt
}

// PlanFromSpec asks the chat's model to expand a loose (request-mode) spec
// into a TestPlan. Only used when the TOML carries no hand-written [[cases]].
// The model is instructed to claim EVERY rule, edge and sequence-message id
// in some case's covers; uncovered ids fail the build mechanically in
// ValidatePlan.
func PlanFromSpec(ctx context.Context, spec *TomlSpec, apiHint string, chat *core.Chat) (*TestPlan, error) {
	chat.WithInstructions(SpecPlannerInstructions)
	if _, err := chat.WithSchema(TestPlan{}); err != nil {
		return nil, err
	}
	msg, err := chat.Ask(ctx, SpecPrompt(spec, apiHint))
	if err != nil {
		return nil, err
	}
	content, ok := msg.Content.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("planner model returned non-JSON content for spec %s", pyRepr(spec.Meta.FeatureName))
	}
	data, err := json.Marshal(content)
	if err != nil {
		return nil, err
	}
	return ParsePlan(data)
}
