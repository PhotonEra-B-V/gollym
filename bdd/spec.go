package bdd

import (
	"errors"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"

	"github.com/BurntSushi/toml"

	"github.com/PhotonEra-B-V/gollym/core"
)

// TOML front end: one spec file, two modes (pyllym's bdd/toml_spec.py).
//
// A TOML spec is the single authoring surface: the API surface, coverable
// rules and edge cases, optional sequenceDiagram blocks (ParseSequence), and,
// in plan mode, hand-written test cases.
//
// Two modes, auto-detected from the presence of [[cases]] unless [meta].mode
// says otherwise:
//
//   - plan mode: the spec carries [[cases]]; TomlSpec.ToTestPlan builds a
//     TestPlan deterministically, with no LLM in the loop.
//   - request mode: the spec is loose (rules and edge cases, no cases);
//     TomlSpec.ToPrompt produces a prompt for the planner LLM, which fills
//     the TestPlan schema. The guarantee still lives in ValidatePlan plus
//     template rendering; the model never writes Go text.

// Build modes.
const (
	ModePlan    = "plan"
	ModeRequest = "request"
)

// Meta is the [meta] table.
type Meta struct {
	FeatureName string `json:"feature_name"`
	Description string `json:"description"`
	// Mode is "plan", "request" or "" (auto-detect).
	Mode          string `json:"mode"`
	AsyncRequired bool   `json:"async_required"`
}

// Api is the [api] table.
type Api struct {
	Module     string   `json:"module"`
	Imports    []string `json:"imports"`
	Signatures []string `json:"signatures"`
}

// ToSurface converts to the plan's ApiSurface.
func (a Api) ToSurface() ApiSurface {
	return ApiSurface{Module: a.Module, Imports: clone(a.Imports), Signatures: clone(a.Signatures)}
}

// Coverable is a rule or edge case carrying a coverable id (rule_1, edge_1).
type Coverable struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

// Case is a hand-written test case (plan mode); mirrors TestCase.
type Case struct {
	TestName string   `json:"test_name"`
	Scenario string   `json:"scenario"`
	Given    []string `json:"given"`
	When     string   `json:"when"`
	Then     []string `json:"then"`
	Fixtures []string `json:"fixtures"`
	Covers   []string `json:"covers"`
	IsAsync  bool     `json:"is_async"`
	EdgeCase bool     `json:"edge_case"`
	Notes    string   `json:"notes"`
}

// ToTestCase converts to the plan's TestCase.
func (c Case) ToTestCase() TestCase {
	return TestCase{
		Scenario: c.Scenario, TestName: c.TestName,
		Given: clone(c.Given), When: c.When, Then: clone(c.Then),
		Fixtures: clone(c.Fixtures), Covers: clone(c.Covers),
		IsAsync: c.IsAsync, EdgeCase: c.EdgeCase, Notes: c.Notes,
	}
}

// Fixture is a hand-written fixture (plan mode); mirrors FixtureDef.
type Fixture struct {
	Name string   `json:"name"`
	Type string   `json:"type,omitempty"`
	Body []string `json:"body"`
}

// TomlSpec is a parsed spec.
type TomlSpec struct {
	Meta       Meta               `json:"meta"`
	API        Api                `json:"api"`
	Rules      []Coverable        `json:"rules"`
	EdgeCases  []Coverable        `json:"edge_cases"`
	Sequences  []*SequenceDiagram `json:"sequences"`
	Cases      []Case             `json:"cases"`
	Fixtures   []Fixture          `json:"fixtures"`
	SourcePath string             `json:"source_path,omitempty"`
}

var slugRe = regexp.MustCompile(`[^a-z0-9]+`)

// Slug is the filesystem/identifier-safe name: "Retry policy" -> "retry_policy".
func (s *TomlSpec) Slug() string {
	slug := strings.Trim(slugRe.ReplaceAllString(strings.ToLower(s.Meta.FeatureName), "_"), "_")
	if slug == "" {
		return "feature"
	}
	return slug
}

// Mode: an explicit [meta].mode wins; else "plan" iff any [[cases]].
func (s *TomlSpec) Mode() string {
	if s.Meta.Mode != "" {
		return s.Meta.Mode
	}
	if len(s.Cases) > 0 {
		return ModePlan
	}
	return ModeRequest
}

// ScenarioNames are the scenarios every plan must cover (from [[cases]]).
func (s *TomlSpec) ScenarioNames() map[string]bool {
	out := map[string]bool{}
	for _, c := range s.Cases {
		out[c.Scenario] = true
	}
	return out
}

// MessageIDs are the sequence-diagram message ids across all diagrams.
func (s *TomlSpec) MessageIDs() map[string]bool {
	out := map[string]bool{}
	for _, d := range s.Sequences {
		for _, m := range d.Messages {
			out[m.ID] = true
		}
	}
	return out
}

// RequiredIDs are the rule and edge-case ids, folded into one set.
func (s *TomlSpec) RequiredIDs() map[string]bool {
	out := map[string]bool{}
	for _, r := range s.Rules {
		out[r.ID] = true
	}
	for _, e := range s.EdgeCases {
		out[e.ID] = true
	}
	return out
}

// ToTestPlan builds a TestPlan deterministically from [[cases]] (plan mode,
// no LLM). It renders and passes ValidatePlan when the hand-written cases
// cover every rule, edge case and sequence-diagram message id.
func (s *TomlSpec) ToTestPlan() *TestPlan {
	plan := &TestPlan{
		Feature:     s.Meta.FeatureName,
		API:         s.API.ToSurface(),
		Fixtures:    []FixtureDef{},
		Cases:       []TestCase{},
		RequiredIDs: sortedKeys(s.RequiredIDs()),
	}
	for _, f := range s.Fixtures {
		plan.Fixtures = append(plan.Fixtures, FixtureDef{Name: f.Name, Type: f.Type, Body: clone(f.Body)})
	}
	for _, c := range s.Cases {
		plan.Cases = append(plan.Cases, c.ToTestCase())
	}
	return plan.normalized()
}

// ToPrompt is the readable prompt block for the planner LLM (request mode):
// the API surface, every rule and edge case with its id, and any sequence
// diagrams, instructing that every id be claimed via 'covers'.
func (s *TomlSpec) ToPrompt() string {
	lines := []string{"Feature: " + s.Meta.FeatureName}
	if s.Meta.Description != "" {
		for _, l := range splitLines(s.Meta.Description) {
			lines = append(lines, "  "+l)
		}
	}
	lines = append(lines, "", "API surface (exercise only this):", "  module: "+s.API.Module)
	for _, imp := range s.API.Imports {
		lines = append(lines, "  import: "+imp)
	}
	for _, sig := range s.API.Signatures {
		lines = append(lines, "  signature: "+sig)
	}
	if len(s.Rules) > 0 {
		lines = append(lines, "", "Rules (cover every id via 'covers'):")
		for _, r := range s.Rules {
			lines = append(lines, fmt.Sprintf("  %s: %s", r.ID, r.Text))
		}
	}
	if len(s.EdgeCases) > 0 {
		lines = append(lines, "", "Edge cases (cover every id via 'covers'):")
		for _, e := range s.EdgeCases {
			lines = append(lines, fmt.Sprintf("  %s: %s", e.ID, e.Text))
		}
	}
	if len(s.Sequences) > 0 {
		lines = append(lines, "", "Sequence-diagram messages (cover every id via 'covers'):")
		for _, d := range s.Sequences {
			lines = append(lines, d.ToAnnotated())
		}
	}
	if s.Meta.AsyncRequired {
		lines = append(lines, "", "This feature is context-aware: set is_async=true and pass ctx "+
			"(the test's context.Context) to the act.")
	}
	lines = append(lines, "", "Every rule id, edge-case id, and sequence message id above MUST be "+
		"claimed by at least one test case via its 'covers' list.")
	return strings.Join(lines, "\n") + "\n"
}

// --- parsing ----------------------------------------------------------------

// Parse parses a TOML spec. It fails when [meta].feature_name or
// [api].module is missing, or [meta].mode is not plan/request.
func Parse(text string) (*TomlSpec, error) {
	var data map[string]any
	if _, err := toml.Decode(text, &data); err != nil {
		return nil, err
	}
	metaRaw := table(data["meta"])
	featureName := metaRaw["feature_name"]
	if !core.Truthy(featureName) {
		return nil, errors.New("TOML spec missing required [meta].feature_name")
	}
	mode := ""
	if m, ok := metaRaw["mode"]; ok && m != nil {
		ms, isStr := m.(string)
		if !isStr || (ms != ModePlan && ms != ModeRequest) {
			return nil, fmt.Errorf("[meta].mode must be 'plan' or 'request', got %s", pyRepr(m))
		}
		mode = ms
	}
	spec := &TomlSpec{Meta: Meta{
		FeatureName:   pyStr(featureName),
		Description:   pyStr(getOr(metaRaw, "description", "")),
		Mode:          mode,
		AsyncRequired: core.Truthy(metaRaw["async_required"]),
	}}

	apiRaw := table(data["api"])
	if !core.Truthy(apiRaw["module"]) {
		return nil, errors.New("TOML spec missing required [api].module")
	}
	spec.API = Api{
		Module:     pyStr(apiRaw["module"]),
		Imports:    strList(apiRaw["imports"]),
		Signatures: strList(apiRaw["signatures"]),
	}

	var err error
	if spec.Rules, err = coverables(data["rules"], "rules"); err != nil {
		return nil, err
	}
	if spec.EdgeCases, err = coverables(data["edge_cases"], "edge_cases"); err != nil {
		return nil, err
	}
	if spec.Sequences, err = sequences(data["sequences"]); err != nil {
		return nil, err
	}
	for _, e := range entries(data["cases"]) {
		if _, ok := e["test_name"]; !ok {
			return nil, missingKey("cases", "test_name")
		}
		spec.Cases = append(spec.Cases, Case{
			TestName: pyStr(e["test_name"]),
			Scenario: pyStr(getOr(e, "scenario", "")),
			Given:    strList(e["given"]),
			When:     pyStr(getOr(e, "when", NoAct)),
			Then:     strList(e["then"]),
			Fixtures: strList(e["fixtures"]),
			Covers:   strList(e["covers"]),
			IsAsync:  core.Truthy(e["is_async"]),
			EdgeCase: core.Truthy(e["edge_case"]),
			Notes:    pyStr(getOr(e, "notes", "")),
		})
	}
	for _, e := range entries(data["fixtures"]) {
		if _, ok := e["name"]; !ok {
			return nil, missingKey("fixtures", "name")
		}
		spec.Fixtures = append(spec.Fixtures, Fixture{
			Name: pyStr(e["name"]),
			Type: pyStr(getOr(e, "type", "")),
			Body: strList(e["body"]),
		})
	}
	return spec, nil
}

// ParseFile parses a spec file, recording its path.
func ParseFile(path string) (*TomlSpec, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	spec, err := Parse(string(b))
	if err != nil {
		return nil, err
	}
	spec.SourcePath = path
	return spec, nil
}

func missingKey(array, key string) error {
	return fmt.Errorf("TOML spec [[%s]] entry missing required key %s", array, pyRepr(key))
}

func coverables(v any, array string) ([]Coverable, error) {
	var out []Coverable
	for _, e := range entries(v) {
		if _, ok := e["id"]; !ok {
			return nil, missingKey(array, "id")
		}
		out = append(out, Coverable{ID: pyStr(e["id"]), Text: pyStr(getOr(e, "text", ""))})
	}
	return out, nil
}

// sequences parses each [[sequences]] mermaid string; each diagram starts one
// past the running message count so M-ids stay unique across all blocks.
func sequences(v any) ([]*SequenceDiagram, error) {
	var out []*SequenceDiagram
	next := 1
	for _, e := range entries(v) {
		if _, ok := e["mermaid"]; !ok {
			return nil, missingKey("sequences", "mermaid")
		}
		d, err := ParseSequence(pyStr(e["mermaid"]), next)
		if err != nil {
			return nil, err
		}
		next += len(d.Messages)
		out = append(out, d)
	}
	return out, nil
}

func table(v any) map[string]any {
	m, _ := v.(map[string]any)
	if m == nil {
		return map[string]any{}
	}
	return m
}

func entries(v any) []map[string]any {
	var out []map[string]any
	switch l := v.(type) {
	case []map[string]any:
		return l
	case []any:
		for _, e := range l {
			out = append(out, table(e))
		}
	}
	return out
}

func getOr(m map[string]any, key string, def any) any {
	if v, ok := m[key]; ok {
		return v
	}
	return def
}

// strList is pyllym's _str_tuple: falsy -> empty, else each item as a string.
// A scalar counts as a one-item list.
func strList(v any) []string {
	if !core.Truthy(v) {
		return []string{}
	}
	l, ok := v.([]any)
	if !ok {
		return []string{pyStr(v)}
	}
	out := make([]string, 0, len(l))
	for _, x := range l {
		out = append(out, pyStr(x))
	}
	return out
}

// pyStr is Python's str() for TOML scalars.
func pyStr(v any) string {
	switch x := v.(type) {
	case nil:
		return "None"
	case string:
		return x
	case bool:
		if x {
			return "True"
		}
		return "False"
	}
	return fmt.Sprint(v)
}

func clone(s []string) []string { return append([]string{}, s...) }

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
