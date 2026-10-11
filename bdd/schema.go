package bdd

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// The structured contract the planner LLM fills in.
//
// The model never writes freeform test code: it fills TestPlan, and the
// renderer turns that into Go test source deterministically. Field
// descriptions reach the provider as part of the JSON schema, so they double
// as prompt instructions.

// ApiSurface is the public API the tests exercise: the design commitment.
type ApiSurface struct {
	// Module is the Go import path of the package under test.
	Module string `json:"module"`
	// Imports are Go import specs the test file needs ("example.com/x/retry"
	// or retry "example.com/x/retry"). Unused ones are dropped on render.
	Imports []string `json:"imports"`
	// Signatures are the exported Go declarations to implement.
	Signatures []string `json:"signatures"`
}

// FixtureDef is shared setup: rendered as a helper `func fixtureName(t
// *testing.T) Type` and called as `name := fixtureName(t)` by the tests that
// request it (Go's stand-in for a pytest fixture).
type FixtureDef struct {
	Name string `json:"name"`
	// Type is the Go result type of the fixture (Go needs one; pytest
	// fixtures are untyped). Empty means `any`.
	Type string   `json:"type,omitempty"`
	Body []string `json:"body"`
}

// TestCase is one rendered test function.
type TestCase struct {
	Scenario string   `json:"scenario"`
	TestName string   `json:"test_name"`
	Given    []string `json:"given"`
	When     string   `json:"when"`
	Then     []string `json:"then"`
	Fixtures []string `json:"fixtures"`
	Covers   []string `json:"covers"`
	// IsAsync: the act needs a context.Context; the test declares
	// `ctx := t.Context()` (Go's analogue of an async pytest test).
	IsAsync  bool   `json:"is_async"`
	EdgeCase bool   `json:"edge_case"`
	Notes    string `json:"notes"`
}

// TestPlan is the whole plan for one feature.
type TestPlan struct {
	Feature     string       `json:"feature"`
	API         ApiSurface   `json:"api"`
	Fixtures    []FixtureDef `json:"fixtures"`
	Cases       []TestCase   `json:"cases"`
	RequiredIDs []string     `json:"required_ids"`
}

func schemaStr(desc string) map[string]any {
	return map[string]any{"type": "string", "description": desc}
}

func schemaStrList(desc string) map[string]any {
	return map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": desc}
}

func titled(m map[string]any, title string) map[string]any { m["title"] = title; return m }

// JSONSchema is TestPlan's JSON Schema, shaped like pydantic's
// model_json_schema ($defs + $ref, titles, defaults) with Go-flavoured
// descriptions. It makes TestPlan a core.SchemaProvider for Chat.WithSchema.
func (TestPlan) JSONSchema() map[string]any {
	apiSurface := map[string]any{
		"description": "The public API the tests exercise — this is the design commitment.",
		"properties": map[string]any{
			"module": titled(schemaStr("Go import path of the package under test, e.g. 'example.com/myapp/retry'"), "Module"),
			"imports": titled(schemaStrList("Go import specs the test file needs, e.g. '\"example.com/myapp/retry\"' "+
				"or 'retry \"example.com/myapp/retry\"'; \"testing\" is always present"), "Imports"),
			"signatures": titled(schemaStrList("Exported Go declarations to implement, "+
				"e.g. 'func Backoff(attempt int, base float64) (float64, error)'"), "Signatures"),
		},
		"required": []any{"module", "signatures"},
		"title":    "ApiSurface",
		"type":     "object",
	}
	fixtureDef := map[string]any{
		"properties": map[string]any{
			"name": titled(schemaStr("Valid Go identifier for the fixture; tests request it by this name"), "Name"),
			"type": titled(map[string]any{"type": "string", "default": "",
				"description": "Go result type of the fixture value, e.g. 'float64' or '*retry.Policy'; empty means 'any'"}, "Type"),
			"body": titled(schemaStrList("Go statements for the fixture body (the test's *testing.T is 't'); "+
				"the last must be a 'return'; register teardown with t.Cleanup"), "Body"),
		},
		"required": []any{"name", "body"},
		"title":    "FixtureDef",
		"type":     "object",
	}
	testCase := map[string]any{
		"properties": map[string]any{
			"scenario":  titled(schemaStr("Name of the scenario this case comes from"), "Scenario"),
			"test_name": titled(schemaStr("snake_case test name, starting with 'test_' (rendered as a TestCamelCase function)"), "Test Name"),
			"given":     titled(schemaStrList("Arrange statements — Go source, one statement per entry"), "Given"),
			"when": titled(schemaStr("The single act expression; its value is assigned with 'result := ...'. "+
				"It must be single-valued: for calls returning an error put the call in 'given' and set this to 'nil'."), "When"),
			"then":     titled(schemaStrList("Boolean Go expressions (no 'if', no t.Fatal), typically about 'result'"), "Then"),
			"fixtures": titled(schemaStrList("Names of fixtures this test requests"), "Fixtures"),
			"covers": titled(schemaStrList("Ids this test exercises: sequence-diagram message ids (e.g. 'M1') "+
				"and/or rule/edge-case ids (e.g. 'rule_1', 'edge_1'); "+
				"required when the feature has sequence diagrams, rules, or edge cases"), "Covers"),
			"is_async": titled(map[string]any{"type": "boolean", "default": false,
				"description": "True when the act needs a context.Context: the test declares 'ctx := t.Context()'"}, "Is Async"),
			"edge_case": titled(map[string]any{"type": "boolean", "default": false,
				"description": "True when this case was derived beyond the literal scenarios (boundary, error path)"}, "Edge Case"),
			"notes": titled(map[string]any{"type": "string", "default": "",
				"description": "One line of implementer-facing context, if useful"}, "Notes"),
		},
		"required": []any{"scenario", "test_name", "when", "then"},
		"title":    "TestCase",
		"type":     "object",
	}
	return map[string]any{
		"$defs": map[string]any{"ApiSurface": apiSurface, "FixtureDef": fixtureDef, "TestCase": testCase},
		"properties": map[string]any{
			"feature":  titled(schemaStr("The feature name, verbatim from the spec"), "Feature"),
			"api":      map[string]any{"$ref": "#/$defs/ApiSurface"},
			"fixtures": map[string]any{"items": map[string]any{"$ref": "#/$defs/FixtureDef"}, "title": "Fixtures", "type": "array"},
			"cases":    map[string]any{"items": map[string]any{"$ref": "#/$defs/TestCase"}, "title": "Cases", "type": "array"},
			"required_ids": titled(schemaStrList("Rule and edge-case ids (e.g. 'rule_1', 'edge_1') that every "+
				"sound plan MUST cover via some TestCase.covers entry"), "Required Ids"),
		},
		"required": []any{"feature", "api", "cases"},
		"title":    "TestPlan",
		"type":     "object",
	}
}

// normalized returns a copy with nil slices replaced by empty ones, so the
// plan JSON carries [] like pydantic's model_dump.
func (p *TestPlan) normalized() *TestPlan {
	out := *p
	out.API.Imports = nonNil(out.API.Imports)
	out.API.Signatures = nonNil(out.API.Signatures)
	out.RequiredIDs = nonNil(out.RequiredIDs)
	out.Fixtures = append([]FixtureDef{}, p.Fixtures...)
	for i := range out.Fixtures {
		out.Fixtures[i].Body = nonNil(out.Fixtures[i].Body)
	}
	out.Cases = append([]TestCase{}, p.Cases...)
	for i := range out.Cases {
		c := &out.Cases[i]
		c.Given, c.Then, c.Fixtures, c.Covers = nonNil(c.Given), nonNil(c.Then), nonNil(c.Fixtures), nonNil(c.Covers)
	}
	return &out
}

func nonNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

// MarshalPlan renders the reviewable plan JSON (indent 2, trailing newline).
func MarshalPlan(p *TestPlan) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(p.normalized()); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// ParsePlan decodes and validates plan JSON (pydantic's model_validate):
// required fields must be present and every field must have the right type.
func ParsePlan(data []byte) (*TestPlan, error) {
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("invalid TestPlan: %w", err)
	}
	if err := requireKeys(raw, "", "feature", "api", "cases"); err != nil {
		return nil, err
	}
	if api, ok := raw["api"].(map[string]any); ok {
		if err := requireKeys(api, "api.", "module", "signatures"); err != nil {
			return nil, err
		}
	}
	for i, f := range asList(raw["fixtures"]) {
		if m, ok := f.(map[string]any); ok {
			if err := requireKeys(m, fmt.Sprintf("fixtures.%d.", i), "name", "body"); err != nil {
				return nil, err
			}
		}
	}
	for i, c := range asList(raw["cases"]) {
		if m, ok := c.(map[string]any); ok {
			if err := requireKeys(m, fmt.Sprintf("cases.%d.", i), "scenario", "test_name", "when", "then"); err != nil {
				return nil, err
			}
		}
	}
	var p TestPlan
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, fmt.Errorf("invalid TestPlan: %w", err)
	}
	return p.normalized(), nil
}

func asList(v any) []any {
	l, _ := v.([]any)
	return l
}

func requireKeys(m map[string]any, prefix string, keys ...string) error {
	for _, k := range keys {
		if _, ok := m[k]; !ok {
			return fmt.Errorf("invalid TestPlan: %s%s: field required", prefix, k)
		}
	}
	return nil
}
