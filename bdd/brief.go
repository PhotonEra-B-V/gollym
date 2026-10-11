package bdd

import (
	"fmt"
	"path/filepath"
	"strings"
)

// RenderBrief renders the implementation brief: the building instruction an
// agent (or human) follows to turn the red suite green (pyllym's
// bdd/brief.py). spec may be nil (no description, sequences, rules or edge
// cases are then shown). Tests are listed by their Go function names.
func RenderBrief(spec *TomlSpec, plan *TestPlan, testsPath string, problems []string) string {
	var sequences []*SequenceDiagram
	var coverables []Coverable
	description := ""
	if spec != nil {
		sequences = spec.Sequences
		coverables = append(append(coverables, spec.Rules...), spec.EdgeCases...)
		description = spec.Meta.Description
	}
	names := TestFuncNames(plan)

	lines := []string{"# BUILD BRIEF: " + plan.Feature, ""}
	if description != "" {
		lines = append(lines, description, "")
	}

	if len(problems) > 0 {
		lines = append(lines, "## ⚠ Plan validation warnings", "")
		for _, p := range problems {
			lines = append(lines, "- "+p)
		}
		lines = append(lines, "")
	}

	lines = append(lines, "## Target API (implement exactly these signatures)", "",
		fmt.Sprintf("Package: `%s`", plan.API.Module), "", "```go")
	lines = append(lines, plan.API.Signatures...)
	lines = append(lines, "```", "")

	if len(sequences) > 0 {
		lines = append(lines, "## Collaboration contract", "",
			"The implementation must produce these interactions, in order —",
			"the suite asserts recorded call sequences against them.", "")
		for _, d := range sequences {
			lines = append(lines, "```mermaid", d.Source, "```", "")
		}
	}

	lines = append(lines, fmt.Sprintf("## Test manifest (%d cases, all currently red)", len(plan.Cases)), "",
		"| Test | Scenario | Kind | Covers |",
		"|------|----------|------|--------|")
	for i, c := range plan.Cases {
		kind := "scenario"
		if c.EdgeCase {
			kind = "edge case"
		}
		covers := "—"
		if len(c.Covers) > 0 {
			covers = strings.Join(c.Covers, ", ")
		}
		lines = append(lines, fmt.Sprintf("| `%s` | %s | %s | %s |", names[i], c.Scenario, kind, covers))
	}
	lines = append(lines, "")

	testsCovering := func(id string) string {
		var tests []string
		for i, c := range plan.Cases {
			for _, cv := range c.Covers {
				if cv == id {
					tests = append(tests, "`"+names[i]+"`")
					break
				}
			}
		}
		if len(tests) == 0 {
			return "**uncovered**"
		}
		return strings.Join(tests, ", ")
	}

	if len(sequences) > 0 {
		lines = append(lines, "### Message traceability", "",
			"| Message | Interaction | Tests |",
			"|---------|-------------|-------|")
		for _, d := range sequences {
			for _, m := range d.Messages {
				lines = append(lines, fmt.Sprintf("| %s | %s → %s: %s | %s |", m.ID, m.Sender, m.Receiver, m.Text, testsCovering(m.ID)))
			}
		}
		lines = append(lines, "")
	}

	if len(coverables) > 0 {
		lines = append(lines, "### Rule/edge traceability", "",
			"| Id | Rule/Edge | Tests |",
			"|----|-----------|-------|")
		for _, item := range coverables {
			lines = append(lines, fmt.Sprintf("| %s | %s | %s |", item.ID, item.Text, testsCovering(item.ID)))
		}
		lines = append(lines, "")
	}

	lines = append(lines, "## Rules", "",
		fmt.Sprintf("- The tests in `%s` are the specification. They are", testsPath),
		"  read-only during implementation — if a test looks wrong, stop and",
		"  flag it; do not adapt the test to the implementation.",
		"- Implement only the API surface listed above; keep everything else",
		"  unexported.",
		fmt.Sprintf("- The file carries the `%s` build tag, so it only builds with", PendingTag),
		fmt.Sprintf("  `-tags %s`; regular `go test ./...` skips it.", PendingTag),
		"",
		"## Definition of done", "",
		fmt.Sprintf("`go test -tags %s %s` passes with zero edits to the test file.", PendingTag, goPackageArg(testsPath)))
	return strings.Join(lines, "\n") + "\n"
}

// goPackageArg is the go test argument for the directory holding a test
// file: "./tests/generated" for relative paths.
func goPackageArg(testsPath string) string {
	dir := filepath.ToSlash(filepath.Dir(testsPath))
	if filepath.IsAbs(dir) || strings.HasPrefix(dir, "./") || strings.HasPrefix(dir, "../") {
		return dir
	}
	if dir == "." {
		return "."
	}
	return "./" + dir
}
