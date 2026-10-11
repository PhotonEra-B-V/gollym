// Package tdg is the test-driven generator (pyllym's pyllym.tdg): it turns a
// TOML spec into a red Go test suite that serves as the building instruction
// for an implementing agent.
//
// It is package bdd's pipeline (TOML spec, plan or request mode, safety gate,
// coverage checks, Go test rendering, brief; see the bdd documentation,
// including its security model) plus two additions:
//
//   - dependency inspection (InspectDependencies, CheckDependencies): the
//     package under test's real imports and declarations are discovered
//     with go list + go/parser, fed to the planner, and cross-checked
//     against the spec's [api].imports and the names its cases use;
//   - timestamped runs (ReadLatest, Reusable): each build writes
//     outDir/<stamp>/, marks it complete with _DONE.json, points
//     outDir/latest.json at it, and the next build reuses specs whose
//     content hash is unchanged.
//
// From Go:
//
//	results, err := tdg.Build(ctx, "features/", "tests/generated",
//		tdg.Options{Options: bdd.Options{Chat: gollym.ChatOptions{Model: "gpt-5.4"}}})
//
// Or from the shell:
//
//	gollym-tdg features/ --out tests/generated --model gpt-5.4
package tdg

import "github.com/PhotonEra-B-V/gollym/bdd"

// The shared pipeline types and functions, re-exported from bdd (pyllym's
// tdg duplicates bdd's modules; Go shares them).
type (
	ApiSurface          = bdd.ApiSurface
	FixtureDef          = bdd.FixtureDef
	TestCase            = bdd.TestCase
	TestPlan            = bdd.TestPlan
	TomlSpec            = bdd.TomlSpec
	SequenceDiagram     = bdd.SequenceDiagram
	SequenceMessage     = bdd.SequenceMessage
	BuildResult         = bdd.BuildResult
	PlanSafetyError     = bdd.PlanSafetyError
	PlanValidationError = bdd.PlanValidationError
	Planner             = bdd.Planner
	ChatPlanner         = bdd.ChatPlanner
	RenderOptions       = bdd.RenderOptions
)

var (
	Parse         = bdd.Parse
	ParseFile     = bdd.ParseFile
	ParseSequence = bdd.ParseSequence
	PlanFromSpec  = bdd.PlanFromSpec
	ValidatePlan  = bdd.ValidatePlan
	ScanPlan      = bdd.ScanPlan
	RenderBrief   = bdd.RenderBrief
)

// RenderTests renders a plan with tdg's header.
func RenderTests(plan *TestPlan, opts RenderOptions) string {
	if opts.Generator == "" {
		opts.Generator = Generator
	}
	return bdd.RenderTests(plan, opts)
}
