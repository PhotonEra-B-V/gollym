package bdd

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/PhotonEra-B-V/gollym/core"
)

// Orchestration: parse -> plan -> render tests + brief to disk (pyllym's
// bdd/builder.py).

// PlanValidationError: the LLM-produced (or hand-written) plan failed the
// deterministic cross-checks.
type PlanValidationError struct {
	SourceName string
	Problems   []string
}

func (e *PlanValidationError) Error() string {
	return fmt.Sprintf("plan for %s failed validation (the plan JSON was still written for inspection):\n%s",
		pyRepr(e.SourceName), details(e.Problems))
}

// PlanSafetyError: a plan's executable strings tripped the AST safety gate.
//
// Raised before rendering, regardless of strictness, so no runnable suite is
// ever written from a plan containing imports, dangerous packages,
// goroutines of unknown code, compiler directives, or non-Go. This is
// defense in depth, not a sandbox (see ScanPlan): a human must still review
// generated tests before they are run in a privileged environment.
type PlanSafetyError struct {
	SourceName string
	Problems   []string
}

func (e *PlanSafetyError) Error() string {
	return fmt.Sprintf("plan for %s failed the safety gate (nothing runnable was written; "+
		"the plan JSON was kept for inspection):\n%s", pyRepr(e.SourceName), details(e.Problems))
}

func details(problems []string) string {
	lines := make([]string, len(problems))
	for i, p := range problems {
		lines[i] = "  - " + p
	}
	return strings.Join(lines, "\n")
}

// BuildResult is one generated spec.
type BuildResult struct {
	Feature   *TomlSpec
	Plan      *TestPlan
	TestPath  string
	PlanPath  string
	BriefPath string
	// Reused: copied forward from an earlier run (tdg only).
	Reused bool
}

// Options configure Build. The zero value matches pyllym's defaults.
type Options struct {
	// APIHint pins the target package / existing signatures (request mode).
	APIHint string
	// Lenient lists validation problems in the brief instead of failing
	// (pyllym's strict=False). Safety problems always fail.
	Lenient bool
	// TomlMode is "plan" or "request" to override detection; "" or "auto"
	// detects ([meta].mode, else plan iff [[cases]]).
	TomlMode string
	// Package is the generated test files' package (default: derived from
	// the output directory's name, see PackageName).
	Package string
	// Planner fills request-mode plans (default: ChatPlanner{Chat}).
	Planner Planner
	// Chat picks the planner model when Planner is nil.
	Chat core.ChatOptions
}

func (o Options) planner() Planner {
	if o.Planner != nil {
		return o.Planner
	}
	return ChatPlanner{Options: o.Chat}
}

// ResolveMode resolves the build mode for a spec: an explicit tomlMode of
// plan/request overrides everything; otherwise the spec's own detection.
func ResolveMode(spec *TomlSpec, tomlMode string) string {
	if tomlMode == ModePlan || tomlMode == ModeRequest {
		return tomlMode
	}
	return spec.Mode()
}

// SpecSource is a discovered spec plus its text (tdg hashes the text).
type SpecSource struct {
	Spec *TomlSpec
	Text string
}

// Discover resolves a Build source: raw TOML text (anything containing a
// newline), a .toml file, or a directory searched recursively for *.toml.
func Discover(source string) ([]SpecSource, error) {
	if strings.Contains(source, "\n") {
		spec, err := Parse(source)
		if err != nil {
			return nil, err
		}
		return []SpecSource{{spec, source}}, nil
	}
	info, err := os.Stat(source)
	if err != nil {
		return nil, err
	}
	var files []string
	if info.IsDir() {
		err := filepath.WalkDir(source, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if !d.IsDir() && strings.HasSuffix(p, ".toml") {
				files = append(files, p)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
		if len(files) == 0 {
			return nil, fmt.Errorf("no .toml specs under %s", source)
		}
	} else {
		files = []string{source}
	}
	var out []SpecSource
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			return nil, err
		}
		spec, err := Parse(string(b))
		if err != nil {
			return nil, fmt.Errorf("%s: %w", f, err)
		}
		spec.SourcePath = f
		out = append(out, SpecSource{spec, string(b)})
	}
	return out, nil
}

// Build turns TOML specs into a red Go test suite plus build briefs.
//
// source is raw TOML text, a .toml spec file, or a directory of them. For
// each spec it writes into outDir:
//
//   - <slug>_test.go: the failing suite (the specification), behind the
//     bdd_pending build tag
//   - <slug>.plan.json: the reviewable plan the tests were rendered from
//   - BRIEF_<slug>.md: the building instruction for the implementer
//
// TOML is the single front end. A spec with hand-written [[cases]] is plan
// mode (spec.ToTestPlan, no LLM); a loose spec with only rules/edge cases is
// request mode (the Planner fills the plan via the model). Mode is
// auto-detected from the presence of [[cases]] unless [meta].mode (or
// opts.TomlMode) is set. Specs may absorb mermaid sequenceDiagram blocks via
// [[sequences]]. The plan is then cross-checked deterministically
// (ValidatePlan): every scenario, every diagram message, and every required
// rule/edge id must be covered by at least one test case. By default a
// failing check returns *PlanValidationError after writing the plan JSON
// for inspection; with opts.Lenient the problems are listed in the brief.
//
// Before rendering, every plan (both modes, regardless of Lenient) passes
// the AST safety gate (ScanPlan): a case's given/when/then, fixture bodies
// and the API imports become executable Go, so imports, dangerous packages,
// directives or non-Go return *PlanSafetyError and nothing runnable is
// written. This is defense in depth, not a sandbox: a human must still review
// the generated suite before running it anywhere with secrets or network
// access.
//
// Review the plan and tests before pointing an implementer at the brief:
// once the suite is treated as the spec, a misread scenario is locked in.
// On error the results built so far are returned with it.
func Build(ctx context.Context, source, outDir string, opts Options) ([]*BuildResult, error) {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return nil, err
	}
	if opts.Package == "" {
		opts.Package = PackageName(outDir)
	}
	specs, err := Discover(source)
	if err != nil {
		return nil, err
	}
	var results []*BuildResult
	for _, s := range specs {
		r, err := Generate(ctx, s.Spec, Step{Dir: outDir, Options: opts, APIHint: opts.APIHint})
		if err != nil {
			return results, err
		}
		results = append(results, r)
	}
	return results, nil
}

// Step is one spec's generation: where to write and what extra the caller
// (tdg) contributes.
type Step struct {
	Dir     string
	Options Options
	// APIHint is passed to the planner (request mode).
	APIHint string
	// Problems are prepended to the validation problems (tdg's dependency
	// problems): fatal unless Lenient, listed in the brief otherwise.
	Problems []string
	// Generator names the tool in the test file header (default gollym/bdd).
	Generator string
}

// Generate plans, gates, validates and renders one spec into step.Dir: the
// per-spec half of Build, shared with tdg.
func Generate(ctx context.Context, spec *TomlSpec, step Step) (*BuildResult, error) {
	opts := step.Options
	var plan *TestPlan
	if ResolveMode(spec, opts.TomlMode) == ModePlan {
		plan = spec.ToTestPlan() // NO LLM
	} else {
		p, err := opts.planner().Plan(ctx, spec, step.APIHint)
		if err != nil {
			return nil, err
		}
		plan = p
	}
	slug := spec.Slug()
	name := spec.Meta.FeatureName
	r := &BuildResult{
		Feature:   spec,
		Plan:      plan,
		TestPath:  filepath.Join(step.Dir, TestFileName(slug)),
		PlanPath:  filepath.Join(step.Dir, slug+".plan.json"),
		BriefPath: filepath.Join(step.Dir, "BRIEF_"+slug+".md"),
	}
	planJSON, err := MarshalPlan(plan)
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(r.PlanPath, planJSON, 0o644); err != nil {
		return nil, err
	}
	// Safety gate first, and unconditionally: an unsafe executable string
	// must never be downgraded to a warning the way a coverage gap can.
	if problems := ScanPlan(plan); len(problems) > 0 {
		return nil, &PlanSafetyError{SourceName: name, Problems: problems}
	}
	problems := append(append([]string{}, step.Problems...),
		ValidatePlan(plan, spec.ScenarioNames(), spec.MessageIDs(), spec.RequiredIDs())...)
	if len(problems) > 0 && !opts.Lenient {
		return nil, &PlanValidationError{SourceName: name, Problems: problems}
	}
	pkg := opts.Package
	if pkg == "" {
		pkg = PackageName(step.Dir)
	}
	src := RenderTests(plan, RenderOptions{Package: pkg, Generator: step.Generator})
	if err := os.WriteFile(r.TestPath, []byte(src), 0o644); err != nil {
		return nil, err
	}
	brief := RenderBrief(spec, plan, r.TestPath, problems)
	if err := os.WriteFile(r.BriefPath, []byte(brief), 0o644); err != nil {
		return nil, err
	}
	return r, nil
}
