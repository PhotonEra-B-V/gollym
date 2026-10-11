package tdg

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/PhotonEra-B-V/gollym/bdd"
)

// Port of pyllym tests/test_tdg_deps.py: dependency inspection and
// cross-checking, against packages of a scratch Go module.

const pkgModule = "example.com/pkgs"

// pkg creates a scratch module; write adds a package file to it.
func pkg(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go tool not available")
	}
	t.Setenv("GOWORK", "off")
	t.Setenv("GOFLAGS", "-mod=mod")
	t.Setenv("GOPROXY", "off")
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "go.mod"), "module "+pkgModule+"\n\ngo 1.24\n")
	return dir
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeModule(t *testing.T, root, name, body string) string {
	t.Helper()
	writeFile(t, filepath.Join(root, name, name+".go"), body)
	return pkgModule + "/" + name
}

func anyContains(problems []string, subs ...string) bool {
	for _, p := range problems {
		ok := true
		for _, s := range subs {
			ok = ok && strings.Contains(p, s)
		}
		if ok {
			return true
		}
	}
	return false
}

// -- static discovery --------------------------------------------------------

func TestStaticDiscoversImportsAndMembers(t *testing.T) {
	root := pkg(t)
	mod := writeModule(t, root, "retrymod", `package retrymod

import (
	"encoding/json"
	"os"

	"github.com/acme/httpx"
)

type RetryPolicy struct{ Attempts int }

func Encode(p RetryPolicy) ([]byte, error) { _ = os.Getpid; _ = httpx.Client; return json.Marshal(p) }
`)
	deps := InspectDependenciesIn(root, mod)
	if deps.Source != SourceStatic || len(deps.Warnings) != 0 {
		t.Fatalf("deps %+v", deps)
	}
	for _, want := range []string{"os", "encoding/json", "github.com/acme/httpx"} {
		if !slices.Contains(deps.Imports, want) {
			t.Fatalf("imports %v missing %s", deps.Imports, want)
		}
	}
	if !slices.Contains(deps.Members, "json.Marshal") || !slices.Contains(deps.Members, "httpx.Client") {
		t.Fatalf("members %v", deps.Members)
	}
	if !slices.Equal(deps.Defined, []string{"Encode", "RetryPolicy"}) || deps.Name != "retrymod" || deps.GoModule != pkgModule {
		t.Fatalf("deps %+v", deps)
	}
	if block := deps.ToPromptBlock(); !strings.Contains(block, "Real dependencies of "+mod+" (discovered via static analysis):") {
		t.Fatalf("block %s", block)
	}
}

// Go has no relative imports; the analogue of pyllym's "relative imports
// contribute nothing" is that test files and build-excluded files are not
// dependencies of the package.
func TestStaticIgnoresTestAndExcludedFiles(t *testing.T) {
	root := pkg(t)
	mod := writeModule(t, root, "relpkg", "package relpkg\n\nimport \"encoding/json\"\n\nvar _ = json.Valid\n")
	writeFile(t, filepath.Join(root, "relpkg", "relpkg_test.go"), "package relpkg\n\nimport \"net/http\"\n\nvar _ = http.Get\n")
	writeFile(t, filepath.Join(root, "relpkg", "tool.go"), "//go:build ignore\n\npackage main\n\nimport \"os/exec\"\n\nvar _ = exec.Command\n")
	deps := InspectDependenciesIn(root, mod)
	if !slices.Equal(deps.Imports, []string{"encoding/json"}) {
		t.Fatalf("imports %v", deps.Imports)
	}
}

// -- go list fallback ----------------------------------------------------------

func TestGoListFallbackWhenSourceUnparseable(t *testing.T) {
	// go list reads the import block; the broken body defeats the static parse.
	root := pkg(t)
	mod := writeModule(t, root, "broken", "package broken\n\nimport \"strings\"\n\nfunc F( {\n")
	deps := InspectDependenciesIn(root, mod)
	if deps.Source != SourceGoList || !slices.Equal(deps.Imports, []string{"strings"}) || len(deps.Defined) != 0 {
		t.Fatalf("deps %+v", deps)
	}
}

func TestUnimportableModuleDegradesToWarning(t *testing.T) {
	root := pkg(t)
	deps := InspectDependenciesIn(root, pkgModule+"/does_not_exist_anywhere_xyz")
	if deps.Source != SourceGoList || len(deps.Warnings) == 0 || len(deps.Imports) != 0 {
		t.Fatalf("deps %+v", deps)
	}
}

// -- cross-checking ------------------------------------------------------------

func TestCheckFlagsImportFromUndeclaredModule(t *testing.T) {
	root := pkg(t)
	mod := writeModule(t, root, "onlyos", "package onlyos\n\nimport \"os\"\n\nvar Pid = os.Getpid\n")
	deps := InspectDependenciesIn(root, mod)
	problems := CheckDependencies([]string{`"encoding/xml"`}, nil, deps)
	if !anyContains(problems, "encoding/xml", "does not depend on") {
		t.Fatalf("problems %v", problems)
	}
}

func TestCheckPassesWhenDeclaredMatchesReal(t *testing.T) {
	root := pkg(t)
	mod := writeModule(t, root, "usesos", "package usesos\n\nimport (\n\t\"encoding/json\"\n\t\"os\"\n)\n\nvar Pid = os.Getpid\nvar Dumps = json.Marshal\n")
	writeModule(t, root, "sibling", "package sibling\n")
	deps := InspectDependenciesIn(root, mod)
	declared := []string{`"os"`, `import "encoding/json"`, `u "` + mod + `"`, `"` + pkgModule + `/sibling"`}
	if problems := CheckDependencies(declared, []string{"Pid", "Dumps"}, deps); len(problems) != 0 {
		t.Fatalf("problems %v", problems)
	}
}

func TestCheckFlagsMissingMemberFromTarget(t *testing.T) {
	root := pkg(t)
	mod := writeModule(t, root, "surface", "package surface\n\nimport \"encoding/json\"\n\nvar Dumps = json.Marshal\n")
	deps := InspectDependenciesIn(root, mod)
	// Loads is not defined by the target package
	problems := CheckDependencies([]string{`"` + mod + `"`}, []string{"Loads"}, deps)
	if !anyContains(problems, "Loads", "not present") {
		t.Fatalf("problems %v", problems)
	}
}

func TestCheckFlagsUnparseableImport(t *testing.T) {
	problems := CheckDependencies([]string{"from x import y"}, nil, Dependencies{Module: "m", Imports: []string{"x"}})
	if !slices.Equal(problems, []string{"could not parse declared import: 'from x import y'"}) {
		t.Fatalf("problems %v", problems)
	}
}

func TestWarningIsSurfacedAsProblem(t *testing.T) {
	root := pkg(t)
	deps := InspectDependenciesIn(root, pkgModule+"/nope_missing_module_abc")
	if problems := CheckDependencies(nil, nil, deps); !anyContains(problems, "could not import") {
		t.Fatalf("problems %v", problems)
	}
}

// -- builder integration (plan mode -> no LLM) -----------------------------------

func planSpec(module, member string) string {
	return `
[meta]
feature_name = "Uses OS"
mode = "plan"

[api]
module = "` + module + `"
imports = ['"` + module + `"']
signatures = ["func Run() int"]

[[rules]]
id = "rule_1"
text = "returns an int"

[[cases]]
test_name = "test_run_returns_int"
scenario = "runs"
when = "planmod.` + member + `()"
then = ["result >= 0"]
covers = ["rule_1"]
`
}

func TestBuildStrictFailsOnDepMismatch(t *testing.T) {
	root := pkg(t)
	mod := writeModule(t, root, "planmod", "package planmod\n\nimport \"os\"\n\nfunc Run() int { return os.Getpid() * 0 }\n")
	_, err := Build(context.Background(), planSpec(mod, "NonexistentThing"), filepath.Join(t.TempDir(), "out"), Options{Dir: root})
	var verr *bdd.PlanValidationError
	if !errors.As(err, &verr) || !anyContains(verr.Problems, "planmod") {
		t.Fatalf("err %v", err)
	}
}

func TestBuildReportsDepMismatchInBriefWhenNotStrict(t *testing.T) {
	root := pkg(t)
	mod := writeModule(t, root, "planmod", "package planmod\n\nimport \"os\"\n\nfunc Run() int { return os.Getpid() * 0 }\n")
	results, err := Build(context.Background(), planSpec(mod, "NonexistentThing"), filepath.Join(t.TempDir(), "out"),
		Options{Options: bdd.Options{Lenient: true}, Dir: root})
	if err != nil {
		t.Fatal(err)
	}
	brief, _ := os.ReadFile(results[0].BriefPath)
	if !strings.Contains(string(brief), "not present") && !strings.Contains(string(brief), "does not depend on") {
		t.Fatalf("brief:\n%s", brief)
	}
	src, _ := os.ReadFile(results[0].TestPath)
	if !strings.Contains(string(src), "// Code generated by gollym/tdg") {
		t.Fatalf("header:\n%s", src)
	}
}

func TestBuildPassesWhenSpecMatchesPackage(t *testing.T) {
	root := pkg(t)
	mod := writeModule(t, root, "planmod", "package planmod\n\nimport \"os\"\n\nfunc Run() int { return os.Getpid() * 0 }\n")
	results, err := Build(context.Background(), planSpec(mod, "Run"), filepath.Join(t.TempDir(), "out"), Options{Dir: root})
	if err != nil || results[0].Reused {
		t.Fatalf("results %v err %v", results, err)
	}
}

func TestBuildNoCheckDepsSkipsInspection(t *testing.T) {
	root := pkg(t)
	mod := writeModule(t, root, "planmod", "package planmod\n\nimport \"os\"\n\nfunc Run() int { return os.Getpid() * 0 }\n")
	// would fail strict if deps were checked; SkipDeps skips it
	results, err := Build(context.Background(), planSpec(mod, "NonexistentThing"), filepath.Join(t.TempDir(), "out"),
		Options{SkipDeps: true, Dir: root})
	if err != nil || len(results[0].Plan.Cases) == 0 {
		t.Fatalf("results %v err %v", results, err)
	}
}

// Request mode hands the discovered dependencies to the planner.
type hintPlanner struct{ hint string }

func (p *hintPlanner) Plan(_ context.Context, spec *bdd.TomlSpec, hint string) (*bdd.TestPlan, error) {
	p.hint = hint
	plan := &bdd.TestPlan{Feature: spec.Meta.FeatureName, API: spec.API.ToSurface(),
		Cases: []bdd.TestCase{{Scenario: "runs", TestName: "test_run", When: "planmod.Run()", Then: []string{"result == 0"}, Covers: []string{"rule_1"}}}}
	return plan, nil
}

func TestRequestModeFeedsDependenciesToPlanner(t *testing.T) {
	root := pkg(t)
	mod := writeModule(t, root, "planmod", "package planmod\n\nimport \"os\"\n\nfunc Run() int { return os.Getpid() * 0 }\n")
	spec := strings.Replace(planSpec(mod, "Run"), `mode = "plan"`, `mode = "request"`, 1)
	planner := &hintPlanner{}
	_, err := Build(context.Background(), spec, filepath.Join(t.TempDir(), "out"),
		Options{Options: bdd.Options{Planner: planner, APIHint: "keep Run"}, Dir: root})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(planner.hint, "keep Run\nReal dependencies of "+mod) || !strings.Contains(planner.hint, "imports: os") {
		t.Fatalf("hint %q", planner.hint)
	}
}
