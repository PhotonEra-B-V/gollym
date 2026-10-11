package tdg

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os/exec"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/PhotonEra-B-V/gollym/bdd"
)

// Inspecting the real dependencies of the package under test (pyllym's
// tdg/deps.py).
//
// The planner otherwise sees only the declared API surface: the import path,
// imports and signatures a human wrote into the TOML spec. Nothing checks
// that those declarations match what the package actually pulls in. This
// file closes that gap: it discovers the package's true imports and
// cross-checks them against the spec, so generated tests reflect the
// package's real collaborators rather than a stale hand-written surface.
//
// Discovery is static-first, go-list-fallback (see InspectDependencies):
//
//   - Static: locate the package with `go list` (Go's find_spec; it resolves
//     the import path in the module context but runs no package code) and
//     parse its non-test .go files with go/parser for imports, the qualified
//     names it uses from them, and its top-level declarations.
//   - Fallback: when the files can't be parsed (syntax errors, no files in
//     the current build context) use go list's own import metadata.
//     pyllym's fallback imports the module and inspects it at runtime; Go
//     cannot load a package at runtime, and never needs to execute it.
//
// CheckDependencies returns human-readable problems in the same shape as
// bdd.ValidatePlan, so the builder folds them into the brief and fails the
// build under strict exactly like a coverage gap.

// Dependency sources.
const (
	SourceStatic = "static" // go/parser over the package's files
	SourceGoList = "golist" // go list's import metadata (fallback)
)

// Dependencies is what a package actually depends on.
type Dependencies struct {
	// Module is the inspected import path.
	Module string
	// Name is the package clause name ("" when unknown).
	Name string
	// GoModule is the path of the Go module containing the package: Go's
	// counterpart of pyllym's top-level package for same-project imports.
	GoModule string
	// Imports are the import paths of the package's non-test files.
	Imports []string
	// Members are the qualified names it uses from its imports
	// ("json.Marshal"): Go's counterpart of `from x import y` names.
	Members []string
	// Defined are the package's top-level declarations (funcs, types, vars,
	// consts): what the tests may select from it.
	Defined []string
	// Source is SourceStatic or SourceGoList.
	Source   string
	Warnings []string
}

// ToPromptBlock is a compact block naming the real dependencies for the
// planner.
func (d Dependencies) ToPromptBlock() string {
	lines := []string{fmt.Sprintf("Real dependencies of %s (discovered via %s analysis):", d.Module, d.Source)}
	if len(d.Imports) > 0 {
		lines = append(lines, "  imports: "+strings.Join(d.Imports, ", "))
	}
	if len(d.Members) > 0 {
		lines = append(lines, "  imported names: "+strings.Join(d.Members, ", "))
	}
	if len(d.Defined) > 0 {
		lines = append(lines, "  defines: "+strings.Join(d.Defined, ", "))
	}
	if len(d.Imports) == 0 && len(d.Members) == 0 {
		lines = append(lines, "  (none — the module imports nothing external)")
	}
	return strings.Join(lines, "\n") + "\n"
}

// listedPackage is the subset of `go list -json` output used here.
type listedPackage struct {
	Dir      string
	Name     string
	GoFiles  []string
	CgoFiles []string
	Imports  []string
	Module   *struct{ Path string }
	Error    *struct{ Err string }
}

// goList locates a package without running any of its code.
func goList(dir, importPath string) (*listedPackage, error) {
	cmd := exec.Command("go", "list", "-e", "-json=Dir,Name,GoFiles,CgoFiles,Imports,Module,Error", "--", importPath)
	cmd.Dir = dir
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("go list: %v: %s", err, strings.TrimSpace(stderr.String()))
	}
	var p listedPackage
	if err := json.Unmarshal(out, &p); err != nil {
		return nil, fmt.Errorf("go list: %v", err)
	}
	if p.Dir == "" || (p.Error != nil && len(p.GoFiles)+len(p.CgoFiles) == 0 && len(p.Imports) == 0) {
		msg := "package not found"
		if p.Error != nil {
			msg = p.Error.Err
		}
		return nil, fmt.Errorf("%s", msg)
	}
	return &p, nil
}

// staticDependencies parses the package's files: no code runs. It returns
// nil (so the caller falls back to go list's metadata) when there is no
// parseable source.
func staticDependencies(module string, p *listedPackage) *Dependencies {
	files := append(append([]string{}, p.GoFiles...), p.CgoFiles...)
	if len(files) == 0 {
		return nil
	}
	imports, members, defined := set{}, set{}, set{}
	fset := token.NewFileSet()
	for _, name := range files {
		f, err := parser.ParseFile(fset, filepath.Join(p.Dir, name), nil, parser.SkipObjectResolution)
		if err != nil {
			return nil
		}
		local := map[string]bool{}
		for _, spec := range f.Imports {
			path, _ := strconv.Unquote(spec.Path.Value)
			imports[path] = true
			n := bdd.AssumedPackageName(path)
			if spec.Name != nil {
				n = spec.Name.Name
			}
			local[n] = true
		}
		ast.Inspect(f, func(n ast.Node) bool {
			if sel, ok := n.(*ast.SelectorExpr); ok {
				if id, ok := sel.X.(*ast.Ident); ok && local[id.Name] {
					members[id.Name+"."+sel.Sel.Name] = true
				}
			}
			return true
		})
		// Top-level names the package defines: what a test may select.
		for _, decl := range f.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				if d.Recv == nil {
					defined[d.Name.Name] = true
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						defined[s.Name.Name] = true
					case *ast.ValueSpec:
						for _, n := range s.Names {
							defined[n.Name] = true
						}
					}
				}
			}
		}
	}
	delete(defined, "_")
	return &Dependencies{
		Module: module, Name: p.Name, GoModule: modulePath(p),
		Imports: imports.sorted(), Members: members.sorted(), Defined: defined.sorted(),
		Source: SourceStatic,
	}
}

func modulePath(p *listedPackage) string {
	if p.Module != nil {
		return p.Module.Path
	}
	return ""
}

// InspectDependencies discovers module's real dependencies from the current
// directory's module context: see InspectDependenciesIn.
func InspectDependencies(module string) Dependencies { return InspectDependenciesIn("", module) }

// InspectDependenciesIn discovers module's real dependencies, resolving the
// import path from dir ("" = current directory): static parse first, go list
// metadata as fallback.
//
// Never fails for an unanalyzable package: the result carries Warnings
// instead, so dependency checking can only add information, never break a
// build that would otherwise succeed (strict builds still fail on them, as
// in pyllym).
func InspectDependenciesIn(dir, module string) Dependencies {
	p, err := goList(dir, module)
	if err != nil {
		return Dependencies{
			Module:   module,
			Source:   SourceGoList,
			Warnings: []string{fmt.Sprintf("could not import %s to inspect dependencies: %v", pyRepr(module), err)},
		}
	}
	if static := staticDependencies(module, p); static != nil {
		return *static
	}
	imports := set{}
	for _, i := range p.Imports {
		imports[i] = true
	}
	return Dependencies{Module: module, Name: p.Name, GoModule: modulePath(p), Imports: imports.sorted(), Source: SourceGoList}
}

// CheckDependencies cross-checks the spec's declared imports, and the names
// its tests select from the package under test, against the real
// dependencies.
//
// declaredImports are the raw [api].imports specs. Each must name the
// package under test, a package it imports, or a package of the same Go
// module. used are the names the tests select from the package under test
// (bdd.SelectedNames); each must be one of its top-level declarations (Go's
// counterpart of pyllym's check that `from module import X` names something
// the module has). That check only runs when the package's own names could
// be read. Inspection warnings are surfaced as problems too, so they reach
// the brief.
func CheckDependencies(declaredImports, used []string, deps Dependencies) []string {
	problems := append([]string{}, deps.Warnings...)
	known := map[string]bool{deps.Module: true}
	for _, i := range deps.Imports {
		known[i] = true
	}
	sameModule := func(path string) bool {
		return deps.GoModule != "" && (path == deps.GoModule || strings.HasPrefix(path, deps.GoModule+"/"))
	}
	for _, statement := range declaredImports {
		_, path, err := bdd.ParseImportSpec(statement)
		if err != nil {
			problems = append(problems, "could not parse declared import: "+pyRepr(statement))
			continue
		}
		if !known[path] && !sameModule(path) {
			problems = append(problems, fmt.Sprintf("declared import references %s, which %s does not depend on",
				pyRepr(path), pyRepr(deps.Module)))
		}
	}
	if len(deps.Defined) > 0 {
		for _, name := range used {
			if !slices.Contains(deps.Defined, name) {
				problems = append(problems, fmt.Sprintf("declared use of %s from %s is not present in %s",
					pyRepr(name), pyRepr(deps.Module), pyRepr(deps.Module)))
			}
		}
	}
	return problems
}

// LocalName is the name the tests use for the package under test: the alias
// of its [api].imports entry, else its package clause name, else the name
// its import path suggests.
func LocalName(spec *bdd.TomlSpec, deps Dependencies) string {
	for _, imp := range spec.API.Imports {
		if name, path, err := bdd.ParseImportSpec(imp); err == nil && path == spec.API.Module && name != "" && name != "_" && name != "." {
			return name
		}
	}
	if deps.Name != "" {
		return deps.Name
	}
	return bdd.AssumedPackageName(spec.API.Module)
}

type set map[string]bool

func (s set) sorted() []string {
	out := make([]string, 0, len(s))
	for k := range s {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// pyRepr quotes like Python's repr for the simple strings used in messages.
func pyRepr(s string) string {
	if strings.ContainsRune(s, '\'') && !strings.ContainsRune(s, '"') {
		return `"` + strings.ReplaceAll(s, `\`, `\\`) + `"`
	}
	return "'" + strings.ReplaceAll(strings.ReplaceAll(s, `\`, `\\`), "'", `\'`) + "'"
}
