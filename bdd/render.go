package bdd

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// Deterministic rendering of a TestPlan into a Go test file (pyllym's
// bdd/renderer.py, rendering Go tests instead of pytest).
//
// The LLM's output stays confined to the plan schema; everything that becomes
// code on disk goes through these templates, so re-planning produces
// meaningful diffs and house style is enforced here, not by the model.
//
// Mapping from pyllym's pytest output:
//
//   - pytestmark = bdd_pending  ->  a //go:build bdd_pending constraint, so
//     the red suite only builds with -tags bdd_pending and the package's
//     regular `go test ./...` stays green until the implementation lands.
//   - def test_x(fixture)  ->  func TestX(t *testing.T) with
//     `fixture := fixtureFixture(t)`; fixtures are typed helpers.
//   - result = <when>  ->  result := <when>; assert <then>  ->
//     if !(<then>) { t.Fatalf(...) }.
//   - @pytest.mark.asyncio  ->  ctx := t.Context() for is_async cases.
//   - imports render verbatim; unused ones are dropped (Go rejects them).
//
// when/then expressions and fixture types are printed from their parsed AST,
// so the code on disk is exactly what ScanPlan screened. Free text that lands
// in comments (feature, scenario, notes, covers) is flattened to one line.

// PendingTag is the build tag every generated test file carries: Go's
// equivalent of pyllym's bdd_pending pytest marker.
const PendingTag = "bdd_pending"

// RenderOptions configure RenderTests.
type RenderOptions struct {
	// Package is the test file's package clause (default "generated").
	Package string
	// Generator names the tool in the header (default "gollym/bdd").
	Generator string
}

// TestFileName is the generated test file's name for a spec slug.
func TestFileName(slug string) string { return slug + "_test.go" }

var (
	nonIdentRe = regexp.MustCompile(`[^0-9a-zA-Z_]+`)
	humpRe     = regexp.MustCompile(`([a-z0-9])([A-Z])`)
	pkgRe      = regexp.MustCompile(`[^a-z0-9_]+`)
)

// PackageName derives a Go package name from an output directory:
// "tests/generated" -> "generated"; unusable names fall back to "generated".
func PackageName(dir string) string {
	name := strings.Trim(pkgRe.ReplaceAllString(strings.ToLower(filepath.Base(dir)), "_"), "_")
	if name == "" || unicode.IsDigit(rune(name[0])) || token.IsKeyword(name) {
		return "generated"
	}
	return name
}

// snake is pyllym's _identifier sanitisation (camelCase humps split first).
func snake(name string) string {
	s := humpRe.ReplaceAllString(strings.TrimSpace(name), "${1}_${2}")
	return strings.ToLower(strings.Trim(nonIdentRe.ReplaceAllString(s, "_"), "_"))
}

func camel(words []string) string {
	var b strings.Builder
	for _, w := range words {
		if w == "" {
			continue
		}
		b.WriteString(strings.ToUpper(w[:1]) + w[1:])
	}
	return b.String()
}

// testFuncName maps a plan test_name (test_first_retry) to a Go test
// function name (TestFirstRetry).
func testFuncName(name string) string {
	words := strings.Split(snake(name), "_")
	if len(words) > 0 && words[0] == "test" {
		words = words[1:]
	}
	if c := camel(words); c != "" {
		return "Test" + c
	}
	return "TestCase"
}

// fixtureVar maps a fixture name to the local variable tests bind it to.
func fixtureVar(name string) string {
	v := snake(name)
	switch {
	case v == "":
		return "fixture"
	case unicode.IsDigit(rune(v[0])) || token.IsKeyword(v):
		return "fixture_" + v
	case v == "t" || v == "result" || v == "ctx" || v == "testing":
		return v + "Fixture"
	}
	return v
}

func unique(name string, taken map[string]bool) string {
	candidate := name
	for n := 2; taken[candidate]; n++ {
		candidate = fmt.Sprintf("%s_%d", name, n)
	}
	taken[candidate] = true
	return candidate
}

// TestFuncNames are the Go test function names RenderTests gives the plan's
// cases, in order (sanitised and de-duplicated).
func TestFuncNames(plan *TestPlan) []string {
	taken := map[string]bool{}
	out := make([]string, len(plan.Cases))
	for i, c := range plan.Cases {
		out[i] = unique(testFuncName(c.TestName), taken)
	}
	return out
}

// oneLine flattens free text for a // comment.
func oneLine(s string) string {
	return strings.Join(strings.FieldsFunc(s, func(r rune) bool {
		return r == '\n' || r == '\r' || r == ' ' || r == ' '
	}), " ")
}

// printExpr re-prints an expression from its AST (comments dropped); the raw
// text is used when it does not parse (ScanPlan rejects such plans).
func printExpr(src string) string {
	expr, err := parseExpr(token.NewFileSet(), src)
	if err != nil {
		return strings.TrimSpace(src)
	}
	var buf bytes.Buffer
	if err := format.Node(&buf, token.NewFileSet(), expr); err != nil {
		return strings.TrimSpace(src)
	}
	return buf.String()
}

// refs collects the identifiers a snippet set uses: plain names (excluding
// selected field/method names) and the package qualifiers of selectors. ok
// is false when any snippet fails to parse.
type refs struct {
	names, qualifiers map[string]bool
	ok                bool
}

func newRefs() *refs { return &refs{names: map[string]bool{}, qualifiers: map[string]bool{}, ok: true} }

func (r *refs) add(node ast.Node) {
	ast.Inspect(node, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.SelectorExpr:
			if id, ok := x.X.(*ast.Ident); ok {
				r.qualifiers[id.Name] = true
			}
			r.add(x.X)
			return false
		case *ast.Ident:
			r.names[x.Name] = true
		}
		return true
	})
}

func (r *refs) stmts(lines []string) {
	if len(lines) == 0 || skippable(strings.Join(lines, "\n")) {
		return
	}
	body, err := parseStmts(token.NewFileSet(), strings.Join(lines, "\n"))
	if err != nil {
		r.ok = false
		return
	}
	r.add(body)
}

func (r *refs) expr(src string) {
	if skippable(src) {
		return
	}
	e, err := parseExpr(token.NewFileSet(), src)
	if err != nil {
		r.ok = false
		return
	}
	r.add(e)
}

// assumedName is the package name an import path is assumed to declare
// (goimports' ImportPathToAssumedName).
func assumedName(importPath string) string {
	base := path.Base(importPath)
	if strings.HasPrefix(base, "v") {
		if _, err := strconv.Atoi(base[1:]); err == nil {
			if dir := path.Dir(importPath); dir != "." {
				base = path.Base(dir)
			}
		}
	}
	base = strings.TrimPrefix(base, "go-")
	if i := strings.IndexFunc(base, func(r rune) bool {
		return !(unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_')
	}); i >= 0 {
		base = base[:i]
	}
	return base
}

type importLine struct{ name, path string }

func (l importLine) String() string {
	if l.name != "" {
		return l.name + " " + strconv.Quote(l.path)
	}
	return strconv.Quote(l.path)
}

// renderImports returns the import block lines: "testing", the API imports
// (plus the module under test) that the snippets reference, std first.
func renderImports(plan *TestPlan, used *refs) []string {
	lines := []importLine{{path: "testing"}}
	specs := append([]string{}, plan.API.Imports...)
	hasModule := false
	for _, imp := range specs {
		if s, err := parseImport(imp); err == nil && importPath(s) == plan.API.Module {
			hasModule = true
		}
	}
	if !hasModule && plan.API.Module != "" {
		specs = append(specs, strconv.Quote(plan.API.Module))
	}
	seen := map[importLine]bool{{path: "testing"}: true}
	for _, imp := range specs {
		spec, err := parseImport(imp)
		if err != nil {
			continue
		}
		l := importLine{path: importPath(spec)}
		name := assumedName(l.path)
		if spec.Name != nil {
			l.name = spec.Name.Name
			name = l.name
		}
		keep := name == "_" || name == "." || !used.ok || used.qualifiers[name]
		if keep && !seen[l] {
			seen[l] = true
			lines = append(lines, l)
		}
	}
	var std, other []string
	for _, l := range lines {
		first, _, _ := strings.Cut(l.path, "/")
		if strings.Contains(first, ".") {
			other = append(other, l.String())
		} else {
			std = append(std, l.String())
		}
	}
	sort.Strings(std)
	sort.Strings(other)
	if len(other) > 0 {
		return append(append(std, ""), other...)
	}
	return std
}

// RenderTests renders a complete, red Go test file for the plan. Run
// ScanPlan first: Build always does.
func RenderTests(plan *TestPlan, opts RenderOptions) string {
	pkg := opts.Package
	if pkg == "" {
		pkg = "generated"
	}
	generator := opts.Generator
	if generator == "" {
		generator = "gollym/bdd"
	}

	// Which packages and names the snippets use (to drop unused imports and
	// silence unused fixtures/results, both compile errors in Go).
	all := newRefs()
	for _, f := range plan.Fixtures {
		all.stmts(f.Body)
		if strings.TrimSpace(f.Type) != "" {
			all.expr(f.Type)
		}
	}
	caseRefs := make([]*refs, len(plan.Cases))
	for i, c := range plan.Cases {
		r := newRefs()
		r.stmts(c.Given)
		if !isNoAct(c.When) {
			r.expr(c.When)
		}
		for _, e := range c.Then {
			r.expr(e)
		}
		caseRefs[i] = r
		for k := range r.qualifiers {
			all.qualifiers[k] = true
		}
		all.ok = all.ok && r.ok
	}

	var out []string
	out = append(out,
		"//go:build "+PendingTag,
		"",
		fmt.Sprintf("// Code generated by %s from feature: %s. DO NOT EDIT.", generator, oneLine(plan.Feature)),
		"// These tests are the specification. Do not edit them during implementation;",
		"// regenerate from the .toml spec instead.",
		"//",
		"// They are red until implemented and only build with the "+PendingTag+" tag:",
		"//",
		"//\tgo test -tags "+PendingTag+" ./...",
		"",
		"package "+pkg,
		"",
		"import (",
	)
	for _, l := range renderImports(plan, all) {
		if l == "" {
			out = append(out, "")
		} else {
			out = append(out, "\t"+l)
		}
	}
	out = append(out, ")")

	// Fixtures: typed helpers called by the tests that request them.
	type fixture struct{ variable, helper string }
	fixtures := map[string]fixture{}
	takenVars, takenHelpers := map[string]bool{}, map[string]bool{}
	for _, f := range plan.Fixtures {
		v := unique(fixtureVar(f.Name), takenVars)
		h := unique("fixture"+camel(strings.Split(snake(v), "_")), takenHelpers)
		if _, dup := fixtures[f.Name]; !dup {
			fixtures[f.Name] = fixture{v, h}
		}
		typ := "any"
		if strings.TrimSpace(f.Type) != "" {
			typ = printExpr(f.Type)
		}
		out = append(out, "",
			fmt.Sprintf("// %s is the %s fixture.", h, strconv.Quote(oneLine(f.Name))),
			fmt.Sprintf("func %s(t *testing.T) %s {", h, typ),
			"\tt.Helper()")
		if len(f.Body) == 0 {
			out = append(out, "\treturn *new("+typ+")")
		}
		for _, line := range f.Body {
			out = append(out, line)
		}
		out = append(out, "}")
	}

	names := TestFuncNames(plan)
	for i, c := range plan.Cases {
		used := caseRefs[i]
		unused := func(name string) bool { return !used.ok || !used.names[name] }
		out = append(out, "")
		if c.EdgeCase {
			out = append(out, "// Edge case derived from: "+oneLine(c.Scenario))
		}
		out = append(out, "func "+names[i]+"(t *testing.T) {")
		for _, name := range c.Fixtures {
			fx, ok := fixtures[name]
			if !ok {
				v := fixtureVar(name)
				fx = fixture{v, "fixture" + camel(strings.Split(snake(v), "_"))}
			}
			out = append(out, fmt.Sprintf("%s := %s(t)", fx.variable, fx.helper))
			if unused(fx.variable) {
				out = append(out, "_ = "+fx.variable)
			}
		}
		if c.IsAsync {
			out = append(out, "ctx := t.Context()")
			if unused("ctx") {
				out = append(out, "_ = ctx")
			}
		}
		if len(c.Covers) > 0 {
			out = append(out, "// Covers: "+oneLine(strings.Join(c.Covers, ", ")))
		}
		if c.Notes != "" {
			out = append(out, "// "+oneLine(c.Notes))
		}
		if len(c.Given) > 0 {
			out = append(out, "// Given: "+oneLine(c.Scenario))
			out = append(out, c.Given...)
		}
		if isNoAct(c.When) {
			// The canonical error-path form puts the act in given.
			out = append(out, "// When: the act is part of Given")
		} else {
			out = append(out, "// When", "result := "+printExpr(c.When))
			if unused("result") {
				out = append(out, "_ = result")
			}
		}
		if len(c.Then) > 0 {
			out = append(out, "// Then")
			for _, expr := range c.Then {
				e := printExpr(expr)
				out = append(out,
					"if !("+e+") {",
					"t.Fatalf(\"then failed: %s\", "+strconv.Quote(e)+")",
					"}")
			}
		}
		out = append(out, "}")
	}

	src := strings.Join(out, "\n") + "\n"
	if formatted, err := format.Source([]byte(src)); err == nil {
		return string(formatted)
	}
	return src
}

// AssumedPackageName is the package name an import path is assumed to
// declare (goimports' heuristic: last element, minus a /vN suffix, a go-
// prefix and anything after the first non-identifier character).
func AssumedPackageName(importPath string) string { return assumedName(importPath) }

// ParseImportSpec parses one API import: `"path"`, `name "path"`, a bare
// path, or a one-spec `import` statement. name is "" without an alias.
func ParseImportSpec(spec string) (name, path string, err error) {
	s, err := parseImport(spec)
	if err != nil {
		return "", "", err
	}
	if s.Name != nil {
		name = s.Name.Name
	}
	return name, importPath(s), nil
}

// SelectedNames are the names the plan's snippets select from qualifier
// (retry.Backoff -> "Backoff" for qualifier "retry"), in first-use order:
// what the tests need the package under test to define. Snippets that do
// not parse are skipped.
func SelectedNames(plan *TestPlan, qualifier string) []string {
	var out []string
	seen := map[string]bool{}
	visit := func(node ast.Node) {
		ast.Inspect(node, func(n ast.Node) bool {
			if sel, ok := n.(*ast.SelectorExpr); ok {
				if id, ok := sel.X.(*ast.Ident); ok && id.Name == qualifier && !seen[sel.Sel.Name] {
					seen[sel.Sel.Name] = true
					out = append(out, sel.Sel.Name)
				}
			}
			return true
		})
	}
	stmts := func(lines []string) {
		if len(lines) == 0 {
			return
		}
		if body, err := parseStmts(token.NewFileSet(), strings.Join(lines, "\n")); err == nil {
			visit(body)
		}
	}
	expr := func(src string) {
		if e, err := parseExpr(token.NewFileSet(), src); err == nil {
			visit(e)
		}
	}
	for _, f := range plan.Fixtures {
		if strings.TrimSpace(f.Type) != "" {
			expr(f.Type)
		}
		stmts(f.Body)
	}
	for _, c := range plan.Cases {
		stmts(c.Given)
		if !isNoAct(c.When) {
			expr(c.When)
		}
		for _, e := range c.Then {
			expr(e)
		}
	}
	return out
}
