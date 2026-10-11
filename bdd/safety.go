package bdd

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"strconv"
	"strings"
	"unicode"
)

// AST safety gate for the executable strings in a TestPlan (pyllym's
// bdd/safety.py, screening Go instead of Python).
//
// The pipeline renders LLM-authored (request mode) or hand-authored (plan
// mode) strings (a case's given/when/then, fixture bodies and types, and the
// API imports) into a Go test file that someone then runs. Every such string
// is therefore executable Go. This file screens them before rendering, with
// go/parser + go/ast, so a hallucinated os/exec call or a prompt-injected
// syscall fails the build loudly instead of landing on disk as runnable code.
//
// This is defense in depth, not a sandbox. It is a denylist over an AST walk:
// it reliably catches accidents and low-effort payloads and is far harder to
// fool than a substring scan, but a determined attacker can construct Go that
// evades any static denylist (for instance through a dependency's API). The
// real trust boundary is procedural: never execute a freshly generated suite
// in an environment with secrets or network access before a human has
// reviewed the diff, and run generation/execution of untrusted specs in a
// sandbox.
//
// The gate also parses each string, so syntactically invalid Go (a common
// failure mode of the request-mode model, e.g. Python leaking through) is
// caught here rather than at compile time. Each snippet is parsed in the same
// lexical context the renderer places it in (statements inside a function
// body, expressions inside parentheses), so what was screened is what runs.

// bannedNames are identifiers whose use is disallowed in a snippet: the
// default package names of the filesystem/process/network/reflection
// packages (unreachable anyway, since snippets cannot import and bannedImports
// screens the API imports, but a second line of defense against packages of
// the same name). Field selectors (x.net) are not affected; common words
// (user, build, debug, signal) are left alone to avoid false positives.
var bannedNames = map[string]bool{
	"os": true, "exec": true, "syscall": true, "unsafe": true, "reflect": true,
	"net": true, "http": true, "httptest": true, "rpc": true, "smtp": true,
	"tls": true, "plugin": true, "runtime": true, "cgo": true, "ioutil": true,
	"filepath": true, "syslog": true, "unix": true, "windows": true, "C": true,
}

// bannedImports are import paths (and, with a trailing slash, path prefixes)
// the API surface may not import: aliasing ("x \"os\"") would otherwise
// sidestep bannedNames.
var bannedImports = []string{
	"C", "os", "os/", "syscall", "unsafe", "reflect", "plugin", "embed",
	"runtime", "runtime/", "net", "net/", "crypto/tls", "io/ioutil",
	"path/filepath", "log/syslog", "debug/", "go/build", "internal/",
	"golang.org/x/sys/",
}

func bannedImport(path string) bool {
	for _, b := range bannedImports {
		if path == b || (strings.HasSuffix(b, "/") && strings.HasPrefix(path, b)) {
			return true
		}
	}
	return false
}

// directive reports a compiler/cgo directive comment (//go:linkname,
// //line, //export, #cgo).
func directive(comment string) bool {
	return strings.HasPrefix(comment, "//go:") || strings.HasPrefix(comment, "//line ") ||
		strings.HasPrefix(comment, "/*line ") || strings.HasPrefix(comment, "//export ") ||
		strings.Contains(comment, "#cgo")
}

// isNoAct reports the "no act" when sentinel: empty, nil, or pyllym's None.
func isNoAct(when string) bool {
	w := strings.TrimSpace(when)
	return w == "" || w == "nil" || w == "None"
}

// NoAct is the canonical when for cases whose act lives in given.
const NoAct = "nil"

// screen collects problems for one snippet. where labels a 0-based line of
// the snippet; line maps a parsed node back to that line.
type screen struct {
	problems []string
	where    func(line int) string
	line     func(token.Pos) int
}

func (s *screen) flag(line int, format string, args ...any) {
	s.problems = append(s.problems, s.where(line)+": "+fmt.Sprintf(format, args...))
}

// tokens flags import/package clauses and directive comments, which the
// parser would only report as a generic syntax error (or keep silently).
func (s *screen) tokens(src string) {
	fset := token.NewFileSet()
	file := fset.AddFile("", -1, len(src))
	var sc scanner.Scanner
	sc.Init(file, []byte(src), func(token.Position, string) {}, scanner.ScanComments)
	for {
		pos, tok, lit := sc.Scan()
		if tok == token.EOF {
			return
		}
		line := file.Line(pos) - 1
		switch tok {
		case token.IMPORT:
			var names []string
			for {
				_, t, l := sc.Scan()
				if t == token.STRING {
					if p, err := strconv.Unquote(l); err == nil {
						names = append(names, p)
					}
					continue
				}
				if t == token.IDENT || t == token.LPAREN || t == token.PERIOD || (t == token.SEMICOLON && l == "\n") {
					continue
				}
				break
			}
			if len(names) == 0 {
				names = []string{"?"}
			}
			s.flag(line, "import statement is not allowed (%s)", strings.Join(names, ", "))
			return
		case token.PACKAGE:
			s.flag(line, "package clause is not allowed")
		case token.COMMENT:
			if directive(lit) {
				s.flag(line, "compiler directive is not allowed (%s)", strings.TrimSpace(firstLine(lit)))
			}
		}
	}
}

// walk screens an AST for banned identifiers and goroutines.
func (s *screen) walk(node ast.Node) {
	ast.Inspect(node, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.SelectorExpr:
			s.walk(x.X) // the selected name itself is a field/method, not a package
			return false
		case *ast.Ident:
			if bannedNames[x.Name] {
				s.flag(s.line(x.Pos()), "disallowed name %s", pyRepr(x.Name))
			}
		case *ast.GoStmt:
			if _, ok := x.Call.Fun.(*ast.FuncLit); !ok {
				s.flag(s.line(x.Pos()), "go statement is only allowed with an inline func literal")
			}
		}
		return true
	})
}

func firstLine(s string) string {
	l, _, _ := strings.Cut(s, "\n")
	return l
}

const stmtPrefix = "package p\nfunc _() {\n"

// parseStmts parses src as statements of a function body: the context the
// renderer places given entries and fixture bodies in. It rejects input that
// closes the body early to smuggle in declarations.
func parseStmts(fset *token.FileSet, src string) (*ast.BlockStmt, error) {
	wrapped := stmtPrefix + src + "\n}\n"
	f, err := parser.ParseFile(fset, "", wrapped, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil {
		return nil, err
	}
	if len(f.Decls) != 1 {
		return nil, fmt.Errorf("snippet must be statements of one function body")
	}
	fn, ok := f.Decls[0].(*ast.FuncDecl)
	if !ok || fn.Body == nil || fset.Position(fn.Body.Rbrace).Offset != len(wrapped)-2 {
		return nil, fmt.Errorf("snippet must be statements of one function body")
	}
	return fn.Body, nil
}

// parseExpr parses src as one expression.
func parseExpr(fset *token.FileSet, src string) (ast.Expr, error) {
	return parser.ParseExprFrom(fset, "", src, parser.SkipObjectResolution)
}

func syntaxMsg(err error) string {
	if list, ok := err.(scanner.ErrorList); ok && len(list) > 0 {
		return list[0].Msg
	}
	return err.Error()
}

// screenStmts screens a statement block; whereLine maps a 0-based line of
// src to the location label.
func screenStmts(src string, whereLine func(int) string) []string {
	fset := token.NewFileSet()
	prefixLines := strings.Count(stmtPrefix, "\n")
	s := &screen{where: whereLine, line: func(pos token.Pos) int {
		return fset.Position(pos).Line - 1 - prefixLines
	}}
	s.tokens(src)
	if len(s.problems) > 0 {
		return s.problems
	}
	body, err := parseStmts(fset, src)
	if err != nil {
		line := 0
		if list, ok := err.(scanner.ErrorList); ok && len(list) > 0 {
			line = list[0].Pos.Line - 1 - prefixLines
		}
		return []string{whereLine(line) + ": not valid Go (" + syntaxMsg(err) + ")"}
	}
	s.walk(body)
	return s.problems
}

// screenExpr screens one expression.
func screenExpr(src, where string) []string {
	fset := token.NewFileSet()
	s := &screen{where: func(int) string { return where }, line: func(token.Pos) int { return 0 }}
	s.tokens(src)
	if len(s.problems) > 0 {
		return s.problems
	}
	expr, err := parseExpr(fset, src)
	if err != nil {
		return []string{where + ": not valid Go (" + syntaxMsg(err) + ")"}
	}
	s.walk(expr)
	return s.problems
}

// parseImport parses one import spec: `"path"`, `name "path"` or a full
// `import ...` statement with a single spec.
func parseImport(spec string) (*ast.ImportSpec, error) {
	s := strings.TrimSpace(spec)
	if rest, ok := strings.CutPrefix(s, "import"); ok && (rest == "" || rest[0] == ' ' || rest[0] == '\t' || rest[0] == '"') {
		s = strings.TrimSpace(rest)
	}
	if !strings.ContainsAny(s, "\"` \t") {
		// a bare path: example.com/x/retry
		s = strconv.Quote(s)
	}
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", "package p\nimport "+s+"\n", parser.ImportsOnly|parser.ParseComments)
	if err != nil {
		return nil, err
	}
	if len(f.Imports) != 1 || len(f.Decls) != 1 || len(f.Comments) != 0 {
		return nil, fmt.Errorf("want exactly one import spec")
	}
	if p := importPath(f.Imports[0]); p == "" || strings.ContainsFunc(p, func(r rune) bool {
		return unicode.IsSpace(r) || !unicode.IsPrint(r) || strings.ContainsRune("!\"#$%&'()*,:;<=>?[\\]^`{|}", r)
	}) {
		return nil, fmt.Errorf("invalid import path %q", p)
	}
	return f.Imports[0], nil
}

func importPath(spec *ast.ImportSpec) string {
	p, _ := strconv.Unquote(spec.Path.Value)
	return p
}

// ScanPlan returns human-readable safety problems; none means the plan is
// clean.
//
// It screens every string the renderer emits as code: each API import, each
// fixture's type and body, and each case's given statements, when act and
// then assertions. Snippets may not import, use os/exec/syscall/unsafe/
// reflect/net/file-access packages, spawn goroutines other than inline func
// literals, or carry //go:, //line or cgo directives; non-Go fails too.
// given entries and fixture bodies are parsed as statements, when and then as
// expressions (matching how the renderer uses them).
//
// Not a sandbox: see the file comment. This turns silent code execution into
// a loud build failure and validates the strings are real Go.
func ScanPlan(plan *TestPlan) []string {
	problems := []string{}

	for i, imp := range plan.API.Imports {
		where := fmt.Sprintf("api imports[%d]", i)
		spec, err := parseImport(imp)
		if err != nil {
			problems = append(problems, where+": not a valid Go import spec ("+syntaxMsg(err)+")")
			continue
		}
		if path := importPath(spec); bannedImport(path) {
			problems = append(problems, fmt.Sprintf("%s: import of %s is not allowed", where, pyRepr(path)))
		}
	}

	for _, f := range plan.Fixtures {
		tag := "fixture " + pyRepr(f.Name)
		if strings.TrimSpace(f.Type) != "" {
			problems = append(problems, screenExpr(f.Type, tag+" type")...)
		}
		if len(f.Body) == 0 {
			continue
		}
		// Screen the body as one block (Go statements span lines) but report
		// the entry each problem sits in, like pyllym's per-line screen.
		lineEntry := entryLines(f.Body)
		problems = append(problems, screenStmts(strings.Join(f.Body, "\n"), func(line int) string {
			return fmt.Sprintf("%s body[%d]", tag, lineEntry(line))
		})...)
	}

	for _, c := range plan.Cases {
		tag := "case " + pyRepr(c.TestName)
		// given entries render as consecutive body lines, so a construct may
		// span several entries; screen them as one block.
		if given := strings.Join(c.Given, "\n"); len(c.Given) > 0 && !skippable(given) {
			problems = append(problems, screenStmts(given, func(int) string { return tag + " given" })...)
		}
		if !isNoAct(c.When) {
			problems = append(problems, screenExpr(c.When, tag+" when")...)
		}
		for i, expr := range c.Then {
			if !skippable(expr) {
				problems = append(problems, screenExpr(expr, fmt.Sprintf("%s then[%d]", tag, i))...)
			}
		}
	}
	return problems
}

// skippable mirrors pyllym's _scan_source short-circuit (empty or "None").
func skippable(src string) bool {
	s := strings.TrimSpace(src)
	return s == "" || s == "None"
}

// entryLines maps a 0-based line of strings.Join(entries, "\n") to the
// entry index containing it.
func entryLines(entries []string) func(int) int {
	var starts []int
	line := 0
	for _, e := range entries {
		starts = append(starts, line)
		line += strings.Count(e, "\n") + 1
	}
	return func(l int) int {
		idx := 0
		for i, s := range starts {
			if l >= s {
				idx = i
			}
		}
		return idx
	}
}
