package bdd

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// compileSpec exercises every rendering feature: a typed fixture (used and
// unused), the error-path form, a context-aware act, an interaction-order
// case, and an unused import that must be dropped.
const compileSpec = `[meta]
feature_name = "Retry policy"

[api]
module = "example.com/m/retry"
imports = ['"example.com/m/retry"', '"errors"', '"slices"', '"fmt"']
signatures = [
  "func Backoff(attempt int, base float64) float64",
  "func Validate(attempt int) error",
  "func Wait(ctx context.Context, attempt int) error",
  "func Run(query func(string))",
]

[[rules]]
id = "rule_1"
text = "The first retry uses the base delay."

[[edge_cases]]
id = "edge_1"
text = "A negative attempt is rejected."

[[fixtures]]
name = "base"
type = "float64"
body = ["t.Cleanup(func() {})", "return 0.5"]

[[cases]]
test_name = "test_first_retry_uses_base_delay"
scenario = "First retry uses the base delay"
when = "retry.Backoff(1, base)"
then = ["result == 0.5"]
fixtures = ["base"]
covers = ["rule_1"]

[[cases]]
test_name = "test_delay_is_capped"
scenario = "Delays are capped"
when = "retry.Backoff(50, 0.5)"
then = ["result <= 30"]
fixtures = ["base"]

[[cases]]
test_name = "test_negative_attempt_rejected"
scenario = "Negative attempt is rejected"
given = ["err := retry.Validate(-1)"]
when = "nil"
then = ["errors.Is(err, retry.ErrNegativeAttempt)"]
edge_case = true
covers = ["edge_1", "M1"]

[[cases]]
test_name = "test_wait_honours_context"
scenario = "Waiting is context-aware"
is_async = true
when = "retry.Wait(ctx, 1)"
then = ["result == nil"]

[[cases]]
test_name = "test_runs_query_once"
scenario = "Run queries once"
given = ["calls := []string{}", "query := func(q string) { calls = append(calls, q) }", "retry.Run(query)"]
when = "nil"
then = ['slices.Equal(calls, []string{"query_users"})']
covers = ["M2"]

[[sequences]]
mermaid = """
sequenceDiagram
    Test->>Retry: Validate
    Retry->>Store: query_users
"""
`

const retryImpl = `package retry

import (
	"context"
	"errors"
)

var ErrNegativeAttempt = errors.New("negative attempt")

func Backoff(attempt int, base float64) float64 {
	d := base
	for i := 1; i < attempt && d < 30; i++ {
		d *= 2
	}
	return min(d, 30)
}

func Validate(attempt int) error {
	if attempt < 0 {
		return ErrNegativeAttempt
	}
	return nil
}

func Wait(ctx context.Context, attempt int) error { return ctx.Err() }

func Run(query func(string)) { query("query_users") }
`

func goTool(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("go", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=-mod=mod", "GOPROXY=off")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go %s: %v\n%s", strings.Join(args, " "), err, out)
	}
	return string(out)
}

// TestGeneratedSuiteCompiles builds a plan into a scratch module next to an
// implementation and runs go vet and go test on it: the rendered file must
// compile, vet clean, and (here, with the implementation present) pass. It
// also checks the bdd_pending tag keeps the suite out of untagged runs.
func TestGeneratedSuiteCompiles(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go tool not available")
	}
	mod := t.TempDir()
	write := func(rel, content string) {
		p := filepath.Join(mod, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/m\n\ngo 1.24\n")
	write("retry/retry.go", retryImpl)

	out := filepath.Join(mod, "tests", "generated")
	results, err := Build(context.Background(), compileSpec, out, Options{})
	if err != nil {
		t.Fatal(err)
	}
	src, _ := os.ReadFile(results[0].TestPath)
	if strings.Contains(string(src), `"fmt"`) {
		t.Fatalf("unused import rendered:\n%s", src)
	}

	goTool(t, mod, "vet", "-tags", PendingTag, "./...")
	if out := goTool(t, mod, "test", "-v", "-tags", PendingTag, "./tests/generated"); !strings.Contains(out, "--- PASS: TestRunsQueryOnce") || !strings.Contains(out, "--- PASS: TestWaitHonoursContext") {
		t.Fatalf("go test output: %s", out)
	}
	// Without the tag the red suite is invisible to the regular run.
	if out := goTool(t, mod, "test", "./..."); strings.Contains(out, "tests/generated") && strings.Contains(out, "FAIL") {
		t.Fatalf("untagged run picked up the suite: %s", out)
	}
	// Rendering into the package's own directory as an external test package.
	plan := results[0].Plan
	write("retry/retry_spec_test.go", RenderTests(plan, RenderOptions{Package: "retry_test"}))
	goTool(t, mod, "vet", "-tags", PendingTag, "./retry")
}
