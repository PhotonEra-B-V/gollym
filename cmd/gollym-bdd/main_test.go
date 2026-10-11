package main

import (
	"os"
	"path/filepath"
	"testing"
)

const spec = `[meta]
feature_name = "Upper"
[api]
module = "strings"
signatures = ["func ToUpper(s string) string"]
[[cases]]
test_name = "test_upper"
scenario = "upper"
when = 'strings.ToUpper("a")'
then = ['result == "A"']
`

func TestRun(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "upper.toml")
	if err := os.WriteFile(path, []byte(spec), 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "gen")
	// flags may follow the positional source, as with argparse
	if code := run([]string{path, "--out", out, "--toml-mode", "plan"}); code != 0 {
		t.Fatalf("exit %d", code)
	}
	if _, err := os.Stat(filepath.Join(out, "upper_test.go")); err != nil {
		t.Fatal(err)
	}
	if code := run([]string{path, "--toml-mode", "bogus"}); code != 2 {
		t.Fatalf("exit %d for a bad mode", code)
	}
	if code := run(nil); code != 2 {
		t.Fatalf("exit %d without a source", code)
	}
}
