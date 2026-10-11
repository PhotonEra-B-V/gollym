package tdg

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// Port of pyllym tests/test_tdg_runs.py: timestamped runs, completion
// markers, and skip-if-unchanged.

// The standard library's "strings" exists everywhere and the spec selects
// nothing from it, so dependency checking stays quiet; that keeps these tests
// about run mechanics, not dependencies.
const runsSpec = `
[meta]
feature_name = "Backoff"
mode = "plan"

[api]
module = "strings"
signatures = ["func Backoff(attempt int) float64"]

[[rules]]
id = "rule_1"
text = "grows with attempt"

[[cases]]
test_name = "test_backoff_grows"
scenario = "grows"
when = "nil"
given = ["if !(1 < 2) { t.Fatal(\"order\") }"]
covers = ["rule_1"]
`

var (
	t1 = time.Date(2026, 7, 11, 14, 30, 5, 0, time.UTC)
	t2 = time.Date(2026, 7, 11, 15, 45, 0, 0, time.UTC)
)

func exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func build(t *testing.T, spec, out string, opts Options) []*BuildResult {
	t.Helper()
	results, err := Build(context.Background(), spec, out, opts)
	if err != nil {
		t.Fatal(err)
	}
	return results
}

func TestBuildWritesTimestampedRunAndMarkers(t *testing.T) {
	out := filepath.Join(t.TempDir(), "gen")
	build(t, runsSpec, out, Options{Now: t1})

	runDir := filepath.Join(out, "2026-07-11-14-30-05")
	if !exists(filepath.Join(runDir, "backoff_test.go")) || !exists(filepath.Join(runDir, DoneMarker)) || !exists(filepath.Join(out, LatestFile)) {
		t.Fatal("run artifacts missing")
	}
	latest := ReadLatest(out)
	if latest == nil || latest.RunDir != "2026-07-11-14-30-05" || latest.Stamp != "2026-07-11-14-30-05" || latest.Hashes["backoff"] != SpecHash(runsSpec) {
		t.Fatalf("latest %+v", latest)
	}
}

func TestSecondBuildUnchangedReuses(t *testing.T) {
	out := filepath.Join(t.TempDir(), "gen")
	build(t, runsSpec, out, Options{Now: t1})
	results := build(t, runsSpec, out, Options{Now: t2})

	if !results[0].Reused || len(results[0].Plan.Cases) != 1 {
		t.Fatalf("result %+v", results[0])
	}
	// a fresh run dir still materializes, with the reused artifact copied in
	if !exists(filepath.Join(out, "2026-07-11-15-45-00", "backoff_test.go")) {
		t.Fatal("reused artifact not copied forward")
	}
	// pointer advances to the newest run
	if ReadLatest(out).RunDir != "2026-07-11-15-45-00" {
		t.Fatal("pointer did not advance")
	}
}

func TestSecondBuildChangedRegenerates(t *testing.T) {
	out := filepath.Join(t.TempDir(), "gen")
	build(t, runsSpec, out, Options{Now: t1})
	changed := replaceOnce(runsSpec, "grows with attempt", "grows monotonically with attempt")
	results := build(t, changed, out, Options{Now: t2})

	if results[0].Reused || ReadLatest(out).Hashes["backoff"] != SpecHash(changed) {
		t.Fatalf("result %+v", results[0])
	}
}

func TestIncompleteRunIsNotReused(t *testing.T) {
	out := filepath.Join(t.TempDir(), "gen")
	build(t, runsSpec, out, Options{Now: t1})
	// simulate a crashed run: drop the done marker of the latest run
	if err := os.Remove(filepath.Join(out, "2026-07-11-14-30-05", DoneMarker)); err != nil {
		t.Fatal(err)
	}
	// ReadLatest refuses a run with no marker
	if ReadLatest(out) != nil {
		t.Fatal("incomplete run accepted")
	}
	// so the next unchanged build regenerates rather than reusing
	if results := build(t, runsSpec, out, Options{Now: t2}); results[0].Reused {
		t.Fatal("incomplete run reused")
	}
}

func TestUseRunsFalseWritesFlat(t *testing.T) {
	out := filepath.Join(t.TempDir(), "flat")
	results := build(t, runsSpec, out, Options{Now: t1, Flat: true})

	if !exists(filepath.Join(out, "backoff_test.go")) { // directly in out, no run dir
		t.Fatal("flat test file missing")
	}
	if exists(filepath.Join(out, LatestFile)) || exists(filepath.Join(out, DoneMarker)) || results[0].Reused {
		t.Fatal("flat build wrote run bookkeeping")
	}
}

func TestReadLatestMissingPointerIsNone(t *testing.T) {
	if ReadLatest(t.TempDir()) != nil {
		t.Fatal("want nil")
	}
}

func TestReadLatestRejectsCorruptPointer(t *testing.T) {
	dir := t.TempDir()
	for _, body := range []string{"not json", `{"run_dir": 3, "hashes": {}}`, `{"run_dir": "x"}`} {
		writeFile(t, filepath.Join(dir, LatestFile), body)
		if ReadLatest(dir) != nil {
			t.Fatalf("accepted %s", body)
		}
	}
}

func TestRunStampAndSpecHash(t *testing.T) {
	if RunStamp(t1) != "2026-07-11-14-30-05" {
		t.Fatal(RunStamp(t1))
	}
	// sha256("abc") prefix
	if SpecHash("abc") != "ba7816bf8f01cfea" {
		t.Fatal(SpecHash("abc"))
	}
}

func replaceOnce(s, old, repl string) string {
	for i := 0; i+len(old) <= len(s); i++ {
		if s[i:i+len(old)] == old {
			return s[:i] + repl + s[i+len(old):]
		}
	}
	return s
}

// Two builds within the same second share a run dir; the second reuses the
// artifacts in place instead of truncating them (pyllym's copy2 would raise
// SameFileError here).
func TestRebuildWithinSameSecondReusesInPlace(t *testing.T) {
	out := filepath.Join(t.TempDir(), "gen")
	build(t, runsSpec, out, Options{Now: t1})
	results := build(t, runsSpec, out, Options{Now: t1})
	if !results[0].Reused || len(results[0].Plan.Cases) != 1 {
		t.Fatalf("result %+v", results[0])
	}
}
