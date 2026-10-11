package tdg

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"time"

	"github.com/PhotonEra-B-V/gollym/bdd"
)

// Timestamped generation runs with a completion marker and skip-if-unchanged
// (pyllym's tdg/runs.py).
//
// Each Build call writes into its own timestamped run directory under outDir
// (2026-07-11-14-30-05/). A run is only ever treated as usable once it
// finishes: on success the builder drops a _DONE.json marker inside the run
// dir and updates outDir/latest.json, a portable pointer (no symlinks) naming
// the newest completed run plus the content hash of every spec it generated.
//
// That pointer powers skip-if-unchanged: before generating a spec, the
// builder hashes it; if the latest completed run already generated an
// identical spec (same hash) and that run's artifacts are still on disk,
// generation is skipped and the prior output is reused. A changed or new spec
// falls through to a fresh generation in the current run.

// Run bookkeeping file names and the run directory format.
const (
	DoneMarker     = "_DONE.json"
	LatestFile     = "latest.json"
	RunStampLayout = "2006-01-02-15-04-05"
)

// UTCNow is the default clock: UTC now. Build takes Options.Now so tests can
// pin a fixed time.
func UTCNow() time.Time { return time.Now().UTC() }

// RunStamp formats a run directory name: 2026-07-11-14-30-05.
func RunStamp(now time.Time) string { return now.Format(RunStampLayout) }

// SpecHash is a stable content hash of a spec's text (sha256, 16 hex chars):
// short enough to read in latest.json, wide enough that a collision between
// two distinct specs is not a practical concern for skip decisions.
func SpecHash(specText string) string {
	sum := sha256.Sum256([]byte(specText))
	return hex.EncodeToString(sum[:])[:16]
}

// LatestPointer is the parsed latest.json: the newest completed run and its
// spec hashes. RunDir is relative to outDir so a whole output tree stays
// movable; Hashes maps a spec slug to the content hash generated in that run.
type LatestPointer struct {
	RunDir string            `json:"run_dir"`
	Stamp  string            `json:"stamp"`
	Hashes map[string]string `json:"hashes"`
}

// ReadLatest loads outDir/latest.json if it points at a still-complete run.
//
// It returns nil when there is no pointer, it is unreadable or corrupt, or
// the run it names is missing its _DONE.json marker (interrupted or deleted):
// in every one of those cases the caller must regenerate rather than trust
// stale state.
func ReadLatest(outDir string) *LatestPointer {
	b, err := os.ReadFile(filepath.Join(outDir, LatestFile))
	if err != nil {
		return nil
	}
	var data map[string]any
	if json.Unmarshal(b, &data) != nil {
		return nil
	}
	runDir, ok := data["run_dir"].(string)
	if !ok {
		return nil
	}
	hashes, ok := data["hashes"].(map[string]any)
	if !ok {
		return nil
	}
	if _, err := os.Stat(filepath.Join(outDir, runDir, DoneMarker)); err != nil {
		return nil // named run is incomplete: do not reuse it
	}
	p := &LatestPointer{RunDir: runDir, Hashes: map[string]string{}}
	if s, ok := data["stamp"]; ok && s != nil {
		p.Stamp = pyStr(s)
	}
	for k, v := range hashes {
		p.Hashes[k] = pyStr(v)
	}
	return p
}

func pyStr(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	b, _ := json.Marshal(v)
	return string(b)
}

// Reuse is a spec whose output can be copied forward from the latest
// completed run.
type Reuse struct {
	Slug         string
	SourceRunDir string
}

// Reusable decides whether slug can be reused from the latest completed run:
// only when the pointer recorded the same hash for it and the prior run's
// rendered test file is still on disk. Any drift (hash changed, new spec,
// artifact deleted) returns nil: regenerate.
func Reusable(latest *LatestPointer, outDir, slug, currentHash string) *Reuse {
	if latest == nil || latest.Hashes[slug] != currentHash {
		return nil
	}
	source := filepath.Join(outDir, latest.RunDir)
	if _, err := os.Stat(filepath.Join(source, bdd.TestFileName(slug))); err != nil {
		return nil
	}
	return &Reuse{Slug: slug, SourceRunDir: source}
}

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

// WriteDoneMarker writes the _DONE.json completion marker inside a finished
// run dir.
func WriteDoneMarker(runDir, stamp string, hashes map[string]string) error {
	return writeJSON(filepath.Join(runDir, DoneMarker), struct {
		Stamp     string            `json:"stamp"`
		Hashes    map[string]string `json:"hashes"`
		Completed bool              `json:"completed"`
	}{stamp, hashes, true})
}

// WriteLatestPointer points outDir/latest.json at a completed run (relative
// run dir).
func WriteLatestPointer(outDir, runDir, stamp string, hashes map[string]string) error {
	return writeJSON(filepath.Join(outDir, LatestFile), LatestPointer{RunDir: filepath.Base(runDir), Stamp: stamp, Hashes: hashes})
}
