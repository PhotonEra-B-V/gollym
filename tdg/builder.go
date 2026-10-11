package tdg

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/PhotonEra-B-V/gollym/bdd"
)

// Orchestration: parse -> inspect deps -> plan -> render tests + brief into a
// timestamped run (pyllym's tdg/builder.py).

// Options configure Build. The zero value matches pyllym's defaults
// (strict, check_deps, use_runs).
type Options struct {
	bdd.Options
	// SkipDeps skips inspecting the package under test (pyllym's
	// check_deps=False).
	SkipDeps bool
	// Flat writes straight into outDir with no run bookkeeping (pyllym's
	// use_runs=False).
	Flat bool
	// Now pins the run timestamp (zero: UTCNow).
	Now time.Time
	// Dir is the directory whose Go module context resolves the package
	// under test ("" = current directory).
	Dir string
}

// Generator is the tool name in tdg's test file headers.
const Generator = "gollym/tdg"

// Build turns TOML specs into a red Go test suite plus build briefs, exactly
// like bdd.Build, with two additions.
//
// Unless opts.SkipDeps, the package under test is inspected for its real
// dependencies (InspectDependenciesIn: static go/parser read, go list
// fallback). In request mode those dependencies are handed to the planner so
// generated tests target the package's actual collaborators; in both modes
// a spec whose declared [api].imports (or, in plan mode, the names its cases
// select from the package) disagree with the package yields dependency
// problems that are listed in the brief and, unless Lenient, fail the build
// with *bdd.PlanValidationError alongside coverage gaps.
//
// Unless opts.Flat, each call writes into a timestamped run dir
// outDir/<YYYY-MM-DD-HH-MM-SS>/ and, on success, drops a _DONE.json marker
// there plus updates outDir/latest.json. The next build reads that pointer
// and skips any spec whose content hash matches the latest completed run
// (copying its artifacts forward: BuildResult.Reused is true); changed or new
// specs regenerate.
//
// The safety gate, coverage checks and security model are bdd's: see
// bdd.Build and the bdd package documentation.
func Build(ctx context.Context, source, outDir string, opts Options) ([]*bdd.BuildResult, error) {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return nil, err
	}
	if opts.Package == "" {
		opts.Package = bdd.PackageName(outDir)
	}

	// Each build lands in its own timestamped run dir; latest.json (written
	// on success) lets the next build skip specs it already generated.
	runDir, stamp := outDir, ""
	var latest *LatestPointer
	if !opts.Flat {
		now := opts.Now
		if now.IsZero() {
			now = UTCNow()
		}
		stamp = RunStamp(now)
		runDir = filepath.Join(outDir, stamp)
		latest = ReadLatest(outDir)
	}
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return nil, err
	}

	specs, err := bdd.Discover(source)
	if err != nil {
		return nil, err
	}
	var results []*bdd.BuildResult
	hashes := map[string]string{}
	for _, s := range specs {
		slug := s.Spec.Slug()
		digest := SpecHash(s.Text)
		hashes[slug] = digest

		// Skip-if-unchanged: copy the latest completed run's artifacts.
		if !opts.Flat {
			if reuse := Reusable(latest, outDir, slug, digest); reuse != nil {
				r, err := copyForward(s.Spec, reuse.SourceRunDir, runDir, slug)
				if err != nil {
					return results, err
				}
				results = append(results, r)
				continue
			}
		}
		r, err := generateSpec(ctx, s.Spec, runDir, opts)
		if err != nil {
			return results, err
		}
		results = append(results, r)
	}

	// Mark the run complete only after every spec succeeded, then publish it
	// as the latest. A run that failed above never gets a marker: never reused.
	if !opts.Flat {
		if err := WriteDoneMarker(runDir, stamp, hashes); err != nil {
			return results, err
		}
		if err := WriteLatestPointer(outDir, runDir, stamp, hashes); err != nil {
			return results, err
		}
	}
	return results, nil
}

// generateSpec inspects dependencies, then plans, gates, validates and
// renders one spec into runDir.
func generateSpec(ctx context.Context, spec *bdd.TomlSpec, runDir string, opts Options) (*bdd.BuildResult, error) {
	step := bdd.Step{Dir: runDir, Options: opts.Options, APIHint: opts.APIHint, Generator: Generator}
	if !opts.SkipDeps {
		deps := InspectDependenciesIn(opts.Dir, spec.API.Module)
		var used []string
		if bdd.ResolveMode(spec, opts.TomlMode) == bdd.ModePlan {
			used = bdd.SelectedNames(spec.ToTestPlan(), LocalName(spec, deps))
		}
		step.Problems = CheckDependencies(spec.API.Imports, used, deps)
		block := deps.ToPromptBlock()
		if opts.APIHint != "" {
			step.APIHint = opts.APIHint + "\n" + block
		} else {
			step.APIHint = block
		}
	}
	return bdd.Generate(ctx, spec, step)
}

// copyForward reuses an unchanged spec's artifacts by copying them from the
// prior run into the current one, so each run dir is self-contained.
func copyForward(spec *bdd.TomlSpec, sourceRunDir, runDir, slug string) (*bdd.BuildResult, error) {
	r := &bdd.BuildResult{
		Feature:   spec,
		TestPath:  filepath.Join(runDir, bdd.TestFileName(slug)),
		PlanPath:  filepath.Join(runDir, slug+".plan.json"),
		BriefPath: filepath.Join(runDir, "BRIEF_"+slug+".md"),
		Reused:    true,
	}
	for _, dst := range []string{r.TestPath, r.PlanPath, r.BriefPath} {
		if filepath.Clean(sourceRunDir) == filepath.Clean(runDir) {
			break // a rebuild within the same second: the artifacts are already here
		}
		if err := copyFile(filepath.Join(sourceRunDir, filepath.Base(dst)), dst); err != nil && !os.IsNotExist(err) {
			return nil, err
		}
	}
	b, err := os.ReadFile(r.PlanPath)
	if err != nil {
		return nil, err
	}
	if r.Plan, err = bdd.ParsePlan(b); err != nil {
		return nil, err
	}
	return r, nil
}

// copyFile copies src to dst keeping its modification time (shutil.copy2).
func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	info, err := in.Stat()
	if err != nil {
		return err
	}
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, info.Mode().Perm())
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	return os.Chtimes(dst, info.ModTime(), info.ModTime())
}
