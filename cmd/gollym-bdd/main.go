// Command gollym-bdd turns .toml specs into a red Go test suite plus build
// briefs (pyllym's python -m pyllym.bdd):
//
//	gollym-bdd features/ --out tests/generated --model gpt-5.4
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/PhotonEra-B-V/gollym/bdd"
)

func main() { os.Exit(run(os.Args[1:])) }

func run(argv []string) int {
	fs := flag.NewFlagSet("gollym-bdd", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: gollym-bdd [flags] source\n\nTurn a .toml spec into a red Go test suite plus build briefs.\nsource is a .toml spec file or a directory containing them.\n\nflags:")
		fs.PrintDefaults()
	}
	out := fs.String("out", "tests/generated", "output directory")
	model := fs.String("model", "", "model id for the planner")
	provider := fs.String("provider", "", "provider slug, if ambiguous")
	apiHint := fs.String("api-hint", "", "pin the target package / existing signatures")
	pkg := fs.String("package", "", "package clause of the generated tests (default: derived from --out)")
	mode := fs.String("toml-mode", "auto", "TOML build mode: 'plan' uses hand-written [[cases]] (no LLM), "+
		"'request' has the planner LLM fill the plan; 'auto' detects from the presence of [[cases]]")
	positional, err := parseInterspersed(fs, argv)
	if err != nil {
		return 2
	}
	if len(positional) != 1 {
		fs.Usage()
		return 2
	}
	if *mode != "auto" && *mode != bdd.ModePlan && *mode != bdd.ModeRequest {
		fmt.Fprintf(os.Stderr, "gollym-bdd: --toml-mode must be auto, plan or request, got %q\n", *mode)
		return 2
	}

	opts := bdd.Options{APIHint: *apiHint, TomlMode: *mode, Package: *pkg}
	opts.Chat.Model, opts.Chat.Provider = *model, *provider
	results, err := bdd.Build(context.Background(), positional[0], *out, opts)
	if err != nil {
		fmt.Fprintln(os.Stderr, "gollym-bdd:", err)
		return 1
	}
	for _, r := range results {
		fmt.Printf("%s: %d tests\n", r.Plan.Feature, len(r.Plan.Cases))
		for _, p := range []string{r.TestPath, r.PlanPath, r.BriefPath} {
			fmt.Printf("  %s\n", p)
		}
	}
	return 0
}

// parseInterspersed allows flags after the positional argument, like
// argparse.
func parseInterspersed(fs *flag.FlagSet, args []string) ([]string, error) {
	var positional []string
	for {
		if err := fs.Parse(args); err != nil {
			return nil, err
		}
		if fs.NArg() == 0 {
			return positional, nil
		}
		positional = append(positional, fs.Arg(0))
		args = fs.Args()[1:]
	}
}
