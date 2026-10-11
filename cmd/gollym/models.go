package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/registry"
)

func init() {
	commands["models"] = command{summary: "refresh or list the model registry", run: runModels}
}

const modelsUsage = `usage:
  gollym models refresh [--remote-only] [--out path] [--registry path]
  gollym models list [--provider p] [--registry path]
`

func runModels(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		fmt.Fprint(stderr, modelsUsage)
		return errors.New("missing subcommand")
	}
	switch args[0] {
	case "refresh":
		return modelsRefresh(ctx, args[1:], stdout, stderr)
	case "list":
		return modelsList(args[1:], stdout, stderr)
	}
	fmt.Fprint(stderr, modelsUsage)
	return fmt.Errorf("unknown subcommand %q", args[0])
}

// loadRegistry reads path, or the embedded registry when path is empty.
func loadRegistry(path string) (*core.Registry, error) {
	reg := core.LoadRegistry(path)
	if path != "" && reg.Len() == 0 {
		return nil, fmt.Errorf("no models in %s", path)
	}
	return reg, nil
}

// refreshProviders overrides the configured providers (tests).
var refreshProviders []*core.ProviderSpec

// modelsRefresh merges every configured provider's model listing into the
// registry and saves it (models.refresh + models.save_to_json).
func modelsRefresh(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("models refresh", flag.ContinueOnError)
	fs.SetOutput(stderr)
	remoteOnly := fs.Bool("remote-only", false, "skip local providers (Ollama, GPUStack, ...)")
	out := fs.String("out", "", "file to write (default: --registry, else ./models.json)")
	from := fs.String("registry", "", "models.json to start from (default: the embedded registry)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	reg, err := loadRegistry(*from)
	if err != nil {
		return err
	}
	dest := *out
	if dest == "" {
		dest = *from
	}
	if dest == "" {
		dest = "models.json"
	}
	opts := registry.Options{Config: core.Global(), RemoteOnly: *remoteOnly, Providers: refreshProviders}
	report := registry.Refresh(ctx, reg, opts)
	if len(report.Fetched) > 0 {
		fmt.Fprintf(stderr, "fetched: %s\n", strings.Join(report.Fetched, ", "))
	} else {
		fmt.Fprintln(stderr, "no configured provider answered; keeping existing models")
	}
	failed := make([]string, 0, len(report.Failed))
	for slug := range report.Failed {
		failed = append(failed, slug)
	}
	sort.Strings(failed)
	for _, slug := range failed {
		fmt.Fprintf(stderr, "skipped %s: %v\n", slug, report.Failed[slug])
	}
	if err := registry.Save(reg, dest); err != nil {
		return err
	}
	fmt.Fprintf(stdout, "saved %d models to %s\n", reg.Len(), dest)
	return nil
}

// modelsList prints the registry as a table.
func modelsList(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("models list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	provider := fs.String("provider", "", "only this provider's models")
	from := fs.String("registry", "", "models.json to read (default: the embedded registry)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	reg, err := loadRegistry(*from)
	if err != nil {
		return err
	}
	if *provider != "" {
		reg = reg.ByProvider(*provider)
	}
	tw := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "PROVIDER\tID\tTYPE\tCONTEXT\tNAME")
	for _, m := range reg.All() {
		ctxWindow := "-"
		if m.ContextWindow != nil {
			ctxWindow = fmt.Sprint(*m.ContextWindow)
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n", m.Provider, m.ID, m.Type(), ctxWindow, m.Name)
	}
	return tw.Flush()
}
