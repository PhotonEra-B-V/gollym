// Command gollym is gollym's command-line tool.
//
//	gollym models refresh [--remote-only] [--out path] [--registry path]
//	gollym models list [--provider p] [--registry path]
//
// Provider credentials come from the environment (OPENAI_API_KEY, ...).
// Commands live in their own files and add themselves to commands.
package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"sort"
	"strings"

	_ "github.com/PhotonEra-B-V/gollym/providers" // register built-in providers
)

// command runs with the arguments after its name.
type command struct {
	summary string
	run     func(ctx context.Context, args []string, stdout, stderr io.Writer) error
}

var commands = map[string]command{}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	code := run(ctx, os.Args[1:], os.Stdout, os.Stderr)
	stop()
	os.Exit(code)
}

func run(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" || args[0] == "help" {
		usage(stderr)
		if len(args) == 0 {
			return 2
		}
		return 0
	}
	cmd, ok := commands[args[0]]
	if !ok {
		fmt.Fprintf(stderr, "gollym: unknown command %q\n", args[0])
		usage(stderr)
		return 2
	}
	if err := cmd.run(ctx, args[1:], stdout, stderr); err != nil {
		fmt.Fprintf(stderr, "gollym %s: %v\n", args[0], err)
		return 1
	}
	return 0
}

func usage(w io.Writer) {
	names := make([]string, 0, len(commands))
	for n := range commands {
		names = append(names, n)
	}
	sort.Strings(names)
	var b strings.Builder
	b.WriteString("usage: gollym <command> [arguments]\n\ncommands:\n")
	for _, n := range names {
		fmt.Fprintf(&b, "  %-10s %s\n", n, commands[n].summary)
	}
	fmt.Fprint(w, b.String())
}
