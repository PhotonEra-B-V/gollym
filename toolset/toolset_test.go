package toolset_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym"
	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
	"github.com/PhotonEra-B-V/gollym/toolset"
)

func mean(data []float64) (float64, error) {
	if len(data) == 0 {
		return 0, errors.New("mean requires at least one data point")
	}
	var sum float64
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data)), nil
}

func median(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	return data[len(data)/2]
}

// stdlib registers stand-ins for Python's statistics module, which the
// pyllym tests use.
func stdlib(t *testing.T) *toolset.Registry {
	t.Helper()
	r := toolset.NewRegistry()
	must(t, r.Register("statistics.mean", mean, toolset.Args("data"),
		toolset.Doc("Return the sample arithmetic mean of data.\n\nLonger text.")))
	must(t, r.Register("statistics.median", median, toolset.Args("data"),
		toolset.Doc("Return the median (middle value) of numeric data.")))
	return r
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func write(t *testing.T, body string) string {
	t.Helper()
	file := filepath.Join(t.TempDir(), "toolset.toml")
	must(t, os.WriteFile(file, []byte(body), 0o644))
	return file
}

func wantErr(t *testing.T, err error, substr string) {
	t.Helper()
	if err == nil || !strings.Contains(err.Error(), substr) {
		t.Fatalf("error %v, want it to contain %q", err, substr)
	}
	if !errors.Is(err, core.ErrConfiguration) {
		t.Fatalf("error %v is not a configuration error", err)
	}
}

// --- ToolFromPath: schema inference & execution ------------------------------

func TestToolFromStdlibCallableInfersAndRuns(t *testing.T) {
	tool, err := stdlib(t).ToolFromPath("statistics.mean", toolset.ToolOptions{})
	must(t, err)
	if tool.Name() != "statistics_mean" {
		t.Fatalf("name %q", tool.Name())
	}
	if tool.Description() != "Return the sample arithmetic mean of data." {
		t.Fatalf("description %q", tool.Description())
	}
	schema := tool.ParamsSchema()
	if _, ok := schema["properties"].(map[string]any)["data"]; !ok {
		t.Fatalf("schema %v", schema)
	}
	out, err := tool.Call(context.Background(), map[string]any{"data": []any{1.0, 2.0, 3.0, 4.0}})
	must(t, err)
	if out != 2.5 {
		t.Fatalf("result %v", out)
	}
}

func TestAnnotationMapsListParamToArray(t *testing.T) {
	r := toolset.NewRegistry()
	total := func(values []float64) float64 {
		s := 0.0
		for _, v := range values {
			s += v
		}
		return s
	}
	must(t, r.Register("toolset_test._total_for_test", total, toolset.Args("values")))
	tool, err := r.ToolFromPath("toolset_test._total_for_test", toolset.ToolOptions{})
	must(t, err)
	prop := tool.ParamsSchema()["properties"].(map[string]any)["values"].(map[string]any)
	if prop["type"] != "array" || !reflect.DeepEqual(prop["items"], map[string]any{"type": "number"}) {
		t.Fatalf("prop %v", prop)
	}
	out, err := tool.Call(context.Background(), map[string]any{"values": []any{1.5, 2.5}})
	must(t, err)
	if out != 4.0 {
		t.Fatalf("result %v", out)
	}
}

func TestNameAndDescriptionOverrides(t *testing.T) {
	tool, err := stdlib(t).ToolFromPath("statistics.median", toolset.ToolOptions{Name: "mid", Description: "the middle value"})
	must(t, err)
	if tool.Name() != "mid" || tool.Description() != "the middle value" {
		t.Fatalf("%q %q", tool.Name(), tool.Description())
	}
}

func TestParamOverrideWinsOverInference(t *testing.T) {
	tool, err := stdlib(t).ToolFromPath("statistics.mean", toolset.ToolOptions{
		Params: map[string]any{"data": map[string]any{"type": "array", "items": "number", "description": "numbers"}},
	})
	must(t, err)
	prop := tool.ParamsSchema()["properties"].(map[string]any)["data"]
	want := map[string]any{"type": "array", "items": map[string]any{"type": "number"}, "description": "numbers"}
	if !reflect.DeepEqual(prop, want) {
		t.Fatalf("prop %v", prop)
	}
}

func TestInferredSchemaShape(t *testing.T) {
	r := toolset.NewRegistry()
	fn := func(_ context.Context, n int, ratio *float64, flags []bool, opts map[string]any, label string) string {
		return label
	}
	must(t, r.Register("pkg.shape", fn, toolset.Args("n", "ratio", "flags", "opts", "label"), toolset.Default("label", "x")))
	tool, err := r.ToolFromPath("pkg.shape", toolset.ToolOptions{})
	must(t, err)
	want := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"n":     map[string]any{"type": "integer"},
			"ratio": map[string]any{"type": "number"},
			"flags": map[string]any{"type": "array", "items": map[string]any{"type": "boolean"}},
			"opts":  map[string]any{"type": "object"},
			"label": map[string]any{"type": "string"},
		},
		"required":             []any{"n", "ratio", "flags", "opts"},
		"additionalProperties": false,
	}
	if !reflect.DeepEqual(tool.ParamsSchema(), want) {
		t.Fatalf("schema %v", tool.ParamsSchema())
	}
	out, err := tool.Call(context.Background(), map[string]any{"n": 2.0, "ratio": nil, "flags": []any{true}, "opts": map[string]any{}})
	must(t, err)
	if out != "x" {
		t.Fatalf("default not applied: %v", out)
	}
	if tool.Description() != "Call pkg.shape." {
		t.Fatalf("fallback description %q", tool.Description())
	}
}

func TestCallErrorsAreGoErrors(t *testing.T) {
	tool, err := stdlib(t).ToolFromPath("statistics.mean", toolset.ToolOptions{})
	must(t, err)
	ctx := context.Background()
	if _, err := tool.Call(ctx, map[string]any{}); err == nil || !strings.Contains(err.Error(), "missing 1 required argument: 'data'") {
		t.Fatalf("missing: %v", err)
	}
	if _, err := tool.Call(ctx, map[string]any{"data": []any{1.0}, "x": 1}); err == nil || !strings.Contains(err.Error(), "unexpected keyword argument 'x'") {
		t.Fatalf("unknown: %v", err)
	}
	if _, err := tool.Call(ctx, map[string]any{"data": "nope"}); err == nil {
		t.Fatal("wrong type should fail")
	}
	if _, err := tool.Call(ctx, map[string]any{"data": []any{}}); err == nil || !strings.Contains(err.Error(), "at least one") {
		t.Fatalf("function error: %v", err)
	}
}

func TestCompositeResultsBecomePlainJSON(t *testing.T) {
	r := toolset.NewRegistry()
	type stats struct {
		Count int     `json:"count"`
		Mean  float64 `json:"mean"`
	}
	must(t, r.Register("pkg.describe", func(data []float64) stats {
		m, _ := mean(data)
		return stats{Count: len(data), Mean: m}
	}, toolset.Args("data")))
	must(t, r.Register("pkg.double", func(data []int) []int {
		for i := range data {
			data[i] *= 2
		}
		return data
	}, toolset.Args("data")))
	describe, _ := r.ToolFromPath("pkg.describe", toolset.ToolOptions{})
	out, err := describe.Call(context.Background(), map[string]any{"data": []any{1.0, 3.0}})
	must(t, err)
	if !reflect.DeepEqual(out, map[string]any{"count": 2.0, "mean": 2.0}) {
		t.Fatalf("describe %#v", out)
	}
	double, _ := r.ToolFromPath("pkg.double", toolset.ToolOptions{})
	out, err = double.Call(context.Background(), map[string]any{"data": []any{1.0, 2.0}})
	must(t, err)
	if !reflect.DeepEqual(out, []any{2.0, 4.0}) {
		t.Fatalf("double %#v", out)
	}
}

func TestRawFuncTakesKeywordArgumentsAndConfigParams(t *testing.T) {
	r := toolset.NewRegistry()
	var got map[string]any
	must(t, r.Register("pkg.raw", func(_ context.Context, args map[string]any) (any, error) {
		got = args
		return "ok", nil
	}))
	tool, err := r.ToolFromPath("pkg.raw", toolset.ToolOptions{})
	must(t, err)
	open := map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": true}
	if !reflect.DeepEqual(tool.ParamsSchema(), open) {
		t.Fatalf("open schema %v", tool.ParamsSchema())
	}
	if _, err := tool.Call(context.Background(), map[string]any{"anything": 1.0}); err != nil || got["anything"] != 1.0 {
		t.Fatalf("raw call %v %v", got, err)
	}

	file := write(t, `
[[tools]]
path = "pkg.raw"
[tools.params.z]
type = "number"
[tools.params.a]
type = "integer"
required = false
[tools.params.b]
description = "bee"
`)
	tools, err := r.LoadToolset(file, toolset.LoadOptions{})
	must(t, err)
	want := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"z": map[string]any{"type": "number"},
			"a": map[string]any{"type": "integer"},
			"b": map[string]any{"type": "string", "description": "bee"},
		},
		"required":             []any{"z", "b"}, // file order, as pyllym's dicts keep it
		"additionalProperties": false,
	}
	if !reflect.DeepEqual(tools[0].ParamsSchema(), want) {
		t.Fatalf("schema %v", tools[0].ParamsSchema())
	}
}

// --- LoadToolset: file parsing & guarantees ----------------------------------

func TestLoadToolsetReadsEntries(t *testing.T) {
	file := write(t, `
[[tools]]
path = "statistics.mean"
description = "average"

[[tools]]
path = "statistics.median"
name = "mid"
`)
	tools, err := stdlib(t).LoadToolset(file, toolset.LoadOptions{})
	must(t, err)
	if len(tools) != 2 || tools[0].Name() != "statistics_mean" || tools[1].Name() != "mid" || tools[0].Description() != "average" {
		t.Fatalf("tools %v", tools)
	}
}

func TestMissingFileRaisesConfigurationError(t *testing.T) {
	_, err := stdlib(t).LoadToolset(filepath.Join(t.TempDir(), "nope.toml"), toolset.LoadOptions{})
	wantErr(t, err, "not found")
}

func TestEntryWithoutPathRaises(t *testing.T) {
	_, err := stdlib(t).LoadToolset(write(t, "[[tools]]\ndescription = \"no path\"\n"), toolset.LoadOptions{})
	wantErr(t, err, "missing 'path'")
}

func TestDuplicateNamesRaise(t *testing.T) {
	file := write(t, `
[[tools]]
path = "statistics.mean"

[[tools]]
path = "statistics.mean"
`)
	_, err := stdlib(t).LoadToolset(file, toolset.LoadOptions{})
	wantErr(t, err, "duplicate tool name")
}

func TestMalformedFilesRaise(t *testing.T) {
	r := stdlib(t)
	_, err := r.LoadToolset(write(t, "[[tools]\n"), toolset.LoadOptions{})
	wantErr(t, err, "invalid TOML in toolset")
	_, err = r.LoadToolset(write(t, "tools = 1\n"), toolset.LoadOptions{})
	wantErr(t, err, "must define an array of [[tools]] tables")
	_, err = r.LoadToolset(write(t, "x = 1\n"), toolset.LoadOptions{})
	wantErr(t, err, "must define an array of [[tools]] tables")
	_, err = r.LoadToolset(write(t, "tools = [1]\n"), toolset.LoadOptions{})
	wantErr(t, err, "entry 0 is not a table")
}

// --- allowlist / safety boundary ---------------------------------------------

func TestUnimportablePathRaises(t *testing.T) {
	_, err := stdlib(t).ToolFromPath("nonexistent_pkg_xyz.func", toolset.ToolOptions{})
	wantErr(t, err, "nonexistent_pkg_xyz")
	_, err = stdlib(t).ToolFromPath("statistics.nope", toolset.ToolOptions{})
	wantErr(t, err, "has no registered attribute 'nope'")
}

func TestMissingOptionalPackageGivesInstallHint(t *testing.T) {
	_, err := stdlib(t).ToolFromPath("torch.mean", toolset.ToolOptions{})
	wantErr(t, err, "not installed")
	var missing *toolset.MissingToolPackageError
	if !errors.As(err, &missing) || missing.Package != "torch" || missing.Dotted != "torch.mean" {
		t.Fatalf("error %#v", err)
	}
}

func TestLoadToolsetSkipsMissingPackageByDefault(t *testing.T) {
	file := write(t, `
[[tools]]
path = "statistics.mean"

[[tools]]
path = "torch.mean"
`)
	tools, err := stdlib(t).LoadToolset(file, toolset.LoadOptions{})
	must(t, err)
	if len(tools) != 1 || tools[0].Name() != "statistics_mean" {
		t.Fatalf("tools %v", tools)
	}
}

func TestLoadToolsetStrictReraisesMissingPackage(t *testing.T) {
	_, err := stdlib(t).LoadToolset(write(t, "[[tools]]\npath = \"torch.mean\"\n"), toolset.LoadOptions{Strict: true})
	var missing *toolset.MissingToolPackageError
	if !errors.As(err, &missing) {
		t.Fatalf("error %v", err)
	}
}

func TestInstalledOptionalPackageIsUsable(t *testing.T) {
	r := toolset.NewRegistry()
	must(t, r.Register("fake_torch_lib.tensor_mean", func(values []float64) float64 {
		m, _ := mean(values)
		return m
	}, toolset.Args("values")))
	tool, err := r.ToolFromPath("fake_torch_lib.tensor_mean", toolset.ToolOptions{})
	must(t, err)
	out, err := tool.Call(context.Background(), map[string]any{"values": []any{2.0, 4.0, 6.0}})
	must(t, err)
	if out != 4.0 {
		t.Fatalf("result %v", out)
	}
}

func TestNonCallablePathRaises(t *testing.T) {
	err := toolset.NewRegistry().Register("statistics.__doc__", "Basic statistics module.")
	wantErr(t, err, "callable")
}

func TestBareNameWithoutDotRaises(t *testing.T) {
	_, err := stdlib(t).ToolFromPath("len", toolset.ToolOptions{})
	wantErr(t, err, "dotted")
	wantErr(t, toolset.NewRegistry().Register("len", func() {}), "dotted")
}

func TestInvalidRegistrations(t *testing.T) {
	r := toolset.NewRegistry()
	wantErr(t, r.Register("pkg.f", func(a, b int) int { return a }, toolset.Args("a")), "Args names 1 parameters, the function has 2")
	wantErr(t, r.Register("pkg.f", func(a int) int { return a }, toolset.Args("a"), toolset.Default("z", 1)), "unknown argument")
	wantErr(t, r.Register("pkg.f", func() (int, int) { return 0, 0 }), "(T, error)")
	must(t, r.Register("pkg.f", func() {}))
	wantErr(t, r.Register("pkg.f", func() {}), "already registered")
}

func TestPackageLevelRegistry(t *testing.T) {
	if !slices.Contains(toolset.DefaultRegistry.Paths(), "toolset_test_pkg.answer") { // -count=N
		toolset.Register("toolset_test_pkg.answer", func() int { return 42 })
	}
	tool, err := toolset.ToolFromPath("toolset_test_pkg.answer", toolset.ToolOptions{})
	must(t, err)
	if out, _ := tool.Call(context.Background(), nil); out != 42 {
		t.Fatalf("result %v", out)
	}
	defer func() {
		if recover() == nil {
			t.Fatal("Register should panic on a bad registration")
		}
	}()
	toolset.Register("nodot", func() {})
}

// The model -> tool -> model loop with a toolset tool.
func TestToolsetToolInChatLoop(t *testing.T) {
	call := map[string]any{
		"model": "glm-5.3",
		"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": nil,
			"tool_calls": []any{map[string]any{"id": "call_1", "type": "function", "function": map[string]any{"name": "statistics_mean", "arguments": `{"data":[1,2,3,4]}`}}}},
			"finish_reason": "tool_calls"}},
	}
	done := map[string]any{
		"model":   "glm-5.3",
		"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": "2.5"}, "finish_reason": "stop"}},
	}
	srv := testserver.New(t, testserver.Reply{Body: call}, testserver.Reply{Body: done})
	ctx := gollym.NewContext(func(c *gollym.Config) {
		c.Set("zhipu_api_key", "test-key")
		c.Set("zhipu_api_base", srv.URL)
		c.MaxRetries = 0
		c.RequestTimeout = 5 * time.Second
	})
	chat, err := ctx.NewChat(gollym.ChatOptions{Model: "glm-5.3", Provider: "zhipu", AssumeModelExists: true})
	must(t, err)
	tools, err := stdlib(t).LoadToolset(write(t, "[[tools]]\npath = \"statistics.mean\"\n"), toolset.LoadOptions{})
	must(t, err)
	chat.WithTools(false, tools...)
	if _, err := chat.Ask(context.Background(), "mean of 1..4?"); err != nil {
		t.Fatal(err)
	}
	first := srv.Requests()[0].JSON(t)
	fn := first["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "statistics_mean" || fn["description"] != "Return the sample arithmetic mean of data." {
		t.Fatalf("tool definition %v", fn)
	}
	msgs := srv.Requests()[1].JSON(t)["messages"].([]any)
	if last := msgs[len(msgs)-1].(map[string]any); last["role"] != "tool" || last["content"] != "2.5" {
		t.Fatalf("tool result %v", last)
	}
}
