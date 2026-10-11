package core_test

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/PhotonEra-B-V/gollym/core"
)

// Ports tests/test_core.py.

func TestUnderscore(t *testing.T) {
	for in, want := range map[string]string{"HTTPProxyTool": "http_proxy_tool", "OpenAI": "open_ai"} {
		if got := core.Underscore(in); got != want {
			t.Errorf("Underscore(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestDeepMerge(t *testing.T) {
	got := core.DeepMerge(map[string]any{"a": map[string]any{"b": 1}}, map[string]any{"a": map[string]any{"c": 2}})
	if want := map[string]any{"a": map[string]any{"b": 1, "c": 2}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v", got)
	}
}

func TestMessageRolesAndValidation(t *testing.T) {
	m, err := core.NewMessage(core.Message{Role: "user", Content: "hi"})
	if err != nil || m.Content != "hi" {
		t.Fatalf("msg %+v err %v", m, err)
	}
	if _, err := core.NewMessage(core.Message{Role: "wizard", Content: "x"}); !errors.Is(err, core.ErrInvalidRole) {
		t.Fatalf("err %v", err)
	} else if !strings.Contains(err.Error(), "Expected role to be one of: assistant, system, tool, user") {
		t.Fatalf("message %q", err)
	}
}

func TestContentFormatCollapsesToText(t *testing.T) {
	if got := core.TextContent("just text").Format(); got != "just text" {
		t.Fatalf("got %v", got)
	}
}

func TestModelsRegistryLookup(t *testing.T) {
	m, err := core.Models().Find("gpt-4o", "")
	if err != nil || m.Provider != "openai" {
		t.Fatalf("gpt-4o: %+v %v", m, err)
	}
	if m, err := core.Models().Find("claude-sonnet-4-6", ""); err != nil || m.Provider != "anthropic" {
		t.Fatalf("claude-sonnet-4-6: %+v %v", m, err)
	}
	if n := core.Models().ChatModels().Len(); n <= 100 {
		t.Fatalf("%d chat models", n)
	}
}

func TestModelNotFound(t *testing.T) {
	_, err := core.Models().Find("nonexistent-model-xyz", "")
	if !errors.Is(err, core.ErrModelNotFound) {
		t.Fatalf("err %v", err)
	}
}

// AddTool stands in for pyllym's `class AddTool(Tool)`: the name comes from
// the type name, the parameters from the execute signature.
type AddTool struct{}

func TestToolSchemaInference(t *testing.T) {
	tool := core.NewTool(core.ToolNameFor(AddTool{}), "Add numbers",
		[]core.Parameter{{Name: "a", Type: "integer", Required: true}, {Name: "b", Type: "integer"}},
		func(_ context.Context, args map[string]any) (any, error) { return nil, nil })
	if tool.Name() != "add" {
		t.Fatalf("name %q", tool.Name())
	}
	schema := tool.ParamsSchema()
	props := schema["properties"].(map[string]any)
	if len(props) != 2 || props["a"] == nil || props["b"] == nil {
		t.Fatalf("properties %v", props)
	}
	if !reflect.DeepEqual(schema["required"], []any{"a"}) {
		t.Fatalf("required %v", schema["required"])
	}
}

func TestToolCallValidationAndExecute(t *testing.T) {
	tool := core.NewTool(core.ToolNameFromString("EchoTool"), "", []core.Parameter{{Name: "text", Required: true}},
		func(_ context.Context, args map[string]any) (any, error) {
			return strings.ToUpper(args["text"].(string)), nil
		})
	out, err := tool.Call(context.Background(), map[string]any{"text": "hi"})
	if err != nil || out != "HI" {
		t.Fatalf("out %v err %v", out, err)
	}
	out, err = tool.Call(context.Background(), map[string]any{"wrong": "x"})
	m, _ := out.(map[string]any)
	if err != nil || m["error"] != "Invalid tool arguments: missing keyword: text" {
		t.Fatalf("out %v err %v", out, err)
	}
	out, _ = tool.Call(context.Background(), map[string]any{"text": "x", "extra": 1})
	if m, _ := out.(map[string]any); m["error"] != "Invalid tool arguments: unknown keyword: extra" {
		t.Fatalf("out %v", out)
	}
}
