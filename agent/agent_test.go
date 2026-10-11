package agent_test

import (
	"context"
	"reflect"
	"testing"

	"github.com/PhotonEra-B-V/gollym"
	"github.com/PhotonEra-B-V/gollym/agent"
	"github.com/PhotonEra-B-V/gollym/core"
)

// Adder mirrors pyllym's test tool class; its name derives from the type.
type Adder struct{}

func (a Adder) Name() string                                    { return core.ToolNameFor(a) }
func (Adder) Description() string                               { return "add" }
func (Adder) ParamsSchema() map[string]any                      { return nil }
func (Adder) Call(context.Context, map[string]any) (any, error) { return 0, nil }

func testContext() *core.Context {
	return gollym.NewContext(func(c *gollym.Config) {
		c.Set("openai_api_key", "test")
		c.Set("anthropic_api_key", "test")
	})
}

// pyllym test_behaviors: test_agent_applies_configuration.
func TestAgentAppliesConfiguration(t *testing.T) {
	helper := agent.Agent{
		Model:        "gpt-4o",
		Instructions: "be concise",
		Tools:        []core.Tool{Adder{}},
		Temperature:  agent.Float(0.3),
		Context:      testContext(),
	}
	chat, err := helper.NewChat(core.ChatOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if chat.Model.ID != "gpt-4o" {
		t.Fatalf("model %q", chat.Model.ID)
	}
	hasSystem := false
	for _, m := range chat.Messages {
		hasSystem = hasSystem || m.Role == core.RoleSystem
	}
	if !hasSystem {
		t.Fatal("no system message")
	}
	if _, ok := chat.Tools.Get("adder"); !ok {
		t.Fatalf("tools %v", chat.Tools.Names())
	}
	if chat.Temperature() == nil || *chat.Temperature() != 0.3 {
		t.Fatalf("temperature %v", chat.Temperature())
	}
}

func TestAgentAppliesEverySetting(t *testing.T) {
	budget := 2048
	a := agent.Agent{
		Model:     "gpt-4o",
		Thinking:  &core.ThinkingConfig{Effort: "high", Budget: &budget},
		Citations: agent.Bool(true),
		Params:    map[string]any{"max_tokens": 77},
		Headers:   map[string]string{"X-Team": "ml"},
		Schema:    map[string]any{"type": "object", "properties": map[string]any{"x": map[string]any{"type": "string"}}},
		Context:   testContext(),
	}
	chat, err := a.NewChat(core.ChatOptions{})
	if err != nil {
		t.Fatal(err)
	}
	req := chat.Request()
	if req.Thinking == nil || req.Thinking.Effort != "high" || *req.Thinking.Budget != 2048 || !req.Citations {
		t.Fatalf("thinking %v citations %v", req.Thinking, req.Citations)
	}
	if req.Headers["X-Team"] != "ml" || req.Params["max_tokens"] != 77 || req.Schema["name"] != "response" {
		t.Fatalf("headers %v params %v schema %v", req.Headers, req.Params, req.Schema)
	}
	// Chats get their own copies of the agent's maps.
	chat.Params["max_tokens"] = 1
	chat.Headers["X-Team"] = "other"
	if a.Params["max_tokens"] != 77 || a.Headers["X-Team"] != "ml" {
		t.Fatal("agent maps were shared with the chat")
	}
	// Nothing set: nothing applied.
	plain, err := agent.Agent{Model: "gpt-4o", Context: testContext()}.NewChat(core.ChatOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(plain.Messages) != 0 || plain.Tools.Len() != 0 || plain.Temperature() != nil || plain.Schema != nil || len(plain.Params) != 0 {
		t.Fatalf("plain chat was configured: %+v", plain.Request())
	}
}

func TestChatOptionsPrecedence(t *testing.T) {
	ctx := testContext()
	a := agent.Agent{Model: "gpt-4o", Context: ctx}
	// The call's options win over the agent's model.
	chat, err := a.NewChat(core.ChatOptions{Model: "claude-sonnet-4-6"})
	if err != nil {
		t.Fatal(err)
	}
	if chat.Model.ID != "claude-sonnet-4-6" || chat.Config() != ctx.Config {
		t.Fatalf("model %q", chat.Model.ID)
	}
	// ChatOptions (chat_kwargs) name a model: it wins over Model.
	a.ChatOptions = core.ChatOptions{Model: "my-model", Provider: "openai", AssumeModelExists: true}
	chat, err = a.NewChat(core.ChatOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if chat.Model.ID != "my-model" || chat.Provider.Slug() != "openai" {
		t.Fatalf("model %q provider %q", chat.Model.ID, chat.Provider.Slug())
	}
	// An explicit context wins over the agent's.
	other := testContext()
	chat, err = agent.Agent{Model: "gpt-4o", Context: ctx}.NewChat(core.ChatOptions{Context: other})
	if err != nil {
		t.Fatal(err)
	}
	if chat.Config() != other.Config {
		t.Fatal("options context ignored")
	}
}

func TestInvalidSettingsAreErrors(t *testing.T) {
	if _, err := (agent.Agent{Model: "gpt-4o", Context: testContext(), Thinking: &core.ThinkingConfig{}}).NewChat(core.ChatOptions{}); err == nil {
		t.Fatal("empty thinking should fail like with_thinking()")
	}
	if _, err := (agent.Agent{Model: "no-such-model-xyz", Context: testContext()}).NewChat(core.ChatOptions{}); err == nil {
		t.Fatal("unknown model should fail")
	}
}

func TestConfigureExistingChat(t *testing.T) {
	chat, err := testContext().NewChat(core.ChatOptions{Model: "gpt-4o"})
	if err != nil {
		t.Fatal(err)
	}
	if err := (agent.Agent{Instructions: "hi"}).Configure(chat); err != nil {
		t.Fatal(err)
	}
	if len(chat.Messages) != 1 || !reflect.DeepEqual(chat.Messages[0].Content, "hi") {
		t.Fatalf("messages %v", chat.Messages)
	}
}
