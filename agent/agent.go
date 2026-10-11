// Package agent declares reusable chat configurations (pyllym's Agent).
//
// pyllym configures an agent with class attributes on a subclass; Go has no
// class attributes, so an Agent is a plain struct value, typically a
// package-level variable:
//
//	var Researcher = agent.Agent{
//		Model:        "claude-sonnet-4-6",
//		Instructions: "You are a meticulous research assistant.",
//		Tools:        []core.Tool{webSearch},
//		Temperature:  agent.Float(0.2),
//	}
//
//	chat, err := Researcher.NewChat(core.ChatOptions{})
//	msg, err := chat.Ask(ctx, "Summarize the latest on ...")
//
// Persistence is handled separately by the persistence package.
package agent

import (
	"maps"

	"github.com/PhotonEra-B-V/gollym/core"
)

// Agent is a declarative chat configuration. Zero fields are left unset, as
// pyllym's None/empty class attributes are.
type Agent struct {
	// Model is pyllym's chat_model; it applies unless ChatOptions or the
	// NewChat options name a model.
	Model string
	// ChatOptions are pyllym's chat_kwargs: defaults for core.NewChat.
	ChatOptions  core.ChatOptions
	Instructions string
	Tools        []core.Tool
	Temperature  *float64
	// Thinking is applied with WithThinking (effort and/or budget).
	Thinking  *core.ThinkingConfig
	Citations *bool
	Params    map[string]any
	Headers   map[string]string
	// Schema is anything Chat.WithSchema accepts.
	Schema any
	// Context binds new chats to a configuration scope unless the options
	// give one.
	Context *core.Context
}

// NewChat creates a conversation pre-configured by the agent (pyllym
// create_chat). Non-zero fields of opts override the agent's ChatOptions.
func (a Agent) NewChat(opts core.ChatOptions) (*core.Chat, error) {
	o := a.ChatOptions
	if a.Model != "" && o.Model == "" {
		o.Model = a.Model
	}
	if opts.Model != "" {
		o.Model = opts.Model
	}
	if opts.Provider != "" {
		o.Provider = opts.Provider
	}
	if opts.AssumeModelExists {
		o.AssumeModelExists = true
	}
	if opts.Context != nil {
		o.Context = opts.Context
	}
	if o.Context == nil {
		o.Context = a.Context
	}
	chat, err := core.NewChat(o)
	if err != nil {
		return nil, err
	}
	if err := a.Configure(chat); err != nil {
		return nil, err
	}
	return chat, nil
}

// Configure applies the agent's settings to an existing chat.
func (a Agent) Configure(chat *core.Chat) error {
	if a.Instructions != "" {
		chat.WithInstructions(a.Instructions)
	}
	if len(a.Tools) > 0 {
		chat.WithTools(false, a.Tools...)
	}
	if a.Temperature != nil {
		chat.WithTemperature(*a.Temperature)
	}
	if a.Thinking != nil {
		if _, err := chat.WithThinking(a.Thinking.Effort, a.Thinking.Budget); err != nil {
			return err
		}
	}
	if a.Citations != nil {
		chat.WithCitations(*a.Citations)
	}
	// Copies, as pyllym's **params/**headers are: chats of one agent don't
	// share mutable maps.
	if len(a.Params) > 0 {
		chat.WithParams(maps.Clone(a.Params))
	}
	if len(a.Headers) > 0 {
		chat.WithHeaders(maps.Clone(a.Headers))
	}
	if a.Schema != nil {
		if _, err := chat.WithSchema(a.Schema); err != nil {
			return err
		}
	}
	return nil
}

// Float returns a pointer, for Temperature.
func Float(v float64) *float64 { return &v }

// Bool returns a pointer, for Citations.
func Bool(v bool) *bool { return &v }
