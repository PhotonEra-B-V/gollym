// Package persistence stores chats, messages and tool calls in PostgreSQL
// (pyllym.persistence). The tables match the ones pyllym's SQLAlchemy
// factory creates, so Python and Go can share them:
//
//	store, _ := persistence.New(pool)          // or a pgx.Tx
//	_ = store.CreateTables(ctx)                // or apply persistence.Schema in a migration
//	rec := &persistence.Chat{ModelID: "gpt-4o", Provider: core.Str("openai")}
//	_ = store.CreateChat(ctx, rec)
//	chat, _ := store.ToChat(rec, persistence.ChatOptions{})
//	_, _ = chat.Ask(ctx, "Hello!")             // user + assistant rows persisted
//
// Load a conversation back with LoadChat and ToChat.
package persistence

import (
	"encoding/json"
	"strings"

	"github.com/PhotonEra-B-V/gollym/core"
)

// Chat is a row of {prefix}chats; Messages are filled by LoadChat.
type Chat struct {
	ID       int64
	ModelID  string
	Provider *string
	Messages []*Message
}

// Message is a row of {prefix}messages with its tool calls.
type Message struct {
	ID                  int64
	ChatID              int64
	Role                string
	Content             *string
	ModelID             *string
	ToolCallID          *string
	InputTokens         *int
	OutputTokens        *int
	CachedTokens        *int
	CacheCreationTokens *int
	ThinkingTokens      *int
	ThinkingText        *string
	ThinkingSignature   *string
	// Citations is the JSON column: a list of citation objects, or nil.
	Citations    []core.Citation
	FinishReason *string
	ToolCalls    []*ToolCall
}

// ToolCall is a row of {prefix}tool_calls. Arguments is the JSON column
// (normally the decoded argument object).
type ToolCall struct {
	ID         int64
	MessageID  *int64
	ToolCallID string
	Name       string
	Arguments  any
}

// ToLLM converts the row to a core.ToolCall (empty arguments become {}).
func (t *ToolCall) ToLLM() *core.ToolCall {
	args := t.Arguments
	if !core.Truthy(args) {
		args = map[string]any{}
	}
	return &core.ToolCall{ID: t.ToolCallID, Name: t.Name, Arguments: args}
}

// ToLLM converts the row to a core.Message. Content that looks like JSON
// (starts with { or [) is decoded, as structured-output replies are stored
// as their JSON text.
func (m *Message) ToLLM() (*core.Message, error) {
	var content any
	if m.Content != nil {
		content = *m.Content
		if s := *m.Content; s == "" || strings.ContainsRune("{[", rune(s[0])) {
			var decoded any
			if err := json.Unmarshal([]byte(s), &decoded); err == nil {
				content = decoded
			}
		}
	}
	var calls *core.ToolCalls
	if len(m.ToolCalls) > 0 {
		calls = &core.ToolCalls{}
		for _, tc := range m.ToolCalls {
			calls.Set(tc.ToolCallID, tc.ToLLM())
		}
	}
	return core.NewMessage(core.Message{
		Role:         m.Role,
		Content:      content,
		ModelID:      core.Deref(m.ModelID),
		ToolCallID:   core.Deref(m.ToolCallID),
		ToolCalls:    calls,
		Tokens:       core.BuildTokens(m.InputTokens, m.OutputTokens, m.CachedTokens, m.CacheCreationTokens, m.ThinkingTokens),
		Thinking:     core.BuildThinking(core.Deref(m.ThinkingText), core.Deref(m.ThinkingSignature)),
		Citations:    m.Citations,
		FinishReason: core.Deref(m.FinishReason),
	})
}

// MessageFromLLM builds the row for a message of chat chatID. Non-string
// content is stored as JSON text (Python's json.dumps format). Text with
// attachments stores only the text: pyllym can't serialise attachments
// either (it raises), and losing the attachment beats failing the chat.
func MessageFromLLM(m *core.Message, chatID int64) *Message {
	rec := &Message{
		ChatID:              chatID,
		Role:                m.Role,
		Content:             contentText(m.Content),
		ModelID:             strOrNil(m.ModelID),
		ToolCallID:          strOrNil(m.ToolCallID),
		InputTokens:         m.InputTokens(),
		OutputTokens:        m.OutputTokens(),
		CachedTokens:        m.CachedTokens(),
		CacheCreationTokens: m.CacheCreationTokens(),
		ThinkingTokens:      m.ThinkingTokens(),
		FinishReason:        strOrNil(m.FinishReason),
	}
	if m.Thinking != nil {
		rec.ThinkingText = strOrNil(m.Thinking.Text)
		rec.ThinkingSignature = strOrNil(m.Thinking.Signature)
	}
	if len(m.Citations) > 0 {
		rec.Citations = append([]core.Citation(nil), m.Citations...)
	}
	for _, tc := range m.ToolCalls.List() {
		rec.ToolCalls = append(rec.ToolCalls, &ToolCall{ToolCallID: tc.ID, Name: tc.Name, Arguments: tc.Arguments})
	}
	return rec
}

func contentText(c any) *string {
	switch v := c.(type) {
	case nil:
		return nil
	case string:
		return &v
	case *core.Content:
		return v.Text
	case *core.SearchResults:
		return v.Text
	case core.RawContent:
		if s, ok := v.Value.(string); ok {
			return &s
		}
		s := core.PyJSON(v.Value)
		return &s
	}
	s := core.PyJSON(c)
	return &s
}

func strOrNil(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}
