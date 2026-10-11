package riverjobs

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"

	"github.com/PhotonEra-B-V/gollym/core"
)

// MessageFromMap builds a message from its to_dict form (pyllym's
// Message(dict)): role (required), content, model_id, tool_call_id,
// tool_calls (keyed by id, or a list), token counts, thinking,
// thinking_signature, citations and finish_reason.
func MessageFromMap(m map[string]any) (*core.Message, error) {
	role, ok := m["role"]
	if !ok {
		return nil, errors.New("message is missing 'role'")
	}
	msg := core.Message{
		Role:         fmt.Sprint(role),
		Content:      m["content"],
		ModelID:      str(m["model_id"]),
		ToolCallID:   str(m["tool_call_id"]),
		FinishReason: str(m["finish_reason"]),
		Tokens: core.BuildTokens(core.AsInt(m["input_tokens"]), core.AsInt(m["output_tokens"]),
			core.AsInt(m["cached_tokens"]), core.AsInt(m["cache_creation_tokens"]), core.AsInt(m["thinking_tokens"])),
		Thinking: core.BuildThinking(str(m["thinking"]), str(m["thinking_signature"])),
	}
	switch calls := m["tool_calls"].(type) {
	case map[string]any:
		msg.ToolCalls = &core.ToolCalls{}
		for _, key := range sortedKeys(calls) {
			tc := toolCallFromMap(calls[key])
			if tc.ID == "" {
				tc.ID = key
			}
			msg.ToolCalls.Set(key, tc)
		}
	case []any:
		msg.ToolCalls = &core.ToolCalls{}
		for _, raw := range calls {
			tc := toolCallFromMap(raw)
			msg.ToolCalls.Set(tc.ID, tc)
		}
	}
	if c, ok := m["citations"]; ok && c != nil {
		b, err := json.Marshal(c)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(b, &msg.Citations); err != nil {
			return nil, fmt.Errorf("message citations: %w", err)
		}
	}
	return core.NewMessage(msg)
}

func toolCallFromMap(v any) *core.ToolCall {
	d, _ := v.(map[string]any)
	tc := &core.ToolCall{ID: str(d["id"]), Name: str(d["name"]), Arguments: d["arguments"],
		ThoughtSignature: str(d["thought_signature"])}
	if s, ok := tc.Arguments.(string); ok {
		var decoded map[string]any
		if json.Unmarshal([]byte(s), &decoded) == nil {
			tc.Arguments = decoded
		}
	}
	if tc.Arguments == nil {
		tc.Arguments = map[string]any{}
	}
	return tc
}

// MessageMap is the message's to_dict form with JSON-safe content: text
// for content with attachments, the raw value for RawContent.
func MessageMap(m *core.Message) map[string]any {
	out := m.Map()
	switch c := m.Content.(type) {
	case *core.Content:
		out["content"] = c.Text
		if c.Text == nil {
			delete(out, "content")
		}
	case *core.SearchResults:
		out["content"] = c.TextOrEmpty()
	case core.RawContent:
		out["content"] = c.Value
	}
	return out
}

func str(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprint(v)
}

// sortedKeys orders a JSON object's keys (Go maps carry no order).
func sortedKeys(m map[string]any) []string { return slices.Sorted(maps.Keys(m)) }
