package core

import (
	"encoding/json"
	"log/slog"
	"strings"

	"github.com/google/uuid"
)

// StreamAccumulator assembles streaming chunks into a complete message,
// splitting <think>...</think> text into Thinking across chunk boundaries.
type StreamAccumulator struct {
	Content           string
	citations         []Citation
	thinkingText      string
	thinkingSignature string
	ToolCalls         ToolCalls
	ModelID           string

	inputTokens, outputTokens, cachedTokens, cacheCreationTokens, thinkingTokens *int

	finishReason     string
	insideThinkTag   bool
	pendingThinkTag  string
	latestToolCallID string
	toolCallIDsByKey map[string]string
}

// NewStreamAccumulator returns an empty accumulator.
func NewStreamAccumulator() *StreamAccumulator {
	return &StreamAccumulator{toolCallIDsByKey: map[string]string{}}
}

// Add folds one chunk in.
func (a *StreamAccumulator) Add(c *Chunk) {
	if a.ModelID == "" {
		a.ModelID = c.ModelID
	}
	a.handleContent(c)
	for _, cit := range c.Citations {
		dup := false
		for _, have := range a.citations {
			if have.Equal(cit) {
				dup = true
				break
			}
		}
		if !dup {
			a.citations = append(a.citations, cit)
		}
	}
	if c.Thinking != nil {
		if c.Thinking.Text != "" {
			a.thinkingText += c.Thinking.Text
		}
		if a.thinkingSignature == "" {
			a.thinkingSignature = c.Thinking.Signature
		}
	}
	if c.FinishReason != "" {
		a.finishReason = c.FinishReason
	}
	if c.Tokens != nil {
		set := func(dst **int, v *int) {
			if v != nil && *v != 0 {
				*dst = v
			}
		}
		set(&a.inputTokens, c.Tokens.Input)
		set(&a.outputTokens, c.Tokens.Output)
		set(&a.cachedTokens, c.Tokens.Cached)
		set(&a.cacheCreationTokens, c.Tokens.CacheCreation)
		set(&a.thinkingTokens, c.Tokens.Thinking)
	}
}

// Message returns the assembled assistant message.
func (a *StreamAccumulator) Message(raw any) *Message {
	a.flushPendingThinkTag()
	var content any
	if a.Content != "" {
		content = a.Content
	}
	m := &Message{
		Role:         RoleAssistant,
		Content:      content,
		Citations:    a.resolvedCitations(),
		Thinking:     BuildThinking(a.thinkingText, a.thinkingSignature),
		Tokens:       BuildTokens(a.inputTokens, a.outputTokens, a.cachedTokens, a.cacheCreationTokens, a.thinkingTokens),
		FinishReason: a.finishReason,
		ModelID:      a.ModelID,
		Raw:          raw,
	}
	if tcs := a.toolCallsFromStream(); tcs.Len() > 0 {
		m.ToolCalls = tcs
		if m.Content == nil {
			m.Content = ""
		}
	}
	return m
}

func (a *StreamAccumulator) flushPendingThinkTag() {
	pending := a.pendingThinkTag
	a.pendingThinkTag = ""
	if pending == "" {
		return
	}
	if a.insideThinkTag {
		a.thinkingText += pending
	} else {
		a.Content += pending
	}
}

func (a *StreamAccumulator) resolvedCitations() []Citation {
	out := make([]Citation, 0, len(a.citations))
	for _, c := range a.citations {
		if c.Text == nil && c.StartIndex != nil && c.EndIndex != nil {
			s, e := *c.StartIndex, *c.EndIndex
			if span := pySlice(a.Content, s, e); span != "" {
				c.Text = &span
			}
		}
		out = append(out, c)
	}
	return out
}

// pySlice slices by code point with Python's clamping.
func pySlice(s string, start, end int) string {
	r := []rune(s)
	n := len(r)
	clamp := func(i int) int {
		if i < 0 {
			i += n
		}
		return max(0, min(i, n))
	}
	start, end = clamp(start), clamp(end)
	if start >= end {
		return ""
	}
	return string(r[start:end])
}

func (a *StreamAccumulator) toolCallsFromStream() *ToolCalls {
	out := &ToolCalls{}
	for _, key := range a.ToolCalls.Keys() {
		tc := a.ToolCalls.Get(key)
		args := tc.Arguments
		if s, ok := args.(string); ok {
			if s == "" {
				args = map[string]any{}
			} else {
				var v any
				if err := json.Unmarshal([]byte(s), &v); err != nil {
					trunc := s
					if len(trunc) > 200 {
						trunc = trunc[:200]
					}
					slog.Warn("Discarding malformed streamed tool-call arguments", "tool", tc.Name, "args", trunc)
					args = map[string]any{}
				} else {
					args = v
				}
			}
		}
		out.Set(key, &ToolCall{ID: tc.ID, Name: tc.Name, Arguments: args, ThoughtSignature: tc.ThoughtSignature})
	}
	return out
}

func (a *StreamAccumulator) handleContent(c *Chunk) {
	if c.IsToolCall() {
		for _, key := range c.ToolCalls.Keys() {
			tc := c.ToolCalls.Get(key)
			if tc.ID != "" {
				a.startToolCall(key, tc)
			} else {
				a.appendToolCallFragment(key, tc)
			}
		}
		return
	}
	switch v := c.Content.(type) {
	case nil:
	case string:
		out, thinking := a.extractThinkTags(v)
		a.Content += out
		a.thinkingText += thinking
	default:
		a.Content += MustJSON(v)
	}
}

func (a *StreamAccumulator) startToolCall(streamKey string, tc *ToolCall) {
	id := tc.ID
	if id == "" {
		id = uuid.NewString()
	}
	args := tc.Arguments
	switch v := args.(type) {
	case nil:
		args = ""
	case string:
	case map[string]any:
		if len(v) == 0 {
			args = ""
		}
	}
	a.ToolCalls.Set(tc.ID, &ToolCall{ID: id, Name: tc.Name, Arguments: args, ThoughtSignature: tc.ThoughtSignature})
	if streamKey != "" {
		a.toolCallIDsByKey[streamKey] = tc.ID
	}
	a.latestToolCallID = tc.ID
}

func (a *StreamAccumulator) appendToolCallFragment(streamKey string, tc *ToolCall) {
	existing := a.findToolCall(streamKey)
	if existing == nil {
		return
	}
	frag, _ := tc.Arguments.(string)
	if s, ok := existing.Arguments.(string); ok {
		existing.Arguments = s + frag
	}
	if tc.ThoughtSignature != "" && existing.ThoughtSignature == "" {
		existing.ThoughtSignature = tc.ThoughtSignature
	}
}

func (a *StreamAccumulator) findToolCall(streamKey string) *ToolCall {
	if streamKey == "" {
		return a.ToolCalls.Get(a.latestToolCallID)
	}
	if mapped, ok := a.toolCallIDsByKey[streamKey]; ok {
		if tc := a.ToolCalls.Get(mapped); tc != nil {
			return tc
		}
	}
	return a.ToolCalls.Get(streamKey)
}

const (
	thinkStart = "<think>"
	thinkEnd   = "</think>"
)

func (a *StreamAccumulator) extractThinkTags(text string) (string, string) {
	remaining := a.pendingThinkTag + text
	a.pendingThinkTag = ""
	var out, thinking strings.Builder
	for remaining != "" {
		if a.insideThinkTag {
			if i := strings.Index(remaining, thinkEnd); i >= 0 {
				thinking.WriteString(remaining[:i])
				a.insideThinkTag = false
				remaining = remaining[i+len(thinkEnd):]
				continue
			}
			n := longestSuffixPrefix(remaining, thinkEnd)
			thinking.WriteString(remaining[:len(remaining)-n])
			a.pendingThinkTag = remaining[len(remaining)-n:]
			remaining = ""
		} else {
			if i := strings.Index(remaining, thinkStart); i >= 0 {
				out.WriteString(remaining[:i])
				a.insideThinkTag = true
				remaining = remaining[i+len(thinkStart):]
				continue
			}
			n := longestSuffixPrefix(remaining, thinkStart)
			out.WriteString(remaining[:len(remaining)-n])
			a.pendingThinkTag = remaining[len(remaining)-n:]
			remaining = ""
		}
	}
	return out.String(), thinking.String()
}

func longestSuffixPrefix(text, tag string) int {
	for l := min(len(text), len(tag)-1); l > 0; l-- {
		if strings.HasSuffix(text, tag[:l]) {
			return l
		}
	}
	return 0
}
