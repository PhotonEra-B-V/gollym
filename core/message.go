package core

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// Content is text plus zero or more attachments.
type Content struct {
	Text        *string
	Attachments []*Attachment
}

// NewContent builds Content from text and attachment sources (each a value
// NewAttachment accepts; nil and blank strings are skipped). It is an error
// for both to be empty.
func NewContent(text *string, attachments ...any) (*Content, error) {
	c := &Content{Text: text}
	for _, src := range attachments {
		if err := c.addSources(src); err != nil {
			return nil, err
		}
	}
	if c.Text == nil && len(c.Attachments) == 0 {
		return nil, errors.New("Text and attachments cannot be both nil")
	}
	return c, nil
}

// TextContent is Content holding only text.
func TextContent(text string) *Content { return &Content{Text: &text} }

func (c *Content) addSources(src any) error {
	switch s := src.(type) {
	case nil:
		return nil
	case string:
		if strings.TrimSpace(s) == "" {
			return nil
		}
	case []string:
		for _, x := range s {
			if err := c.addSources(x); err != nil {
				return err
			}
		}
		return nil
	case []any:
		for _, x := range s {
			if err := c.addSources(x); err != nil {
				return err
			}
		}
		return nil
	case map[string]any:
		keys := make([]string, 0, len(s))
		for k := range s {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			if err := c.addSources(s[k]); err != nil {
				return err
			}
		}
		return nil
	}
	return c.AddAttachment(src, "")
}

// AddAttachment appends one attachment.
func (c *Content) AddAttachment(source any, filename string) error {
	a, err := NewAttachment(source, filename)
	if err != nil {
		return err
	}
	c.Attachments = append(c.Attachments, a)
	return nil
}

// Format returns the bare text when there are no attachments, else c.
func (c *Content) Format() any {
	if c.Text != nil && len(c.Attachments) == 0 {
		return *c.Text
	}
	return c
}

// IsEmpty reports no attachments and no (or empty) text.
func (c *Content) IsEmpty() bool {
	return len(c.Attachments) == 0 && (c.Text == nil || *c.Text == "")
}

// TextOrEmpty returns the text ("" when nil).
func (c *Content) TextOrEmpty() string {
	if c == nil || c.Text == nil {
		return ""
	}
	return *c.Text
}

// RawContent is a provider-specific payload that bypasses formatting.
type RawContent struct{ Value any }

// SearchResults are tool results the model can cite: providers with native
// citation support receive citable blocks, others the plain-text rendering.
type SearchResults struct {
	Content
	Results []map[string]string
}

// NewSearchResults validates and renders results (each needs title and
// text; url optional).
func NewSearchResults(results ...map[string]string) (*SearchResults, error) {
	if len(results) == 0 {
		return nil, errors.New("SearchResults requires at least one result")
	}
	var parts []string
	norm := make([]map[string]string, 0, len(results))
	for _, r := range results {
		if r["title"] == "" || r["text"] == "" {
			return nil, errors.New("Search results require 'title' and 'text'")
		}
		e := map[string]string{"title": r["title"], "text": r["text"]}
		if u, ok := r["url"]; ok {
			e["url"] = u
		}
		norm = append(norm, e)
		attrs := fmt.Sprintf("title='%s'", e["title"])
		if e["url"] != "" {
			attrs += fmt.Sprintf(" url='%s'", e["url"])
		}
		parts = append(parts, fmt.Sprintf("<search_result %s>\n%s\n</search_result>", attrs, e["text"]))
	}
	text := strings.Join(parts, "\n\n")
	return &SearchResults{Content: Content{Text: &text}, Results: norm}, nil
}

// Roles.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

var roles = map[string]bool{RoleSystem: true, RoleUser: true, RoleAssistant: true, RoleTool: true}

var (
	finishStopped = map[string]bool{"stop": true, "end_turn": true, "stop_sequence": true}
	finishMax     = map[string]bool{"length": true, "max_tokens": true, "max_output_tokens": true, "model_context_window_exceeded": true}
	finishTool    = map[string]bool{"tool_calls": true, "tool_use": true, "function_call": true}
	finishFilter  = map[string]bool{
		"blocklist": true, "content_filter": true, "content_filtered": true, "guardrail_intervened": true,
		"image_recitation": true, "image_safety": true, "model_armor": true, "prohibited_content": true,
		"recitation": true, "safety": true, "spii": true,
	}
)

// Message is one message in a conversation.
//
// Content holds what pyllym's message.content returns: a string, *Content
// (text with attachments), *SearchResults, RawContent, or decoded JSON
// (map/slice) for structured-output responses. Nil is allowed (an assistant
// message that only calls tools).
type Message struct {
	Role         string
	Content      any
	ModelID      string
	ToolCalls    *ToolCalls
	ToolCallID   string
	Tokens       *Tokens
	Raw          any
	Thinking     *Thinking
	Citations    []Citation
	FinishReason string

	modelInfo       *Info
	modelInfoLoaded bool
}

// NewMessage validates the role and normalises content the way pyllym's
// Message.__init__ does (strings and *Content collapse via Content.format).
func NewMessage(m Message) (*Message, error) {
	if !roles[m.Role] {
		return nil, fmt.Errorf("%w: Expected role to be one of: assistant, system, tool, user", ErrInvalidRole)
	}
	if m.Role == RoleAssistant && m.Content == nil && m.ToolCalls.Len() > 0 {
		m.Content = ""
	}
	if c, ok := m.Content.(*Content); ok {
		m.Content = c.Format()
	}
	out := m
	return &out, nil
}

// MustMessage is NewMessage for roles known to be valid.
func MustMessage(m Message) *Message {
	out, err := NewMessage(m)
	if err != nil {
		panic(err)
	}
	return out
}

// Text returns the message text: the string content, the text of *Content,
// or the JSON of structured content.
func (m *Message) Text() string {
	switch c := m.Content.(type) {
	case nil:
		return ""
	case string:
		return c
	case *Content:
		return c.TextOrEmpty()
	case *SearchResults:
		return c.TextOrEmpty()
	case RawContent:
		if s, ok := c.Value.(string); ok {
			return s
		}
		return MustJSON(c.Value)
	}
	return MustJSON(m.Content)
}

func (m *Message) IsToolCall() bool   { return m.ToolCalls.Len() > 0 }
func (m *Message) IsToolResult() bool { return m.ToolCallID != "" }

func (m *Message) finishKey() string {
	return strings.ReplaceAll(strings.ToLower(m.FinishReason), "-", "_")
}

func (m *Message) IsStopped() bool         { return finishStopped[m.finishKey()] }
func (m *Message) IsMaxTokens() bool       { return finishMax[m.finishKey()] }
func (m *Message) IsToolCallStop() bool    { return finishTool[m.finishKey()] }
func (m *Message) IsContentFiltered() bool { return finishFilter[m.finishKey()] }

func (m *Message) tok(f func(*Tokens) *int) *int {
	if m.Tokens == nil {
		return nil
	}
	return f(m.Tokens)
}

func (m *Message) InputTokens() *int  { return m.tok(func(t *Tokens) *int { return t.Input }) }
func (m *Message) OutputTokens() *int { return m.tok(func(t *Tokens) *int { return t.Output }) }
func (m *Message) CachedTokens() *int { return m.tok(func(t *Tokens) *int { return t.Cached }) }
func (m *Message) CacheCreationTokens() *int {
	return m.tok(func(t *Tokens) *int { return t.CacheCreation })
}
func (m *Message) ThinkingTokens() *int { return m.tok(func(t *Tokens) *int { return t.Thinking }) }

// ModelInfo looks the message's model up in the registry (nil if unknown).
func (m *Message) ModelInfo() *Info {
	if m.ModelID == "" {
		return nil
	}
	if !m.modelInfoLoaded {
		m.modelInfoLoaded = true
		if info, err := Models().Find(m.ModelID, ""); err == nil {
			m.modelInfo = info
		}
	}
	return m.modelInfo
}

// Cost prices this message's tokens with model (or its own model).
func (m *Message) Cost(model *Info) *Cost {
	if model == nil {
		model = m.ModelInfo()
	}
	return NewCost(m.Tokens, model)
}

// Map is the to_dict form.
func (m *Message) Map() map[string]any {
	out := map[string]any{"role": m.Role}
	if m.Content != nil {
		out["content"] = m.Content
	}
	if m.ModelID != "" {
		out["model_id"] = m.ModelID
	}
	if m.ToolCalls.Len() > 0 {
		calls := map[string]any{}
		for _, k := range m.ToolCalls.Keys() {
			calls[k] = m.ToolCalls.Get(k).Map()
		}
		out["tool_calls"] = calls
	}
	if m.ToolCallID != "" {
		out["tool_call_id"] = m.ToolCallID
	}
	if m.Thinking != nil {
		if m.Thinking.Text != "" {
			out["thinking"] = m.Thinking.Text
		}
		if m.Thinking.Signature != "" {
			out["thinking_signature"] = m.Thinking.Signature
		}
	}
	if len(m.Citations) > 0 {
		out["citations"] = m.Citations
	}
	if m.FinishReason != "" {
		out["finish_reason"] = m.FinishReason
	}
	if m.Tokens != nil {
		for k, v := range m.Tokens.Map() {
			out[k] = v
		}
	}
	return out
}

// Chunk is a streaming partial message.
type Chunk = Message
