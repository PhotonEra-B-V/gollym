package bdd

import (
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"
)

// Minimal Mermaid sequenceDiagram parser producing value objects (pyllym's
// bdd/sequence.py).
//
// Sequence diagrams complement a TOML spec's rules and cases: rules say what
// must be true, diagrams say who talks to whom, in what order. Each message
// gets a stable id (M1, M2, ...) so test cases can claim coverage of it and
// the builder can verify, mechanically, that no interaction was dropped.
//
// Covers the common subset: participants/actors (with aliases), call arrows
// (->>, ->), reply arrows (-->>, -->), async arrows (-), --)), +/- activation
// markers, and alt/else/opt/loop/par/critical blocks (recorded as context on
// each message). Notes and styling directives are ignored.

var arrowRe = regexp.MustCompile(
	`^(?P<sender>[^-<>:]+?)\s*` +
		`(?P<arrow>-->>|->>|--\)|-\)|-->|->)\s*` +
		`(?P<activation>[+-]?)\s*` +
		`(?P<receiver>[^:]+?)\s*:\s*` +
		`(?P<text>.*)$`)

var arrowKinds = map[string]string{
	"->>": "call", "->": "call",
	"-->>": "reply", "-->": "reply",
	"-)": "async", "--)": "async",
}

var blockOpeners = []string{"alt", "opt", "loop", "par", "critical", "rect", "break"}

// SequenceMessage is one arrow of a diagram.
type SequenceMessage struct {
	ID       string   `json:"id"`
	Sender   string   `json:"sender"`
	Receiver string   `json:"receiver"`
	Text     string   `json:"text"`
	Kind     string   `json:"kind"`    // call | reply | async
	Context  []string `json:"context"` // enclosing alt/opt/loop labels
}

// SequenceDiagram is a parsed diagram.
type SequenceDiagram struct {
	Participants []string          `json:"participants"`
	Messages     []SequenceMessage `json:"messages"`
	Source       string            `json:"source"` // normalized mermaid, for verbatim re-emission
}

// ToAnnotated renders one line per message with its id: the planner prompt
// format.
func (d *SequenceDiagram) ToAnnotated() string {
	lines := make([]string, 0, len(d.Messages))
	for _, m := range d.Messages {
		where := ""
		if len(m.Context) > 0 {
			where = " [" + strings.Join(m.Context, " / ") + "]"
		}
		lines = append(lines, fmt.Sprintf("%s: %s -[%s]-> %s: %s%s", m.ID, m.Sender, m.Kind, m.Receiver, m.Text, where))
	}
	return strings.Join(lines, "\n")
}

// ParseSequence parses a mermaid sequenceDiagram block. start sets the first
// message number, so several diagrams of one feature keep unique ids.
func ParseSequence(text string, start int) (*SequenceDiagram, error) {
	var participants []string
	var messages []SequenceMessage
	var context, sourceLines []string
	seenHeader := false
	n := start

	note := func(name string) {
		if !slices.Contains(participants, name) {
			participants = append(participants, name)
		}
	}

	for _, raw := range splitLines(text) {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "%%") {
			continue
		}
		sourceLines = append(sourceLines, line)
		if line == "sequenceDiagram" {
			seenHeader = true
			continue
		}
		word, rest, _ := strings.Cut(line, " ")
		switch {
		case word == "participant" || word == "actor":
			// "participant A as Alias" declares A; messages use the bare name.
			name, _, _ := strings.Cut(rest, " as ")
			note(strings.TrimSpace(name))
			continue
		case slices.Contains(blockOpeners, word):
			context = append(context, strings.TrimSpace(word+" "+strings.TrimSpace(rest)))
			continue
		case word == "else":
			if len(context) > 0 {
				context[len(context)-1] = strings.TrimSpace("else " + strings.TrimSpace(rest))
			}
			continue
		case word == "end":
			if len(context) > 0 {
				context = context[:len(context)-1]
			}
			continue
		}
		m := arrowRe.FindStringSubmatch(line)
		if m == nil {
			continue // notes, activate/deactivate lines, styling
		}
		sender := strings.TrimSpace(m[arrowRe.SubexpIndex("sender")])
		receiver := strings.TrimSpace(m[arrowRe.SubexpIndex("receiver")])
		note(sender)
		note(receiver)
		messages = append(messages, SequenceMessage{
			ID:       fmt.Sprintf("M%d", n),
			Sender:   sender,
			Receiver: receiver,
			Text:     strings.TrimSpace(m[arrowRe.SubexpIndex("text")]),
			Kind:     arrowKinds[m[arrowRe.SubexpIndex("arrow")]],
			Context:  append([]string{}, context...),
		})
		n++
	}
	if !seenHeader {
		return nil, errors.New("no 'sequenceDiagram' header found in mermaid source")
	}
	return &SequenceDiagram{
		Participants: nonNil(participants),
		Messages:     messages,
		Source:       strings.Join(sourceLines, "\n"),
	}, nil
}

// splitLines splits like Python's str.splitlines for the common line endings.
func splitLines(s string) []string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	lines := strings.Split(s, "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}
