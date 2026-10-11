package core

import (
	"sort"
	"strings"
)

func sortStrings(s []string) { sort.Strings(s) }

// SSEParser splits a server-sent-events byte stream into (event, data)
// pairs: multi-line data fields are joined with "\n" and an event is emitted
// on each blank line; comments and id:/retry: fields are ignored.
type SSEParser struct {
	buf       strings.Builder
	eventType string
	data      []string
	emit      func(event, data string) error
}

// NewSSEParser calls emit for each complete event.
func NewSSEParser(emit func(event, data string) error) *SSEParser {
	return &SSEParser{eventType: "message", emit: emit}
}

// Write feeds raw bytes.
func (p *SSEParser) Write(chunk []byte) error {
	p.buf.WriteString(strings.ToValidUTF8(string(chunk), "�"))
	pending := p.buf.String()
	p.buf.Reset()
	for {
		i := strings.IndexByte(pending, '\n')
		if i < 0 {
			break
		}
		line := strings.TrimRight(pending[:i], "\r")
		pending = pending[i+1:]
		if err := p.line(line); err != nil {
			return err
		}
	}
	p.buf.WriteString(pending)
	return nil
}

func (p *SSEParser) line(line string) error {
	switch {
	case line == "":
		if len(p.data) > 0 {
			data := strings.Join(p.data, "\n")
			p.data = nil
			ev := p.eventType
			p.eventType = "message"
			return p.emit(ev, data)
		}
		p.eventType = "message"
	case strings.HasPrefix(line, ":"):
	case strings.HasPrefix(line, "event:"):
		p.eventType = strings.TrimSpace(line[len("event:"):])
	case strings.HasPrefix(line, "data:"):
		p.data = append(p.data, strings.TrimLeft(line[len("data:"):], " \t"))
	}
	return nil
}

// Close flushes a trailing event without a final blank line.
func (p *SSEParser) Close() error {
	if rest := p.buf.String(); rest != "" {
		p.buf.Reset()
		if err := p.line(strings.TrimRight(rest, "\r")); err != nil {
			return err
		}
	}
	if len(p.data) > 0 {
		data := strings.Join(p.data, "\n")
		p.data = nil
		return p.emit(p.eventType, data)
	}
	return nil
}
