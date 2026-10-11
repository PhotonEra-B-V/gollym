// Package testserver is a recording HTTP stub for provider tests (pyllym's
// stub_llm_server): it records every request and replays scripted replies.
package testserver

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

// Request is one recorded request.
type Request struct {
	Method  string
	Path    string
	Query   string
	Headers http.Header
	Body    []byte
}

// JSON decodes the recorded body.
func (r Request) JSON(t testing.TB) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(r.Body, &m); err != nil {
		t.Fatalf("request body is not JSON: %v\n%s", err, r.Body)
	}
	return m
}

// Reply is one scripted response.
type Reply struct {
	Status  int
	Headers map[string]string
	Body    any // JSON-encoded unless string or []byte
	// Hang closes the connection without answering (connection error).
	Hang bool
}

// Server records requests and serves replies in order (the last repeats).
type Server struct {
	*httptest.Server
	mu       sync.Mutex
	requests []Request
	replies  []Reply
}

// New starts a server serving replies in order.
func New(t testing.TB, replies ...Reply) *Server {
	t.Helper()
	s := &Server{replies: replies}
	s.Server = httptest.NewServer(http.HandlerFunc(s.handle))
	t.Cleanup(s.Close)
	return s
}

func (s *Server) handle(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	s.mu.Lock()
	s.requests = append(s.requests, Request{Method: r.Method, Path: r.URL.Path, Query: r.URL.RawQuery, Headers: r.Header.Clone(), Body: body})
	n := len(s.requests)
	var rep Reply
	if len(s.replies) > 0 {
		rep = s.replies[min(n, len(s.replies))-1]
	}
	s.mu.Unlock()
	if rep.Hang {
		if hj, ok := w.(http.Hijacker); ok {
			conn, _, _ := hj.Hijack()
			conn.Close()
			return
		}
	}
	for k, v := range rep.Headers {
		w.Header().Set(k, v)
	}
	var out []byte
	switch b := rep.Body.(type) {
	case nil:
	case string:
		out = []byte(b)
	case []byte:
		out = b
	default:
		out, _ = json.Marshal(b)
		if w.Header().Get("Content-Type") == "" {
			w.Header().Set("Content-Type", "application/json")
		}
	}
	status := rep.Status
	if status == 0 {
		status = 200
	}
	w.WriteHeader(status)
	_, _ = w.Write(out)
}

// Requests returns the recorded requests.
func (s *Server) Requests() []Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Request(nil), s.requests...)
}

// Last returns the last recorded request.
func (s *Server) Last(t testing.TB) Request {
	t.Helper()
	r := s.Requests()
	if len(r) == 0 {
		t.Fatal("no requests recorded")
	}
	return r[len(r)-1]
}
