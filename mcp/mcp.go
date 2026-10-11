// Package mcp adapts tools served by an external MCP (Model Context
// Protocol) server into core.Tool values, so they drop straight into
// chat.WithTools (pyllym's mcp module).
//
//	server := mcp.Stdio("npx", []string{"-y", "@modelcontextprotocol/server-filesystem", "/tmp"}, mcp.StdioOptions{})
//	if err := server.Connect(ctx); err != nil { ... }
//	defer server.Close()
//	chat.WithTools(false, server.Tools...)
//
//	remote := mcp.HTTP("https://example.com/mcp", mcp.HTTPOptions{Headers: map[string]string{...}})
//
// Transports come from the official Go SDK
// (github.com/modelcontextprotocol/go-sdk), imported here as sdk.
//
// Design notes (as in pyllym):
//   - Server owns the session lifecycle (transport + initialize). Python's
//     "async with" becomes Connect/Close. Tools are snapshotted on Connect;
//     call Refresh to re-list after a tools/list_changed.
//   - The adapter core, ToolsFromSession, works against anything with
//     ListTools/CallTool (the Session interface, which *sdk.ClientSession
//     satisfies), so it is testable without a server.
//   - Tool results: structuredContent wins when present; otherwise text
//     blocks are joined; non-text blocks become small maps. A result with
//     isError set comes back as {"error": ...}, the convention core tools use
//     for invalid arguments, so the model can recover.
//
// A stdio server inherits only a minimal environment (HOME, LOGNAME, PATH,
// SHELL, TERM, USER; the Windows equivalents on Windows) plus
// StdioOptions.Env, as the Python SDK does, so API keys in the parent's
// environment don't leak to it.
//
// Security model: an MCP server is an arbitrary external program or endpoint.
// A stdio server is code you execute; an HTTP server is a service you trust
// with your tool arguments (which may contain conversation data). Connect
// only to servers you would run by hand; gollym adds no sandbox.
package mcp

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strings"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/PhotonEra-B-V/gollym/core"
)

// Session is the slice of an MCP client session the adapter needs.
// *sdk.ClientSession implements it.
type Session interface {
	ListTools(ctx context.Context, params *sdk.ListToolsParams) (*sdk.ListToolsResult, error)
	CallTool(ctx context.Context, params *sdk.CallToolParams) (*sdk.CallToolResult, error)
}

// Tool is a remote MCP tool adapted to core.Tool. Name, description and JSON
// schema come from the server's tools/list entry; Call forwards the
// arguments over the live session.
type Tool struct {
	session     Session
	remoteName  string
	name        string
	description string
	schema      map[string]any
}

// NewTool wraps one remote tool. prefix namespaces the advertised name; the
// remote call still uses the server's name.
func NewTool(session Session, name, description string, schema map[string]any, prefix string) *Tool {
	if description == "" {
		description = "MCP tool " + name + "."
	}
	return &Tool{session: session, remoteName: name, name: prefix + name, description: description, schema: schema}
}

func (t *Tool) Name() string                 { return t.name }
func (t *Tool) Description() string          { return t.description }
func (t *Tool) ParamsSchema() map[string]any { return t.schema }

// RemoteName is the tool's name on the server (without the prefix).
func (t *Tool) RemoteName() string { return t.remoteName }

// Call runs the remote tool. Like pyllym's MCPTool (which takes **kwargs) it
// does no local validation; the server checks the arguments. Transport and
// protocol errors are returned as errors; tool errors (isError) as
// {"error": ...} results.
func (t *Tool) Call(ctx context.Context, args map[string]any) (any, error) {
	if args == nil {
		args = map[string]any{}
	}
	slog.Debug("Tool called", "tool", t.name, "args", args)
	result, err := t.session.CallTool(ctx, &sdk.CallToolParams{Name: t.remoteName, Arguments: args})
	if err != nil {
		return nil, err
	}
	out := convertResult(result)
	slog.Debug("Tool returned", "tool", t.name, "result", out)
	return out, nil
}

// ToolsOptions configure ToolsFromSession.
type ToolsOptions struct {
	// Prefix namespaces the advertised tool names (useful when combining
	// servers whose tool names could collide).
	Prefix string
	// Only restricts to the named remote tools. nil means all; a non-nil
	// empty slice selects none (pyllym: only=None vs only=set()).
	Only []string
}

// ToolsFromSession lists a session's tools (following pagination) and wraps
// each as a *Tool.
func ToolsFromSession(ctx context.Context, session Session, o ToolsOptions) ([]core.Tool, error) {
	var wanted map[string]bool
	if o.Only != nil {
		wanted = map[string]bool{}
		for _, n := range o.Only {
			wanted[n] = true
		}
	}
	var tools []core.Tool
	found := map[string]bool{}
	cursor := ""
	for {
		var params *sdk.ListToolsParams
		if cursor != "" {
			params = &sdk.ListToolsParams{Cursor: cursor}
		}
		listing, err := session.ListTools(ctx, params)
		if err != nil {
			return nil, err
		}
		if listing == nil {
			break
		}
		for _, entry := range listing.Tools {
			if entry == nil || (wanted != nil && !wanted[entry.Name]) {
				continue
			}
			found[entry.Name] = true
			tools = append(tools, NewTool(session, entry.Name, entry.Description, schemaMap(entry.InputSchema), o.Prefix))
		}
		cursor = listing.NextCursor
		if cursor == "" {
			break
		}
	}
	if wanted != nil {
		var missing []string
		for n := range wanted {
			if !found[n] {
				missing = append(missing, n)
			}
		}
		if len(missing) > 0 {
			sort.Strings(missing)
			slog.Warn(fmt.Sprintf("MCP server did not offer requested tools: %s", pyList(missing)))
		}
	}
	return tools, nil
}

// schemaMap normalises a tool's input schema to a JSON map (nil = none).
func schemaMap(schema any) map[string]any {
	switch s := schema.(type) {
	case nil:
		return nil
	case map[string]any:
		return s
	}
	raw, err := json.Marshal(schema)
	if err != nil {
		return nil
	}
	var m map[string]any
	if json.Unmarshal(raw, &m) != nil {
		return nil
	}
	return m
}

// convertResult flattens a CallToolResult into plain JSON values (pyllym
// _convert_result).
func convertResult(result *sdk.CallToolResult) any {
	if result == nil {
		return nil
	}
	var parts []any
	for _, block := range result.Content {
		switch b := block.(type) {
		case *sdk.TextContent:
			parts = append(parts, b.Text)
		case *sdk.ImageContent:
			parts = append(parts, map[string]any{
				"type": "image", "mime_type": b.MIMEType, "data": base64.StdEncoding.EncodeToString(b.Data),
			})
		case *sdk.EmbeddedResource:
			uri, text := "", any(nil)
			if b.Resource != nil {
				uri = b.Resource.URI
				if b.Resource.Text != "" {
					text = b.Resource.Text
				}
			}
			parts = append(parts, map[string]any{"type": "resource", "uri": uri, "text": text})
		default:
			parts = append(parts, blockString(block))
		}
	}
	var texts []string
	allText := true
	for _, p := range parts {
		if s, ok := p.(string); ok {
			texts = append(texts, s)
		} else {
			allText = false
		}
	}
	var payload any
	switch {
	case result.StructuredContent != nil:
		payload = result.StructuredContent
	case len(parts) == 1:
		payload = parts[0]
	case allText:
		payload = strings.Join(texts, "\n")
	default:
		payload = parts
	}
	if result.IsError {
		if isEmpty(payload) {
			payload = "MCP tool call failed"
		}
		return map[string]any{"error": payload}
	}
	return payload
}

// blockString stands in for Python's str(block) on other content blocks.
func blockString(block sdk.Content) string {
	raw, err := json.Marshal(block)
	if err != nil {
		return fmt.Sprint(block)
	}
	return string(raw)
}

// isEmpty is pyllym's `payload in (None, "", [])`.
func isEmpty(v any) bool {
	switch x := v.(type) {
	case nil:
		return true
	case string:
		return x == ""
	case []any:
		return len(x) == 0
	}
	return false
}

func pyList(items []string) string {
	q := make([]string, len(items))
	for i, s := range items {
		q[i] = "'" + s + "'"
	}
	return "[" + strings.Join(q, ", ") + "]"
}

// --- Server ------------------------------------------------------------------

// StdioOptions configure a stdio server.
type StdioOptions struct {
	// Env is merged over the minimal inherited environment.
	Env map[string]string
	// Dir is the working directory ("" = the current one).
	Dir    string
	Prefix string
	Only   []string
}

// HTTPOptions configure a streamable-HTTP server.
type HTTPOptions struct {
	// Headers are sent with every request (e.g. Authorization).
	Headers map[string]string
	// HTTPClient replaces the default client; Headers are added on top.
	HTTPClient *http.Client
	Prefix     string
	Only       []string
}

// Server is a connection to one MCP server. Build it with Stdio or HTTP,
// Connect, read Tools, and Close when done.
type Server struct {
	// Tools is the snapshot of the server's tools (set by Connect/Refresh).
	Tools []core.Tool

	transport  string
	command    string
	args       []string
	env        map[string]string
	dir        string
	url        string
	headers    map[string]string
	httpClient *http.Client
	prefix     string
	only       []string

	session *sdk.ClientSession
}

// Stdio describes a server spawned as a subprocess speaking MCP over stdio.
func Stdio(command string, args []string, o StdioOptions) *Server {
	return &Server{transport: "stdio", command: command, args: args, env: o.Env, dir: o.Dir, prefix: o.Prefix, only: o.Only}
}

// HTTP describes a remote server speaking MCP over streamable HTTP.
func HTTP(url string, o HTTPOptions) *Server {
	return &Server{transport: "http", url: url, headers: o.Headers, httpClient: o.HTTPClient, prefix: o.Prefix, only: o.Only}
}

// Connect opens the transport, initializes the session and lists the tools.
// On failure everything opened so far is closed.
func (s *Server) Connect(ctx context.Context) error {
	if s.session != nil {
		return &core.ConfigurationError{Message: "MCP server is already connected"}
	}
	var transport sdk.Transport
	if s.transport == "stdio" {
		cmd := exec.Command(s.command, s.args...)
		cmd.Env = stdioEnv(s.env)
		cmd.Dir = s.dir
		cmd.Stderr = os.Stderr
		transport = &sdk.CommandTransport{Command: cmd}
	} else {
		transport = &sdk.StreamableClientTransport{Endpoint: s.url, HTTPClient: s.client()}
	}
	// The name/version gollym reports to servers (it tracks pyllym 1.16.0).
	client := sdk.NewClient(&sdk.Implementation{Name: "gollym", Version: "1.16.0"}, nil)
	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		return err
	}
	s.session = session
	if _, err := s.Refresh(ctx); err != nil {
		_ = s.Close()
		return err
	}
	return nil
}

// Close ends the session (and stops a stdio server).
func (s *Server) Close() error {
	session := s.session
	s.session, s.Tools = nil, nil
	if session == nil {
		return nil
	}
	return session.Close()
}

// Session returns the live session (nil before Connect).
func (s *Server) Session() Session {
	if s.session == nil {
		return nil
	}
	return s.session
}

// Refresh re-lists the server's tools (e.g. after a tools/list_changed).
func (s *Server) Refresh(ctx context.Context) ([]core.Tool, error) {
	if s.session == nil {
		return nil, &core.ConfigurationError{Message: "MCP server is not connected; call Connect first"}
	}
	tools, err := ToolsFromSession(ctx, s.session, ToolsOptions{Prefix: s.prefix, Only: s.only})
	if err != nil {
		return nil, err
	}
	s.Tools = tools
	return tools, nil
}

// client returns the HTTP client for the transport: nil (the SDK default)
// without headers or a custom client.
func (s *Server) client() *http.Client {
	if len(s.headers) == 0 {
		return s.httpClient
	}
	base := http.DefaultClient
	if s.httpClient != nil {
		base = s.httpClient
	}
	c := *base
	rt := c.Transport
	if rt == nil {
		rt = http.DefaultTransport
	}
	c.Transport = &headerTransport{base: rt, headers: s.headers}
	return &c
}

type headerTransport struct {
	base    http.RoundTripper
	headers map[string]string
}

func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}
	return t.base.RoundTrip(req)
}

// inheritedEnv is the Python SDK's DEFAULT_INHERITED_ENV_VARS.
func inheritedEnv() []string {
	if runtime.GOOS == "windows" {
		return []string{"APPDATA", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "PATH", "PATHEXT",
			"PROCESSOR_ARCHITECTURE", "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "USERNAME", "USERPROFILE"}
	}
	return []string{"HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"}
}

// stdioEnv is the safe default environment with extra merged over it.
func stdioEnv(extra map[string]string) []string {
	env := map[string]string{}
	for _, k := range inheritedEnv() {
		if v, ok := os.LookupEnv(k); ok && !strings.HasPrefix(v, "()") {
			env[k] = v
		}
	}
	for k, v := range extra {
		env[k] = v
	}
	keys := make([]string, 0, len(env))
	for k := range env {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]string, len(keys))
	for i, k := range keys {
		out[i] = k + "=" + env[k]
	}
	return out
}
