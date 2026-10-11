package mcp

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/PhotonEra-B-V/gollym"
	"github.com/PhotonEra-B-V/gollym/core"
)

func toolEntry(name, description string) *sdk.Tool {
	return &sdk.Tool{Name: name, Description: description, InputSchema: map[string]any{
		"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string"}}, "required": []any{"path"},
	}}
}

func textResult(text string, isError bool) *sdk.CallToolResult {
	return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: text}}, IsError: isError}
}

type call struct {
	name string
	args any
}

// fakeSession stands in for *sdk.ClientSession.
type fakeSession struct {
	tools   []*sdk.Tool
	results map[string]*sdk.CallToolResult
	pages   func(cursor string) *sdk.ListToolsResult
	cursors []string
	calls   []call
	callErr error
}

func (s *fakeSession) ListTools(_ context.Context, p *sdk.ListToolsParams) (*sdk.ListToolsResult, error) {
	cursor := ""
	if p != nil {
		cursor = p.Cursor
	}
	s.cursors = append(s.cursors, cursor)
	if s.pages != nil {
		return s.pages(cursor), nil
	}
	return &sdk.ListToolsResult{Tools: s.tools}, nil
}

func (s *fakeSession) CallTool(_ context.Context, p *sdk.CallToolParams) (*sdk.CallToolResult, error) {
	s.calls = append(s.calls, call{p.Name, p.Arguments})
	if s.callErr != nil {
		return nil, s.callErr
	}
	if r, ok := s.results[p.Name]; ok {
		return r, nil
	}
	return textResult("ok", false), nil
}

func tools(t *testing.T, s Session, o ToolsOptions) []core.Tool {
	t.Helper()
	out, err := ToolsFromSession(context.Background(), s, o)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func callTool(t *testing.T, tool core.Tool, args map[string]any) any {
	t.Helper()
	out, err := tool.Call(context.Background(), args)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// --- ToolsFromSession --------------------------------------------------------

func TestWrapsListedToolsWithMetadata(t *testing.T) {
	ts := tools(t, &fakeSession{tools: []*sdk.Tool{toolEntry("read_file", "Read a file.")}}, ToolsOptions{})
	if len(ts) != 1 {
		t.Fatalf("tools %v", ts)
	}
	tool, ok := ts[0].(*Tool)
	if !ok || tool.Name() != "read_file" || tool.Description() != "Read a file." {
		t.Fatalf("tool %#v", ts[0])
	}
	if _, ok := tool.ParamsSchema()["properties"].(map[string]any)["path"]; !ok {
		t.Fatalf("schema %v", tool.ParamsSchema())
	}
}

func TestPrefixNamespacesNameButCallsRemoteName(t *testing.T) {
	s := &fakeSession{tools: []*sdk.Tool{toolEntry("search", "")}}
	ts := tools(t, s, ToolsOptions{Prefix: "docs_"})
	if ts[0].Name() != "docs_search" || ts[0].(*Tool).RemoteName() != "search" {
		t.Fatalf("name %q", ts[0].Name())
	}
	callTool(t, ts[0], map[string]any{"path": "x"})
	if !reflect.DeepEqual(s.calls, []call{{"search", map[string]any{"path": "x"}}}) {
		t.Fatalf("calls %v", s.calls)
	}
}

func TestOnlyFiltersTools(t *testing.T) {
	s := &fakeSession{tools: []*sdk.Tool{toolEntry("a", ""), toolEntry("b", ""), toolEntry("c", "")}}
	var names []string
	for _, tool := range tools(t, s, ToolsOptions{Only: []string{"a", "c", "missing"}}) {
		names = append(names, tool.Name())
	}
	sort.Strings(names)
	if !reflect.DeepEqual(names, []string{"a", "c"}) {
		t.Fatalf("names %v", names)
	}
	if got := tools(t, s, ToolsOptions{Only: []string{}}); len(got) != 0 {
		t.Fatalf("an empty Only selects nothing: %v", got)
	}
}

func TestPaginationFollowsCursor(t *testing.T) {
	s := &fakeSession{pages: func(cursor string) *sdk.ListToolsResult {
		if cursor == "" {
			return &sdk.ListToolsResult{Tools: []*sdk.Tool{toolEntry("first", "")}, NextCursor: "page2"}
		}
		return &sdk.ListToolsResult{Tools: []*sdk.Tool{toolEntry("second", "")}}
	}}
	ts := tools(t, s, ToolsOptions{})
	if len(ts) != 2 || ts[0].Name() != "first" || ts[1].Name() != "second" {
		t.Fatalf("tools %v", ts)
	}
	if !reflect.DeepEqual(s.cursors, []string{"", "page2"}) {
		t.Fatalf("cursors %v", s.cursors)
	}
}

func TestMissingDescriptionGetsFallback(t *testing.T) {
	ts := tools(t, &fakeSession{tools: []*sdk.Tool{toolEntry("bare", "")}}, ToolsOptions{})
	if ts[0].Description() != "MCP tool bare." {
		t.Fatalf("description %q", ts[0].Description())
	}
}

func TestListErrorPropagates(t *testing.T) {
	boom := errors.New("closed")
	_, err := ToolsFromSession(context.Background(), errSession{boom}, ToolsOptions{})
	if !errors.Is(err, boom) {
		t.Fatalf("err %v", err)
	}
}

type errSession struct{ err error }

func (s errSession) ListTools(context.Context, *sdk.ListToolsParams) (*sdk.ListToolsResult, error) {
	return nil, s.err
}
func (s errSession) CallTool(context.Context, *sdk.CallToolParams) (*sdk.CallToolResult, error) {
	return nil, s.err
}

// --- Tool.Call ---------------------------------------------------------------

func TestCallForwardsArgumentsAndReturnsText(t *testing.T) {
	s := &fakeSession{tools: []*sdk.Tool{toolEntry("echo", "")}, results: map[string]*sdk.CallToolResult{"echo": textResult("hello", false)}}
	tool := tools(t, s, ToolsOptions{})[0]
	if out := callTool(t, tool, map[string]any{"path": "p"}); out != "hello" {
		t.Fatalf("out %v", out)
	}
	if !reflect.DeepEqual(s.calls, []call{{"echo", map[string]any{"path": "p"}}}) {
		t.Fatalf("calls %v", s.calls)
	}
	callTool(t, tool, nil)
	if !reflect.DeepEqual(s.calls[1].args, map[string]any{}) {
		t.Fatalf("nil args should be sent as {}: %v", s.calls[1].args)
	}
}

func TestErrorResultBecomesErrorDict(t *testing.T) {
	s := &fakeSession{tools: []*sdk.Tool{toolEntry("boom", "")}, results: map[string]*sdk.CallToolResult{"boom": textResult("kaput", true)}}
	out := callTool(t, tools(t, s, ToolsOptions{})[0], map[string]any{"path": "p"})
	if !reflect.DeepEqual(out, map[string]any{"error": "kaput"}) {
		t.Fatalf("out %v", out)
	}
}

func TestStructuredContentWinsOverText(t *testing.T) {
	result := &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: "{}"}}, StructuredContent: map[string]any{"count": 3.0}}
	s := &fakeSession{tools: []*sdk.Tool{toolEntry("stats", "")}, results: map[string]*sdk.CallToolResult{"stats": result}}
	out := callTool(t, tools(t, s, ToolsOptions{})[0], map[string]any{"path": "p"})
	if !reflect.DeepEqual(out, map[string]any{"count": 3.0}) {
		t.Fatalf("out %v", out)
	}
}

func TestTransportErrorIsAGoError(t *testing.T) {
	boom := errors.New("connection lost")
	s := &fakeSession{tools: []*sdk.Tool{toolEntry("x", "")}, callErr: boom}
	if _, err := tools(t, s, ToolsOptions{})[0].Call(context.Background(), nil); !errors.Is(err, boom) {
		t.Fatalf("err %v", err)
	}
}

// --- convertResult -----------------------------------------------------------

func TestMultipleTextBlocksJoin(t *testing.T) {
	r := &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: "one"}, &sdk.TextContent{Text: "two"}}}
	if out := convertResult(r); out != "one\ntwo" {
		t.Fatalf("out %v", out)
	}
}

func TestImageBlockBecomesDict(t *testing.T) {
	// The SDK decodes base64 into bytes; the dict carries base64 as pyllym's.
	r := &sdk.CallToolResult{Content: []sdk.Content{&sdk.ImageContent{MIMEType: "image/png", Data: []byte("hello")}}}
	want := map[string]any{"type": "image", "mime_type": "image/png", "data": "aGVsbG8="}
	if out := convertResult(r); !reflect.DeepEqual(out, want) {
		t.Fatalf("out %v", out)
	}
}

func TestEmptyErrorResultGetsPlaceholderMessage(t *testing.T) {
	out := convertResult(&sdk.CallToolResult{IsError: true})
	if !reflect.DeepEqual(out, map[string]any{"error": "MCP tool call failed"}) {
		t.Fatalf("out %v", out)
	}
}

func TestMixedBlocksStayAList(t *testing.T) {
	r := &sdk.CallToolResult{Content: []sdk.Content{
		&sdk.TextContent{Text: "see"},
		&sdk.EmbeddedResource{Resource: &sdk.ResourceContents{URI: "file:///a.txt", Text: "body"}},
		&sdk.EmbeddedResource{Resource: &sdk.ResourceContents{URI: "file:///b.bin", Blob: []byte{1}}},
	}}
	want := []any{
		"see",
		map[string]any{"type": "resource", "uri": "file:///a.txt", "text": "body"},
		map[string]any{"type": "resource", "uri": "file:///b.bin", "text": nil},
	}
	if out := convertResult(r); !reflect.DeepEqual(out, want) {
		t.Fatalf("out %#v", out)
	}
	// Other blocks are stringified (pyllym: str(block)).
	r = &sdk.CallToolResult{Content: []sdk.Content{&sdk.AudioContent{MIMEType: "audio/wav", Data: []byte("x")}}}
	if out, ok := convertResult(r).(string); !ok || !strings.Contains(out, `"type":"audio"`) {
		t.Fatalf("out %v", convertResult(r))
	}
	// No content at all: an empty string (all of zero parts are text).
	if out := convertResult(&sdk.CallToolResult{}); out != "" {
		t.Fatalf("out %#v", out)
	}
}

// --- Server ------------------------------------------------------------------

func TestRefreshBeforeConnectRaises(t *testing.T) {
	_, err := Stdio("some-server", nil, StdioOptions{}).Refresh(context.Background())
	if !errors.Is(err, core.ErrConfiguration) {
		t.Fatalf("err %v", err)
	}
	if err := Stdio("some-server", nil, StdioOptions{}).Close(); err != nil {
		t.Fatalf("Close before Connect: %v", err)
	}
}

func TestMCPToolsPlugIntoChatRegistry(t *testing.T) {
	ts := tools(t, &fakeSession{tools: []*sdk.Tool{toolEntry("lookup", "")}}, ToolsOptions{})
	ctx := gollym.NewContext(func(c *gollym.Config) { c.Set("openai_api_key", "test") })
	chat, err := ctx.NewChat(gollym.ChatOptions{Model: "gpt-4o", Provider: "openai", AssumeModelExists: true})
	if err != nil {
		t.Fatal(err)
	}
	chat.WithTools(false, ts...)
	if _, ok := chat.Tools.Get("lookup"); !ok {
		t.Fatal("lookup not registered")
	}
}

func TestHTTPClientOnlyWhenNeeded(t *testing.T) {
	if c := HTTP("https://x/mcp", HTTPOptions{}).client(); c != nil {
		t.Fatal("no headers: the SDK builds its own client")
	}
	custom := &http.Client{Timeout: time.Second}
	if c := HTTP("https://x/mcp", HTTPOptions{HTTPClient: custom}).client(); c != custom {
		t.Fatal("a custom client is used as is")
	}
	c := HTTP("https://x/mcp", HTTPOptions{HTTPClient: custom, Headers: map[string]string{"A": "b"}}).client()
	if c == custom || c.Timeout != time.Second {
		t.Fatal("headers wrap a copy of the custom client")
	}
}

func TestStdioEnvIsMinimal(t *testing.T) {
	t.Setenv("GOLLYM_SECRET", "s3cret")
	t.Setenv("PATH", "/bin")
	env := stdioEnv(map[string]string{"EXTRA": "1", "PATH": "/custom"})
	joined := strings.Join(env, "\n")
	if strings.Contains(joined, "GOLLYM_SECRET") || !strings.Contains(joined, "EXTRA=1") || !strings.Contains(joined, "PATH=/custom") {
		t.Fatalf("env %v", env)
	}
}

// testServer is a small MCP server: echo (text), stats (structured), fail
// (isError) and env (reads an environment variable).
func testServer() *sdk.Server {
	s := sdk.NewServer(&sdk.Implementation{Name: "test", Version: "v0"}, nil)
	type echoIn struct {
		Text string `json:"text"`
	}
	sdk.AddTool(s, &sdk.Tool{Name: "echo", Description: "Echo text."}, func(_ context.Context, _ *sdk.CallToolRequest, in echoIn) (*sdk.CallToolResult, any, error) {
		return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: in.Text}}}, nil, nil
	})
	type statsOut struct {
		Count int `json:"count"`
	}
	sdk.AddTool(s, &sdk.Tool{Name: "stats"}, func(_ context.Context, _ *sdk.CallToolRequest, _ struct{}) (*sdk.CallToolResult, statsOut, error) {
		return nil, statsOut{Count: 3}, nil
	})
	sdk.AddTool(s, &sdk.Tool{Name: "fail"}, func(_ context.Context, _ *sdk.CallToolRequest, _ struct{}) (*sdk.CallToolResult, any, error) {
		return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: "kaput"}}, IsError: true}, nil, nil
	})
	type envIn struct {
		Name string `json:"name"`
	}
	sdk.AddTool(s, &sdk.Tool{Name: "env"}, func(_ context.Context, _ *sdk.CallToolRequest, in envIn) (*sdk.CallToolResult, any, error) {
		return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: os.Getenv(in.Name) + "|"}}}, nil, nil
	})
	return s
}

// TestMain doubles as the stdio test server when re-executed.
func TestMain(m *testing.M) {
	if os.Getenv("GOLLYM_MCP_TEST_SERVER") == "1" {
		if err := testServer().Run(context.Background(), &sdk.StdioTransport{}); err != nil {
			os.Exit(1)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func byName(t *testing.T, server *Server) map[string]core.Tool {
	t.Helper()
	out := map[string]core.Tool{}
	for _, tool := range server.Tools {
		out[tool.Name()] = tool
	}
	return out
}

func checkServer(t *testing.T, server *Server) {
	t.Helper()
	tools := byName(t, server)
	echo, stats, fail := tools["t_echo"], tools["t_stats"], tools["t_fail"]
	if echo == nil || stats == nil || fail == nil {
		t.Fatalf("tools %v", tools)
	}
	if echo.Description() != "Echo text." || echo.ParamsSchema()["type"] != "object" {
		t.Fatalf("echo %q %v", echo.Description(), echo.ParamsSchema())
	}
	if out := callTool(t, echo, map[string]any{"text": "hi"}); out != "hi" {
		t.Fatalf("echo %v", out)
	}
	if out := callTool(t, stats, nil); !reflect.DeepEqual(out, map[string]any{"count": 3.0}) {
		t.Fatalf("stats %#v", out)
	}
	if out := callTool(t, fail, nil); !reflect.DeepEqual(out, map[string]any{"error": "kaput"}) {
		t.Fatalf("fail %#v", out)
	}
}

func TestHTTPServerEndToEnd(t *testing.T) {
	var mu sync.Mutex
	var auth []string
	handler := sdk.NewStreamableHTTPHandler(func(*http.Request) *sdk.Server { return testServer() }, nil)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		auth = append(auth, r.Header.Get("Authorization"))
		mu.Unlock()
		handler.ServeHTTP(w, r)
	}))
	defer srv.Close()

	server := HTTP(srv.URL, HTTPOptions{Headers: map[string]string{"Authorization": "Bearer t"}, Prefix: "t_", Only: []string{"echo", "stats", "fail"}})
	ctx := context.Background()
	if err := server.Connect(ctx); err != nil {
		t.Fatal(err)
	}
	defer server.Close() // before srv.Close, which waits for the SSE stream
	if len(server.Tools) != 3 {
		t.Fatalf("Only should select 3 tools: %d", len(server.Tools))
	}
	checkServer(t, server)
	if _, err := server.Refresh(ctx); err != nil || len(server.Tools) != 3 {
		t.Fatalf("refresh %v %d", err, len(server.Tools))
	}
	if err := server.Connect(ctx); !errors.Is(err, core.ErrConfiguration) {
		t.Fatalf("second Connect: %v", err)
	}
	if err := server.Close(); err != nil {
		t.Fatal(err)
	}
	if server.Tools != nil || server.Session() != nil {
		t.Fatal("Close clears the tools and session")
	}
	mu.Lock()
	defer mu.Unlock()
	if len(auth) == 0 {
		t.Fatal("no requests")
	}
	for _, a := range auth {
		if a != "Bearer t" {
			t.Fatalf("headers %v", auth)
		}
	}
}

func TestStdioServerEndToEnd(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Skip(err)
	}
	t.Setenv("GOLLYM_SECRET", "s3cret")
	server := Stdio(exe, nil, StdioOptions{Env: map[string]string{"GOLLYM_MCP_TEST_SERVER": "1"}, Prefix: "t_"})
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := server.Connect(ctx); err != nil {
		t.Fatal(err)
	}
	defer server.Close()
	checkServer(t, server)
	envTool := byName(t, server)["t_env"]
	// The child sees only the minimal environment plus Env.
	if out := callTool(t, envTool, map[string]any{"name": "GOLLYM_SECRET"}); out != "|" {
		t.Fatalf("secret leaked: %v", out)
	}
	if out := callTool(t, envTool, map[string]any{"name": "GOLLYM_MCP_TEST_SERVER"}); out != "1|" {
		t.Fatalf("env %v", out)
	}
}

func TestConnectFailureLeavesServerClosed(t *testing.T) {
	server := HTTP("http://127.0.0.1:1/mcp", HTTPOptions{})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := server.Connect(ctx); err == nil {
		t.Fatal("connect to a closed port should fail")
	}
	if server.Session() != nil || server.Tools != nil {
		t.Fatal("failed Connect left state behind")
	}
}
