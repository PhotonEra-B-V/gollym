package converse_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"

	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
	"github.com/PhotonEra-B-V/gollym/protocols/converse"
	_ "github.com/PhotonEra-B-V/gollym/providers"
)

const (
	sonnet4  = "us.anthropic.claude-sonnet-4-20250514-v1:0"   // reasoningSupported.embedded = true
	sonnet45 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0" // embedded = false
)

// newChat is pyllym's conftest: dummy Bedrock credentials, pointed at base.
func newChat(t *testing.T, base, model string, cfgFns ...func(*core.Config)) *core.Chat {
	t.Helper()
	ctx := core.NewContext(func(c *core.Config) {
		c.Set("bedrock_api_key", "AKIA-test")
		c.Set("bedrock_secret_key", "secret-test")
		c.Set("bedrock_region", "us-east-1")
		c.Set("bedrock_session_token", "")
		c.Set("bedrock_api_base", base)
		c.MaxRetries = 2
		c.RetryInterval = time.Millisecond
		c.RateLimitRetryInterval = time.Millisecond
		c.RequestTimeout = 5 * time.Second
		for _, f := range cfgFns {
			f(c)
		}
	})
	chat, err := ctx.NewChat(core.ChatOptions{Model: model, Provider: "bedrock"})
	if err != nil {
		t.Fatal(err)
	}
	return chat
}

// converseReply is tests/factories.py bedrock_converse.
func converseReply(text string, toolUse map[string]any, in, out int) map[string]any {
	content := []any{map[string]any{"text": text}}
	stop := "end_turn"
	if toolUse != nil {
		content = append(content, map[string]any{"toolUse": toolUse})
		stop = "tool_use"
	}
	return map[string]any{
		"output":     map[string]any{"message": map[string]any{"role": "assistant", "content": content}},
		"stopReason": stop,
		"usage":      map[string]any{"inputTokens": in, "outputTokens": out, "totalTokens": in + out},
	}
}

func firstBedrockModel(t *testing.T) string {
	t.Helper()
	all := core.Models().ByProvider("bedrock").All()
	if len(all) == 0 {
		t.Fatal("expected bedrock models in the registry")
	}
	return all[0].ID
}

func weatherTool(calls *[]map[string]any) *core.FuncTool {
	return core.NewTool("weather", "Get the weather",
		[]core.Parameter{{Name: "city", Type: "string", Description: "City", Required: true}},
		func(_ context.Context, args map[string]any) (any, error) {
			if calls != nil {
				*calls = append(*calls, args)
			}
			return "sunny in " + args["city"].(string), nil
		})
}

func normalize(t *testing.T, v any) any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func assertJSONEqual(t *testing.T, got, want any) {
	t.Helper()
	g, w := normalize(t, got), normalize(t, want)
	if !reflect.DeepEqual(g, w) {
		gb, _ := json.MarshalIndent(g, "", " ")
		wb, _ := json.MarshalIndent(w, "", " ")
		t.Fatalf("payload mismatch\n got: %s\nwant: %s", gb, wb)
	}
}

// --- tests/test_bedrock.py ----------------------------------------------------

func TestBedrockConverseRoundtrip(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: converseReply("Bedrock says hi", nil, 9, 7)})
	model := firstBedrockModel(t)
	msg, err := newChat(t, srv.URL, model).Ask(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	req := srv.Last(t)
	if req.Path != "/model/"+model+"/converse" {
		t.Fatalf("path %s", req.Path)
	}
	if msg.Text() != "Bedrock says hi" || *msg.InputTokens() != 9 || *msg.OutputTokens() != 7 || msg.FinishReason != "end_turn" {
		t.Fatalf("message %+v", msg)
	}
	if auth := req.Headers.Get("Authorization"); !strings.HasPrefix(auth, "AWS4-HMAC-SHA256") {
		t.Fatalf("request was not SigV4-signed: %q", auth)
	}
}

func TestBedrockRenderPayloadShape(t *testing.T) {
	chat := newChat(t, "", firstBedrockModel(t))
	if err := chat.AddUserMessage("hello"); err != nil {
		t.Fatal(err)
	}
	payload, err := chat.Render()
	if err != nil {
		t.Fatal(err)
	}
	first := payload["messages"].([]any)[0].(map[string]any)
	if first["role"] != "user" {
		t.Fatalf("role %v", first["role"])
	}
	if _, ok := first["content"].([]any); !ok {
		t.Fatalf("content %T", first["content"])
	}
}

// --- signing -----------------------------------------------------------------

// The recorded request carries a valid SigV4 signature over exactly the body
// that was sent: re-signing it at its X-Amz-Date reproduces Authorization.
func TestRequestsAreSigV4SignedOverTheSentBody(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: converseReply("ok", nil, 1, 1)})
	if _, err := newChat(t, srv.URL, sonnet45).Ask(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	rec := srv.Last(t)
	sum := sha256.Sum256(rec.Body)
	hash := hex.EncodeToString(sum[:])
	if rec.Headers.Get("X-Amz-Content-Sha256") != hash {
		t.Fatalf("content hash %q, body hash %q", rec.Headers.Get("X-Amz-Content-Sha256"), hash)
	}
	if rec.Headers.Get("X-Amz-Security-Token") != "" {
		t.Fatal("unexpected session token header")
	}
	auth := rec.Headers.Get("Authorization")
	date := rec.Headers.Get("X-Amz-Date")
	when, err := time.Parse("20060102T150405Z", date)
	if err != nil {
		t.Fatalf("X-Amz-Date %q: %v", date, err)
	}
	if !strings.Contains(auth, "Credential=AKIA-test/"+date[:8]+"/us-east-1/bedrock/aws4_request") {
		t.Fatalf("Authorization %q", auth)
	}
	for _, h := range []string{"host", "x-amz-content-sha256", "x-amz-date"} {
		if !strings.Contains(auth, h) {
			t.Fatalf("%s not signed: %q", h, auth)
		}
	}
	again, _ := http.NewRequest(http.MethodPost, srv.URL+rec.Path, bytes.NewReader(rec.Body))
	again.Header.Set("Content-Type", rec.Headers.Get("Content-Type"))
	again.Header.Set("X-Amz-Content-Sha256", hash)
	creds := aws.Credentials{AccessKeyID: "AKIA-test", SecretAccessKey: "secret-test"}
	if err := v4.NewSigner().SignHTTP(context.Background(), creds, again, hash, "bedrock", "us-east-1", when); err != nil {
		t.Fatal(err)
	}
	if again.Header.Get("Authorization") != auth {
		t.Fatalf("signature does not verify\n got %s\nwant %s", auth, again.Header.Get("Authorization"))
	}
}

func TestSessionTokenIsSentAndSigned(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: converseReply("ok", nil, 1, 1)})
	chat := newChat(t, srv.URL, sonnet45, func(c *core.Config) { c.Set("bedrock_session_token", "TOKEN-123") })
	if _, err := chat.Ask(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	rec := srv.Last(t)
	if rec.Headers.Get("X-Amz-Security-Token") != "TOKEN-123" || !strings.Contains(rec.Headers.Get("Authorization"), "x-amz-security-token") {
		t.Fatalf("headers %v", rec.Headers)
	}
}

func TestProviderConfiguration(t *testing.T) {
	spec, ok := core.LookupProvider("bedrock")
	if !ok {
		t.Fatal("bedrock not registered")
	}
	wantOpts := []string{"bedrock_api_key", "bedrock_secret_key", "bedrock_region", "bedrock_session_token", "bedrock_api_base"}
	if !reflect.DeepEqual(spec.Options, wantOpts) || !reflect.DeepEqual(spec.Requirements, wantOpts[:3]) {
		t.Fatalf("options %v requirements %v", spec.Options, spec.Requirements)
	}
	cfg := core.NewConfig()
	for _, k := range wantOpts {
		t.Setenv(strings.ToUpper(k), "")
	}
	if _, err := core.NewProvider(spec, cfg); !errors.Is(err, core.ErrConfiguration) {
		t.Fatalf("err %v", err)
	}
	cfg.Set("bedrock_api_key", "k").Set("bedrock_secret_key", "s").Set("bedrock_region", "eu-central-1")
	p, err := core.NewProvider(spec, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if base, _ := p.APIBase(); base != "https://bedrock-runtime.eu-central-1.amazonaws.com" {
		t.Fatalf("api base %q", base)
	}
	if p.Name() != "Bedrock" || len(p.Headers()) != 0 {
		t.Fatalf("name %q headers %v", p.Name(), p.Headers())
	}
	if name, _ := p.ProtocolName("", nil); name != "converse" {
		t.Fatalf("protocol %q", name)
	}
}

// --- payload parity with pyllym ----------------------------------------------

func loadGolden(t *testing.T) map[string]any {
	t.Helper()
	b, err := os.ReadFile("testdata/pyllym_payloads.json")
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func render(t *testing.T, chat *core.Chat) map[string]any {
	t.Helper()
	p, err := chat.Render()
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func addMessage(t *testing.T, chat *core.Chat, m *core.Message) {
	t.Helper()
	if _, err := chat.AddMessage(m); err != nil {
		t.Fatal(err)
	}
}

// A conversation exercising system, attachments (image, two same-named
// PDFs, inline text files), thinking + tool calls, tool results (incl. empty
// output), redacted thinking and raw content renders exactly as pyllym does.
func TestRenderFullConversationMatchesPyllym(t *testing.T) {
	golden := loadGolden(t)
	chat := newChat(t, "", sonnet4)
	chat.WithInstructions("Be brief.").WithTemperature(0.3).WithTool(weatherTool(nil))
	if err := chat.WithToolChoice("weather"); err != nil {
		t.Fatal(err)
	}
	f := func(name string) string { return filepath.Join("testdata", "files", name) }
	if err := chat.AddUserMessage("What's the weather?", f("cat.png"), f("report.pdf"), f("b/report.pdf"), f("data.csv"), f("notes.txt")); err != nil {
		t.Fatal(err)
	}
	calls := &core.ToolCalls{}
	calls.Set("t1", &core.ToolCall{ID: "t1", Name: "weather", Arguments: map[string]any{"city": "Paris"}})
	calls.Set("t2", &core.ToolCall{ID: "t2", Name: "weather", Arguments: map[string]any{"city": "Oslo"}})
	addMessage(t, chat, &core.Message{Role: core.RoleAssistant, Content: "", ToolCalls: calls, Thinking: &core.Thinking{Text: "let me think", Signature: "sig1"}})
	addMessage(t, chat, &core.Message{Role: core.RoleTool, Content: "Sunny", ToolCallID: "t1"})
	addMessage(t, chat, &core.Message{Role: core.RoleTool, Content: "", ToolCallID: "t2"})
	addMessage(t, chat, &core.Message{Role: core.RoleAssistant, Content: "Paris is sunny; Oslo unknown.", Thinking: &core.Thinking{Signature: "redacted-blob"}})
	addMessage(t, chat, &core.Message{Role: core.RoleUser, Content: core.RawContent{Value: []any{
		map[string]any{"text": "raw user"}, map[string]any{"reasoningContent": map[string]any{"x": 1}}, "junk",
	}}})
	addMessage(t, chat, &core.Message{Role: core.RoleAssistant, Content: core.RawContent{Value: map[string]any{"text": "raw assistant"}}})
	assertJSONEqual(t, render(t, chat), golden["full"])
}

func TestRenderOptionsMatchPyllym(t *testing.T) {
	golden := loadGolden(t)
	cases := map[string]func(*core.Chat) error{
		"effort_embedded": func(c *core.Chat) error { _, err := c.WithThinking("high", nil); return err },
		"effort_plain":    func(c *core.Chat) error { _, err := c.WithThinking("low", nil); return err },
		"budget":          func(c *core.Chat) error { _, err := c.WithThinking("", core.Int(2048)); return err },
		"effort_none_budget": func(c *core.Chat) error {
			_, err := c.WithThinking("none", core.Int(1024))
			return err
		},
		"choice_auto":     func(c *core.Chat) error { c.WithTool(weatherTool(nil)); return c.WithToolChoice("auto") },
		"choice_none":     func(c *core.Chat) error { c.WithTool(weatherTool(nil)); return c.WithToolChoice("none") },
		"choice_required": func(c *core.Chat) error { c.WithTool(weatherTool(nil)); return c.WithToolChoice("required") },
	}
	for name, setup := range cases {
		t.Run(name, func(t *testing.T) {
			model := sonnet45
			if name == "effort_embedded" {
				model = sonnet4
			}
			chat := newChat(t, "", model)
			if err := setup(chat); err != nil {
				t.Fatal(err)
			}
			if err := chat.AddUserMessage("hi"); err != nil {
				t.Fatal(err)
			}
			assertJSONEqual(t, render(t, chat), golden[name])
		})
	}
}

// The schema is sent as a JSON string; Go orders keys differently, so the
// strings are compared decoded.
func TestRenderSchemaMatchesPyllym(t *testing.T) {
	golden := loadGolden(t)["schema"].(map[string]any)
	chat := newChat(t, "", sonnet45)
	if _, err := chat.WithSchema(map[string]any{
		"name": "Person card!", "strict": true,
		"schema": map[string]any{"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string"}}, "required": []any{"name"}, "additionalProperties": false},
	}); err != nil {
		t.Fatal(err)
	}
	if err := chat.AddUserMessage("hi"); err != nil {
		t.Fatal(err)
	}
	got := normalize(t, render(t, chat)).(map[string]any)
	decode := func(p map[string]any) {
		js := core.DigMap(p, "outputConfig", "textFormat", "structure", "jsonSchema")
		var v any
		if err := json.Unmarshal([]byte(js["schema"].(string)), &v); err != nil {
			t.Fatal(err)
		}
		js["schema"] = v
	}
	decode(got)
	decode(golden)
	assertJSONEqual(t, got, golden)
}

func TestParamsDeepMergeOverPayload(t *testing.T) {
	chat := newChat(t, "", sonnet45)
	chat.WithTemperature(0.5).WithParams(map[string]any{"inferenceConfig": map[string]any{"maxTokens": 100}})
	if err := chat.AddUserMessage("hi"); err != nil {
		t.Fatal(err)
	}
	if got := render(t, chat)["inferenceConfig"]; !reflect.DeepEqual(got, map[string]any{"temperature": 0.5, "maxTokens": 100}) {
		t.Fatalf("inferenceConfig %v", got)
	}
}

func TestUnsupportedAttachments(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"song.mp3", "slides.pptx", "clip.mp4"} {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte("data"), 0o600); err != nil {
			t.Fatal(err)
		}
		chat := newChat(t, "", sonnet45)
		if err := chat.AddUserMessage("look", path); err != nil {
			t.Fatal(err)
		}
		if _, err := chat.Render(); !errors.Is(err, core.ErrUnsupportedContent) {
			t.Fatalf("%s: err %v", name, err)
		}
	}
}

func TestProviderFileIsAnS3Document(t *testing.T) {
	chat := newChat(t, "", sonnet45)
	files := []any{
		&core.UploadedFile{ID: "file-1", URI: "s3://bucket/q3 report.pdf", Filename: "q3 report.pdf", MIMEType: "application/pdf"},
		&core.UploadedFile{ID: "s3://bucket/other.docx", Filename: "q3 report.docx"},
	}
	if err := chat.AddUserMessage("summarise", files...); err != nil {
		t.Fatal(err)
	}
	content := render(t, chat)["messages"].([]any)[0].(map[string]any)["content"]
	assertJSONEqual(t, content, []any{
		map[string]any{"text": "summarise"},
		map[string]any{"document": map[string]any{"format": "pdf", "name": "q3_report", "source": map[string]any{"s3Location": map[string]any{"uri": "s3://bucket/q3 report.pdf"}}}},
		map[string]any{"document": map[string]any{"format": "docx", "name": "q3_report_2", "source": map[string]any{"s3Location": map[string]any{"uri": "s3://bucket/other.docx"}}}},
	})
}

func TestToolResultContentShapes(t *testing.T) {
	chat := newChat(t, "", sonnet45)
	if err := chat.AddUserMessage("go"); err != nil {
		t.Fatal(err)
	}
	calls := &core.ToolCalls{}
	calls.Set("a", &core.ToolCall{ID: "a", Name: "weather"})
	addMessage(t, chat, &core.Message{Role: core.RoleAssistant, ToolCalls: calls})
	addMessage(t, chat, &core.Message{Role: core.RoleTool, ToolCallID: "a", Content: map[string]any{"temp": 21}})
	addMessage(t, chat, &core.Message{Role: core.RoleTool, ToolCallID: "b", Content: core.RawContent{Value: []any{map[string]any{"json": map[string]any{"k": 1}}, map[string]any{"other": 1}}}})
	addMessage(t, chat, &core.Message{Role: core.RoleTool, ToolCallID: "c", Content: core.RawContent{Value: "plain raw"}})
	attach, err := core.NewContent(core.Str(""), []byte("hello"))
	if err != nil {
		t.Fatal(err)
	}
	addMessage(t, chat, &core.Message{Role: core.RoleTool, ToolCallID: "d", Content: attach})
	msgs := render(t, chat)["messages"].([]any)
	assertJSONEqual(t, msgs[1], map[string]any{"role": "assistant", "content": []any{
		map[string]any{"toolUse": map[string]any{"toolUseId": "a", "name": "weather", "input": map[string]any{}}},
	}})
	results := msgs[2].(map[string]any)["content"].([]any)
	content := func(i int) any { return results[i].(map[string]any)["toolResult"].(map[string]any)["content"] }
	assertJSONEqual(t, content(0), []any{map[string]any{"json": map[string]any{"temp": 21}}})
	assertJSONEqual(t, content(1), []any{map[string]any{"json": map[string]any{"k": 1}}})
	assertJSONEqual(t, content(2), []any{map[string]any{"text": "plain raw"}})
	if s := content(3).([]any)[0].(map[string]any)["text"].(string); !strings.Contains(s, "hello") {
		t.Fatalf("attachment tool result %q", s)
	}
}

func TestSanitizeDocumentName(t *testing.T) {
	for in, want := range map[string]string{
		"report.pdf":            "report",
		"dir/Q3 results!.xlsx":  "Q3_results_",
		"archive.tar.gz":        "archive_tar",
		".hidden":               "_hidden",
		"":                      "document",
		"dir/":                  "document",
		"résumé-2026_v1.docx":   "résumé-2026_v1",
		"/abs/path/notes.final": "notes",
		"..x":                   "__x",
		"a..":                   "a_",
		"...":                   "___",
	} {
		if got := converse.SanitizeDocumentName(in); got != want {
			t.Errorf("%q: got %q want %q", in, got, want)
		}
	}
}

// --- parsing -----------------------------------------------------------------

func parse(t *testing.T, body map[string]any) *core.Message {
	t.Helper()
	srv := testserver.New(t, testserver.Reply{Body: body})
	chat := newChat(t, srv.URL, sonnet45)
	if err := chat.AddUserMessage("hi"); err != nil {
		t.Fatal(err)
	}
	msg, err := chat.Send(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	return msg
}

func TestParseThinkingToolCallsAndUsage(t *testing.T) {
	msg := parse(t, map[string]any{
		"modelId": "anthropic.claude",
		"output": map[string]any{"message": map[string]any{"role": "assistant", "content": []any{
			map[string]any{"reasoningContent": map[string]any{"reasoningText": map[string]any{"text": "step 1. ", "signature": "sig-a"}}},
			map[string]any{"reasoningContent": map[string]any{"redactedContent": "blob"}},
			map[string]any{"reasoningContent": map[string]any{"reasoningText": map[string]any{"text": "step 2."}}},
			map[string]any{"text": "Hello "},
			map[string]any{"text": "world"},
			map[string]any{"toolUse": map[string]any{"toolUseId": "tu1", "name": "weather", "input": map[string]any{"city": "Paris"}}},
			map[string]any{"toolUse": map[string]any{"toolUseId": "tu2", "name": "noargs"}},
		}}},
		"stopReason": "tool_use",
		"usage": map[string]any{
			"inputTokens": 100, "outputTokens": 20, "cacheReadInputTokens": 30, "cacheWriteInputTokens": 10,
			"outputTokensDetails": map[string]any{"reasoningTokens": 5},
		},
	})
	if msg.Text() != "Hello world" || msg.ModelID != "anthropic.claude" || msg.FinishReason != "tool_use" || !msg.IsToolCallStop() {
		t.Fatalf("message %+v", msg)
	}
	if msg.Thinking == nil || msg.Thinking.Text != "step 1. step 2." || msg.Thinking.Signature != "sig-a" {
		t.Fatalf("thinking %+v", msg.Thinking)
	}
	if *msg.InputTokens() != 60 || *msg.OutputTokens() != 20 || *msg.CachedTokens() != 30 || *msg.CacheCreationTokens() != 10 || *msg.ThinkingTokens() != 5 {
		t.Fatalf("tokens %+v", msg.Tokens.Map())
	}
	if msg.ToolCalls.Len() != 2 || !reflect.DeepEqual(msg.ToolCalls.Get("tu1").Arguments, map[string]any{"city": "Paris"}) ||
		!reflect.DeepEqual(msg.ToolCalls.Get("tu2").Arguments, map[string]any{}) {
		t.Fatalf("tool calls %v", msg.ToolCalls.List())
	}
}

func TestParseRedactedOnlyAndToolOnly(t *testing.T) {
	msg := parse(t, map[string]any{
		"output": map[string]any{"message": map[string]any{"content": []any{
			map[string]any{"reasoningContent": map[string]any{"redactedContent": "blob"}},
			map[string]any{"toolUse": map[string]any{"toolUseId": "x", "name": "weather", "input": map[string]any{}}},
		}}},
		"usage": map[string]any{"inputTokens": 5, "outputTokens": 1, "reasoningTokens": 0},
	})
	if msg.Content != "" || msg.Thinking == nil || msg.Thinking.Text != "" || msg.Thinking.Signature != "blob" {
		t.Fatalf("message %+v thinking %+v", msg, msg.Thinking)
	}
	if msg.ThinkingTokens() != nil {
		t.Fatalf("thinking tokens %v", *msg.ThinkingTokens())
	}
}

func TestEmptyResponseIsAnError(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: map[string]any{}})
	_, err := newChat(t, srv.URL, sonnet45).Ask(context.Background(), "hi")
	if !errors.Is(err, core.ErrAPI) || !strings.Contains(err.Error(), "Empty or unparseable") {
		t.Fatalf("err %v", err)
	}
}

// --- tool loop and errors ----------------------------------------------------

func TestToolLoop(t *testing.T) {
	srv := testserver.New(t,
		testserver.Reply{Body: converseReply("Let me check", map[string]any{"toolUseId": "tu1", "name": "weather", "input": map[string]any{"city": "Paris"}}, 10, 5)},
		testserver.Reply{Body: converseReply("It is sunny in Paris", nil, 20, 6)},
	)
	var calls []map[string]any
	chat := newChat(t, srv.URL, sonnet45)
	chat.WithTool(weatherTool(&calls))
	msg, err := chat.Ask(context.Background(), "Weather in Paris?")
	if err != nil {
		t.Fatal(err)
	}
	if msg.Text() != "It is sunny in Paris" || len(calls) != 1 || calls[0]["city"] != "Paris" {
		t.Fatalf("msg %q calls %v", msg.Text(), calls)
	}
	body := srv.Last(t).JSON(t)
	assertJSONEqual(t, body["messages"], []any{
		map[string]any{"role": "user", "content": []any{map[string]any{"text": "Weather in Paris?"}}},
		map[string]any{"role": "assistant", "content": []any{
			map[string]any{"text": "Let me check"},
			map[string]any{"toolUse": map[string]any{"toolUseId": "tu1", "name": "weather", "input": map[string]any{"city": "Paris"}}},
		}},
		map[string]any{"role": "user", "content": []any{
			map[string]any{"toolResult": map[string]any{"toolUseId": "tu1", "content": []any{map[string]any{"text": "sunny in Paris"}}}},
		}},
	})
	if body["toolConfig"] == nil {
		t.Fatal("toolConfig missing")
	}
}

func TestHTTPErrors(t *testing.T) {
	cases := []struct {
		status int
		body   any
		target error
		msg    string
		tries  int
	}{
		{400, map[string]any{"message": "The provided model identifier is invalid."}, core.ErrBadRequest, "The provided model identifier is invalid.", 1},
		{400, map[string]any{"message": "Input is too long for requested model."}, core.ErrContextLengthExceeded, "Input is too long for requested model.", 1},
		{403, map[string]any{"Message": "User is not authorized"}, core.ErrForbidden, "User is not authorized", 1},
		{429, map[string]any{"message": "Too many requests, please wait before trying again."}, core.ErrRateLimit, "Too many requests, please wait before trying again.", 3},
		{503, map[string]any{"__type": "ServiceUnavailableException"}, core.ErrServiceUnavailable, "ServiceUnavailableException", 3},
	}
	for _, tc := range cases {
		srv := testserver.New(t, testserver.Reply{Status: tc.status, Body: tc.body})
		_, err := newChat(t, srv.URL, sonnet45).Ask(context.Background(), "hi")
		if !errors.Is(err, tc.target) || err.Error() != tc.msg || len(srv.Requests()) != tc.tries {
			t.Errorf("%d: err %v after %d requests", tc.status, err, len(srv.Requests()))
		}
	}
}

func TestThrottlingIsRetried(t *testing.T) {
	srv := testserver.New(t,
		testserver.Reply{Status: 429, Body: map[string]any{"message": "Too many requests"}},
		testserver.Reply{Body: converseReply("ok", nil, 1, 1)},
	)
	msg, err := newChat(t, srv.URL, sonnet45).Ask(context.Background(), "hi")
	if err != nil || msg.Text() != "ok" {
		t.Fatalf("msg %v err %v", msg, err)
	}
	reqs := srv.Requests()
	if len(reqs) != 2 || reqs[1].Headers.Get("Authorization") == "" {
		t.Fatalf("%d requests", len(reqs))
	}
}
