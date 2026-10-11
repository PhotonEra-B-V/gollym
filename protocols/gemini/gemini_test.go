package gemini_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym"
	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
	"github.com/PhotonEra-B-V/gollym/protocols/gemini"
	"github.com/PhotonEra-B-V/gollym/providers"
)

// --- helpers (pyllym tests/factories.py) ---------------------------------------

func geminiResponse(text *string, functionCall map[string]any, prompt, candidates int) map[string]any {
	parts := []any{}
	if text != nil {
		parts = append(parts, map[string]any{"text": *text})
	}
	if functionCall != nil {
		parts = append(parts, map[string]any{"functionCall": functionCall})
	}
	return map[string]any{
		"modelVersion":  "gemini-2.5-flash",
		"candidates":    []any{map[string]any{"content": map[string]any{"role": "model", "parts": parts}, "finishReason": "STOP"}},
		"usageMetadata": map[string]any{"promptTokenCount": prompt, "candidatesTokenCount": candidates},
	}
}

func textResponse(text string) map[string]any { return geminiResponse(&text, nil, 8, 4) }

func geminiSSE(chunks ...string) string {
	var sb strings.Builder
	for _, c := range chunks {
		b, _ := json.Marshal(map[string]any{"candidates": []any{map[string]any{"content": map[string]any{"parts": []any{map[string]any{"text": c}}}}}, "modelVersion": "gemini-2.5-flash"})
		sb.WriteString("data: " + string(b) + "\n\n")
	}
	b, _ := json.Marshal(map[string]any{
		"candidates":    []any{map[string]any{"content": map[string]any{"parts": []any{}}, "finishReason": "STOP"}},
		"usageMetadata": map[string]any{"promptTokenCount": 1, "candidatesTokenCount": len(chunks)},
		"modelVersion":  "gemini-2.5-flash",
	})
	sb.WriteString("data: " + string(b) + "\n\n")
	return sb.String()
}

func geminiContext(base string) *gollym.Context {
	return gollym.NewContext(func(c *gollym.Config) {
		c.Set("gemini_api_key", "gm-test")
		c.Set("gemini_api_base", base)
		c.MaxRetries = 0
		c.RetryInterval = time.Millisecond
		c.RequestTimeout = 5 * time.Second
	})
}

func geminiChat(t *testing.T, base, model string) *gollym.Chat {
	t.Helper()
	chat, err := geminiContext(base).NewChat(gollym.ChatOptions{Model: model, Provider: "gemini"})
	if err != nil {
		t.Fatal(err)
	}
	return chat
}

func render(t *testing.T, chat *gollym.Chat) map[string]any {
	t.Helper()
	payload, err := chat.Render()
	if err != nil {
		t.Fatal(err)
	}
	return normalize(t, payload)
}

// normalize round-trips v through JSON so it compares like a request body.
func normalize(t *testing.T, v any) map[string]any {
	t.Helper()
	b, err := core.MarshalPayload(v)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func jsonEqual(t *testing.T, what string, got, want any) {
	t.Helper()
	g, _ := json.Marshal(got)
	w, _ := json.Marshal(want)
	var gv, wv any
	_ = json.Unmarshal(g, &gv)
	_ = json.Unmarshal(w, &wv)
	if !reflect.DeepEqual(gv, wv) {
		t.Fatalf("%s:\n got %s\nwant %s", what, g, w)
	}
}

func weatherTool(got *map[string]any) *gollym.FuncTool {
	return gollym.NewTool("weather", "weather", []gollym.Parameter{{Name: "city", Required: true}},
		func(_ context.Context, args map[string]any) (any, error) {
			if got != nil {
				*got = args
			}
			return "Sunny in " + args["city"].(string), nil
		})
}

// --- pyllym tests/test_chat.py, test_behaviors.py, test_gemini_extra.py -------

func TestGeminiChat(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: textResponse("Hi from Gemini")})
	msg, err := geminiChat(t, srv.URL, "gemini-2.5-flash").Ask(context.Background(), "Hello")
	if err != nil {
		t.Fatal(err)
	}
	if msg.Text() != "Hi from Gemini" || *msg.InputTokens() != 8 || *msg.OutputTokens() != 4 {
		t.Fatalf("msg %q in=%v out=%v", msg.Text(), *msg.InputTokens(), *msg.OutputTokens())
	}
	if !msg.IsStopped() || msg.ModelID != "gemini-2.5-flash" {
		t.Fatalf("finish %q model %q", msg.FinishReason, msg.ModelID)
	}
	req := srv.Last(t)
	if req.Path != "/models/gemini-2.5-flash:generateContent" || req.Headers.Get("x-goog-api-key") != "gm-test" {
		t.Fatalf("path %s key %q", req.Path, req.Headers.Get("x-goog-api-key"))
	}
	jsonEqual(t, "payload", req.JSON(t), map[string]any{
		"contents":         []any{map[string]any{"role": "user", "parts": []any{map[string]any{"text": "Hello"}}}},
		"generationConfig": map[string]any{},
	})
}

func TestRenderGeminiShape(t *testing.T) {
	chat := geminiChat(t, "http://unused", "gemini-2.5-flash")
	if err := chat.AddUserMessage("hi"); err != nil {
		t.Fatal(err)
	}
	if _, ok := render(t, chat)["contents"]; !ok {
		t.Fatal("no contents")
	}
}

func TestGeminiToolLoop(t *testing.T) {
	call := geminiResponse(nil, map[string]any{"name": "weather", "args": map[string]any{"city": "Tokyo"}}, 8, 4)
	srv := testserver.New(t, testserver.Reply{Body: call}, testserver.Reply{Body: textResponse("Tokyo is sunny.")})
	chat := geminiChat(t, srv.URL, "gemini-2.5-flash")
	var got map[string]any
	chat.WithTool(weatherTool(&got))
	msg, err := chat.Ask(context.Background(), "weather in Tokyo?")
	if err != nil {
		t.Fatal(err)
	}
	if msg.Text() != "Tokyo is sunny." || got["city"] != "Tokyo" {
		t.Fatalf("msg %q args %v", msg.Text(), got)
	}
	var roles []string
	for _, m := range chat.Messages {
		roles = append(roles, m.Role)
	}
	if strings.Join(roles, ",") != "user,assistant,tool,assistant" {
		t.Fatalf("roles %v", roles)
	}
	first := srv.Requests()[0].JSON(t)
	jsonEqual(t, "tools", first["tools"], []any{map[string]any{"functionDeclarations": []any{map[string]any{
		"name": "weather", "description": "weather",
		"parameters": map[string]any{"type": "OBJECT", "properties": map[string]any{"city": map[string]any{"type": "STRING"}}, "required": []any{"city"}},
	}}}})
	second := srv.Requests()[1].JSON(t)
	jsonEqual(t, "contents", second["contents"], []any{
		map[string]any{"role": "user", "parts": []any{map[string]any{"text": "weather in Tokyo?"}}},
		map[string]any{"role": "model", "parts": []any{map[string]any{"functionCall": map[string]any{"name": "weather", "args": map[string]any{"city": "Tokyo"}}}}},
		map[string]any{"role": "user", "parts": []any{map[string]any{"functionResponse": map[string]any{
			"name": "weather", "response": map[string]any{"name": "weather", "content": []any{map[string]any{"text": "Sunny in Tokyo"}}},
		}}}},
	})
}

func TestGeminiStreaming(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Headers: map[string]string{"Content-Type": "text/event-stream"}, Body: geminiSSE("Hel", "lo!")})
	chat := geminiChat(t, srv.URL, "gemini-2.5-flash")
	var chunks []string
	msg, err := chat.Ask(context.Background(), "hi", gollym.OnChunk(func(c *gollym.Chunk) error {
		chunks = append(chunks, c.Text())
		return nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(chunks, "") != "Hello!" || chat.Messages[len(chat.Messages)-1].Text() != "Hello!" {
		t.Fatalf("chunks %q msg %q", chunks, msg.Text())
	}
	if *msg.InputTokens() != 1 || *msg.OutputTokens() != 2 || msg.FinishReason != "STOP" || msg.ModelID != "gemini-2.5-flash" {
		t.Fatalf("msg %+v", msg)
	}
	req := srv.Last(t)
	if req.Path != "/models/gemini-2.5-flash:streamGenerateContent" || req.Query != "alt=sse" {
		t.Fatalf("path %s?%s", req.Path, req.Query)
	}
	if _, ok := req.JSON(t)["stream"]; ok {
		t.Fatal("gemini payload has no stream flag")
	}
}

func TestGeminiSystemMessagesUseSystemInstruction(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: textResponse("Oui.")})
	chat := geminiChat(t, srv.URL, "gemini-2.5-flash")
	chat.WithInstructions("Be terse")
	if _, err := chat.Ask(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	body := srv.Last(t).JSON(t)
	jsonEqual(t, "systemInstruction", body["systemInstruction"], map[string]any{"parts": []any{map[string]any{"text": "Be terse"}}})
	contents := body["contents"].([]any)
	if len(contents) != 1 || contents[0].(map[string]any)["role"] != "user" {
		t.Fatalf("contents %v", contents)
	}
}

func geminiProtocol(t *testing.T, model string) (core.Protocol, *core.Info) {
	t.Helper()
	info, p, err := core.ResolveModel(model, "gemini", false, geminiContext("http://unused").Config)
	if err != nil {
		t.Fatal(err)
	}
	proto, err := p.Protocol("", info)
	if err != nil {
		t.Fatal(err)
	}
	return proto, info
}

func TestGeminiMultipleSystemMessagesConcatenate(t *testing.T) {
	proto, info := geminiProtocol(t, "gemini-2.5-flash")
	payload, err := core.RenderRequest(proto, &core.CompletionRequest{Model: info, Messages: []*core.Message{
		{Role: "system", Content: "Be terse"},
		{Role: "system", Content: "Answer in French"},
		{Role: "user", Content: "hi"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, "systemInstruction", payload["systemInstruction"], map[string]any{"parts": []any{map[string]any{"text": "Be terse\n\nAnswer in French"}}})
	if c := payload["contents"].([]any); len(c) != 1 || c[0].(map[string]any)["role"] != "user" {
		t.Fatalf("contents %v", c)
	}
}

// --- Rekryon's calls -----------------------------------------------------------

// Rekryon's scoring call: system + user, temperature 0.1, maxOutputTokens and
// JSON mode via params deep-merged into generationConfig.
func TestRekryonStyleJSONCall(t *testing.T) {
	text := `{"overall_score": 7}`
	reply := geminiResponse(&text, nil, 120, 45)
	reply["usageMetadata"].(map[string]any)["totalTokenCount"] = 165
	srv := testserver.New(t, testserver.Reply{Body: reply})
	chat := geminiChat(t, srv.URL, "gemini-2.5-flash")
	chat.WithTemperature(0.1).WithParams(map[string]any{"generationConfig": map[string]any{"maxOutputTokens": 4096, "responseMimeType": "application/json"}})
	chat.WithInstructions("You are a recruiter.")
	msg, err := chat.Ask(context.Background(), "Score this candidate")
	if err != nil {
		t.Fatal(err)
	}
	req := srv.Last(t)
	if req.Headers.Get("x-goog-api-key") != "gm-test" || req.Headers.Get("Authorization") != "" {
		t.Fatalf("headers %v", req.Headers)
	}
	jsonEqual(t, "payload", req.JSON(t), map[string]any{
		"systemInstruction": map[string]any{"parts": []any{map[string]any{"text": "You are a recruiter."}}},
		"contents":          []any{map[string]any{"role": "user", "parts": []any{map[string]any{"text": "Score this candidate"}}}},
		"generationConfig":  map[string]any{"temperature": 0.1, "maxOutputTokens": 4096, "responseMimeType": "application/json"},
	})
	if msg.Text() != text || *msg.InputTokens() != 120 || *msg.OutputTokens() != 45 {
		t.Fatalf("msg %q in=%d out=%d", msg.Text(), *msg.InputTokens(), *msg.OutputTokens())
	}
	if msg.CachedTokens() != nil || msg.ThinkingTokens() != nil {
		t.Fatalf("cached %v thinking %v", msg.CachedTokens(), msg.ThinkingTokens())
	}
}

// Rekryon's extraction call: a schema plus a PDF (and an image) sent inline,
// temperature 0.
func TestRekryonStyleExtraction(t *testing.T) {
	text := `{"name": "Ada"}`
	srv := testserver.New(t, testserver.Reply{Body: geminiResponse(&text, nil, 900, 20)})
	chat := geminiChat(t, srv.URL, "gemini-2.5-flash")
	schema := map[string]any{
		"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string"}},
		"required": []any{"name"}, "additionalProperties": false,
	}
	if _, err := chat.WithSchema(schema); err != nil {
		t.Fatal(err)
	}
	chat.WithTemperature(0)
	pdf := []byte("%PDF-1.4 fake cv")
	png := []byte("\x89PNG\r\n\x1a\nfake")
	msg, err := chat.Ask(context.Background(), "Extract the candidate",
		gollym.With(core.MustAttachment(pdf, "cv.pdf"), core.MustAttachment(png, "photo.png")))
	if err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, "payload", srv.Last(t).JSON(t), map[string]any{
		"contents": []any{map[string]any{"role": "user", "parts": []any{
			map[string]any{"text": "Extract the candidate"},
			map[string]any{"inline_data": map[string]any{"mime_type": "application/pdf", "data": base64.StdEncoding.EncodeToString(pdf)}},
			map[string]any{"inline_data": map[string]any{"mime_type": "image/png", "data": base64.StdEncoding.EncodeToString(png)}},
		}}},
		"generationConfig": map[string]any{"temperature": 0, "responseMimeType": "application/json", "responseJsonSchema": schema},
	})
	got, ok := msg.Content.(map[string]any)
	if !ok || got["name"] != "Ada" {
		t.Fatalf("content %#v", msg.Content)
	}
}

// --- render ----------------------------------------------------------------------

func TestResponseSchemaForOlderModels(t *testing.T) {
	chat := geminiChat(t, "http://unused", "gemini-2.0-flash")
	_, err := chat.WithSchema(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"person": map[string]any{"$ref": "#/$defs/Person", "description": "who"},
			"tags":   map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "maxItems": 3},
			"age":    map[string]any{"anyOf": []any{map[string]any{"type": "integer"}, map[string]any{"type": "null"}}},
		},
		"required":             []any{"person", "person"},
		"additionalProperties": false,
		"$defs": map[string]any{"Person": map[string]any{
			"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string", "enum": []any{"a", "b"}}},
			"additionalProperties": false,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := chat.AddUserMessage("hi"); err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, "generationConfig", render(t, chat)["generationConfig"], map[string]any{
		"responseMimeType": "application/json",
		"responseSchema": map[string]any{
			"type": "OBJECT",
			"properties": map[string]any{
				"person": map[string]any{"type": "OBJECT", "description": "who", "properties": map[string]any{"name": map[string]any{"type": "STRING", "enum": []any{"a", "b"}}}},
				"tags":   map[string]any{"type": "ARRAY", "items": map[string]any{"type": "STRING"}, "maxItems": 3},
				"age":    map[string]any{"type": "INTEGER", "nullable": true},
			},
			"required": []any{"person"},
		},
	})
}

func TestRecursiveRefsAreCutOff(t *testing.T) {
	got := gemini.ConvertResponseSchema(map[string]any{
		"$ref": "#/definitions/Node",
		"definitions": map[string]any{"Node": map[string]any{
			"type": "object", "properties": map[string]any{"child": map[string]any{"$ref": "#/definitions/Node"}},
		}},
	})
	jsonEqual(t, "schema", got, map[string]any{"type": "OBJECT", "properties": map[string]any{
		"child": map[string]any{"type": "STRING"},
	}})
	if gemini.ConvertResponseSchema(map[string]any{}) != nil {
		t.Fatal("empty schema should convert to nil")
	}
}

func TestToolSchemaConversion(t *testing.T) {
	got, err := gemini.ConvertToolSchema(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"limit": map[string]any{"anyOf": []any{map[string]any{"type": "integer", "minimum": 1}, map[string]any{"type": "null"}}, "description": "max"},
			"ids":   map[string]any{"type": "array", "items": map[string]any{"type": "number"}, "min_items": 1},
			"where": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string", "enum": []any{"Oulu"}}}, "required": []any{"city"}},
			"flag":  map[string]any{"type": "boolean"},
		},
		"required": []any{"limit"},
	})
	if err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, "parameters", got, map[string]any{
		"type": "OBJECT",
		"properties": map[string]any{
			"limit": map[string]any{"type": "INTEGER", "description": "max", "minimum": 1, "nullable": true},
			"ids":   map[string]any{"type": "ARRAY", "items": map[string]any{"type": "NUMBER"}, "minItems": 1},
			"where": map[string]any{"type": "OBJECT", "properties": map[string]any{"city": map[string]any{"type": "STRING", "enum": []any{"Oulu"}}}, "required": []any{"city"}},
			"flag":  map[string]any{"type": "BOOLEAN"},
		},
		"required": []any{"limit"},
	})
	if _, err := gemini.ConvertToolSchema(map[string]any{"type": "string"}); err == nil {
		t.Fatal("non-object parameters must fail")
	}
}

func TestToolChoiceAndThinkingConfig(t *testing.T) {
	for choice, want := range map[string]map[string]any{
		"auto":     {"mode": "auto"},
		"none":     {"mode": "none"},
		"required": {"mode": "any"},
		"weather":  {"mode": "any", "allowedFunctionNames": []any{"weather"}},
	} {
		chat := geminiChat(t, "http://unused", "gemini-2.5-flash")
		chat.WithTool(weatherTool(nil))
		if err := chat.WithToolChoice(choice); err != nil {
			t.Fatal(err)
		}
		_ = chat.WithToolCalls("one") // Gemini can't control parallel calls: ignored.
		if err := chat.AddUserMessage("hi"); err != nil {
			t.Fatal(err)
		}
		payload := render(t, chat)
		jsonEqual(t, "toolConfig "+choice, payload["toolConfig"], map[string]any{"functionCallingConfig": want})
	}

	chat := geminiChat(t, "http://unused", "gemini-2.5-flash")
	if _, err := chat.WithThinking("high", core.Int(1024)); err != nil {
		t.Fatal(err)
	}
	if err := chat.AddUserMessage("hi"); err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, "thinkingConfig", render(t, chat)["generationConfig"], map[string]any{
		"thinkingConfig": map[string]any{"includeThoughts": true, "thinkingLevel": "high", "thinkingBudget": 1024},
	})
}

func TestAssistantHistoryRendering(t *testing.T) {
	proto, info := geminiProtocol(t, "gemini-2.5-flash")
	calls := &core.ToolCalls{}
	calls.Set("c1", &core.ToolCall{ID: "c1", Name: "weather", Arguments: map[string]any{"city": "Oulu"}})
	calls.Set("c2", &core.ToolCall{ID: "c2", Name: "time", Arguments: map[string]any{}, ThoughtSignature: "own-sig"})
	calls.Set("c3", &core.ToolCall{ID: "c3", Name: "news", Arguments: map[string]any{}})
	payload, err := core.RenderRequest(proto, &core.CompletionRequest{Model: info, Messages: []*core.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "Hello", Thinking: &core.Thinking{Text: "pondering", Signature: "sig-1"}},
		{Role: "assistant", Content: "", ToolCalls: calls, Thinking: &core.Thinking{Signature: "fallback"}},
		{Role: "tool", ToolCallID: "c1", Content: "-20C"},
		{Role: "tool", ToolCallID: "c2", Content: ""},
		{Role: "tool", ToolCallID: "c3", Content: "news!"},
		{Role: "user", Content: core.RawContent{Value: []any{map[string]any{"text": "raw"}}}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, "contents", payload["contents"], []any{
		map[string]any{"role": "user", "parts": []any{map[string]any{"text": "hi"}}},
		map[string]any{"role": "model", "parts": []any{
			map[string]any{"thought": true, "text": "pondering", "thoughtSignature": "sig-1"},
			map[string]any{"text": "Hello"},
		}},
		map[string]any{"role": "model", "parts": []any{
			map[string]any{"functionCall": map[string]any{"name": "weather", "args": map[string]any{"city": "Oulu"}}, "thoughtSignature": "fallback"},
			map[string]any{"functionCall": map[string]any{"name": "time", "args": map[string]any{}}, "thoughtSignature": "own-sig"},
			map[string]any{"functionCall": map[string]any{"name": "news", "args": map[string]any{}}},
		}},
		map[string]any{"role": "user", "parts": []any{
			map[string]any{"functionResponse": map[string]any{"name": "weather", "response": map[string]any{"name": "weather", "content": []any{map[string]any{"text": "-20C"}}}}},
			map[string]any{"functionResponse": map[string]any{"name": "time", "response": map[string]any{"name": "time", "content": []any{map[string]any{"text": "(no output)"}}}}},
			map[string]any{"functionResponse": map[string]any{"name": "news", "response": map[string]any{"name": "news", "content": []any{map[string]any{"text": "news!"}}}}},
		}},
		map[string]any{"role": "user", "parts": []any{map[string]any{"text": "raw"}}},
	})
}

func TestAttachmentsRendering(t *testing.T) {
	proto, info := geminiProtocol(t, "gemini-2.5-flash")
	text := "see files"
	content := &core.Content{Text: &text, Attachments: []*core.Attachment{
		core.MustAttachment([]byte("hello"), "notes.txt"),
		core.MustAttachment(&core.UploadedFile{ID: "files/abc", URI: "https://generativelanguage.googleapis.com/v1beta/files/abc", MIMEType: "video/mp4"}, ""),
		core.MustAttachment(&core.UploadedFile{ID: "files/xyz", MIMEType: "audio/mpeg"}, ""),
	}}
	payload, err := core.RenderRequest(proto, &core.CompletionRequest{Model: info, Messages: []*core.Message{{Role: "user", Content: content}}})
	if err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, "parts", payload["contents"].([]any)[0].(map[string]any)["parts"], []any{
		map[string]any{"text": "see files"},
		map[string]any{"text": "<file name='notes.txt' mime_type='text/plain'>hello</file>"},
		map[string]any{"file_data": map[string]any{"mime_type": "video/mp4", "file_uri": "https://generativelanguage.googleapis.com/v1beta/files/abc"}},
		map[string]any{"file_data": map[string]any{"mime_type": "audio/mpeg", "file_uri": "files/xyz"}},
	})

	docx := &core.Content{Attachments: []*core.Attachment{core.MustAttachment([]byte("PK"), "cv.docx")}}
	_, err = core.RenderRequest(proto, &core.CompletionRequest{Model: info, Messages: []*core.Message{{Role: "user", Content: docx}}})
	if !errors.Is(err, core.ErrUnsupportedContent) {
		t.Fatalf("docx err %v", err)
	}
}

// --- parse -----------------------------------------------------------------------

func parse(t *testing.T, body map[string]any) *core.Message {
	t.Helper()
	proto, _ := geminiProtocol(t, "gemini-2.5-flash")
	msg, err := proto.ParseCompletionResponse(&core.Response{Status: 200, Body: normalize(t, body)})
	if err != nil {
		t.Fatal(err)
	}
	return msg
}

func TestUsageThinkingAndSignatures(t *testing.T) {
	msg := parse(t, map[string]any{
		"candidates": []any{map[string]any{"content": map[string]any{"parts": []any{
			map[string]any{"text": "Let me think. ", "thought": true},
			map[string]any{"text": "More.", "thought": true, "thoughtSignature": "sig-a"},
			map[string]any{"text": "Answer"},
		}}, "finishReason": "MAX_TOKENS"}},
		"usageMetadata": map[string]any{"promptTokenCount": 100, "cachedContentTokenCount": 30, "candidatesTokenCount": 10, "thoughtsTokenCount": 5, "totalTokenCount": 115},
	})
	if msg.Text() != "Answer" || msg.Thinking == nil || msg.Thinking.Text != "Let me think. More." || msg.Thinking.Signature != "sig-a" {
		t.Fatalf("content %q thinking %v", msg.Text(), msg.Thinking)
	}
	if *msg.InputTokens() != 70 || *msg.OutputTokens() != 15 || *msg.CachedTokens() != 30 || *msg.ThinkingTokens() != 5 {
		t.Fatalf("tokens %+v", msg.Tokens)
	}
	if !msg.IsMaxTokens() || msg.ModelID != "gemini-2.5-flash" {
		t.Fatalf("finish %q model %q", msg.FinishReason, msg.ModelID)
	}
}

func TestToolCallParseKeepsThoughtSignature(t *testing.T) {
	msg := parse(t, map[string]any{"candidates": []any{map[string]any{"content": map[string]any{"parts": []any{
		map[string]any{"text": "calling"},
		map[string]any{"functionCall": map[string]any{"name": "weather", "args": map[string]any{"city": "Paris"}}, "thoughtSignature": "sig-tc"},
		map[string]any{"functionCall": map[string]any{"name": "time"}},
	}}}}})
	if msg.Content != "" || msg.ToolCalls.Len() != 2 {
		t.Fatalf("content %#v calls %d", msg.Content, msg.ToolCalls.Len())
	}
	first, second := msg.ToolCalls.List()[0], msg.ToolCalls.List()[1]
	if first.Name != "weather" || first.ThoughtSignature != "sig-tc" || first.ID == "" || first.Arguments.(map[string]any)["city"] != "Paris" {
		t.Fatalf("first %+v", first)
	}
	if len(second.Arguments.(map[string]any)) != 0 {
		t.Fatalf("second args %v", second.Arguments)
	}
	if msg.Thinking == nil || msg.Thinking.Signature != "sig-tc" {
		t.Fatalf("thinking %v", msg.Thinking)
	}
}

func TestInlineAndFileDataResponses(t *testing.T) {
	img := []byte("\x89PNG\r\n\x1a\nimage")
	msg := parse(t, map[string]any{"candidates": []any{map[string]any{"content": map[string]any{"parts": []any{
		map[string]any{"text": "Here"},
		map[string]any{"inlineData": map[string]any{"mimeType": "image/png", "data": base64.StdEncoding.EncodeToString(img)}},
		map[string]any{"fileData": map[string]any{"mimeType": "image/jpeg", "fileUri": "https://example.com/x"}},
	}}}}})
	c, ok := msg.Content.(*core.Content)
	if !ok || c.TextOrEmpty() != "Here" || len(c.Attachments) != 2 {
		t.Fatalf("content %#v", msg.Content)
	}
	a, b := c.Attachments[0], c.Attachments[1]
	data, _ := a.Content()
	if a.Filename != "gemini_attachment_2.png" || a.MIMEType != "image/png" || string(data) != string(img) {
		t.Fatalf("inline %s %s", a.Filename, a.MIMEType)
	}
	if b.Filename != "gemini_attachment_3.jpg" || b.URL != "https://example.com/x" {
		t.Fatalf("file %s %s", b.Filename, b.URL)
	}

	empty := parse(t, map[string]any{"candidates": []any{}})
	if empty.Content != "" || *empty.OutputTokens() != 0 || empty.InputTokens() != nil {
		t.Fatalf("empty %#v", empty)
	}
}

func TestGroundingCitations(t *testing.T) {
	text := "Päivää! Oulu is cold."
	msg := parse(t, map[string]any{"candidates": []any{map[string]any{
		"content": map[string]any{"parts": []any{map[string]any{"text": text}}},
		"groundingMetadata": map[string]any{
			"groundingChunks": []any{
				map[string]any{"web": map[string]any{"uri": "https://a.example", "title": "A"}},
				map[string]any{"retrievedContext": map[string]any{"uri": "gs://b", "title": "B"}},
			},
			"groundingSupports": []any{map[string]any{
				// "Päivää! " is 8 characters but 11 bytes.
				"segment":               map[string]any{"startIndex": 11, "endIndex": 25, "text": "Oulu is cold."},
				"groundingChunkIndices": []any{0, 1, 7},
			}},
		},
	}}})
	if len(msg.Citations) != 2 {
		t.Fatalf("citations %+v", msg.Citations)
	}
	c := msg.Citations[0]
	if *c.URL != "https://a.example" || *c.Title != "A" || *c.Text != "Oulu is cold." || *c.StartIndex != 8 || *c.EndIndex != 21 || *c.SourceIndex != 0 {
		t.Fatalf("citation %+v", c)
	}
	if *msg.Citations[1].URL != "gs://b" || *msg.Citations[1].SourceIndex != 1 {
		t.Fatalf("second %+v", msg.Citations[1])
	}

	chunksOnly := parse(t, map[string]any{"candidates": []any{map[string]any{
		"content":           map[string]any{"parts": []any{map[string]any{"text": "x"}}},
		"groundingMetadata": map[string]any{"groundingChunks": []any{"bad", map[string]any{"web": map[string]any{"uri": "https://c"}}}},
	}}})
	if len(chunksOnly.Citations) != 1 || *chunksOnly.Citations[0].SourceIndex != 1 || chunksOnly.Citations[0].Title != nil {
		t.Fatalf("chunk citations %+v", chunksOnly.Citations)
	}
}

func TestStreamingToolCallAndThinking(t *testing.T) {
	events := []map[string]any{
		{"candidates": []any{map[string]any{"content": map[string]any{"parts": []any{map[string]any{"text": "hmm", "thought": true, "thoughtSignature": "s1"}}}}}},
		{"candidates": []any{map[string]any{"content": map[string]any{"parts": []any{map[string]any{"functionCall": map[string]any{"name": "weather", "args": map[string]any{"city": "Oulu"}}}}}, "finishReason": "STOP"}},
			"usageMetadata": map[string]any{"promptTokenCount": 3, "candidatesTokenCount": 2, "thoughtsTokenCount": 4}},
	}
	var sb strings.Builder
	for _, e := range events {
		b, _ := json.Marshal(e)
		sb.WriteString("data: " + string(b) + "\r\n\r\n")
	}
	srv := testserver.New(t,
		testserver.Reply{Headers: map[string]string{"Content-Type": "text/event-stream"}, Body: sb.String()},
		testserver.Reply{Headers: map[string]string{"Content-Type": "text/event-stream"}, Body: geminiSSE("done")},
	)
	chat := geminiChat(t, srv.URL, "gemini-2.5-flash")
	var got map[string]any
	chat.WithTool(weatherTool(&got))
	msg, err := chat.Ask(context.Background(), "weather?", gollym.OnChunk(func(*gollym.Chunk) error { return nil }))
	if err != nil {
		t.Fatal(err)
	}
	if msg.Text() != "done" || got["city"] != "Oulu" {
		t.Fatalf("msg %q args %v", msg.Text(), got)
	}
	call := chat.Messages[1]
	if call.ToolCalls.Len() != 1 || call.Thinking == nil || call.Thinking.Text != "hmm" || *call.OutputTokens() != 6 || *call.ThinkingTokens() != 4 {
		t.Fatalf("call message %+v", call)
	}
	second := srv.Requests()[1].JSON(t)
	model := second["contents"].([]any)[1].(map[string]any)
	jsonEqual(t, "model turn", model, map[string]any{"role": "model", "parts": []any{
		map[string]any{"functionCall": map[string]any{"name": "weather", "args": map[string]any{"city": "Oulu"}}, "thoughtSignature": "s1"},
	}})
}

func TestHTTPErrorsUseGeminiMessage(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Status: 400, Body: map[string]any{"error": map[string]any{"code": 400, "message": "API key not valid.", "status": "INVALID_ARGUMENT"}}})
	_, err := geminiChat(t, srv.URL, "gemini-2.5-flash").Ask(context.Background(), "hi")
	if !errors.Is(err, gollym.ErrBadRequest) || err.Error() != "API key not valid." {
		t.Fatalf("err %v", err)
	}
	srv = testserver.New(t, testserver.Reply{Status: 429, Body: map[string]any{"error": map[string]any{"code": 429, "message": "Resource has been exhausted", "status": "RESOURCE_EXHAUSTED"}}})
	_, err = geminiChat(t, srv.URL, "gemini-2.5-flash").Ask(context.Background(), "hi")
	if !errors.Is(err, gollym.ErrRateLimit) {
		t.Fatalf("err %v", err)
	}
}

// --- embeddings and models ---------------------------------------------------

func TestEmbeddings(t *testing.T) {
	srv := testserver.New(t,
		testserver.Reply{Body: map[string]any{"embeddings": []any{map[string]any{"values": []any{0.1, 0.2}}}}},
		testserver.Reply{Body: map[string]any{"embeddings": []any{map[string]any{"values": []any{1}}, map[string]any{"values": []any{2}}}}},
	)
	ctx := geminiContext(srv.URL)
	opts := gollym.EmbedOptions{ModelOptions: gollym.ModelOptions{Model: "text-embedding-004", Provider: "gemini", AssumeModelExists: true, Context: ctx}, Dimensions: core.Int(2)}
	emb, err := gollym.Embed(context.Background(), "hello", opts)
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := emb.Vectors.([]float64); !ok || len(v) != 2 || v[1] != 0.2 || emb.InputTokens != 0 {
		t.Fatalf("vectors %#v", emb.Vectors)
	}
	req := srv.Last(t)
	if req.Path != "/models/text-embedding-004:batchEmbedContents" {
		t.Fatalf("path %s", req.Path)
	}
	jsonEqual(t, "payload", req.JSON(t), map[string]any{"requests": []any{map[string]any{
		"model": "models/text-embedding-004", "content": map[string]any{"parts": []any{map[string]any{"text": "hello"}}}, "outputDimensionality": 2,
	}}})

	opts.Dimensions = nil
	emb, err = gollym.Embed(context.Background(), []string{"a", "b"}, opts)
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := emb.Vectors.([][]float64); !ok || len(v) != 2 || v[1][0] != 2 {
		t.Fatalf("vectors %#v", emb.Vectors)
	}
	if reqs := srv.Last(t).JSON(t)["requests"].([]any); len(reqs) != 2 || reqs[1].(map[string]any)["outputDimensionality"] != nil {
		t.Fatalf("requests %v", reqs)
	}
}

func TestListModels(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: map[string]any{"models": []any{
		map[string]any{"name": "models/gemini-2.0-flash", "displayName": "Gemini 2.0 Flash", "version": "2.0", "inputTokenLimit": 1000, "supportedGenerationMethods": []any{"generateContent"}},
		map[string]any{"name": "models/text-embedding-004"},
	}}})
	_, p, err := core.ResolveModel("gemini-2.5-flash", "gemini", false, geminiContext(srv.URL).Config)
	if err != nil {
		t.Fatal(err)
	}
	models, err := gollym.ListModels(context.Background(), p)
	if err != nil {
		t.Fatal(err)
	}
	if srv.Last(t).Path != "/models" || len(models) != 2 {
		t.Fatalf("path %s models %d", srv.Last(t).Path, len(models))
	}
	m := models[0]
	if m.ID != "gemini-2.0-flash" || m.Name != "Gemini 2.0 Flash" || *m.ContextWindow != 1000 || *m.MaxOutputTokens != 8192 || m.Provider != "gemini" {
		t.Fatalf("model %+v", m)
	}
	if strings.Join(m.Capabilities, ",") != "function_calling,structured_output,vision" || m.Metadata["version"] != "2.0" {
		t.Fatalf("caps %v metadata %v", m.Capabilities, m.Metadata)
	}
	if *m.InputPricePerMillion() != 0.10 || *m.OutputPricePerMillion() != 0.40 {
		t.Fatalf("pricing %v/%v", *m.InputPricePerMillion(), *m.OutputPricePerMillion())
	}
	e := models[1]
	if e.Name != "text-embedding-004" || *e.ContextWindow != 2048 || *e.MaxOutputTokens != 768 || len(e.Capabilities) != 0 {
		t.Fatalf("embedding model %+v", e)
	}
}

func TestGeminiCapabilities(t *testing.T) {
	c := providers.GeminiCapabilities{}
	if *c.ContextWindowFor("gemini-1.5-pro-002") != 2_097_152 || *c.ContextWindowFor("gemini-3-pro") != 32_768 || c.ContextWindowFor("imagen-3.0") != nil {
		t.Fatal("context window")
	}
	if *c.MaxTokensFor("gemini-2.5-pro-exp-03-25") != 64_000 || *c.MaxTokensFor("imagen-3.0") != 4 || c.MaxTokensFor("gemini-embedding-exp") != nil {
		t.Fatal("max tokens")
	}
	if got := c.CriticalCapabilitiesFor("gemini-2.0-flash-lite"); strings.Join(got, ",") != "vision" {
		t.Fatalf("flash-lite caps %v", got)
	}
	if got := c.CriticalCapabilitiesFor("aqa"); len(got) != 0 {
		t.Fatalf("aqa caps %v", got)
	}
	// pyllym's `input or price or default`: zero-priced families get defaults.
	p := c.PricingFor("text-embedding-004")["text_tokens"].(map[string]any)["standard"].(map[string]any)
	if p["input_per_million"] != 0.075 || p["output_per_million"] != 0.30 {
		t.Fatalf("embedding pricing %v", p)
	}
	p = c.PricingFor("imagen-3.0")["text_tokens"].(map[string]any)["standard"].(map[string]any)
	if p["input_per_million"] != 0.03 || p["output_per_million"] != 0.03 {
		t.Fatalf("imagen pricing %v", p)
	}
	if c.PricingFamily("gemini-2.0-flash-lite-001") != "flash_lite_2" || c.PricingFamily("unknown") != "base" {
		t.Fatal("pricing family")
	}
}

// --- Vertex AI -------------------------------------------------------------------

func vertexContext(base, token, location string) *gollym.Context {
	return gollym.NewContext(func(c *gollym.Config) {
		c.Set("vertexai_project_id", "proj")
		c.Set("vertexai_location", location)
		c.Set("vertexai_access_token", token)
		c.Set("vertexai_api_base", base)
		c.MaxRetries = 0
		c.RequestTimeout = 5 * time.Second
	})
}

func TestVertexAIChatAndStream(t *testing.T) {
	srv := testserver.New(t,
		testserver.Reply{Body: textResponse("Hi from Vertex")},
		testserver.Reply{Headers: map[string]string{"Content-Type": "text/event-stream"}, Body: geminiSSE("a", "b")},
		testserver.Reply{Body: map[string]any{"embeddings": []any{map[string]any{"values": []any{0.5}}}}},
	)
	ctx := vertexContext(srv.URL, "ya29.token", "us-central1")
	chat, err := ctx.NewChat(gollym.ChatOptions{Model: "gemini-2.5-flash", Provider: "vertexai"})
	if err != nil {
		t.Fatal(err)
	}
	msg, err := chat.Ask(context.Background(), "hi")
	if err != nil || msg.Text() != "Hi from Vertex" {
		t.Fatalf("msg %v err %v", msg, err)
	}
	const path = "/projects/proj/locations/us-central1/publishers/google/models/gemini-2.5-flash"
	req := srv.Last(t)
	if req.Path != path+":generateContent" || req.Headers.Get("Authorization") != "Bearer ya29.token" || req.Headers.Get("x-goog-api-key") != "" {
		t.Fatalf("path %s headers %v", req.Path, req.Headers)
	}
	if _, err := chat.Ask(context.Background(), "again", gollym.OnChunk(func(*gollym.Chunk) error { return nil })); err != nil {
		t.Fatal(err)
	}
	if req := srv.Last(t); req.Path != path+":streamGenerateContent" || req.Query != "alt=sse" {
		t.Fatalf("stream %s?%s", req.Path, req.Query)
	}
	_, err = gollym.Embed(context.Background(), "x", gollym.EmbedOptions{ModelOptions: gollym.ModelOptions{Model: "text-embedding-005", Provider: "vertexai", AssumeModelExists: true, Context: ctx}})
	if err != nil {
		t.Fatal(err)
	}
	if req := srv.Last(t); req.Path != "/projects/proj/locations/us-central1/publishers/google/models/text-embedding-005:predict" {
		t.Fatalf("embed path %s", req.Path)
	}
}

func TestVertexAIBaseAndMissingToken(t *testing.T) {
	spec, ok := core.LookupProvider("vertexai")
	if !ok {
		t.Fatal("vertexai not registered")
	}
	for location, want := range map[string]string{
		"global":       "https://aiplatform.googleapis.com/v1beta1",
		"europe-west4": "https://europe-west4-aiplatform.googleapis.com/v1beta1",
	} {
		p, err := core.NewProvider(spec, vertexContext("", "tok", location).Config)
		if err != nil {
			t.Fatal(err)
		}
		if base, err := p.APIBase(); err != nil || base != want {
			t.Fatalf("%s: base %q err %v", location, base, err)
		}
	}

	srv := testserver.New(t, testserver.Reply{Body: textResponse("never")})
	ctx := vertexContext(srv.URL, "", "us-central1")
	ctx.Config.Set("vertexai_access_token", "")
	t.Setenv("VERTEXAI_ACCESS_TOKEN", "")
	chat, err := ctx.NewChat(gollym.ChatOptions{Model: "gemini-2.5-flash", Provider: "vertexai"})
	if err != nil {
		t.Fatal(err)
	}
	_, err = chat.Ask(context.Background(), "hi")
	var cfgErr *core.ConfigurationError
	if !errors.As(err, &cfgErr) || !strings.Contains(err.Error(), "gcloud auth print-access-token") {
		t.Fatalf("err %v", err)
	}
	if len(srv.Requests()) != 0 {
		t.Fatal("no request should be sent without a token")
	}
}

func TestGeminiRequiresAPIKey(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "")
	ctx := gollym.NewContext(func(c *gollym.Config) { c.Set("gemini_api_key", "") })
	_, err := ctx.NewChat(gollym.ChatOptions{Model: "gemini-2.5-flash", Provider: "gemini"})
	if !errors.Is(err, gollym.ErrConfiguration) {
		t.Fatalf("err %v", err)
	}
}
