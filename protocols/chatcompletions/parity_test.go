package chatcompletions_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"mime"
	"mime/multipart"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
	"github.com/PhotonEra-B-V/gollym/protocols/chatcompletions"
	_ "github.com/PhotonEra-B-V/gollym/providers"
)

// Parity with pyllym itself: testdata/pyllym_golden.json holds payloads
// rendered and messages parsed by pyllym 1.16.0 (testdata/golden.py
// regenerates it). Each case builds the same inputs in Go and must produce
// JSON-equal output.

type golden struct {
	Render map[string]any `json:"render"`
	Parse  map[string]struct {
		Response map[string]any `json:"response"`
		Message  map[string]any `json:"message"`
	} `json:"parse"`
	Stream map[string]struct {
		Chunks  []map[string]any `json:"chunks"`
		Message map[string]any   `json:"message"`
	} `json:"stream"`
	Features map[string]any `json:"features"`
}

func loadGolden(t *testing.T) golden {
	t.Helper()
	b, err := os.ReadFile("testdata/pyllym_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g golden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

// jsonEqual compares two values as JSON documents.
func jsonEqual(t *testing.T, name string, got, want any) {
	t.Helper()
	norm := func(v any) any {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		var out any
		_ = json.Unmarshal(b, &out)
		return out
	}
	g, w := norm(got), norm(want)
	if !reflect.DeepEqual(g, w) {
		gb, _ := json.MarshalIndent(g, "", " ")
		wb, _ := json.MarshalIndent(w, "", " ")
		t.Fatalf("%s differs from pyllym\n--- go ---\n%s\n--- pyllym ---\n%s", name, gb, wb)
	}
}

func parityCtx(t *testing.T) *core.Context {
	t.Helper()
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("DEEPSEEK_API_KEY", "")
	t.Setenv("OPENAI_USE_SYSTEM_ROLE", "")
	return core.NewContext(func(c *core.Config) {
		c.Set("openai_api_key", "sk-test")
		c.Set("deepseek_api_key", "sk-test")
	})
}

func parityChat(t *testing.T, model, provider string) *core.Chat {
	t.Helper()
	chat, err := parityCtx(t).NewChat(core.ChatOptions{Model: model, Provider: provider, AssumeModelExists: provider != ""})
	if err != nil {
		t.Fatal(err)
	}
	return chat
}

func noop(context.Context, map[string]any) (any, error) { return nil, nil }

func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

func render(t *testing.T, chat *core.Chat) map[string]any {
	t.Helper()
	p, err := chat.Render()
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestParityRender(t *testing.T) {
	g := loadGolden(t)

	t.Run("tools_params", func(t *testing.T) {
		weather := core.NewTool("weather", "Get weather", []core.Parameter{
			{Name: "city", Description: "City name", Required: true}, {Name: "days", Type: "integer"},
		}, noop)
		ping := core.NewTool(core.ToolNameFromString("Ping"), "", nil, noop)
		tagged := &core.FuncTool{ToolName: "tagged", Desc: "Tagged", Params: []core.Parameter{{Name: "q", Required: true}},
			Extra: map[string]any{"function": map[string]any{"strict": false}, "x_extra": 1}, Fn: noop}
		chat := parityChat(t, "gpt-4o", "").WithInstructions("be terse").WithTemperature(0.7)
		chat.WithTools(false, weather, ping, tagged)
		if err := chat.WithToolChoice("weather"); err != nil {
			t.Fatal(err)
		}
		if err := chat.WithToolCalls("one"); err != nil {
			t.Fatal(err)
		}
		chat.WithParams(map[string]any{"max_tokens": 100, "metadata": map[string]any{"a": map[string]any{"b": 1}}})
		_ = chat.AddUserMessage("hi")
		jsonEqual(t, "tools_params", render(t, chat), g.Render["tools_params"])
	})

	t.Run("schema_thinking", func(t *testing.T) {
		chat := parityChat(t, "o3", "").WithTemperature(0.2)
		must(chat.WithThinking("high", nil))
		must(chat.WithSchema(map[string]any{"name": "My Schema!", "strict": false, "description": "d",
			"schema": map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "string"}}, "required": []any{"a"}}}))
		_ = chat.AddUserMessage("hi")
		jsonEqual(t, "schema_thinking", render(t, chat), g.Render["schema_thinking"])
	})

	t.Run("bare_schema", func(t *testing.T) {
		chat := parityChat(t, "gpt-4o", "")
		must(chat.WithSchema(map[string]any{"type": "object", "properties": map[string]any{"x": map[string]any{"type": "integer"}}, "strict": false}))
		_ = chat.AddUserMessage("hi")
		jsonEqual(t, "bare_schema", render(t, chat), g.Render["bare_schema"])
	})

	t.Run("search_temperature", func(t *testing.T) {
		chat := parityChat(t, "gpt-4o-search-preview", "openai").WithTemperature(0.5)
		_ = chat.AddUserMessage("hi")
		jsonEqual(t, "search_temperature", render(t, chat), g.Render["search_temperature"])
	})

	t.Run("conversation", func(t *testing.T) {
		dir := t.TempDir()
		files := map[string][]byte{"notes.txt": []byte("hello file\n"), "doc.pdf": []byte("%PDF-1.4 test"), "pic.png": []byte("\x89PNG\r\n\x1a\nfakepng")}
		for name, data := range files {
			if err := os.WriteFile(filepath.Join(dir, name), data, 0o644); err != nil {
				t.Fatal(err)
			}
		}
		chat := parityChat(t, "deepseek-chat", "deepseek").WithInstructions("sys")
		calls := &core.ToolCalls{}
		calls.Set("call_1", &core.ToolCall{ID: "call_1", Name: "weather", Arguments: map[string]any{"city": "Paris", "note": "é<>&"}})
		for _, m := range []*core.Message{
			{Role: "user", Content: "weather?"},
			{Role: "assistant", ToolCalls: calls, Thinking: core.BuildThinking("let me check", "sig1")},
			{Role: "tool", Content: "Sunny", ToolCallID: "call_1"},
			{Role: "assistant", Thinking: core.BuildThinking("done thinking", "")},
		} {
			must(chat.AddMessage(m))
		}
		if err := chat.AddUserMessage("look", filepath.Join(dir, "notes.txt"), filepath.Join(dir, "doc.pdf"),
			"https://example.com/cat.png", filepath.Join(dir, "pic.png")); err != nil {
			t.Fatal(err)
		}
		must(chat.AddMessage(&core.Message{Role: "user", Content: core.RawContent{Value: map[string]any{"k": []any{1, 2}}}}))
		must(chat.AddMessage(&core.Message{Role: "user", Content: core.RawContent{Value: []any{map[string]any{"type": "text", "text": "raw"}}}}))
		jsonEqual(t, "conversation", render(t, chat), g.Render["conversation"])
	})

	t.Run("stream", func(t *testing.T) {
		chat := parityChat(t, "gpt-4o", "")
		proto := chatcompletions.New(chat.Provider, chat.Model)
		user := core.MustMessage(core.Message{Role: "user", Content: "hi"})
		payload, err := core.RenderRequest(proto, &core.CompletionRequest{Messages: []*core.Message{user}, Tools: &core.ToolSet{}, Model: chat.Model, Stream: true})
		if err != nil {
			t.Fatal(err)
		}
		jsonEqual(t, "stream", payload, g.Render["stream"])
	})
}

// msgDict is golden.py's msg_dict: to_dict with tool calls and citations
// as dicts.
func msgDict(m *core.Message) map[string]any {
	d := m.Map()
	if m.ToolCalls.Len() > 0 {
		calls := map[string]any{}
		for _, k := range m.ToolCalls.Keys() {
			calls[k] = m.ToolCalls.Get(k).Map()
		}
		d["tool_calls"] = calls
	}
	cits := []any{}
	for _, c := range m.Citations {
		cits = append(cits, c.Map())
	}
	d["citations"] = cits
	return d
}

func TestParityParse(t *testing.T) {
	g := loadGolden(t)
	chat := parityChat(t, "gpt-4o", "")
	proto := chatcompletions.New(chat.Provider, chat.Model)
	for name, c := range g.Parse {
		t.Run(name, func(t *testing.T) {
			msg, err := proto.ParseCompletionResponse(&core.Response{Status: 200, Body: c.Response})
			if err != nil {
				t.Fatal(err)
			}
			jsonEqual(t, name, msgDict(msg), c.Message)
		})
	}
}

func TestParityStream(t *testing.T) {
	g := loadGolden(t)
	chat := parityChat(t, "gpt-4o", "")
	proto := chatcompletions.New(chat.Provider, chat.Model).(core.Streamer)
	for name, c := range g.Stream {
		t.Run(name, func(t *testing.T) {
			acc := core.NewStreamAccumulator()
			for _, data := range c.Chunks {
				chunk, err := proto.BuildChunk(data)
				if err != nil {
					t.Fatal(err)
				}
				acc.Add(chunk)
			}
			jsonEqual(t, name, msgDict(acc.Message(nil)), c.Message)
		})
	}
}

// featureCtx points openai and deepseek at srv.
func featureCtx(t *testing.T, srv *testserver.Server) *core.Context {
	t.Helper()
	ctx := parityCtx(t)
	t.Setenv("OPENAI_API_BASE", "")
	t.Setenv("DEEPSEEK_API_BASE", "")
	ctx.Config.Set("openai_api_base", srv.URL)
	ctx.Config.Set("deepseek_api_base", srv.URL)
	return ctx
}

func TestParityListModels(t *testing.T) {
	g := loadGolden(t)
	for _, slug := range []string{"openai", "deepseek"} {
		t.Run(slug, func(t *testing.T) {
			srv := testserver.New(t, testserver.Reply{Body: g.Features["models_body"]})
			spec, _ := core.LookupProvider(slug)
			p, err := core.NewProvider(spec, featureCtx(t, srv).Config)
			if err != nil {
				t.Fatal(err)
			}
			models, err := core.ListModels(context.Background(), p)
			if err != nil {
				t.Fatal(err)
			}
			var got []any
			for _, m := range models {
				got = append(got, m.Map())
			}
			if srv.Last(t).Method != "GET" || srv.Last(t).Path != "/models" {
				t.Fatalf("%s %s", srv.Last(t).Method, srv.Last(t).Path)
			}
			jsonEqual(t, "list_models_"+slug, got, g.Features["list_models_"+slug])
		})
	}
}

func TestParityEmbeddingAndSpeechPayloads(t *testing.T) {
	g := loadGolden(t)
	srv := testserver.New(t, testserver.Reply{Body: map[string]any{"data": []any{}, "usage": map[string]any{}}})
	ctx := featureCtx(t, srv)
	if _, err := ctx.Embed(context.Background(), []string{"a", "b"}, core.EmbedOptions{
		ModelOptions: core.ModelOptions{Model: "text-embedding-3-small"}, Dimensions: core.Int(256)}); err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, "embedding_payload", srv.Last(t).JSON(t), g.Features["embedding_payload"])

	audio := testserver.New(t, testserver.Reply{Headers: map[string]string{"Content-Type": "audio/wav"}, Body: []byte("RIFFwav")})
	speech, err := featureCtx(t, audio).Speak(context.Background(), "hello", core.SpeakRequest{
		ModelOptions: core.ModelOptions{Model: "gpt-4o-mini-tts"}, Format: "wav", Instructions: "calm", Speed: core.Float(1.25),
		Params: map[string]any{"extra": 1}})
	if err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, "speech_payload", audio.Last(t).JSON(t), g.Features["speech_payload"])
	if string(speech.Data) != "RIFFwav" || speech.Voice != "alloy" || speech.Format != "wav" {
		t.Fatalf("speech %+v", speech)
	}
}

func TestParityTranscriptionForm(t *testing.T) {
	g := loadGolden(t)
	srv := testserver.New(t, testserver.Reply{Body: map[string]any{"text": "hei", "language": "fi"}})
	path := filepath.Join(t.TempDir(), "clip.mp3")
	if err := os.WriteFile(path, []byte("ID3audio"), 0o644); err != nil {
		t.Fatal(err)
	}
	tr, err := featureCtx(t, srv).Transcribe(context.Background(), path, core.TranscribeRequest{
		ModelOptions: core.ModelOptions{Model: "whisper-1"}, Language: "fi", Temperature: core.Float(0),
		Prompt: "names", ResponseFormat: "verbose_json"})
	if err != nil {
		t.Fatal(err)
	}
	if tr.Text != "hei" || tr.Language != "fi" {
		t.Fatalf("transcription %+v", tr)
	}
	req := srv.Last(t)
	_, params, err := mime.ParseMediaType(req.Headers.Get("Content-Type"))
	if err != nil || req.Path != "/audio/transcriptions" {
		t.Fatalf("%s %v", req.Path, err)
	}
	fields, file := map[string]any{}, ""
	mr := multipart.NewReader(bytes.NewReader(req.Body), params["boundary"])
	for {
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		b, _ := io.ReadAll(part)
		if part.FormName() == "file" {
			file = part.FileName()
			if string(b) != "ID3audio" {
				t.Fatalf("file bytes %q", b)
			}
			continue
		}
		fields[part.FormName()] = string(b)
	}
	jsonEqual(t, "transcription_data", fields, g.Features["transcription_data"])
	if file != g.Features["transcription_file"] {
		t.Fatalf("filename %q", file)
	}
}
