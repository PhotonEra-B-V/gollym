package zhipulayout_test

import (
	"context"
	"encoding/base64"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym"
	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
	"github.com/PhotonEra-B-V/gollym/protocols/zhipulayout"
)

func layoutBody() map[string]any {
	return map[string]any{
		"id":         "task-123",
		"created":    1757000000,
		"model":      "glm-ocr",
		"md_results": "# Invoice\n\nTotal: $42.00",
		"layout_details": []any{map[string]any{
			"index": 0, "label": "title", "bbox_2d": []any{10, 10, 200, 40},
			"content": "Invoice", "height": 30, "width": 190,
		}},
		"data_info": map[string]any{"num_pages": 2, "pages": []any{map[string]any{"page_id": 0}, map[string]any{"page_id": 1}}},
		"usage": map[string]any{
			"prompt_tokens": 1200, "completion_tokens": 300,
			"prompt_tokens_details": map[string]any{"cached_tokens": 100}, "total_tokens": 1500,
		},
	}
}

func zhipuContext(base string) *gollym.Context {
	return gollym.NewContext(func(c *gollym.Config) {
		c.Set("zhipu_api_key", "zp-test")
		c.Set("zhipu_api_base", base)
		c.Set("openai_api_key", "sk-test")
		c.MaxRetries = 0
		c.RetryInterval = time.Millisecond
		c.RequestTimeout = 5 * time.Second
	})
}

func parse(t *testing.T, file string, opts map[string]any) (*gollym.LayoutParsing, *testserver.Server) {
	t.Helper()
	srv := testserver.New(t, testserver.Reply{Body: layoutBody()})
	res, err := gollym.ParseLayout(context.Background(), file, gollym.LayoutRequest{
		ModelOptions: gollym.ModelOptions{Provider: "zhipu", Context: zhipuContext(srv.URL)}, Options: opts,
	})
	if err != nil {
		t.Fatal(err)
	}
	return res, srv
}

func TestParseLayoutReturnsTextAndElements(t *testing.T) {
	res, srv := parse(t, "https://cdn.example.com/invoice.pdf", nil)
	req := srv.Last(t)
	if req.Method != "POST" || req.Path != "/layout_parsing" || req.Headers.Get("Authorization") != "Bearer zp-test" {
		t.Fatalf("request %s %s auth %q", req.Method, req.Path, req.Headers.Get("Authorization"))
	}
	if res.Text != "# Invoice\n\nTotal: $42.00" || res.Markdown() != res.Text {
		t.Fatalf("text %q", res.Text)
	}
	if res.Model != "glm-ocr" || res.ID != "task-123" || res.PageCount == nil || *res.PageCount != 2 {
		t.Fatalf("model %q id %q pages %v", res.Model, res.ID, res.PageCount)
	}
	if res.Created != 1757000000.0 || len(res.Pages.([]any)) != 2 {
		t.Fatalf("created %v pages %v", res.Created, res.Pages)
	}
	if len(res.Elements) != 1 {
		t.Fatalf("elements %v", res.Elements)
	}
	e := res.Elements[0]
	if e.Label != "title" || !reflect.DeepEqual(e.BBox, []any{10.0, 10.0, 200.0, 40.0}) || e.Content != "Invoice" {
		t.Fatalf("element %+v", e)
	}
	if e.Map()["index"] != 0.0 {
		t.Fatalf("element map %v", e.Map())
	}
}

func TestParseLayoutReportsUsage(t *testing.T) {
	res, _ := parse(t, "https://cdn.example.com/invoice.pdf", nil)
	if *res.InputTokens != 1200 || *res.OutputTokens != 300 || *res.CachedTokens != 100 {
		t.Fatalf("tokens %d %d %d", *res.InputTokens, *res.OutputTokens, *res.CachedTokens)
	}
}

func TestParseLayoutDefaultsToGLMOCRAndOmitsUnsetOptions(t *testing.T) {
	_, srv := parse(t, "https://cdn.example.com/invoice.pdf", nil)
	body := srv.Last(t).JSON(t)
	want := map[string]any{"model": "glm-ocr", "file": "https://cdn.example.com/invoice.pdf"}
	if !reflect.DeepEqual(body, want) {
		t.Fatalf("payload %v", body)
	}
}

func TestParseLayoutForwardsPageRangeOptions(t *testing.T) {
	_, srv := parse(t, "https://cdn.example.com/invoice.pdf", map[string]any{
		"start_page_id": 2, "end_page_id": 5, "need_layout_visualization": true, "ignored": "x",
	})
	body := srv.Last(t).JSON(t)
	if body["start_page_id"] != 2.0 || body["end_page_id"] != 5.0 || body["need_layout_visualization"] != true {
		t.Fatalf("payload %v", body)
	}
	if _, ok := body["ignored"]; ok {
		t.Fatalf("unknown option forwarded: %v", body)
	}
}

func TestParseLayoutEncodesLocalFileAsDataURI(t *testing.T) {
	doc := filepath.Join(t.TempDir(), "scan.PNG")
	content := []byte("\x89PNG\r\n\x1a\nFAKE")
	if err := os.WriteFile(doc, content, 0o644); err != nil {
		t.Fatal(err)
	}
	_, srv := parse(t, doc, nil)
	want := "data:image/png;base64," + base64.StdEncoding.EncodeToString(content)
	if got := srv.Last(t).JSON(t)["file"]; got != want {
		t.Fatalf("file %v", got)
	}
}

func TestParseLayoutUnknownExtensionAndMissingPath(t *testing.T) {
	doc := filepath.Join(t.TempDir(), "scan.tiff")
	if err := os.WriteFile(doc, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	p, err := zhipulayout.RenderLayoutPayload(doc, "glm-ocr", nil)
	if err != nil || p["file"] != "data:application/octet-stream;base64,eA==" {
		t.Fatalf("payload %v err %v", p, err)
	}
	p, _ = zhipulayout.RenderLayoutPayload("/no/such/file.pdf", "glm-ocr", nil)
	if p["file"] != "/no/such/file.pdf" {
		t.Fatalf("missing path should pass through: %v", p)
	}
	p, _ = zhipulayout.RenderLayoutPayload("data:application/pdf;base64,AAAA", "glm-ocr", nil)
	if p["file"] != "data:application/pdf;base64,AAAA" {
		t.Fatalf("data URI should pass through: %v", p)
	}
}

func TestParseLayoutPassesURLsThroughUnchanged(t *testing.T) {
	_, srv := parse(t, "https://cdn.example.com/invoice.pdf", nil)
	if got := srv.Last(t).JSON(t)["file"]; got != "https://cdn.example.com/invoice.pdf" {
		t.Fatalf("file %v", got)
	}
}

func TestParseLayoutUnsupportedProviderErrors(t *testing.T) {
	_, err := gollym.ParseLayout(context.Background(), "https://x/y.pdf", gollym.LayoutRequest{
		ModelOptions: gollym.ModelOptions{Model: "gpt-5.4", Provider: "openai", Context: zhipuContext("http://unused")},
	})
	if !errors.Is(err, gollym.ErrUnsupported) || !strings.Contains(err.Error(), "layout parsing") {
		t.Fatalf("err %v", err)
	}
}

func TestGLMOCRIsRegisteredWithPricing(t *testing.T) {
	info, err := gollym.Models().Find("glm-ocr", "zhipu")
	if err != nil {
		t.Fatal(err)
	}
	if info.Name != "GLM-OCR" {
		t.Fatalf("name %q", info.Name)
	}
	std := info.Pricing.TextTokens()
	if *std.Input() != 0.03 || *std.Output() != 0.03 {
		t.Fatalf("pricing %v %v", *std.Input(), *std.Output())
	}
	found := false
	for _, m := range info.Modalities.Input {
		found = found || m == "pdf"
	}
	if !found {
		t.Fatalf("modalities %v", info.Modalities.Input)
	}
}

// zhipu's default protocol is this one; chat must stay plain Chat
// Completions (Rekryon's JSON-mode glm-5.3 calls).
func TestZhipuChatIsPlainChatCompletions(t *testing.T) {
	completion := map[string]any{
		"model":   "glm-5.3",
		"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": `{"ok": true}`}, "finish_reason": "stop"}},
		"usage":   map[string]any{"prompt_tokens": 10, "completion_tokens": 5},
	}
	srv := testserver.New(t, testserver.Reply{Body: completion})
	chat, err := zhipuContext(srv.URL).NewChat(gollym.ChatOptions{Model: "glm-5.3", Provider: "zhipu", AssumeModelExists: true})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := mustProtocol(t, chat.Provider).(*zhipulayout.Protocol); !ok {
		t.Fatal("zhipu default protocol is not zhipulayout")
	}
	chat.WithInstructions("be terse").WithParams(map[string]any{"max_tokens": 512, "response_format": map[string]any{"type": "json_object"}})
	msg, err := chat.Ask(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	req := srv.Last(t)
	body := req.JSON(t)
	if req.Path != "/chat/completions" || body["model"] != "glm-5.3" || body["max_tokens"] != 512.0 || body["stream"] != false {
		t.Fatalf("path %s payload %v", req.Path, body)
	}
	if body["response_format"].(map[string]any)["type"] != "json_object" {
		t.Fatalf("response_format %v", body["response_format"])
	}
	if body["messages"].([]any)[0].(map[string]any)["role"] != "system" {
		t.Fatalf("system role %v", body["messages"])
	}
	if msg.Text() != `{"ok": true}` || *msg.InputTokens() != 10 {
		t.Fatalf("msg %q", msg.Text())
	}
}

func TestZhipuDefaultBase(t *testing.T) {
	t.Setenv("ZHIPU_API_BASE", "")
	ctx := gollym.NewContext(func(c *gollym.Config) { c.Set("zhipu_api_key", "zp-test"); c.Set("zhipu_api_base", "") })
	_, p, err := core.ResolveModel("glm-5.3", "zhipu", true, ctx.Config)
	if err != nil {
		t.Fatal(err)
	}
	base, err := p.(core.Endpoint).APIBase()
	if err != nil || base != "https://api.z.ai/api/paas/v4" {
		t.Fatalf("base %q err %v", base, err)
	}
	if p.Name() != "Zhipu" || !p.Spec().AssumeModels {
		t.Fatalf("spec %+v", p.Spec())
	}
}

func mustProtocol(t *testing.T, p gollym.Provider) core.Protocol {
	t.Helper()
	proto, err := core.ProtocolFor(p)
	if err != nil {
		t.Fatal(err)
	}
	return proto
}
