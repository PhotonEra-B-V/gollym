// Package zhipulayout is z.ai's Chat Completions plus glm-ocr layout parsing
// (pyllym protocols/zhipu_layout.py).
//
// z.ai serves document OCR from a dedicated layout_parsing endpoint rather
// than chat/completions, returning Markdown text plus positional layout
// elements. Everything else on this protocol is plain Chat Completions.
package zhipulayout

import (
	"context"
	"encoding/base64"
	"os"
	"path/filepath"
	"strings"

	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/protocols/chatcompletions"
)

// mimeTypes are the extensions the endpoint accepts as inline base64 input.
var mimeTypes = map[string]string{
	".pdf":  "application/pdf",
	".png":  "image/png",
	".jpg":  "image/jpeg",
	".jpeg": "image/jpeg",
}

// layoutOptions are forwarded to the endpoint when set.
var layoutOptions = []string{
	"return_crop_images", "need_layout_visualization", "start_page_id",
	"end_page_id", "request_id", "user_id",
}

// Protocol is Chat Completions plus ParseLayout.
type Protocol struct {
	*chatcompletions.Protocol
}

// New is the core.ProtocolFactory.
func New(p core.Provider, model *core.Info) core.Protocol {
	return &Protocol{Protocol: &chatcompletions.Protocol{Provider: p, Model: model}}
}

// LayoutParsingURL is the OCR endpoint, relative to the api base.
func (z *Protocol) LayoutParsingURL() string { return "layout_parsing" }

// ParseLayout runs glm-ocr on a URL, data URI or local file.
func (z *Protocol) ParseLayout(ctx context.Context, file string, model string, opts map[string]any) (*core.LayoutParsing, error) {
	payload, err := RenderLayoutPayload(file, model, opts)
	if err != nil {
		return nil, err
	}
	resp, err := z.Provider.Connection().Post(ctx, z.LayoutParsingURL(), payload, nil)
	if err != nil {
		return nil, err
	}
	return ParseLayoutResponse(resp, model), nil
}

// RenderLayoutPayload builds the layout_parsing body; unset options are
// omitted rather than sent as null.
func RenderLayoutPayload(file string, model string, opts map[string]any) (map[string]any, error) {
	encoded, err := encodeFile(file)
	if err != nil {
		return nil, err
	}
	payload := map[string]any{"model": model, "file": encoded}
	for _, k := range layoutOptions {
		payload[k] = opts[k]
	}
	return core.Compact(payload), nil
}

// ParseLayoutResponse reads a layout_parsing response.
func ParseLayoutResponse(resp *core.Response, model string) *core.LayoutParsing {
	body := resp.BodyMap()
	usage := core.DigMap(body, "usage")
	var elements []core.LayoutElement
	for _, e := range core.ToSafeArray(body["layout_details"]) {
		if m, ok := e.(map[string]any); ok {
			elements = append(elements, core.NewLayoutElement(m))
		}
	}
	text, _ := core.Or(body["md_results"], "").(string)
	modelID, _ := core.Or(body["model"], model).(string)
	id, _ := body["id"].(string)
	return &core.LayoutParsing{
		Text:          text,
		Model:         modelID,
		ID:            id,
		Created:       body["created"],
		Elements:      elements,
		Visualization: body["layout_visualization"],
		PageCount:     core.AsInt(core.Dig(body, "data_info", "num_pages")),
		Pages:         core.Dig(body, "data_info", "pages"),
		InputTokens:   core.AsInt(usage["prompt_tokens"]),
		OutputTokens:  core.AsInt(usage["completion_tokens"]),
		CachedTokens:  core.AsInt(core.Dig(usage, "prompt_tokens_details", "cached_tokens")),
	}
}

// encodeFile passes URLs and pre-encoded data through and reads local paths
// as base64 data URIs. A path that doesn't exist is sent as given.
func encodeFile(file string) (string, error) {
	for _, prefix := range []string{"http://", "https://", "data:"} {
		if strings.HasPrefix(file, prefix) {
			return file, nil
		}
	}
	path := expandHome(file)
	if _, err := os.Stat(path); err != nil {
		return file, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	mime, ok := mimeTypes[strings.ToLower(filepath.Ext(path))]
	if !ok {
		mime = "application/octet-stream"
	}
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data), nil
}

func expandHome(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, p[1:])
		}
	}
	return p
}
