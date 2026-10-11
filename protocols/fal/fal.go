// Package fal is the fal.ai image and video generation protocol (pyllym
// protocols/fal.py).
//
// fal serves models behind two shapes:
//
//   - Synchronous run (https://fal.run/{model}), used for image models
//     (FLUX.2, HunyuanImage, Qwen-Image, ...), which return quickly.
//   - Queue (https://queue.fal.run/{model}): submit returns a status_url and
//     response_url; we poll until COMPLETED, then fetch the result. Used for
//     video models (LTX, Wan, HunyuanVideo, ...), which take longer.
package fal

import (
	"context"
	"strconv"
	"strings"
	"time"

	"github.com/PhotonEra-B-V/gollym/core"
)

// QueueBase is the public queue endpoint.
const QueueBase = "https://queue.fal.run"

// PollInterval is the queue poll cadence (a variable so tests can speed it
// up, like pyllym's Fal.poll_interval).
var PollInterval = time.Second

var terminalBad = map[string]bool{"FAILED": true, "ERROR": true, "CANCELLED": true}

// Protocol speaks fal for one provider.
type Protocol struct {
	Provider core.Provider
	Model    *core.Info
}

// New is the core.ProtocolFactory.
func New(p core.Provider, model *core.Info) core.Protocol {
	return &Protocol{Provider: p, Model: model}
}

func (f *Protocol) conn() *core.Connection { return f.Provider.Connection() }

// fal has no chat endpoint.
func (f *Protocol) CompletionURL() string { return "" }

func (f *Protocol) RenderPayload(*core.CompletionRequest) (map[string]any, error) {
	return nil, core.NotSupported("fal does not support chat")
}

func (f *Protocol) ParseCompletionResponse(*core.Response) (*core.Message, error) {
	return nil, core.NotSupported("fal does not support chat")
}

// --- image (synchronous run) --------------------------------------------------

// Paint runs an image model synchronously. With sets image_url unless the
// params already do; Mask is ignored, as in pyllym.
func (f *Protocol) Paint(ctx context.Context, prompt string, o core.PaintOptions) (*core.Image, error) {
	payload := map[string]any{"prompt": prompt}
	for k, v := range SizePayload(o.Size) {
		payload[k] = v
	}
	for k, v := range o.Params {
		payload[k] = v
	}
	if o.With != nil {
		if _, ok := payload["image_url"]; !ok {
			payload["image_url"] = o.With
		}
	}
	resp, err := f.conn().Post(ctx, "/"+strings.TrimLeft(o.Model, "/"), payload, nil)
	if err != nil {
		return nil, err
	}
	return ParseImage(resp.BodyMap(), o.Model)
}

// ParseImage reads the first image of a run response.
func ParseImage(data map[string]any, model string) (*core.Image, error) {
	first := core.DigMap(data, "images", 0)
	if len(first) == 0 {
		return nil, core.NewError(core.KindError, nil, "fal image response contained no images")
	}
	usage := map[string]any{}
	for k, v := range data {
		if k != "images" {
			usage[k] = v
		}
	}
	url, _ := first["url"].(string)
	b64, _ := first["b64_json"].(string)
	mime, _ := core.Or(first["content_type"], "image/png").(string)
	prompt, _ := data["prompt"].(string)
	return &core.Image{URL: url, Data: b64, MIMEType: mime, RevisedPrompt: prompt, ModelID: model, Usage: usage}, nil
}

// --- video (queue + poll) -----------------------------------------------------

// Animate submits a video job to the queue, polls it to completion and
// fetches the result.
func (f *Protocol) Animate(ctx context.Context, prompt string, model string, params map[string]any) (*core.Video, error) {
	payload := map[string]any{"prompt": prompt}
	for k, v := range params {
		payload[k] = v
	}
	submit, err := f.conn().Post(ctx, f.queueBase()+"/"+strings.TrimLeft(model, "/"), payload, nil)
	if err != nil {
		return nil, err
	}
	statusURL, _ := core.Dig(submit.Body, "status_url").(string)
	responseURL, _ := core.Dig(submit.Body, "response_url").(string)
	if statusURL == "" || responseURL == "" {
		return nil, core.NewError(core.KindError, submit, "fal queue submission did not return poll URLs")
	}
	if err := f.awaitCompletion(ctx, statusURL); err != nil {
		return nil, err
	}
	result, err := f.conn().Get(ctx, responseURL, nil)
	if err != nil {
		return nil, err
	}
	return ParseVideo(result.BodyMap(), model)
}

// queueBase honours gateway overrides like the sync base does: fal_queue_base
// wins, else a custom fal_api_base (proxy / self-hosted gateway), else the
// public queue.
func (f *Protocol) queueBase() string {
	cfg := f.Provider.Config()
	if v := cfg.Get("fal_queue_base"); v != "" {
		return v
	}
	if v := cfg.Get("fal_api_base"); v != "" {
		return v
	}
	return QueueBase
}

// awaitCompletion polls status_url about RequestTimeout / PollInterval times.
func (f *Protocol) awaitCompletion(ctx context.Context, statusURL string) error {
	interval := max(PollInterval, 10*time.Millisecond)
	attempts := max(1, int(f.Provider.Config().RequestTimeout/interval))
	for range attempts {
		resp, err := f.conn().Get(ctx, statusURL, nil)
		if err != nil {
			return err
		}
		status, _ := core.Dig(resp.Body, "status").(string)
		if status == "COMPLETED" {
			return nil
		}
		if terminalBad[status] {
			return core.NewError(core.KindError, resp, "fal job "+strings.ToLower(status))
		}
		if PollInterval > 0 {
			t := time.NewTimer(PollInterval)
			select {
			case <-ctx.Done():
				t.Stop()
				return ctx.Err()
			case <-t.C:
			}
		}
	}
	return core.NewError(core.KindError, nil, "fal job timed out before completing")
}

// ParseVideo reads the video (or first of videos) of a queue result.
func ParseVideo(data map[string]any, model string) (*core.Video, error) {
	video := core.DigMap(data, "video")
	if len(video) == 0 {
		video = core.DigMap(data, "videos", 0)
	}
	url, _ := video["url"].(string)
	if url == "" {
		return nil, core.NewError(core.KindError, nil, "fal video response contained no video")
	}
	mime, _ := core.Or(video["content_type"], "video/mp4").(string)
	return &core.Video{URL: url, MIMEType: mime, ModelID: model, Raw: data}, nil
}

// SizePayload maps an OpenAI-style "WxH" size to fal's image_size; any other
// string (e.g. "landscape_16_9") passes through as a fal size preset.
func SizePayload(size string) map[string]any {
	if size == "" {
		return map[string]any{}
	}
	if w, h, ok := strings.Cut(size, "x"); ok && isDigits(w) && isDigits(h) {
		width, _ := strconv.Atoi(w)
		height, _ := strconv.Atoi(h)
		return map[string]any{"image_size": map[string]any{"width": width, "height": height}}
	}
	return map[string]any{"image_size": size}
}

func isDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}
