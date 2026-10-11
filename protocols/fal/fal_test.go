package fal_test

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym"
	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
	"github.com/PhotonEra-B-V/gollym/protocols/fal"
)

func falContext(apiBase, queueBase string) *gollym.Context {
	return gollym.NewContext(func(c *gollym.Config) {
		c.Set("fal_api_key", "fal-test")
		c.Set("fal_api_base", apiBase)
		c.Set("fal_queue_base", queueBase)
		c.MaxRetries = 0
		c.RetryInterval = time.Millisecond
		c.RequestTimeout = 5 * time.Second
	})
}

func fastPolling(t *testing.T) {
	prev := fal.PollInterval
	fal.PollInterval = 0
	t.Cleanup(func() { fal.PollInterval = prev })
}

func TestFalImagePaint(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: map[string]any{
		"images": []any{map[string]any{"url": "https://cdn.fal/img.png", "content_type": "image/png"}},
		"prompt": "a red panda", "seed": 42,
	}})
	img, err := gollym.Paint(context.Background(), "a red panda", gollym.PaintRequest{
		ModelOptions: gollym.ModelOptions{Model: "fal-ai/flux/dev", Provider: "fal", Context: falContext(srv.URL, "")},
		Size:         "1024x768",
	})
	if err != nil {
		t.Fatal(err)
	}
	req := srv.Last(t)
	if req.Path != "/fal-ai/flux/dev" || req.Headers.Get("Authorization") != "Key fal-test" {
		t.Fatalf("path %s auth %q", req.Path, req.Headers.Get("Authorization"))
	}
	want := map[string]any{"prompt": "a red panda", "image_size": map[string]any{"width": 1024.0, "height": 768.0}}
	if got := req.JSON(t); !reflect.DeepEqual(got, want) {
		t.Fatalf("payload %v", got)
	}
	if img.URL != "https://cdn.fal/img.png" || img.MIMEType != "image/png" || img.RevisedPrompt != "a red panda" || img.ModelID != "fal-ai/flux/dev" {
		t.Fatalf("image %+v", img)
	}
	if !reflect.DeepEqual(img.Usage, map[string]any{"prompt": "a red panda", "seed": 42.0}) {
		t.Fatalf("usage %v", img.Usage)
	}
}

func TestFalPaintPresetSizeParamsAndReference(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: map[string]any{"images": []any{map[string]any{"b64_json": "aGk="}}}})
	img, err := gollym.Paint(context.Background(), "p", gollym.PaintRequest{
		ModelOptions: gollym.ModelOptions{Model: "/fal-ai/qwen-image", Provider: "fal", Context: falContext(srv.URL, "")},
		Size:         "landscape_16_9", With: "https://ref/img.png",
		Params: map[string]any{"num_inference_steps": 4},
	})
	if err != nil {
		t.Fatal(err)
	}
	req := srv.Last(t)
	want := map[string]any{"prompt": "p", "image_size": "landscape_16_9", "num_inference_steps": 4.0, "image_url": "https://ref/img.png"}
	if got := req.JSON(t); req.Path != "/fal-ai/qwen-image" || !reflect.DeepEqual(got, want) {
		t.Fatalf("path %s payload %v", req.Path, got)
	}
	if img.Data != "aGk=" || img.MIMEType != "image/png" || !img.IsBase64() {
		t.Fatalf("image %+v", img)
	}
}

func TestFalPaintWithoutImagesErrors(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: map[string]any{"images": []any{}}})
	_, err := gollym.Paint(context.Background(), "p", gollym.PaintRequest{
		ModelOptions: gollym.ModelOptions{Model: "fal-ai/flux/dev", Provider: "fal", Context: falContext(srv.URL, "")},
	})
	if !errors.Is(err, gollym.ErrAPI) || err.Error() != "fal image response contained no images" {
		t.Fatalf("err %v", err)
	}
}

func TestFalVideoAnimateQueuePoll(t *testing.T) {
	fastPolling(t)
	jobs := testserver.New(t,
		testserver.Reply{Body: map[string]any{"status": "IN_PROGRESS"}},
		testserver.Reply{Body: map[string]any{"status": "COMPLETED"}},
		testserver.Reply{Body: map[string]any{"video": map[string]any{"url": "https://cdn.fal/out.mp4", "content_type": "video/mp4"}}},
	)
	queue := testserver.New(t, testserver.Reply{Body: map[string]any{
		"request_id": "r1", "status_url": jobs.URL + "/r1/status", "response_url": jobs.URL + "/r1",
	}})
	// Animate defaults to provider fal.
	video, err := gollym.Animate(context.Background(), "a timelapse sunrise", gollym.AnimateRequest{
		ModelOptions: gollym.ModelOptions{Model: "fal-ai/ltx-video-13b-distilled", Context: falContext("", queue.URL)},
		Params:       map[string]any{"num_frames": 121},
	})
	if err != nil {
		t.Fatal(err)
	}
	submit := queue.Last(t)
	if submit.Path != "/fal-ai/ltx-video-13b-distilled" || submit.Headers.Get("Authorization") != "Key fal-test" {
		t.Fatalf("submit %s auth %q", submit.Path, submit.Headers.Get("Authorization"))
	}
	if got := submit.JSON(t); !reflect.DeepEqual(got, map[string]any{"prompt": "a timelapse sunrise", "num_frames": 121.0}) {
		t.Fatalf("submit payload %v", got)
	}
	var paths []string
	for _, r := range jobs.Requests() {
		if r.Method != "GET" || r.Headers.Get("Authorization") != "Key fal-test" {
			t.Fatalf("job request %s auth %q", r.Method, r.Headers.Get("Authorization"))
		}
		paths = append(paths, r.Path)
	}
	// Polled until COMPLETED, then fetched the result.
	if !reflect.DeepEqual(paths, []string{"/r1/status", "/r1/status", "/r1"}) {
		t.Fatalf("job requests %v", paths)
	}
	if video.URL != "https://cdn.fal/out.mp4" || video.MIMEType != "video/mp4" || video.ModelID != "fal-ai/ltx-video-13b-distilled" || video.Raw == nil {
		t.Fatalf("video %+v", video)
	}
}

func TestFalQueueFallsBackToAPIBase(t *testing.T) {
	fastPolling(t)
	jobs := testserver.New(t,
		testserver.Reply{Body: map[string]any{"status": "COMPLETED"}},
		testserver.Reply{Body: map[string]any{"videos": []any{map[string]any{"url": "https://cdn.fal/v.mp4"}}}},
	)
	gateway := testserver.New(t, testserver.Reply{Body: map[string]any{"status_url": jobs.URL + "/s", "response_url": jobs.URL + "/r"}})
	video, err := gollym.Animate(context.Background(), "p", gollym.AnimateRequest{
		ModelOptions: gollym.ModelOptions{Model: "fal-ai/wan-t2v", Context: falContext(gateway.URL, "")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if gateway.Last(t).Path != "/fal-ai/wan-t2v" || video.URL != "https://cdn.fal/v.mp4" || video.MIMEType != "video/mp4" {
		t.Fatalf("path %s video %+v", gateway.Last(t).Path, video)
	}
}

func TestFalQueueErrors(t *testing.T) {
	fastPolling(t)
	queue := testserver.New(t, testserver.Reply{Body: map[string]any{"request_id": "r1"}})
	_, err := gollym.Animate(context.Background(), "p", gollym.AnimateRequest{
		ModelOptions: gollym.ModelOptions{Model: "fal-ai/wan-t2v", Context: falContext("", queue.URL)},
	})
	if err == nil || err.Error() != "fal queue submission did not return poll URLs" {
		t.Fatalf("err %v", err)
	}

	jobs := testserver.New(t, testserver.Reply{Body: map[string]any{"status": "FAILED"}})
	queue = testserver.New(t, testserver.Reply{Body: map[string]any{"status_url": jobs.URL + "/s", "response_url": jobs.URL + "/r"}})
	_, err = gollym.Animate(context.Background(), "p", gollym.AnimateRequest{
		ModelOptions: gollym.ModelOptions{Model: "fal-ai/wan-t2v", Context: falContext("", queue.URL)},
	})
	if err == nil || err.Error() != "fal job failed" {
		t.Fatalf("err %v", err)
	}

	jobs = testserver.New(t, testserver.Reply{Body: map[string]any{"status": "COMPLETED"}}, testserver.Reply{Body: map[string]any{"video": map[string]any{}}})
	queue = testserver.New(t, testserver.Reply{Body: map[string]any{"status_url": jobs.URL + "/s", "response_url": jobs.URL + "/r"}})
	_, err = gollym.Animate(context.Background(), "p", gollym.AnimateRequest{
		ModelOptions: gollym.ModelOptions{Model: "fal-ai/wan-t2v", Context: falContext("", queue.URL)},
	})
	if err == nil || err.Error() != "fal video response contained no video" {
		t.Fatalf("err %v", err)
	}
}

func TestFalQueueTimesOut(t *testing.T) {
	prev := fal.PollInterval
	fal.PollInterval = 20 * time.Millisecond
	t.Cleanup(func() { fal.PollInterval = prev })
	jobs := testserver.New(t, testserver.Reply{Body: map[string]any{"status": "IN_QUEUE"}})
	queue := testserver.New(t, testserver.Reply{Body: map[string]any{"status_url": jobs.URL + "/s", "response_url": jobs.URL + "/r"}})
	ctx := falContext("", queue.URL)
	ctx.Config.RequestTimeout = 100 * time.Millisecond
	_, err := gollym.Animate(context.Background(), "p", gollym.AnimateRequest{
		ModelOptions: gollym.ModelOptions{Model: "fal-ai/wan-t2v", Context: ctx},
	})
	if err == nil || err.Error() != "fal job timed out before completing" {
		t.Fatalf("err %v", err)
	}
	// RequestTimeout / PollInterval polls.
	if n := len(jobs.Requests()); n != 5 {
		t.Fatalf("%d polls, want 5", n)
	}
}

func TestSizePayload(t *testing.T) {
	cases := map[string]map[string]any{
		"":           {},
		"1024x1024":  {"image_size": map[string]any{"width": 1024, "height": 1024}},
		"square_hd":  {"image_size": "square_hd"},
		"x768":       {"image_size": "x768"},
		"1024x":      {"image_size": "1024x"},
		"10a24x768":  {"image_size": "10a24x768"},
		"1024x768x2": {"image_size": "1024x768x2"},
	}
	for size, want := range cases {
		if got := fal.SizePayload(size); !reflect.DeepEqual(got, want) {
			t.Fatalf("%q -> %v, want %v", size, got, want)
		}
	}
}

func TestFalSpecAndNoChat(t *testing.T) {
	t.Setenv("FAL_API_BASE", "")
	spec, ok := core.LookupProvider("fal")
	if !ok || !spec.AssumeModels || spec.Local || spec.DisplayName() != "Fal" {
		t.Fatalf("spec %+v", spec)
	}
	if !reflect.DeepEqual(spec.Options, []string{"fal_api_key", "fal_api_base", "fal_queue_base"}) || !reflect.DeepEqual(spec.Requirements, []string{"fal_api_key"}) {
		t.Fatalf("options %v requirements %v", spec.Options, spec.Requirements)
	}
	ctx := falContext("", "")
	_, p, err := core.ResolveModel("fal-ai/flux/dev", "fal", false, ctx.Config)
	if err != nil {
		t.Fatal(err)
	}
	if base, _ := p.(core.Endpoint).APIBase(); base != "https://fal.run" {
		t.Fatalf("base %q", base)
	}
	chat, err := ctx.NewChat(gollym.ChatOptions{Model: "fal-ai/flux/dev", Provider: "fal"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := chat.Ask(context.Background(), "hi"); !errors.Is(err, gollym.ErrUnsupported) {
		t.Fatalf("err %v", err)
	}
}
