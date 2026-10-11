//go:build live

package fal_test

import (
	"context"
	"os"
	"testing"

	"github.com/PhotonEra-B-V/gollym"
)

// Live smoke test against fal.ai: go test -tags live ./protocols/fal/

func TestLiveFalPaint(t *testing.T) {
	key := os.Getenv("FAL_KEY")
	if key == "" {
		t.Skip("FAL_KEY not set")
	}
	ctx := gollym.NewContext(func(c *gollym.Config) { c.Set("fal_api_key", key) })
	img, err := gollym.Paint(context.Background(), "a red panda, watercolor", gollym.PaintRequest{
		ModelOptions: gollym.ModelOptions{Model: "fal-ai/flux/schnell", Provider: "fal", Context: ctx},
		Size:         "512x512",
	})
	if err != nil {
		t.Fatal(err)
	}
	if img.URL == "" && img.Data == "" {
		t.Fatalf("image %+v", img)
	}
	t.Logf("image %s (%s)", img.URL, img.MIMEType)
}
