//go:build live

package systemone_test

import (
	"context"
	"os"
	"testing"

	"github.com/PhotonEra-B-V/gollym"
)

// Live smoke test against TypeSafe: go test -tags live ./protocols/systemone/

func TestLiveJevDecide(t *testing.T) {
	key := os.Getenv("TYPESAFE_API_KEY")
	if key == "" {
		t.Skip("TYPESAFE_API_KEY not set")
	}
	ctx := gollym.NewContext(func(c *gollym.Config) { c.Set("typesafe_api_key", key) })
	d, err := gollym.Decide(context.Background(), "Charged twice for one order.", questions(t), gollym.DecideRequest{
		ModelOptions: gollym.ModelOptions{Context: ctx},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range d.Names {
		t.Logf("%s: %v", name, d.Get(name).Map())
	}
	if len(d.Answers) != 3 {
		t.Fatalf("answers %v", d.Answers)
	}
}
