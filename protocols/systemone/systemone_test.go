package systemone_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym"
	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
)

func questions(t *testing.T) *gollym.Questions {
	t.Helper()
	team, err := core.ChoiceQuestion([]string{"billing", "technical"},
		map[string]string{"billing": "Charges or refunds", "technical": "Product errors"}, "Which team should investigate?")
	if err != nil {
		t.Fatal(err)
	}
	severity, err := core.ScoreQuestion([]string{"No impact", "Limited impact", "Material impact"}, "")
	if err != nil {
		t.Fatal(err)
	}
	qs := &gollym.Questions{}
	qs.Add("team", team).Add("severity", severity).Add("needs_human", core.YesNoQuestion("Does this need a human?", "An account change", ""))
	return qs
}

func jevResponse() map[string]any {
	return map[string]any{
		"model": "jev-1.13.0",
		"answers": map[string]any{
			"team": map[string]any{"type": "choice", "choice": "billing",
				"probabilities": map[string]any{"billing": 0.85, "technical": 0.15}, "confidence": 0.82},
			"severity": map[string]any{"type": "score", "score": 1.2,
				"legend":        map[string]any{"0": "No impact", "1": "Limited impact", "2": "Material impact"},
				"probabilities": map[string]any{"0": 0.1, "1": 0.6, "2": 0.3}, "confidence": 0.7},
			"needs_human": map[string]any{"type": "noul", "noul": 0.91},
		},
		"usage": map[string]any{"input_tokens": 312, "output_tokens": 48},
	}
}

func typesafeContext(base string) *gollym.Context {
	return gollym.NewContext(func(c *gollym.Config) {
		c.Set("typesafe_api_key", "ts-test")
		c.Set("typesafe_api_base", base)
		c.Set("openai_api_key", "sk-test")
		c.MaxRetries = 0
		c.RetryInterval = time.Millisecond
		c.RequestTimeout = 5 * time.Second
	})
}

func decide(t *testing.T, srv *testserver.Server, state any, qs *gollym.Questions, model string) (*gollym.Decision, error) {
	t.Helper()
	return gollym.Decide(context.Background(), state, qs, gollym.DecideRequest{
		ModelOptions: gollym.ModelOptions{Model: model, Context: typesafeContext(srv.URL)},
	})
}

func TestJevDecideRoundTrip(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: jevResponse()})
	d, err := decide(t, srv, "Charged twice for one order.", questions(t), "")
	if err != nil {
		t.Fatal(err)
	}
	req := srv.Last(t)
	if req.Path != "/systemone" || req.Headers.Get("Authorization") != "Bearer ts-test" {
		t.Fatalf("path %s auth %q", req.Path, req.Headers.Get("Authorization"))
	}
	want := map[string]any{
		"model": "jev-latest",
		"state": "Charged twice for one order.",
		"questions": map[string]any{
			"team": map[string]any{"type": "choice", "instructions": "Which team should investigate?",
				"criteria": map[string]any{"billing": "Charges or refunds", "technical": "Product errors"}},
			"severity":    map[string]any{"type": "score", "criteria": []any{"No impact", "Limited impact", "Material impact"}},
			"needs_human": map[string]any{"type": "noul", "instructions": "Does this need a human?", "criteria": map[string]any{"true": "An account change"}},
		},
	}
	if got := req.JSON(t); !reflect.DeepEqual(got, want) {
		t.Fatalf("payload\n got %v\nwant %v", got, want)
	}
	if d.Model != "jev-1.13.0" || d.InputTokens() != 312 || d.OutputTokens() != 48 {
		t.Fatalf("decision %+v", d)
	}
	team := d.Get("team")
	if *team.Choice != "billing" || team.Value() != "billing" || *team.Confidence != 0.82 {
		t.Fatalf("team %+v", team)
	}
	sev := d.Get("severity")
	if *sev.Score != 1.2 || sev.Level() != "Limited impact" {
		t.Fatalf("severity %+v", sev)
	}
	nh := d.Get("needs_human")
	if nh.Type != "yes_no" || *nh.Probability != 0.91 || !*nh.Yes() {
		t.Fatalf("needs_human %+v", nh)
	}
	if !reflect.DeepEqual(d.Names, []string{"team", "severity", "needs_human"}) {
		t.Fatalf("names %v", d.Names)
	}
}

func TestJevAcceptsDictQuestionsAndNullSpelling(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: map[string]any{
		"model": "jev-latest", "answers": map[string]any{"ok": map[string]any{"type": "null", "null": 0.2}},
	}})
	q, err := core.BuildQuestion(map[string]any{"type": "noul", "instructions": "Is it fine?"})
	if err != nil {
		t.Fatal(err)
	}
	d, err := decide(t, srv, "state", (&gollym.Questions{}).Add("ok", q), "jev-preview")
	if err != nil {
		t.Fatal(err)
	}
	sent := srv.Last(t).JSON(t)
	if sent["model"] != "jev-preview" || core.Dig(sent, "questions", "ok", "type") != "noul" {
		t.Fatalf("payload %v", sent)
	}
	ok := d.Get("ok")
	if *ok.Probability != 0.2 || *ok.Yes() || d.Tokens != nil {
		t.Fatalf("answer %+v tokens %v", ok, d.Tokens)
	}
}

func TestJevUnregisteredVersionAndCustomBase(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: jevResponse()})
	d, err := gollym.Decide(context.Background(), "s", questions(t), gollym.DecideRequest{
		ModelOptions: gollym.ModelOptions{Model: "jev-2.0.0", Provider: "typesafe", Context: typesafeContext(srv.URL + "/typesafe/v1")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if p := srv.Last(t).Path; p != "/typesafe/v1/systemone" {
		t.Fatalf("path %s", p)
	}
	if srv.Last(t).JSON(t)["model"] != "jev-2.0.0" || *d.Get("team").Choice != "billing" {
		t.Fatalf("decision %+v", d)
	}
}

func TestJevParamsAreDeepMerged(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: jevResponse()})
	_, err := gollym.Decide(context.Background(), map[string]any{"ticket": 1}, questions(t), gollym.DecideRequest{
		ModelOptions: gollym.ModelOptions{Context: typesafeContext(srv.URL)},
		Options:      map[string]any{"params": map[string]any{"seed": 7, "questions": map[string]any{"team": map[string]any{"instructions": "Who?"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	sent := srv.Last(t).JSON(t)
	if sent["seed"] != 7.0 || core.Dig(sent, "questions", "team", "instructions") != "Who?" || core.Dig(sent, "questions", "team", "type") != "choice" {
		t.Fatalf("payload %v", sent)
	}
	if !reflect.DeepEqual(sent["state"], map[string]any{"ticket": 1.0}) {
		t.Fatalf("state %v", sent["state"])
	}
}

func TestJevRejectsTooManyOptions(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: jevResponse()})
	labels := make([]string, 256)
	for i := range labels {
		labels[i] = fmt.Sprintf("o%d", i)
	}
	choice, _ := core.ChoiceQuestion(labels, nil, "")
	if _, err := decide(t, srv, "s", (&gollym.Questions{}).Add("q", choice), ""); err == nil || !strings.Contains(err.Error(), "255") {
		t.Fatalf("err %v", err)
	}
	levels := make([]string, 11)
	for i := range levels {
		levels[i] = fmt.Sprint(i)
	}
	score, _ := core.ScoreQuestion(levels, "")
	if _, err := decide(t, srv, "s", (&gollym.Questions{}).Add("q", score), ""); err == nil || !strings.Contains(err.Error(), "10 levels") {
		t.Fatalf("err %v", err)
	}
	if n := len(srv.Requests()); n != 0 {
		t.Fatalf("%d requests sent", n)
	}
}

func TestJevCannotChat(t *testing.T) {
	srv := testserver.New(t)
	chat, err := typesafeContext(srv.URL).NewChat(gollym.ChatOptions{Model: "jev-latest"})
	if err != nil {
		t.Fatal(err)
	}
	_, err = chat.Ask(context.Background(), "hi")
	if !errors.Is(err, gollym.ErrAPI) || !strings.Contains(err.Error(), "Decide") {
		t.Fatalf("err %v", err)
	}
	_, err = chat.Ask(context.Background(), "hi", gollym.OnChunk(func(*gollym.Chunk) error { return nil }))
	if err == nil || !strings.Contains(err.Error(), "Decide") {
		t.Fatalf("stream err %v", err)
	}
}

func TestProvidersWithoutDecisionsSaySo(t *testing.T) {
	srv := testserver.New(t)
	_, err := decide(t, srv, "s", questions(t), "gpt-5.4")
	if !errors.Is(err, gollym.ErrUnsupported) || !strings.Contains(err.Error(), "does not support decisions") {
		t.Fatalf("err %v", err)
	}
}

func TestDecideNeedsQuestions(t *testing.T) {
	srv := testserver.New(t)
	if _, err := decide(t, srv, "s", &gollym.Questions{}, ""); err == nil || !strings.Contains(err.Error(), "at least one question") {
		t.Fatalf("err %v", err)
	}
}

func TestTypeSafeSpec(t *testing.T) {
	t.Setenv("TYPESAFE_API_BASE", "")
	ctx := gollym.NewContext(func(c *gollym.Config) { c.Set("typesafe_api_key", "ts-test") })
	_, p, err := core.ResolveModel("jev-9.9.9", "typesafe", false, ctx.Config)
	if err != nil {
		t.Fatal(err)
	}
	base, _ := p.(core.Endpoint).APIBase()
	if base != "https://api.typesafe.ai/v1" || p.Name() != "TypeSafe" {
		t.Fatalf("base %q name %q", base, p.Name())
	}
	if name, _ := p.ProtocolName("", nil); name != "systemone" {
		t.Fatalf("protocol %q", name)
	}
	if !reflect.DeepEqual(p.Spec().Requirements, []string{"typesafe_api_key"}) {
		t.Fatalf("requirements %v", p.Spec().Requirements)
	}
}

// --- value objects (test_decisions.py) ------------------------------------------

func TestQuestionValidation(t *testing.T) {
	if _, err := core.ChoiceQuestion([]string{"only"}, nil, ""); err == nil || !strings.Contains(err.Error(), "at least two") {
		t.Fatalf("choice err %v", err)
	}
	if _, err := core.ScoreQuestion([]string{"one"}, ""); err == nil || !strings.Contains(err.Error(), "at least two") {
		t.Fatalf("score err %v", err)
	}
	if _, err := core.BuildQuestion(map[string]any{"type": "essay"}); err == nil || !strings.Contains(err.Error(), "Unknown question type") {
		t.Fatalf("build err %v", err)
	}
}

func TestQuestionRoundTrip(t *testing.T) {
	qs := questions(t)
	for _, name := range qs.Names {
		q := qs.ByKey[name]
		back, err := core.BuildQuestion(q.Map())
		if err != nil || !reflect.DeepEqual(back, q) {
			t.Fatalf("%s: %+v != %+v (%v)", name, back, q, err)
		}
	}
	ab, _ := core.ChoiceQuestion([]string{"a", "b"}, nil, "")
	if !reflect.DeepEqual(ab.CriteriaValue(), map[string]any{"a": "a", "b": "b"}) {
		t.Fatalf("criteria %v", ab.CriteriaValue())
	}
	lohi, _ := core.ScoreQuestion([]string{"lo", "hi"}, "")
	if !reflect.DeepEqual(lohi.Options(), []string{"lo", "hi"}) {
		t.Fatalf("options %v", lohi.Options())
	}
	if m := core.YesNoQuestion("ok?", "", "").Map(); !reflect.DeepEqual(m, map[string]any{"type": "yes_no", "instructions": "ok?"}) {
		t.Fatalf("yes_no map %v", m)
	}
}

func TestAnswerRoundTrip(t *testing.T) {
	a := core.Answer{Type: "score", Score: core.Float(1.5), Probabilities: map[string]float64{"0": 0.5, "1": 0.5}, Legend: map[string]string{"0": "lo"}}
	m := a.Map()
	// to_dict -> JSON -> build, as a stored answer would travel.
	back := core.BuildAnswer(roundTripJSON(t, m))
	if !reflect.DeepEqual(back, a) {
		t.Fatalf("%+v != %+v", back, a)
	}
	d := &gollym.Decision{Answers: map[string]gollym.Answer{"s": a}, Model: "jev-latest", Tokens: &gollym.Tokens{Input: core.Int(3)}}
	if d.OutputTokens() != 0 || d.InputTokens() != 3 {
		t.Fatalf("tokens %d %d", d.InputTokens(), d.OutputTokens())
	}
}

func TestJevModelsAreRegistered(t *testing.T) {
	m, err := gollym.Models().Find("jev-latest", "")
	if err != nil {
		t.Fatal(err)
	}
	if m.Provider != "typesafe" || m.Type() != "decision" {
		t.Fatalf("provider %q type %q", m.Provider, m.Type())
	}
	ids := map[string]bool{}
	for _, i := range gollym.Models().DecisionModels().All() {
		ids[i.ID] = true
	}
	if !ids["jev-latest"] || !ids["jev-1.13.0"] {
		t.Fatalf("decision models %v", ids)
	}
	if core.NewConfig().DefaultDecisionModel != "jev-latest" {
		t.Fatal("default decision model")
	}
}

func roundTripJSON(t *testing.T, m map[string]any) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal([]byte(core.MustJSON(m)), &out); err != nil {
		t.Fatal(err)
	}
	return out
}
