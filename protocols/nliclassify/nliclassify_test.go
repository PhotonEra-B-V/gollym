package nliclassify_test

import (
	"context"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/PhotonEra-B-V/gollym"
	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/internal/testserver"
	"github.com/PhotonEra-B-V/gollym/protocols/nliclassify"
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

func classifyResponse(rows [][]float64, tokens int) map[string]any {
	data := make([]any, len(rows))
	for i, probs := range rows {
		ps := make([]any, len(probs))
		for j, p := range probs {
			ps[j] = p
		}
		data[i] = map[string]any{"index": i, "label": "x", "probs": ps, "num_classes": 3}
	}
	return map[string]any{
		"object": "list", "model": "openjev-4b", "data": data,
		"usage": map[string]any{"prompt_tokens": tokens, "total_tokens": tokens, "completion_tokens": 0},
	}
}

func vllmContext(base, key string) *gollym.Context {
	return gollym.NewContext(func(c *gollym.Config) {
		c.Set("vllm_api_base", base)
		c.Set("vllm_api_key", key)
		c.MaxRetries = 0
		c.RetryInterval = time.Millisecond
		c.RequestTimeout = 5 * time.Second
	})
}

func decide(t *testing.T, srv *testserver.Server, state any, qs *gollym.Questions, model string, opts map[string]any) *gollym.Decision {
	t.Helper()
	t.Setenv("VLLM_API_KEY", "")
	d, err := gollym.Decide(context.Background(), state, qs, gollym.DecideRequest{
		ModelOptions: gollym.ModelOptions{Model: model, Provider: "vllm", Context: vllmContext(srv.URL+"/v1", "")},
		Options:      opts,
	})
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func approxMap(got map[string]float64, want map[string]float64) bool {
	if len(got) != len(want) {
		return false
	}
	for k, v := range want {
		if !approx(got[k], v) {
			return false
		}
	}
	return true
}

func TestOpenJevOnVLLM(t *testing.T) {
	// Rows are [contradiction, entailment, neutral], one per hypothesis in
	// question order: 2 team options, 3 severity levels, 1 yes/no.
	srv := testserver.New(t, testserver.Reply{Body: classifyResponse([][]float64{
		{0.1, 0.6, 0.3}, {0.7, 0.2, 0.1}, {0.8, 0.1, 0.1}, {0.2, 0.3, 0.5}, {0.4, 0.4, 0.2}, {0.25, 0.75, 0.0},
	}, 120)})
	d := decide(t, srv, map[string]any{"ticket": "Charged twice."}, questions(t), "openjev-4b", nil)

	req := srv.Last(t)
	if req.Path != "/classify" || req.Headers.Get("Authorization") != "" {
		t.Fatalf("path %s auth %q", req.Path, req.Headers.Get("Authorization"))
	}
	sent := req.JSON(t)
	inputs := sent["input"].([]any)
	if sent["model"] != "openjev-4b" || len(inputs) != 6 {
		t.Fatalf("payload %v", sent)
	}
	if inputs[0] != "Premise: {\"ticket\": \"Charged twice.\"}\nHypothesis: This is about Charges or refunds." {
		t.Fatalf("input[0] %q", inputs[0])
	}
	if !strings.HasSuffix(inputs[5].(string), "Hypothesis: An account change") {
		t.Fatalf("input[5] %q", inputs[5])
	}

	team := d.Get("team")
	if *team.Choice != "billing" || !approxMap(team.Probabilities, map[string]float64{"billing": 0.75, "technical": 0.25}) || !approx(*team.Confidence, 0.75) {
		t.Fatalf("team %+v", team)
	}
	sev := d.Get("severity")
	if !approxMap(sev.Probabilities, map[string]float64{"0": 0.125, "1": 0.375, "2": 0.5}) || !approx(*sev.Score, 0.375+2*0.5) || sev.Level() != "Material impact" {
		t.Fatalf("severity %+v", sev)
	}
	if !approx(*d.Get("needs_human").Probability, 0.75) || d.InputTokens() != 120 || d.Model != "openjev-4b" {
		t.Fatalf("decision %+v", d)
	}
	if !reflect.DeepEqual(d.Names, []string{"team", "severity", "needs_human"}) {
		t.Fatalf("names %v", d.Names)
	}
}

func TestNLILabelOrderAndTemplatesAreConfigurable(t *testing.T) {
	// bart-large-mnli order: [contradiction, neutral, entailment]
	srv := testserver.New(t, testserver.Reply{Body: classifyResponse([][]float64{{0.1, 0.1, 0.8}, {0.6, 0.3, 0.1}}, 10)})
	intent, _ := core.ChoiceQuestion([]string{"refund", "upgrade"}, nil, "Intent?")
	d := decide(t, srv, "I want my money back", (&gollym.Questions{}).Add("intent", intent), "facebook/bart-large-mnli", map[string]any{
		"nli_labels":          []any{"contradiction", "neutral", "entailment"},
		"hypothesis_template": "{instructions} {option}",
		"pair_template":       "{premise} || {hypothesis}",
	})
	sent := srv.Last(t).JSON(t)
	if !reflect.DeepEqual(sent["input"], []any{"I want my money back || Intent? refund", "I want my money back || Intent? upgrade"}) {
		t.Fatalf("input %v", sent["input"])
	}
	a := d.Get("intent")
	if *a.Choice != "refund" || !approx(a.Probabilities["refund"], 0.8/0.9) {
		t.Fatalf("intent %+v", a)
	}
}

func TestNLIBatchesOverVLLMInputCap(t *testing.T) {
	labels := make([]string, 300)
	for i := range labels {
		labels[i] = fmt.Sprintf("o%d", i)
	}
	first := make([][]float64, 256)
	for i := range first {
		first[i] = []float64{0.5, 0.5, 0}
	}
	second := make([][]float64, 44)
	for i := range second {
		second[i] = []float64{0.5, 0.5, 0}
	}
	second[43] = []float64{0, 1, 0}
	srv := testserver.New(t, testserver.Reply{Body: classifyResponse(first, 5)}, testserver.Reply{Body: classifyResponse(second, 5)})
	q, _ := core.ChoiceQuestion(labels, nil, "")
	d := decide(t, srv, "s", (&gollym.Questions{}).Add("q", q), "openjev-4b", nil)
	var sizes []int
	for _, r := range srv.Requests() {
		sizes = append(sizes, len(r.JSON(t)["input"].([]any)))
	}
	if !reflect.DeepEqual(sizes, []int{256, 44}) {
		t.Fatalf("batch sizes %v", sizes)
	}
	if *d.Get("q").Choice != "o299" || d.InputTokens() != 10 {
		t.Fatalf("choice %v tokens %d", *d.Get("q").Choice, d.InputTokens())
	}
}

func TestNLIRowsAreSortedByIndexAndKeyIsSent(t *testing.T) {
	body := classifyResponse([][]float64{{0.9, 0.1, 0}, {0.1, 0.9, 0}}, 0)
	data := body["data"].([]any)
	data[0], data[1] = data[1], data[0]
	srv := testserver.New(t, testserver.Reply{Body: body})
	q, _ := core.ChoiceQuestion([]string{"a", "b"}, nil, "")
	d, err := gollym.Decide(context.Background(), "s", (&gollym.Questions{}).Add("q", q), gollym.DecideRequest{
		ModelOptions: gollym.ModelOptions{Model: "openjev-4b", Provider: "vllm", Context: vllmContext(srv.URL+"/", "secret")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if *d.Get("q").Choice != "b" || d.Tokens != nil {
		t.Fatalf("choice %v tokens %v", *d.Get("q").Choice, d.Tokens)
	}
	req := srv.Last(t)
	if req.Path != "/classify" || req.Headers.Get("Authorization") != "Bearer secret" {
		t.Fatalf("path %s auth %q", req.Path, req.Headers.Get("Authorization"))
	}
}

func TestNLIYesNoFallsBackToInstructionsAndEvenOdds(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: classifyResponse([][]float64{{0, 0, 1}}, 0)})
	d := decide(t, srv, "s", (&gollym.Questions{}).Add("ok", core.YesNoQuestion("Is it fine?", "", "")), "openjev-4b", nil)
	if got := srv.Last(t).JSON(t)["input"].([]any)[0]; got != "Premise: s\nHypothesis: Is it fine?" {
		t.Fatalf("input %q", got)
	}
	if *d.Get("ok").Probability != 0.5 {
		t.Fatalf("probability %v", *d.Get("ok").Probability)
	}
}

func TestClassifyURL(t *testing.T) {
	for base, want := range map[string]string{
		"http://h:8000/v1":  "http://h:8000/classify",
		"http://h:8000/v1/": "http://h:8000/classify",
		"http://h:8000":     "http://h:8000/classify",
	} {
		_, p, err := core.ResolveModel("m", "vllm", true, vllmContext(base, "").Config)
		if err != nil {
			t.Fatal(err)
		}
		proto, _ := core.ProtocolFor(p)
		got, _ := proto.(*nliclassify.Protocol).ClassifyURL()
		if got != want {
			t.Fatalf("%s -> %s, want %s", base, got, want)
		}
	}
}

// --- test_provider_matrix.py: vLLM chat --------------------------------------

func completion(content string) map[string]any {
	return map[string]any{
		"model":   "some-model",
		"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": content}, "finish_reason": "stop"}},
		"usage":   map[string]any{"prompt_tokens": 10, "completion_tokens": 5},
	}
}

func TestVLLMLocalChat(t *testing.T) {
	t.Setenv("VLLM_API_KEY", "")
	srv := testserver.New(t, testserver.Reply{Body: completion("reply from vllm")})
	// vLLM assumes models: no assume flag needed.
	chat, err := vllmContext(srv.URL+"/v1", "").NewChat(gollym.ChatOptions{Model: "meta-llama/Llama-3.1-8B-Instruct", Provider: "vllm"})
	if err != nil {
		t.Fatal(err)
	}
	msg, err := chat.Ask(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	req := srv.Last(t)
	if req.Path != "/v1/chat/completions" || msg.Text() != "reply from vllm" || *msg.InputTokens() != 10 {
		t.Fatalf("path %s msg %q", req.Path, msg.Text())
	}
	if _, ok := req.Headers["Authorization"]; ok {
		t.Fatalf("unexpected auth header %q", req.Headers.Get("Authorization"))
	}
}

func TestVLLMSystemRole(t *testing.T) {
	srv := testserver.New(t, testserver.Reply{Body: completion("ok")})
	chat, err := vllmContext(srv.URL+"/v1", "").NewChat(gollym.ChatOptions{Model: "some-model", Provider: "vllm", AssumeModelExists: true})
	if err != nil {
		t.Fatal(err)
	}
	chat.WithInstructions("be terse")
	if _, err := chat.Ask(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	if role := core.Dig(srv.Last(t).JSON(t), "messages", 0, "role"); role != "system" {
		t.Fatalf("role %v", role)
	}
}

func TestVLLMSpec(t *testing.T) {
	spec, ok := core.LookupProvider("vllm")
	if !ok || !spec.Local || !spec.AssumeModels || spec.DisplayName() != "VLLM" {
		t.Fatalf("spec %+v", spec)
	}
	if !reflect.DeepEqual(spec.Requirements, []string{"vllm_api_base"}) || !reflect.DeepEqual(spec.Options, []string{"vllm_api_base", "vllm_api_key"}) {
		t.Fatalf("options %v requirements %v", spec.Options, spec.Requirements)
	}
}
