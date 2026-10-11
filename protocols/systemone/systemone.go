// Package systemone is TypeSafe's decision protocol for Jev (pyllym
// protocols/systemone.py).
//
// Jev never writes text: it serves a single POST /v1/systemone endpoint that
// takes a state plus named, typed questions and answers each with a
// calibrated distribution. There is no chat, tools or streaming.
//
//	{"model": "jev-latest", "state": "...",
//	 "questions": {"team": {"type": "choice", "instructions": "...",
//	                        "criteria": {"billing": "...", "technical": "..."}}}}
//	-> {"model": "jev-1.13.0",
//	    "answers": {"team": {"type": "choice", "choice": "billing",
//	                         "probabilities": {...}, "confidence": 0.82}},
//	    "usage": {"input_tokens": 312, "output_tokens": 48}}
//
// Score questions send criteria as an ordered list (2-10 levels) and answer
// with score, legend and probabilities; the yes/no type is spelled "noul" on
// the wire and answers with "noul", the probability of yes.
package systemone

import (
	"context"
	"fmt"
	"sort"

	"github.com/PhotonEra-B-V/gollym/core"
)

// Jev's limits.
const (
	MaxChoiceOptions = 255
	MaxScoreLevels   = 10
)

var wireTypes = map[string]string{core.QuestionChoice: "choice", core.QuestionScore: "score", core.QuestionYesNo: "noul"}

// Protocol speaks systemone for one provider.
type Protocol struct {
	Provider core.Provider
	Model    *core.Info
}

// New is the core.ProtocolFactory.
func New(p core.Provider, model *core.Info) core.Protocol {
	return &Protocol{Provider: p, Model: model}
}

// DecisionURL is the endpoint, relative to the api base.
func (s *Protocol) DecisionURL() string { return "systemone" }

// CompletionURL is unused: Jev cannot chat.
func (s *Protocol) CompletionURL() string { return s.DecisionURL() }

// RenderPayload fails: Jev is a decision model.
func (s *Protocol) RenderPayload(*core.CompletionRequest) (map[string]any, error) {
	return nil, core.NewError(core.KindError, nil, "Jev is a decision model and cannot chat; use gollym.Decide()")
}

// ParseCompletionResponse fails: Jev is a decision model.
func (s *Protocol) ParseCompletionResponse(*core.Response) (*core.Message, error) {
	return nil, core.NewError(core.KindError, nil, "Jev is a decision model and cannot chat; use gollym.Decide()")
}

// Decide sends state and questions and parses the answers. opts["params"]
// is deep-merged over the payload.
func (s *Protocol) Decide(ctx context.Context, state any, questions *core.Questions, model string, opts map[string]any) (*core.Decision, error) {
	payload, err := RenderDecisionPayload(state, questions, model, opts)
	if err != nil {
		return nil, err
	}
	resp, err := s.Provider.Connection().Post(ctx, s.DecisionURL(), payload, nil)
	if err != nil {
		return nil, err
	}
	return ParseDecisionResponse(resp, model, questions), nil
}

// RenderDecisionPayload builds the systemone body.
func RenderDecisionPayload(state any, questions *core.Questions, model string, opts map[string]any) (map[string]any, error) {
	qs := map[string]any{}
	for _, name := range questions.Names {
		q, err := FormatQuestion(questions.ByKey[name])
		if err != nil {
			return nil, err
		}
		qs[name] = q
	}
	payload := map[string]any{"model": model, "state": state, "questions": qs}
	params, _ := opts["params"].(map[string]any)
	return core.DeepMerge(payload, params), nil
}

// FormatQuestion is the question's to_dict form with the wire type, after
// checking Jev's option limits.
func FormatQuestion(q core.Question) (map[string]any, error) {
	count := len(q.Options())
	if q.Type == core.QuestionChoice && count > MaxChoiceOptions {
		return nil, fmt.Errorf("Jev choice questions take at most %d options", MaxChoiceOptions)
	}
	if q.Type == core.QuestionScore && count > MaxScoreLevels {
		return nil, fmt.Errorf("Jev score questions take at most %d levels", MaxScoreLevels)
	}
	wire, ok := wireTypes[q.Type]
	if !ok {
		return nil, fmt.Errorf("Unknown question type: %q", q.Type)
	}
	out := q.Map()
	out["type"] = wire
	return out, nil
}

// ParseDecisionResponse reads the answers. Names follow the question order
// for answered questions, then any extra answers sorted (Go maps lose the
// response's key order).
func ParseDecisionResponse(resp *core.Response, model string, questions *core.Questions) *core.Decision {
	body := resp.BodyMap()
	raw := core.DigMap(body, "answers")
	answers := map[string]core.Answer{}
	for name, data := range raw {
		if m, ok := data.(map[string]any); ok {
			answers[name] = ParseAnswer(m)
		}
	}
	var names []string
	seen := map[string]bool{}
	if questions != nil {
		for _, n := range questions.Names {
			if _, ok := answers[n]; ok {
				names = append(names, n)
				seen[n] = true
			}
		}
	}
	var extra []string
	for n := range answers {
		if !seen[n] {
			extra = append(extra, n)
		}
	}
	sort.Strings(extra)
	usage := core.DigMap(body, "usage")
	modelID, _ := core.Or(body["model"], model).(string)
	return &core.Decision{
		Answers: answers,
		Names:   append(names, extra...),
		Model:   modelID,
		Tokens:  core.BuildTokens(core.AsInt(usage["input_tokens"]), core.AsInt(usage["output_tokens"]), nil, nil, nil),
	}
}

// ParseAnswer reads one wire answer; noul (or null) answers carry P(yes)
// under their type name.
func ParseAnswer(data map[string]any) core.Answer {
	if t := data["type"]; t == "noul" || t == "null" {
		p, ok := data["noul"]
		if !ok {
			p = data["null"]
		}
		return core.BuildAnswer(map[string]any{"type": core.QuestionYesNo, "probability": p})
	}
	return core.BuildAnswer(data)
}
