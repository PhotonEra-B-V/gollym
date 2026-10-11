// Package nliclassify answers Decide with an open NLI classifier served by
// vLLM, e.g. OpenJev (pyllym protocols/nli_classify.py).
//
// NLI classifiers take a (premise, hypothesis) pair and return probabilities
// for entailment / contradiction / neutral. vLLM serves them on /classify
// (vllm serve <model> --runner pooling --convert classify), at the server
// root rather than under /v1.
//
// Decide maps each typed question onto zero-shot NLI: the state is the
// premise and every option becomes a hypothesis, all sent in one batch.
//
//   - choice / score: each option's entailment probability, normalised across
//     the options, is its probability; confidence is the top one.
//   - yes_no: the hypothesis is the true criterion (else the instructions);
//     P(yes) is entailment / (entailment + contradiction).
//
// Options (Decide's opts):
//
//   - nli_labels: the classifier's output order; defaults to OpenJev's
//     contradiction, entailment, neutral. MNLI checkpoints such as
//     bart-large-mnli use contradiction, neutral, entailment.
//   - hypothesis_template: formats each option; {option} and {instructions}
//     are available. Defaults to "This is about {option}.".
//   - pair_template: joins premise and hypothesis into one input; defaults to
//     OpenJev's "Premise: {premise}\nHypothesis: {hypothesis}".
//
// Everything else on this protocol is plain Chat Completions.
package nliclassify

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/PhotonEra-B-V/gollym/core"
	"github.com/PhotonEra-B-V/gollym/protocols/chatcompletions"
)

// Defaults (OpenJev).
var DefaultLabels = []string{"contradiction", "entailment", "neutral"}

const (
	DefaultHypothesisTemplate = "This is about {option}."
	DefaultPairTemplate       = "Premise: {premise}\nHypothesis: {hypothesis}"
	// MaxInputs is vLLM's per-request cap on classify inputs.
	MaxInputs = 256
)

// Protocol is Chat Completions plus Decide over /classify.
type Protocol struct {
	*chatcompletions.Protocol
}

// New is the core.ProtocolFactory.
func New(p core.Provider, model *core.Info) core.Protocol {
	return &Protocol{Protocol: &chatcompletions.Protocol{Provider: p, Model: model}}
}

// ClassifyURL is <server root>/classify: a trailing /v1 is dropped.
func (n *Protocol) ClassifyURL() (string, error) {
	base, err := n.Provider.APIBase()
	if err != nil {
		return "", err
	}
	base = strings.TrimRight(base, "/")
	base = strings.TrimSuffix(base, "/v1")
	return base + "/classify", nil
}

// Decide answers questions about state with one batched classify call (more
// above MaxInputs).
func (n *Protocol) Decide(ctx context.Context, state any, questions *core.Questions, model string, opts map[string]any) (*core.Decision, error) {
	labels := stringList(opts["nli_labels"])
	if len(labels) == 0 {
		labels = DefaultLabels
	}
	entailment, contradiction := indexOf(labels, "entailment"), indexOf(labels, "contradiction")
	if entailment < 0 || contradiction < 0 {
		return nil, fmt.Errorf("nli_labels must include entailment and contradiction, got %v", labels)
	}
	hypothesisTemplate, _ := core.Or(opts["hypothesis_template"], DefaultHypothesisTemplate).(string)
	pairTemplate, _ := core.Or(opts["pair_template"], DefaultPairTemplate).(string)

	premise, ok := state.(string)
	if !ok {
		premise = core.PyJSON(state)
	}
	hypotheses := map[string][]string{}
	var inputs []any
	for _, name := range questions.Names {
		hs, err := Hypotheses(questions.ByKey[name], hypothesisTemplate)
		if err != nil {
			return nil, err
		}
		hypotheses[name] = hs
		for _, h := range hs {
			input, err := format(pairTemplate, map[string]string{"premise": premise, "hypothesis": h})
			if err != nil {
				return nil, err
			}
			inputs = append(inputs, input)
		}
	}

	url, err := n.ClassifyURL()
	if err != nil {
		return nil, err
	}
	var rows [][]float64
	inputTokens := 0
	for start := 0; start < len(inputs); start += MaxInputs {
		end := min(start+MaxInputs, len(inputs))
		resp, err := n.Provider.Connection().Post(ctx, url, map[string]any{"model": model, "input": inputs[start:end]}, nil)
		if err != nil {
			return nil, err
		}
		body := resp.BodyMap()
		data := core.ToSafeArray(body["data"])
		sort.SliceStable(data, func(i, j int) bool {
			return rowIndex(data[i]) < rowIndex(data[j])
		})
		for _, row := range data {
			probs := floats(core.Dig(row, "probs"))
			if len(probs) <= max(entailment, contradiction) {
				return nil, core.NewError(core.KindError, resp, fmt.Sprintf("classify row has %d probabilities for labels %v", len(probs), labels))
			}
			rows = append(rows, probs)
		}
		inputTokens += core.Deref(core.AsInt(core.Dig(body, "usage", "prompt_tokens")))
	}
	if len(rows) != len(inputs) {
		return nil, core.NewError(core.KindError, nil, fmt.Sprintf("classify returned %d rows for %d inputs", len(rows), len(inputs)))
	}

	answers := map[string]core.Answer{}
	cursor := 0
	for _, name := range questions.Names {
		q := questions.ByKey[name]
		chunk := rows[cursor : cursor+len(hypotheses[name])]
		cursor += len(hypotheses[name])
		if q.Type == core.QuestionYesNo {
			answers[name] = yesNoAnswer(chunk[0], entailment, contradiction)
			continue
		}
		ent := make([]float64, len(chunk))
		for i, p := range chunk {
			ent[i] = p[entailment]
		}
		answers[name] = rankedAnswer(q, ent)
	}
	var input *int
	if inputTokens != 0 {
		input = &inputTokens
	}
	return &core.Decision{
		Answers: answers, Names: append([]string(nil), questions.Names...), Model: model,
		Tokens: core.BuildTokens(input, nil, nil, nil, nil),
	}, nil
}

// Hypotheses are the texts tested against the premise for one question:
// the yes/no true criterion (else the instructions), or one formatted
// hypothesis per choice description / score level.
func Hypotheses(q core.Question, template string) ([]string, error) {
	if q.Type == core.QuestionYesNo {
		return []string{core.Or(q.True, q.Instructions).(string)}, nil
	}
	var texts []string
	switch q.Type {
	case core.QuestionChoice:
		for _, l := range q.Labels {
			texts = append(texts, q.Choices[l])
		}
	default:
		texts = q.Levels
	}
	out := make([]string, len(texts))
	for i, t := range texts {
		h, err := format(template, map[string]string{"option": t, "instructions": q.Instructions})
		if err != nil {
			return nil, err
		}
		out[i] = h
	}
	return out, nil
}

func normalize(scores []float64) []float64 {
	total := 0.0
	for _, s := range scores {
		total += s
	}
	out := make([]float64, len(scores))
	for i, s := range scores {
		if total <= 0 {
			out[i] = 1 / float64(len(scores))
		} else {
			out[i] = s / total
		}
	}
	return out
}

func rankedAnswer(q core.Question, entailments []float64) core.Answer {
	probs := normalize(entailments)
	best := 0
	for i, p := range probs {
		if p > probs[best] {
			best = i
		}
	}
	confidence := probs[best]
	if q.Type == core.QuestionChoice {
		labels := q.Options()
		choice := labels[best]
		ps := map[string]float64{}
		for i, l := range labels {
			ps[l] = probs[i]
		}
		return core.Answer{Type: core.QuestionChoice, Choice: &choice, Probabilities: ps, Confidence: &confidence, Legend: map[string]string{}}
	}
	levels := q.Options()
	score := 0.0
	ps, legend := map[string]float64{}, map[string]string{}
	for i, p := range probs {
		score += float64(i) * p
		k := strconv.Itoa(i)
		ps[k] = p
		legend[k] = levels[i]
	}
	return core.Answer{Type: core.QuestionScore, Score: &score, Probabilities: ps, Legend: legend, Confidence: &confidence}
}

func yesNoAnswer(probs []float64, entailment, contradiction int) core.Answer {
	yes, no := probs[entailment], probs[contradiction]
	p := 0.5
	if yes+no > 0 {
		p = yes / (yes + no)
	}
	return core.Answer{Type: core.QuestionYesNo, Probability: &p, Probabilities: map[string]float64{}, Legend: map[string]string{}}
}

// format is Python's str.format with named fields only ({{ and }} escape).
func format(template string, fields map[string]string) (string, error) {
	var b strings.Builder
	for i := 0; i < len(template); i++ {
		c := template[i]
		switch {
		case c == '{' && i+1 < len(template) && template[i+1] == '{':
			b.WriteByte('{')
			i++
		case c == '}' && i+1 < len(template) && template[i+1] == '}':
			b.WriteByte('}')
			i++
		case c == '{':
			end := strings.IndexByte(template[i:], '}')
			if end < 0 {
				return "", fmt.Errorf("Single '{' encountered in format string %q", template)
			}
			name := template[i+1 : i+end]
			v, ok := fields[name]
			if !ok {
				return "", fmt.Errorf("unknown field {%s} in template %q", name, template)
			}
			b.WriteString(v)
			i += end
		case c == '}':
			return "", fmt.Errorf("Single '}' encountered in format string %q", template)
		default:
			b.WriteByte(c)
		}
	}
	return b.String(), nil
}

func stringList(v any) []string {
	switch l := v.(type) {
	case []string:
		return l
	case []any:
		out := make([]string, len(l))
		for i, x := range l {
			out[i] = fmt.Sprint(x)
		}
		return out
	}
	return nil
}

func indexOf(list []string, s string) int {
	for i, x := range list {
		if x == s {
			return i
		}
	}
	return -1
}

func rowIndex(row any) int {
	return core.Deref(core.AsInt(core.Dig(row, "index")))
}

func floats(v any) []float64 {
	var out []float64
	for _, x := range core.ToSafeArray(v) {
		f := core.AsFloat(x)
		if f == nil {
			return nil
		}
		out = append(out, *f)
	}
	return out
}
