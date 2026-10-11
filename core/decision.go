package core

import (
	"errors"
	"fmt"
	"sort"
)

// Decision question types.
const (
	QuestionChoice = "choice"
	QuestionScore  = "score"
	QuestionYesNo  = "yes_no"
)

var yesNoAliases = map[string]bool{"yes_no": true, "noul": true, "null": true}

// Question is one typed question for Decide. Criteria depends on Type:
// choice -> Choices (label -> description, ordered by Labels), score ->
// Levels (lowest first), yes_no -> optional True/False descriptions.
type Question struct {
	Type         string
	Instructions string
	Labels       []string          // choice labels, in order
	Choices      map[string]string // choice label -> description
	Levels       []string          // score levels, lowest first
	True, False  string            // yes_no descriptions
}

// ChoiceQuestion picks one label (labels in order, each with a description).
func ChoiceQuestion(labels []string, descriptions map[string]string, instructions string) (Question, error) {
	if len(labels) < 2 {
		return Question{}, errors.New("A choice question needs at least two options")
	}
	ch := map[string]string{}
	for _, l := range labels {
		if d, ok := descriptions[l]; ok {
			ch[l] = d
		} else {
			ch[l] = l
		}
	}
	return Question{Type: QuestionChoice, Instructions: instructions, Labels: labels, Choices: ch}, nil
}

// ScoreQuestion places the state on an ordered scale.
func ScoreQuestion(levels []string, instructions string) (Question, error) {
	if len(levels) < 2 {
		return Question{}, errors.New("A score question needs at least two options")
	}
	return Question{Type: QuestionScore, Instructions: instructions, Levels: levels}, nil
}

// YesNoQuestion asks for P(yes).
func YesNoQuestion(instructions, trueDesc, falseDesc string) Question {
	return Question{Type: QuestionYesNo, Instructions: instructions, True: trueDesc, False: falseDesc}
}

// Options returns choice labels or score levels (empty for yes/no).
func (q Question) Options() []string {
	switch q.Type {
	case QuestionChoice:
		return append([]string(nil), q.Labels...)
	case QuestionScore:
		return append([]string(nil), q.Levels...)
	}
	return nil
}

// CriteriaValue is the wire/to_dict criteria value.
func (q Question) CriteriaValue() any {
	switch q.Type {
	case QuestionChoice:
		m := map[string]any{}
		for _, l := range q.Labels {
			m[l] = q.Choices[l]
		}
		return m
	case QuestionScore:
		out := make([]any, len(q.Levels))
		for i, l := range q.Levels {
			out[i] = l
		}
		return out
	}
	m := map[string]any{}
	if q.True != "" {
		m["true"] = q.True
	}
	if q.False != "" {
		m["false"] = q.False
	}
	if len(m) == 0 {
		return nil
	}
	return m
}

// Map is the to_dict form.
func (q Question) Map() map[string]any {
	out := map[string]any{"type": q.Type}
	if q.Instructions != "" {
		out["instructions"] = q.Instructions
	}
	if c := q.CriteriaValue(); c != nil {
		out["criteria"] = c
	}
	return out
}

// BuildQuestion coerces a to_dict- or wire-shaped map. Choice criteria maps
// have no order in Go, so labels are sorted; pass Labels explicitly (via
// ChoiceQuestion) when order matters.
func BuildQuestion(d map[string]any) (Question, error) {
	t, _ := d["type"].(string)
	instr, _ := d["instructions"].(string)
	switch {
	case yesNoAliases[t]:
		q := Question{Type: QuestionYesNo, Instructions: instr}
		if c, ok := d["criteria"].(map[string]any); ok {
			q.True, _ = c["true"].(string)
			q.False, _ = c["false"].(string)
		}
		return q, nil
	case t == QuestionScore:
		var levels []string
		for _, l := range ToSafeArray(d["criteria"]) {
			levels = append(levels, fmt.Sprint(l))
		}
		return ScoreQuestion(levels, instr)
	case t == QuestionChoice:
		switch c := d["criteria"].(type) {
		case map[string]any:
			labels := sortedKeys(c)
			desc := map[string]string{}
			for _, l := range labels {
				desc[l] = fmt.Sprint(c[l])
			}
			return ChoiceQuestion(labels, desc, instr)
		default:
			var labels []string
			for _, l := range ToSafeArray(c) {
				labels = append(labels, fmt.Sprint(l))
			}
			return ChoiceQuestion(labels, nil, instr)
		}
	}
	return Question{}, fmt.Errorf("Unknown question type: %q", t)
}

// Answer is the model's answer to one Question.
type Answer struct {
	Type          string
	Choice        *string
	Score         *float64
	Probability   *float64
	Probabilities map[string]float64
	Confidence    *float64
	Legend        map[string]string
}

// Value is the headline answer: chosen label, score, or P(yes).
func (a Answer) Value() any {
	switch a.Type {
	case QuestionChoice:
		if a.Choice == nil {
			return nil
		}
		return *a.Choice
	case QuestionScore:
		if a.Score == nil {
			return nil
		}
		return *a.Score
	}
	if a.Probability == nil {
		return nil
	}
	return *a.Probability
}

// Level is the text of the most likely score level ("" otherwise).
func (a Answer) Level() string {
	if a.Type != QuestionScore || len(a.Probabilities) == 0 {
		return ""
	}
	best, bestP := "", -1.0
	for _, k := range sortedKeysF(a.Probabilities) {
		if a.Probabilities[k] > bestP {
			best, bestP = k, a.Probabilities[k]
		}
	}
	if l, ok := a.Legend[best]; ok {
		return l
	}
	return best
}

// Yes reports P(yes) >= 0.5 (nil when no probability).
func (a Answer) Yes() *bool {
	if a.Probability == nil {
		return nil
	}
	v := *a.Probability >= 0.5
	return &v
}

// BuildAnswer reads a to_dict/wire-shaped answer.
func BuildAnswer(d map[string]any) Answer {
	t, _ := d["type"].(string)
	if yesNoAliases[t] {
		t = QuestionYesNo
	}
	a := Answer{Type: t, Score: AsFloat(d["score"]), Probability: AsFloat(d["probability"]), Confidence: AsFloat(d["confidence"]),
		Probabilities: map[string]float64{}, Legend: map[string]string{}}
	if c, ok := d["choice"].(string); ok {
		a.Choice = &c
	}
	if p, ok := d["probabilities"].(map[string]any); ok {
		for k, v := range p {
			if f := AsFloat(v); f != nil {
				a.Probabilities[k] = *f
			}
		}
	}
	if l, ok := d["legend"].(map[string]any); ok {
		for k, v := range l {
			a.Legend[k] = fmt.Sprint(v)
		}
	}
	return a
}

// Map is the to_dict form.
func (a Answer) Map() map[string]any {
	out := map[string]any{"type": a.Type}
	if a.Choice != nil {
		out["choice"] = *a.Choice
	}
	if a.Score != nil {
		out["score"] = *a.Score
	}
	if a.Probability != nil {
		out["probability"] = *a.Probability
	}
	if len(a.Probabilities) > 0 {
		out["probabilities"] = a.Probabilities
	}
	if a.Confidence != nil {
		out["confidence"] = *a.Confidence
	}
	if len(a.Legend) > 0 {
		out["legend"] = a.Legend
	}
	return out
}

// Decision holds answers keyed by question name.
type Decision struct {
	Answers map[string]Answer
	Names   []string // question order
	Model   string
	Tokens  *Tokens
}

// Get returns one answer.
func (d *Decision) Get(name string) Answer { return d.Answers[name] }

func (d *Decision) InputTokens() int {
	if d.Tokens == nil {
		return 0
	}
	return Deref(d.Tokens.Input)
}

func (d *Decision) OutputTokens() int {
	if d.Tokens == nil {
		return 0
	}
	return Deref(d.Tokens.Output)
}

// Questions is an ordered set of named questions.
type Questions struct {
	Names []string
	ByKey map[string]Question
}

// Add appends a named question.
func (q *Questions) Add(name string, question Question) *Questions {
	if q.ByKey == nil {
		q.ByKey = map[string]Question{}
	}
	if _, ok := q.ByKey[name]; !ok {
		q.Names = append(q.Names, name)
	}
	q.ByKey[name] = question
	return q
}

func sortedKeys(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func sortedKeysF(m map[string]float64) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
