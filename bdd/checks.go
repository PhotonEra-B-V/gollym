package bdd

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// Deterministic cross-checks between the spec and the (LLM-produced or
// hand-written) plan.
//
// This is the airtight half of the pipeline: coverage accounting happens in
// plain code, never by asking the model whether it covered everything. A plan
// that drops a scenario, ignores a sequence-diagram message, skips a required
// rule/edge id, or claims an id that doesn't exist fails the build.

// messageLess sorts M1, M2, ... numerically, before any other id (lexical).
func messageLess(a, b string) bool {
	na, aok := messageNum(a)
	nb, bok := messageNum(b)
	switch {
	case aok && bok:
		if na == nb {
			return a < b
		}
		if len(na) != len(nb) {
			return len(na) < len(nb)
		}
		return na < nb
	case aok != bok:
		return aok
	}
	return a < b
}

// messageNum returns the digits of an "M<digits>" id without leading zeros.
func messageNum(id string) (string, bool) {
	if !strings.HasPrefix(id, "M") || len(id) < 2 {
		return "", false
	}
	digits := id[1:]
	for _, r := range digits {
		if r < '0' || r > '9' {
			return "", false
		}
	}
	digits = strings.TrimLeft(digits, "0")
	return digits, true
}

// ValidatePlan returns human-readable problems; none means the plan is sound.
//
// scenarioNames are the scenarios every plan must cover (matched against
// TestCase.Scenario). messageIDs (sequence-diagram message ids) and
// requiredIDs (rule/edge ids) are BOTH valid targets for TestCase.Covers;
// every id in either set must be claimed by some case, and any claimed id
// outside their union is rejected as unknown.
func ValidatePlan(plan *TestPlan, scenarioNames, messageIDs, requiredIDs map[string]bool) []string {
	problems := []string{}

	covered := map[string]bool{}
	claimed := map[string]bool{}
	for _, c := range plan.Cases {
		covered[c.Scenario] = true
		for _, id := range c.Covers {
			claimed[id] = true
		}
	}
	for _, s := range sortedKeys(scenarioNames) {
		if !covered[s] {
			problems = append(problems, "scenario not covered by any test case: "+pyRepr(s))
		}
	}
	for _, id := range sortedKeys(claimed) {
		if !messageIDs[id] && !requiredIDs[id] {
			problems = append(problems, "test case claims unknown id: "+pyRepr(id))
		}
	}
	var uncoveredMsgs []string
	for id := range messageIDs {
		if !claimed[id] {
			uncoveredMsgs = append(uncoveredMsgs, id)
		}
	}
	sort.Slice(uncoveredMsgs, func(i, j int) bool { return messageLess(uncoveredMsgs[i], uncoveredMsgs[j]) })
	for _, id := range uncoveredMsgs {
		problems = append(problems, "sequence message not covered by any test case: "+id)
	}
	for _, id := range sortedKeys(requiredIDs) {
		if !claimed[id] {
			problems = append(problems, "rule/edge case not covered by any test case: "+id)
		}
	}
	seen := map[string]bool{}
	for _, c := range plan.Cases {
		if seen[c.TestName] {
			problems = append(problems, "duplicate test name in plan: "+pyRepr(c.TestName))
		}
		seen[c.TestName] = true
	}
	return problems
}

// pyRepr renders a value like Python's repr() so messages match pyllym's
// ({x!r}): strings get Python's quoting and escapes.
func pyRepr(v any) string {
	s, ok := v.(string)
	if !ok {
		return pyStr(v)
	}
	quote := '\''
	if strings.ContainsRune(s, '\'') && !strings.ContainsRune(s, '"') {
		quote = '"'
	}
	var b strings.Builder
	b.WriteRune(quote)
	for _, r := range s {
		switch {
		case r == '\\':
			b.WriteString(`\\`)
		case r == quote:
			b.WriteRune('\\')
			b.WriteRune(r)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case !unicode.IsPrint(r) && r != ' ':
			switch {
			case r < 0x100:
				fmt.Fprintf(&b, `\x%02x`, r)
			case r < 0x10000:
				fmt.Fprintf(&b, `\u%04x`, r)
			default:
				fmt.Fprintf(&b, `\U%08x`, r)
			}
		default:
			b.WriteRune(r)
		}
	}
	b.WriteRune(quote)
	return b.String()
}

// goQuote is strconv.Quote, named for symmetry with pyRepr.
func goQuote(s string) string { return strconv.Quote(s) }
