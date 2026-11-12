# Prism17 rubric

The rubric describes the **strength of a disclosure**, not the strength of the
company's sustainability performance. It is deterministic: the same five boolean
dimensions and the same `negated` flag always produce the same level. There is no
LLM call in `prism17/rubric.py`.

## The five dimensions

Every judged claim carries five booleans. They refer to the verified quote.

| Key | Question |
| --- | --- |
| `specificity` | Does the quote describe a concrete action, commitment, or result rather than a general statement? |
| `measurable_indicator` | Is there a number, percentage, intensity, or other measurable indicator? |
| `deadline` | Is there a target date, reporting year, or time-bound commitment? |
| `baseline` | Is there a prior-year value, reference point, or baseline for comparison? |
| `reported_progress` | Is there a reported result or change against a baseline, not only a future intention? |

A dimension is `present` only when the judge finds it in the quote. Absence is a
**disclosure gap in that quote**, not evidence that the company does nothing.
The quote may be one passage from a longer report that discloses the missing
dimension elsewhere.

## Level rules

Rules are applied in order. The first match wins.

| Level | Rule |
| --- | --- |
| 1 | The quote says the action is not applicable, not done, or not measured (`negated` is true). |
| 1 | `specificity` is false: the statement is relevant but general. |
| 4 | All five dimensions are present. |
| 3 | `specificity`, `measurable_indicator`, and `deadline` are present. |
| 2 | `specificity` is present and at least one of `measurable_indicator`, `deadline`, `baseline`, or `reported_progress` is present. |
| 1 | `specificity` is present but no measurable, time-bound, baseline, or progress information is present. |

`negated` is read from the judge's `extracted` object. It covers statements such
as "not applicable", "not measured", or "we do not report this". Those statements
can be useful disclosures of a gap, but they are not evidence of action, so they
receive Level 1.

## Worked examples

These examples are illustrative. Real labels come from the source page.

| Quote | Dimensions | Level | Why |
| --- | --- | --- | --- |
| "We care about the environment." | none | 1 | General; no concrete action. |
| "We are installing 20 MW of solar capacity by 2027." | specific, measurable, deadline | 3 | Concrete action with a number and a date, but no baseline or reported progress. |
| "We reduced Scope 1 emissions by 42% in FY2024 against a FY2020 baseline of 100,000 tCO2e." | all five | 4 | Specific, measurable, dated, baseline-anchored, and reports a result. |
| "We plan to reduce water use." | specific only | 1 | No measurable indicator, deadline, baseline, or progress. |
| "This indicator is not applicable to our operations." | negated | 1 | The company states the action is not applicable. |
| "We trained 1,200 employees in 2024 as part of our safety programme." | specific, measurable, deadline | 3 | A completed count in a reporting year; no explicit baseline or progress comparison. |

A future commitment can reach Level 4 if it is specific, measurable, time-bound,
baseline-anchored, and reports progress against that baseline. Level 4 is not a
claim that the target was achieved.

## Disclosure gaps

For every finding the rubric returns a list of gaps for the missing dimensions:

- "The disclosure is general rather than specific."
- "No measurable indicator is disclosed."
- "No target date or deadline is disclosed."
- "No baseline or prior-year comparison is disclosed."
- "No reported progress or result is disclosed."

User-facing reports use neutral flag text such as "low specificity" and
"disclosure gap: no deadline".

## Scoring gate

A finding is scored only when all of the following hold:

1. the judge marked it relevant and schema-valid;
2. the quote was verified on the cited page (Stage 1);
3. Stage 2 did not return `uncertain` or `not_supported`;
4. the rubric assigned a level;
5. no review reason is attached.

Stage 2 `not_checked` does **not** block scoring. That keeps the audit usable on a
laptop that has not downloaded the optional NLI model.

## Coverage matrix

The report's evidence matrix shows the same five dimensions per claim as
`present`, `absent`, or `n/a` (not assessable). It is a presentation of the
judged dimensions, not a second scoring system. A claim can be Level 2 while
still showing several absent dimensions.

## Calibration and limitations

- The dimensions come from the local judge. A prior-year column in a table can be
  visible in the quote while the judge still marks `baseline` false.
- The rubric measures the evidence in one quote. It does not search the whole
  report for a missing dimension.
- The rubric does not reward the size, ambition, or real-world effect of a
  target. It rewards specificity and evidence.
- The rubric is deterministic, but the judge is not. Re-running with another
  model can change the dimensions and therefore the level.
- No human gold annotations or measured rubric agreement results are bundled.
  Use `eval/make_gold_template.py` and `eval/score_benchmark.py` to collect and
  score them. Do not report agreement numbers before annotations exist.

## Related files

- `prism17/rubric.py` — implementation; no model calls.
- `tests/test_rubric.py` — branch coverage for every rule.
- `data/gold/ANNOTATION_GUIDELINES.md` — how to label reviewer levels.
- `README.md` — run commands and report behaviour.
