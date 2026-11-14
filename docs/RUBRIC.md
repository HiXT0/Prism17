# Prism17 disclosure rubric

Prism17 scores **disclosure**, not real-world performance. A Level 4 passage is
not evidence that a company is sustainable; it is evidence that the report
states a specific, measurable, time-bound, baseline-anchored action and reports
progress against it. A Level 1 passage is not evidence of inaction.

The rubric is deterministic. It lives in `prism17/rubric.py` and uses only the
five boolean dimensions returned by the judge plus an explicit `negated` flag.
No model call happens during grading.

## Five dimensions

| Dimension | True when the passage... |
| --- | --- |
| `specificity` | names a concrete action, project, technology, policy, or commitment. A general aspiration is not specific. |
| `measurable_indicator` | gives a number with a unit or metric, such as `42%`, `1,200 tCO2e`, `15 ML`, or `3,000 employees`. A year alone is not an indicator. |
| `deadline` | gives a target date or year by which an action or result is to be achieved. The report's own reporting year is not a deadline. |
| `baseline` | gives an explicit starting value, prior-year value, or benchmark against which change is measured. |
| `reported_progress` | reports actual results or progress achieved, not only a future plan or commitment. |

## Level rules

Rules are applied in order; the first matching rule wins.

| Level | Rule | Neutral interpretation |
| --- | --- | --- |
| 1 | The passage states the action is not applicable, not done, or not measured (`negated: true`). | Explicit non-disclosure or exclusion. |
| 1 | `specificity` is false. | Relevant but general disclosure. |
| 4 | All five dimensions are true. | Specific, measurable, time-bound, baseline-anchored, and reports progress. |
| 3 | `specificity`, `measurable_indicator`, and `deadline` are true. | Specific and quantified, but baseline or progress evidence is incomplete. |
| 2 | `specificity` is true and at least one of `measurable_indicator`, `deadline`, `baseline`, or `reported_progress` is true. | Specific but partially evidenced. |
| 1 | `specificity` is true and no other dimension is true. | Specific action with no measurable, time-bound, baseline, or progress information. |

Every graded finding also records `disclosure_gaps`: neutral sentences for each
missing dimension. The report uses these to generate guidance, not accusations.

## Scoring preconditions

A finding is scored only when all of the following hold:

- the judge marked the passage relevant and its output passed schema validation;
- the supporting quote was verified against the cited source page (Stage 1);
- Stage 2 semantic support, when enabled, is `supported` or `not_checked`;
- a rubric level was assigned;
- no review reason is attached.

A finding that fails any precondition keeps `level: null` and is shown under
"needs review". It is never counted as scored, and it never contributes to
coverage, tracking, or comparison metrics.

## Worked examples

These are illustrative shape examples, not gold annotations.

| Passage (abridged) | Dimensions | Level |
| --- | --- | --- |
| "We care about the climate." | none | 1 (not specific) |
| "We installed LED lighting at three plants." | specificity | 1 (specific, nothing else) |
| "We installed LED lighting at three plants, cutting electricity use by 12%." | specificity, measurable | 2 |
| "We will cut electricity use by 12% by 2027." | specificity, measurable, deadline | 3 |
| "We cut electricity use by 12% against a 2021 baseline, and report annual progress toward our 2027 target." | all five dimensions | 4 |
| "This target is not applicable to our operations." | negated | 1 |

## Calibration and change control

- The rubric has no tunable thresholds. Do not tune it against benchmark or test
  reports.
- Changing a rule requires updating `prism17/rubric.py`, this document, and the
  rubric tests in `tests/test_rubric.py` together.
- Judge dimension flags can disagree with a human reading. For example, a
  prior-year column may be visible in a table quote while the judge marks
  `baseline` false. Read the listed gaps alongside the quote and record any
  disagreement during annotation.
- The annotation benchmark measures agreement with human labels. It does not
  validate the rubric as a measure of real-world performance.
