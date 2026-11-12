# Annotation guidelines

This benchmark measures **disclosure quality**, not real-world sustainability
performance. Annotators label what a report discloses and how specific that
disclosure is. They do not judge whether the company's activities are good,
sufficient, or sincere.

No gold annotations are bundled with the repository. The template is blank on
purpose. Do not report benchmark numbers until the reviewer columns have been
filled by people.

## Files

- `data/gold/annotation_template.csv` — generated from audit JSON with blank
  `reviewer_*` columns.
- `data/gold/annotations.csv` — a filled copy. Keep one row per annotated
  finding.
- `eval/score_benchmark.py` — turns a filled CSV into a metrics JSON report.

Generate a template:

```powershell
python eval/make_gold_template.py outputs/sjs.phase2.json `
  --out data/gold/annotation_template.csv
```

Score a filled copy:

```powershell
python eval/score_benchmark.py data/gold/annotations.csv `
  --out eval/benchmark.json
```

## What is in the template

The template includes scored findings and findings that need human review. It
excludes passages the judge cleanly marked not relevant, because they are not
claims about a target. Use `--include-not-relevant` if a hard-negative sample is
needed, and `--scored-only` for a scored-only sample.

The template is not a representative random sample of a report. If the benchmark
is meant to estimate performance, define the sampling frame and record it in
the run notes. A convenience sample can support error analysis but not a broad
performance claim.

## Work one row at a time

1. Open the PDF at `pdf_path` and go to `page` (1-indexed, matching a PDF
   viewer).
2. Read `quote` on that page. If the quote is not present, do not reconstruct
   it from memory. Mark `reviewer_quote_ok` as `0` and add a note.
3. Read `claim` and compare it with the quote and the surrounding page.
4. Enter the reviewer columns. Leave a field blank when it cannot be assessed;
   blank means "not assessed", not "no".

Do not edit the `model_*`, `claim`, `quote`, `page`, `target_id`, or
`passage_id` columns. They record the output being evaluated.

## Fields

| Field | Accepted values | Meaning |
| --- | --- | --- |
| `reviewer_relevant` | `1`, `0`, blank | `1` if the passage addresses the SDG target in a substantive way, even if the disclosure is weak. `0` if it is off-topic or only a keyword match. |
| `reviewer_level` | `1`–`4`, blank | Rubric level for the **disclosure** in the quote. Use the rules below. Blank if the quote cannot be assessed. |
| `reviewer_quote_ok` | `1`, `0`, blank | `1` if the quote appears on the cited page and is faithful to the source. `0` if it is absent, altered, or stitched together misleadingly. |
| `reviewer_semantic_support` | `supported`, `uncertain`, `not_supported`, blank | Whether the quote supports the claim attributed to it. `supported` = the claim is traceable to the quote; `uncertain` = a material detail is missing or contradicted; `not_supported` = the quote does not support the claim. |
| `reviewer_notes` | free text | Short, neutral reason for an override or uncertainty. Avoid accusatory language. |
| `annotator` | initials or identifier | Who annotated the row. |
| `annotated_at_utc` | ISO-8601 timestamp | When the row was annotated. |

## Rubric levels

Apply the first matching rule.

| Level | Rule |
| --- | --- |
| 1 | The passage says the action is not applicable, not done, or not measured; or the disclosure is not specific. |
| 2 | Specific action plus at least one of: measurable indicator, deadline, baseline, reported progress. |
| 3 | Specific action plus a measurable indicator and a deadline. |
| 4 | Specific action plus a measurable indicator, deadline, baseline, and reported progress. |

Level 4 is not a claim that the target was achieved. It means the disclosure
contains all five evidence dimensions. A future commitment can reach Level 4 if
it is specific, measurable, time-bound, baseline-anchored, and reports progress
against that baseline.

## Relevance and support are different

- **Relevance** asks whether the passage is about the target at all.
- **Quote faithfulness** asks whether the quoted words are on the cited page.
- **Semantic support** asks whether those words support the claim the system
  attributed to them.

A row can be relevant with a faithful quote that still does not support the
claim. Record that combination instead of collapsing the fields.

## Double annotation and agreement

For any benchmark claim, use at least two independent annotators on the same
sample:

1. Annotators fill separate copies without seeing each other's labels.
2. A third person adjudicates disagreements and records the final labels.
3. Report the number of double-annotated rows and agreement statistics
   (exact agreement, within-one agreement, Cohen's kappa, quadratic weighted
   kappa, and semantic-support agreement where it was annotated).
4. Report the sampling frame, the audit runs, and the model versions.

`eval/score_benchmark.py` computes these statistics from the filled CSV. It does
not decide which annotation is correct.

## Neutral language

Use "disclosure gap", "low specificity", "not found in the retrieved sample",
and "needs review". Do not write that a company "failed", "lied", or "does not
care". The benchmark is about text and evidence, not intent.

## Privacy and licensing

Annotate only reports you are permitted to use. Do not put confidential report
text into a public repository. If a gold file contains third-party report text,
record its source and licence in the run notes.
