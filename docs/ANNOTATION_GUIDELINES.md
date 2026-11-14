# Annotation guidelines

These guidelines cover the human-annotation benchmark for Prism17. The
benchmark measures **agreement on disclosure labels**, not real-world
sustainability performance, and it must not be used to tune prompts or
thresholds on test data.

## Sampling frame

- Annotate **development (DEV) reports only**. Do not annotate test reports.
- Use a sample of roughly 150-200 passages per evaluation round.
- The sample should cover every rubric level, passages the judge rejected as not
  relevant, and table passages. Record how many passages came from each stratum.
- Record the dataset version, the report manifest, the report files used, and
  the model/configuration that produced the model columns. Evaluation runs must
  log provenance, not just aggregate numbers.
- A `manifest.csv` with a `split` column is required before a DEV-only export
  can be built. The export must filter on `split == "dev"`.

## Independence

- Annotate from the source PDF and the quoted passage, not from the model
  columns. The model columns are frozen predictions supplied for comparison.
- Do not copy `model_relevant`, `model_level`, `model_quote_ok`, or
  `model_semantic_support` into the reviewer columns.
- If two people annotate the same sample, they must work independently and
  record their own `annotator` value. Resolve disagreements afterwards and keep
  both original labels. Report agreement statistics.
- Never invent a label to fill a gap. Leave the cell blank if you cannot decide;
  blank cells are reported as missing, not as a negative label.

## Template

Build a stratified DEV sample directly from the report PDFs:

```powershell
python eval/export_candidates.py --judge --sample-size 175 --per-report 25
```

The export filters `data/reports/manifest.csv` on `split == "dev"`, writes
`eval/annotate_template.csv`, and records provenance (manifest digest, PDFs,
split, seed, per-report limits, and strata) in
`eval/annotate_template.provenance.json`. It never opens test or excluded PDFs.
If judged findings are too few to fill the requested sample, the remaining slots
are filled with retrieval-only table and body passages, interleaved across DEV
reports so one long report cannot consume the whole sample. Those rows have
blank model columns and cannot be paired in benchmark scoring. Use `--no-judge`
to export retrieval candidates with blank model columns throughout.

Alternatively, generate a template from audit JSON with:

```powershell
python eval/make_gold_template.py examples/sjs.audit.json `
  --labels "S.J.S. FY2023-24" `
  --out data/gold/annotations.csv
```

The script refuses to overwrite an existing file unless `--force` is supplied.
Fill only the `reviewer_*` columns, `annotator`, and `annotated_at_utc`.

## Fields

| Column | Accepted values | Meaning |
| --- | --- | --- |
| `reviewer_relevant` | `1` / `0` (also `yes` / `no`, `true` / `false`); blank | Does the passage explicitly support the candidate SDG target? |
| `reviewer_level` | `1`, `2`, `3`, `4`; blank | Disclosure level under the rubric in `docs/RUBRIC.md`. Fractional values are invalid. |
| `reviewer_quote_ok` | `1` / `0`; blank | Is the quoted text present on the cited page, after normal whitespace and hyphenation handling? This is the Stage 1 question only. |
| `reviewer_semantic_support` | `supported`, `uncertain`, `not_supported`; blank | Does the quote actually support the claim? Leave blank when Stage 2 was not assessed. `not_checked` is treated as missing. |
| `reviewer_notes` | free text | Short reason for a disagreement or an ambiguous case. |
| `annotator` | short identifier | Who produced the label. |
| `annotated_at_utc` | ISO-8601 UTC timestamp | When the label was produced. |

### Relevance

Mark `1` only when the passage explicitly describes an action, commitment,
policy, or result that fits the target. A passage that merely mentions a related
theme is `0`. If the fit is arguable, mark `0` and explain the ambiguity in
`reviewer_notes`; do not stretch the target.

### Rubric level

Apply the rules in `docs/RUBRIC.md` in order. A level is required when
`reviewer_relevant` is `1`. Leave the level blank when the passage is not
relevant. A year alone is not a measurable indicator, and the report's own
reporting year is not a deadline.

### Quote verification

`reviewer_quote_ok` answers only whether the quote exists on the cited page. It
does not ask whether the quote supports the claim; that is the semantic-support
question. If the quote is a table row, check the parsed table passage as well as
the raw page, because layout extraction can differ.

### Semantic support

Use the same three labels as the model:

- `supported`: the claim is a faithful restatement of information in the quote;
- `uncertain`: the quote is ambiguous, incomplete, or needs a human decision;
- `not_supported`: the quote does not support the claim.

Do not use outside knowledge and do not assess whether the quoted action really
happened.

## Scoring

Score a filled file with:

```powershell
python eval/score_benchmark.py data/gold/annotations.csv `
  --out eval/benchmark.json
```

Scoring is paired: a metric is computed only from rows where both the reviewer
value and a valid model value are present. The report status is one of:

| Status | Meaning |
| --- | --- |
| `no_annotations` | No reviewer values are present. All metrics are null. |
| `invalid_annotations` | Reviewer values are present but invalid. All metrics are null. Correct the values first. |
| `no_pairs` | Reviewer values are valid but no model value is available for a paired metric. All metrics are null. |
| `ok` | At least one paired metric was computed. |

Metrics include binary accuracy, precision, recall, F1, and a confusion matrix
for relevance and quote verification; exact and within-one agreement, mean
absolute difference, quadratic weighted kappa, and a confusion matrix for the
rubric; and agreement, Cohen's kappa, and a confusion matrix for semantic
support.

## Reporting rules

- Do not report a metric, confidence interval, or sample size that the scoring
  report did not produce.
- State the sample frame, the number of paired rows, the dataset version, and
  the report files.
- If the status is not `ok`, report that no benchmark metrics were computed.
- Prefer independent double annotation with agreement statistics. A single
  annotator can describe agreement with the model but cannot support a claim
  about annotation reliability.
- Keep the benchmark separate from prompt development. Do not tune prompts or
  thresholds on the sample used for the reported evaluation.
