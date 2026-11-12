# Prism17

**Evidence-grounded SDG disclosure auditing.** Prism17 reads a corporate
sustainability report, BRSR filing, or annual report and produces an SDG
Alignment Report in which every finding is tied to a verified quote and page
number.

> Prism17 assesses **disclosure quality**, not real-world sustainability
> performance. It is decision support: a human reviewer must confirm or override
> every finding before it is used.

## Status

Phases 1-4 are implemented and tested: parsing, dense retrieval over the UN SDG
targets, a strict-JSON local Ollama judge, stage 1 quote verification, a
deterministic L1-L4 rubric, the CLI pipeline, Markdown/HTML reports, and a local
Streamlit review UI.

Phase 5 adds two things. First, **Stage 2 semantic verification** checks whether
a verified quote supports the claim attributed to it. It combines a deterministic
consistency guard with an optional local NLI cross-encoder or local Ollama check.
It is **disabled by default** so a first CPU run does not download another model.
Second, evaluation helpers and annotation tooling cover binary metrics, Cohen's
kappa, quadratic weighted kappa, rubric agreement, and a gold-annotation
template/scorer. **No human gold annotations or measured benchmark results are
bundled.** Do not report benchmark numbers until real annotations exist.

Phase 6 adds an evidence coverage matrix, narrow multi-year tracking of
quantified targets, and a descriptive cross-company comparison. Tracking and
comparison report disclosure only; they do not establish real-world performance,
and the comparison is not a ranking.

Phase 2 is deliberately conservative: a finding is scored only when its quote is
verified against the source page, Stage 2 does not veto it, and the rubric can
assign a level. Malformed model output and unverifiable quotes are retained as
"needs review", not silently dropped. Phase 2 has been verified end to end with
the real local model; see "Phase 2 verification on the sample report" below.

### Roadmap

- [x] Parse reports into pages, passages, and tables.
- [x] Retrieve SDG targets and judge candidate passages locally.
- [x] Verify quote existence (Stage 1) and apply the deterministic rubric.
- [x] Render coverage reports and review findings in the Streamlit UI.
- [x] Add Stage 2 semantic support checking (optional, disabled by default).
- [x] Add the five-dimension evidence coverage matrix.
- [x] Add narrow multi-year tracking for quantified targets.
- [x] Add descriptive cross-company comparison.
- [x] Add evaluation metrics, annotation templates, and annotation guidelines.
- [ ] Collect independent human gold annotations and report measured benchmark
      metrics.
- [ ] Validate Stage 2 semantic verification end to end on the bundled sample
      with the NLI model enabled.

## Requirements

- Python 3.11 or newer
- A local [Ollama](https://ollama.com/) installation (needed from Phase 2)
- No GPU required

The Stage 2 NLI check uses the installed `sentence-transformers` package and
downloads `cross-encoder/nli-deberta-v3-small` on first use. It is optional and
disabled by default.

## Setup (Windows PowerShell)

```powershell
cd prism17
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux:

```bash
cd prism17
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install Ollama separately, then pull the default model:

```bash
ollama pull qwen2.5:7b-instruct
```

Optional comparison models:

```bash
ollama pull llama3.1:8b
ollama pull gemma2:9b
```

## Project layout

```text
prism17/
  config.yaml
  data/
    sdg_targets.json
    reports/
    gold/
      ANNOTATION_GUIDELINES.md
  docs/
    RUBRIC.md
  prism17/
    config.py
    parse.py
    models.py
    retrieve.py
    judge.py
    verify.py
    semantic.py
    rubric.py
    coverage.py
    guidance.py
    report.py
    review.py
    tracking.py
    compare.py
    benchmark.py
    evaluation.py
    ui.py
    pipeline.py
  eval/
    osdg_check.py
    make_gold_template.py
    score_benchmark.py
  app.py
  prompts/
    judge_system.txt
    judge_examples.txt
  tests/
  outputs/
```

## Parse a report (Phase 1)

```powershell
python -m prism17.parse data/reports/sjs_india_brsr_fy2023_24.pdf
```

Write the full parsed document to JSON:

```powershell
python -m prism17.parse data/reports/sjs_india_brsr_fy2023_24.pdf --out outputs/sjs.parse.json
```

The parser:

- extracts text page by page, with 1-indexed page numbers matching a PDF viewer;
- splits body text into passages of roughly 50-250 words;
- extracts tables separately with `is_table: true`;
- records unreadable pages instead of crashing;
- keeps raw page text for later quote verification.

Phase 1 verification on the bundled sample: 59 pages, 326 passages (313 body,
13 tables), 0 unreadable pages, about 11 seconds on CPU. A blank-page test
confirms that unreadable pages are reported without a crash.

## Audit a report (Phase 2)

Make sure Ollama is running and the default model is available:

```powershell
ollama pull qwen2.5:7b-instruct
```

Run the full local pipeline:

```powershell
python -m prism17.pipeline data/reports/sjs_india_brsr_fy2023_24.pdf --out outputs/sjs.audit.json
```

For a quicker first run, restrict the goals, pages, and number of candidates:

```powershell
python -m prism17.pipeline data/reports/sjs_india_brsr_fy2023_24.pdf `
  --sdg 13 --max-pages 20 --max-candidates 20 --out outputs/sjs.phase2.json
```

The pipeline:

1. parses the PDF into page-level passages;
2. embeds passages and the SDG targets locally with
   `BAAI/bge-small-en-v1.5`, then retrieves candidates with FAISS on CPU;
3. asks the local Ollama judge for strict JSON on each candidate;
4. verifies the returned quote against the source page (exact match after
   whitespace/hyphenation normalisation, then RapidFuzz partial ratio >= 95);
5. optionally checks semantic support for the claim/quote pair (Stage 2,
   disabled by default);
6. applies the deterministic rubric;
7. prints scored findings with page numbers and verified quotes, and writes the
   full JSON, including findings that need human review;
8. optionally renders a Markdown and/or HTML SDG Alignment Report when
   `--report` or `--html-report` is used.

Useful options: `--model llama3.1:8b`, `--top-k`, `--min-score`,
`--fuzzy-threshold`, `--max-candidates`, `--no-tables`, `--scored-only`,
`--semantic`/`--no-semantic`, and `--verbose`. The first run downloads the
embedding model and caches embeddings under `.cache/prism17/`.

### Phase 2 verification on the sample report

The bundled 59-page BRSR was run end to end with `qwen2.5:7b-instruct` on CPU:

```powershell
python -m prism17.pipeline data/reports/sjs_india_brsr_fy2023_24.pdf `
  --sdg 12 --max-candidates 12 --out outputs/sjs.phase2.json
```

- 59 pages, 326 passages (13 tables), 0 unreadable pages.
- 978 Goal 12 candidates retrieved from 11 targets; the top 12 were judged.
- 2 findings scored, 10 judged not relevant, 0 needs review.
- Scored: target 12.5, page 49, Level 2, quote verified with `page_fuzzy`
  96.84; target 12.6, page 25, Level 1, quote verified with `page_exact` 100.0.
- About 10.5 minutes of wall-clock time, dominated by 12 local judge calls.

The artifact is `outputs/sjs.phase2.json`. Every scored entry carries
`source_page`, `quote_verified: true`, `quote_method`, `quote_score`, and the
rubric level. The run also confirms the review split: passages the judge marks
not relevant are counted separately from findings that need human review.

### Rubric (disclosure quality, not performance)

Rules are applied in order; the first matching rule wins.

| Level | Deterministic rule |
| --- | --- |
| 1 | The passage says the action is not applicable, not done, or not measured; or the disclosure is not specific. |
| 2 | Specific action plus at least one of: measurable indicator, deadline, baseline, reported progress. |
| 3 | Specific action plus a measurable indicator and a deadline. |
| 4 | Specific action plus a measurable indicator, deadline, baseline, and reported progress. |

Every finding also carries a list of disclosure gaps for the dimensions that
are absent. A finding is never scored without a verified quote and a page
number.

## Generate the SDG Alignment Report (Phase 3)

The report is deterministic and can be rendered during an audit run or from an
existing audit JSON, without re-running the judge.

During a run:

```powershell
python -m prism17.pipeline data/reports/sjs_india_brsr_fy2023_24.pdf `
  --sdg 12 --max-candidates 12 `
  --out outputs/sjs.phase2.json `
  --report outputs/sjs.report.md `
  --html-report outputs/sjs.report.html
```

From an existing audit JSON:

```powershell
python -m prism17.report outputs/sjs.phase2.json `
  --out outputs/sjs.report.md `
  --html outputs/sjs.report.html
```

Use `--company "Example Ltd"` and `--title "..."` to override the header.

To render a reviewed report from the CLI, pass the review file saved by the UI:

```powershell
python -m prism17.report outputs/sjs.phase2.json `
  --reviews outputs/sjs.phase2.reviews.json `
  --out outputs/sjs.reviewed.md `
  --html outputs/sjs.reviewed.html
```

The report contains:

- report details and a disclosure-scope disclaimer;
- a summary of scored findings, level distribution, and coverage;
- coverage by goal, including targets with and without scored disclosure;
- scored findings grouped by goal and target, each with page, verified quote,
  level, assessment, disclosure gaps, and suggested improvements;
- goals without scored findings, with neutral guidance and a retrieval caveat;
- findings that need human review, listed separately;
- method, limitations, and responsible-use notes.

The bundled sample's report is `outputs/sjs.report.md`, with an HTML companion
at `outputs/sjs.report.html`. Coverage counts only what was retrieved and
judged: a target with no scored finding is reported as "not identified", not as
evidence that the company does nothing.

## Human review in the Streamlit UI (Phase 4)

Start the local UI from the project root:

```powershell
streamlit run app.py
```

The UI opens on the bundled audit (`outputs/sjs.phase2.json`) if it exists. Use
the sidebar to load another audit JSON, or to run a new audit from a PDF path or
upload. A new audit uses the same local pipeline and the same Ollama model as the
CLI; run limits for pages and candidates are exposed so a first run stays short.

The four tabs are:

1. **Summary & coverage** — scored findings, targets with disclosure, judged
   candidates, review status, and per-goal and per-target coverage.
2. **Findings review** — one editable row per scored finding. Target, page,
   rubric level, and the verified quote are read-only. A reviewer can set the
   status to `pending`, `confirmed`, `overridden`, or `flagged`, add a reviewer
   level, reviewer name, and note, then save.
3. **Needs review** — findings that could not be scored because the quote was
   unverified or the judge output failed validation, plus the candidates that
   were judged not relevant.
4. **Report** — the reviewed report preview and Markdown/HTML downloads. A
   button writes reviewed artifacts to `outputs/`.

Review decisions are stored separately from the audit, by default next to the
audit JSON as `<audit-stem>.reviews.json`. Each decision is keyed by
`passage_id|target_id|page`. The rubric level is never overwritten: an override
records a reviewer level alongside it, so the original assessment stays
auditable. Saving a review does not re-run retrieval or the judge, and the
coverage counts still describe the original scored findings.

The UI is single-user and local. It has no authentication and is not intended
to be exposed on a network.

## Stage 2 semantic verification (Phase 5)

Stage 1 checks that a quote exists on the cited page. Stage 2 checks whether the
quote supports the claim attributed to it. It combines:

1. a deterministic consistency guard for years, numbers, units, negation, and
   commitment status (planned, ongoing, achieved);
2. a local semantic decision from the NLI cross-encoder
   `cross-encoder/nli-deberta-v3-small`, or optionally from the local Ollama
   model.

No text is sent to a remote AI service. Enable it for one run:

```powershell
python -m prism17.pipeline data/reports/sjs_india_brsr_fy2023_24.pdf `
  --sdg 12 --max-candidates 12 --semantic `
  --out outputs/sjs.semantic.json
```

Or enable it in `config.yaml`:

```yaml
verification:
  semantic_enabled: true
  semantic_method: nli          # nli or llm
  nli_supported_threshold: 0.60
  nli_uncertain_threshold: 0.35
```

Outcomes and their effect on scoring:

| Status | Meaning | Can be scored? |
| --- | --- | --- |
| `supported` | The quote supports the claim. | Yes |
| `not_checked` | The optional model was unavailable or the input was empty. Stage 1 still applies. | Yes |
| `uncertain` | The guard found a mismatch, or the model score is between the two thresholds. | No, needs human review |
| `not_supported` | The model found that the quote does not support the claim. | No, needs human review |

The consistency guard is one-directional: everything in the claim must be
traceable to the quote, while the quote may contain extra context. A paraphrase
that drops a number or changes a year is flagged as `uncertain`, not silently
accepted. Stage 2 is conservative by design, and its effect has not yet been
benchmarked against human labels.

## Evidence coverage matrix

The report adds a five-dimension matrix for every reviewed claim:
specificity, measurable indicator, deadline, baseline, and reported progress.
Each dimension is `present`, `absent`, or `n/a`:

- **present** — the judge identified the dimension in the verified quote;
- **absent** — a disclosure gap in that quote, not proof that the company does
  nothing;
- **n/a** — the claim could not be assessed and needs human review.

The matrix is a view of the same judged findings, not a second score. It is
available in the Markdown report, the HTML report, the audit JSON, and the
Streamlit summary tab. Neutral flags use wording such as "low specificity" and
"disclosure gap".

## Multi-year tracking (Phase 6)

Compare quantified targets across two audits:

```powershell
python -m prism17.tracking `
  outputs/earlier.audit.json outputs/later.audit.json `
  --out outputs/tracking.json `
  --markdown outputs/tracking.md
```

Tracking is deliberately narrow:

- only scored Level 3 or Level 4 findings are compared by default (change with
  `--min-level` or `tracking.min_level`);
- one representative finding per target is chosen deterministically: highest
  rubric level, then judge confidence, then page and passage id;
- values are compared only when a recognised unit was extracted from both
  reports and the units match, including multipliers such as `million`; when
  both reports record a baseline, those baselines must also agree; a missing
  unit, mismatched unit, or changed baseline is reported as `not comparable`;
- a missing later target is labelled `not found in later report (needs review)`
  and is **never** described as discontinued;
- direction comes from the target wording; if the favourable direction cannot
  be inferred, the result is `changed`, not `progressing`.

The Markdown output states these limits. Review both source pages before using a
comparison.

## Cross-company comparison (Phase 6)

Place two or more audits side by side:

```powershell
python -m prism17.compare `
  outputs/company_a.audit.json outputs/company_b.audit.json `
  --labels "Company A" "Company B" --sdg 12 --sdg 13 `
  --out outputs/compare.json `
  --markdown outputs/compare.md
```

The comparison reports scored findings, high-level counts, best level, targets
with scored disclosure, per-goal metrics, level distribution, and review status.
High-level means rubric Level 3 or above by default; change it with
`compare.min_level` in `config.yaml` or `--min-level`, and the rendered label
follows the threshold (for example, `L2+`). The comparison counts only scored
findings and states that differences may come from report scope, retrieval depth,
or the number of judged candidates. It is **descriptive, not a ranking**, and it
does not assess real-world performance.

## Evaluation and annotation (Phase 5)

### Retrieval and relevance check on OSDG

```powershell
python eval/osdg_check.py --sample-size 200
python eval/osdg_check.py --sample-size 20 --skip-judge  # smoke test
```

Download the OSDG Community Dataset (version 2024.04) first, as described in the
script. This evaluates retrieval and relevance only, not the rubric.

### Human gold annotations

No gold annotations are bundled. Generate a blank template from an existing
audit:

```powershell
python eval/make_gold_template.py outputs/sjs.phase2.json `
  --out data/gold/annotation_template.csv
```

The `reviewer_*` columns are empty. Follow
[`data/gold/ANNOTATION_GUIDELINES.md`](data/gold/ANNOTATION_GUIDELINES.md),
fill a copy by hand, then score it:

```powershell
python eval/score_benchmark.py data/gold/annotations.csv `
  --out eval/benchmark.json
```

The scorer reports relevance precision/recall/F1, quote-verification agreement,
rubric exact agreement, within-one agreement, mean absolute difference, Cohen's
kappa, quadratic weighted kappa, semantic-support agreement, confusion matrices,
and the number of annotated pairs. With blank reviewer columns it writes
`status: "no_annotations"` and null metrics; it never fills in placeholder
numbers. Invalid reviewer values yield `status: "invalid_annotations"`, and
annotations without a model counterpart yield `status: "no_pairs"`, again with
null metrics. For a benchmark claim, use independent double annotation and
report agreement and the sampling frame.

See [`docs/RUBRIC.md`](docs/RUBRIC.md) for the rubric definition and calibration
notes.

## Run tests

```powershell
python -m pytest
```

## SDG data

`data/sdg_targets.json` contains all 17 goals and all 169 targets, with each
target's `id`, `goal`, and `text`. Target texts follow the UN 2030 Agenda
framework. The source note is stored in the JSON file.

## Configuration

Edit `config.yaml` to change the embedding model, retrieval depth, LLM model,
verification thresholds, or report defaults. Paths are resolved relative to the
project root.

## Current limitations

- Stage 2 semantic verification has not yet been validated end to end on the
  bundled sample with the NLI model enabled. It is disabled by default to avoid a
  model download on a first CPU run.
- The deterministic consistency guard is conservative. A faithful paraphrase
  that omits a number, year, or unit can be marked `uncertain` and sent to human
  review.
- No human gold annotations are bundled, so there are no measured benchmark
  results for relevance, rubric levels, or semantic support. The annotation
  scripts are tooling, not evidence.
- Quote verification checks that the quote exists on the source page. Stage 2
  checks support, but neither stage proves that the company's statement is true.
- Mapping a company passage to a UN SDG target is an approximation. A human
  reviewer should confirm that the target is a reasonable fit.
- The local judge can misclassify a passage. Findings that fail schema checks or
  quote verification are kept as "needs review" and never scored.
- The default 7B judge is conservative. In local testing it marked many
  plausibly related passages as not relevant, so recall is lower than a larger
  model would give. Raise `--max-candidates` and review the unscored output
  before drawing conclusions from a zero-score run.
- Scanned pages without a text layer are reported as unreadable. OCR is not part
  of the current pipeline.
- Table extraction is best-effort; a table quote may be verified against the
  parsed table passage when the raw page layout differs.
- Dimension flags come from the judge. A prior-year column in a table can be
  present in the quote while the judge still marks `baseline` false, so the
  listed disclosure gaps and the coverage matrix should be read alongside the
  quote.
- Judging every candidate on a laptop is slow. Use `--sdg`, `--max-pages`, and
  `--max-candidates` for quick runs.
- Coverage is limited to the focus goals, targets, and candidates that were
  retrieved and judged in a run. It is not a complete inventory of the report.
- Report guidance is deterministic template text keyed to rubric gaps. It is
  not tailored to the company's sector, strategy, or prior commitments.
- Human-review decisions annotate the report; they do not change the rubric
  level, coverage counts, or the audit JSON. Treat them as an audit trail for a
  reviewer, not as a second scoring system.
- Multi-year tracking compares quantified Level 3/4 disclosure by default. It does
  not match reworded indicators, adjust for restatements, or establish a trend. A
  target absent from a later report may have been moved to another section or
  simply not retrieved.
- Cross-company comparison is descriptive. It does not normalize for report
  length, page coverage, sector, or retrieval settings, so it must not be used
  as a league table.
- The Streamlit UI is single-user and local, with no authentication. Do not
  expose it on a network. A review file must be kept with its audit JSON because
  decisions are keyed to that run's passages and pages.
- Reports are Markdown and HTML; there is no PDF export.

## Responsible use

- A disclosure gap is not proof of poor performance.
- A high level reflects the presence of measurable, time-bound, baseline-
  anchored disclosure, not the quality of the underlying activity.
- Quotes are checked against the source page, but semantic support and rubric
  decisions should be reviewed by a human.
- A missing target in a later report is not evidence that the target was
  discontinued.
- A cross-company table is a comparison of disclosure, not a ranking.
- The tool must not be used to make automated accusations about a company.

## Sample document

`data/reports/sjs_india_brsr_fy2023_24.pdf` is a publicly filed BRSR used for
local testing. It remains the property of its publisher and is included only as
a development fixture.
