# Prism17

**Evidence-grounded SDG disclosure auditing.** Prism17 reads a corporate
sustainability report, BRSR filing, or annual report and produces an SDG
Alignment Report in which every finding is tied to a verified quote and page
number.

> Prism17 assesses **disclosure quality**, not real-world sustainability
> performance. It is decision support: a human reviewer must confirm or override
> every finding before it is used.

## Status

Phases 1-6 are implemented: dense retrieval over the UN SDG targets, a
strict-JSON local Ollama judge, quote verification, an optional semantic-support
check, a deterministic L1-L4 rubric, an SDG Alignment Report, a local Streamlit
review UI, multi-year tracking, descriptive cross-company comparison, and
human-annotation benchmark tooling.

Phase 2 is deliberately conservative: a finding is scored only when its quote is
verified against the source page and the rubric can assign a level. Malformed
model output and unverifiable quotes are retained as "needs review", not
silently dropped. Stage 2 semantic support is opt-in and off by default. The
Phase 2 pipeline was verified end to end with the real local model on the
bundled sample. Stage 2 was also checked on five quote-verified findings; it
remains opt-in because the small NLI model is conservative on some paraphrases.

No human gold annotations exist in this repository, so no benchmark metrics are
claimed. The tooling writes null metrics and an explicit status until valid
paired annotations exist.

## Requirements

- Python 3.11 or newer
- A local [Ollama](https://ollama.com/) installation (needed from Phase 2)
- No GPU required

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
    osdg/
    reports/            manifest.csv and the bundled S.J.S. sample
    dataset_prism17/    local DEV/test/excluded PDFs and OSDG data
    gold/
  docs/
    RUBRIC.md
    ANNOTATION_GUIDELINES.md
  eval/
    export_candidates.py
    osdg_check.py
    make_gold_template.py
    score_benchmark.py
  examples/
    sjs.audit.json
    sjs.parse.json
    sjs.report.md
    sjs.report.html
  track.py
  compare.py
  prism17/
    config.py
    parse.py
    models.py
    retrieve.py
    judge.py
    semantic.py
    verify.py
    rubric.py
    coverage.py
    guidance.py
    report.py
    review.py
    ui.py
    pipeline.py
    tracking.py
    compare.py
    benchmark.py
    evaluation.py
    audit_io.py
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
python -m prism17.parse data/reports/sjs_india_brsr_fy2023_24.pdf --out examples/sjs.parse.json
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
python -m prism17.pipeline data/reports/sjs_india_brsr_fy2023_24.pdf --out examples/sjs.audit.json
```

For a quicker first run, restrict the goals, pages, and number of candidates:

```powershell
python -m prism17.pipeline data/reports/sjs_india_brsr_fy2023_24.pdf `
  --sdg 13 --max-pages 20 --max-candidates 20 --out examples/sjs.audit.json
```

The pipeline:

1. parses the PDF into page-level passages;
2. embeds passages and the SDG targets locally with
   `BAAI/bge-small-en-v1.5`, then retrieves candidates with FAISS on CPU;
3. asks the local Ollama judge for strict JSON on each candidate;
4. verifies the returned quote against the source page (exact match after
   whitespace/hyphenation normalisation, then RapidFuzz partial ratio >= 95);
5. optionally runs Stage 2 semantic-support verification (`--semantic`), then
   applies the deterministic rubric;
6. prints scored findings with page numbers and verified quotes, and writes the
   full JSON, including findings that need human review;
7. optionally renders a Markdown and/or HTML SDG Alignment Report when
   `--report` or `--html-report` is used.

Useful options: `--model llama3.1:8b`, `--top-k`, `--min-score`,
`--fuzzy-threshold`, `--max-candidates`, `--no-tables`, `--scored-only`,
`--semantic`, `--no-semantic`, `--semantic-method nli|llm`, and `--verbose`. The
first run downloads the embedding model and caches embeddings under
`.cache/prism17/`.

### Phase 2 verification on the sample report

The bundled 59-page BRSR was run end to end with `qwen2.5:7b-instruct` on CPU:

```powershell
python -m prism17.pipeline data/reports/sjs_india_brsr_fy2023_24.pdf `
  --sdg 12 --max-candidates 12 --out examples/sjs.audit.json
```

- 59 pages, 326 passages (13 tables), 0 unreadable pages.
- 978 Goal 12 candidates retrieved from 11 targets; the top 12 were judged.
- 2 findings scored, 10 judged not relevant, 0 needs review.
- Scored: target 12.5, page 49, Level 2, quote verified with `page_fuzzy`
  96.84; target 12.6, page 25, Level 1, quote verified with `page_exact` 100.0.
- About 10.5 minutes of wall-clock time, dominated by 12 local judge calls.

The artifact is `examples/sjs.audit.json`. Every scored entry carries
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
  --out examples/sjs.audit.json `
  --report examples/sjs.report.md `
  --html-report examples/sjs.report.html
```

From an existing audit JSON:

```powershell
python -m prism17.report examples/sjs.audit.json `
  --out examples/sjs.report.md `
  --html examples/sjs.report.html
```

Use `--company "Example Ltd"` and `--title "..."` to override the header.

To render a reviewed report from the CLI, pass the review file saved by the UI:

```powershell
python -m prism17.report examples/sjs.audit.json `
  --reviews outputs/sjs.phase2.reviews.json `
  --out outputs/sjs.reviewed.md `
  --html outputs/sjs.reviewed.html
```

The report contains:

- report details and a disclosure-scope disclaimer;
- a summary of scored findings, level distribution, and coverage;
- coverage by goal, including targets with and without scored disclosure;
- an evidence coverage matrix showing, per goal, how many scored findings
  disclose each rubric dimension (specificity, measurable indicator, deadline,
  baseline, and reported progress);
- scored findings grouped by goal and target, each with page, verified quote,
  level, assessment, disclosure gaps, and suggested improvements;
- goals without scored findings, with neutral guidance and a retrieval caveat;
- findings that need human review, listed separately;
- method, limitations, and responsible-use notes.

The bundled sample's report is `examples/sjs.report.md`, with an HTML companion
at `examples/sjs.report.html`. Coverage counts only what was retrieved and
judged: a target with no scored finding is reported as "not identified", not as
evidence that the company does nothing.

## Human review in the Streamlit UI (Phase 4)

Start the local UI from the project root:

```powershell
streamlit run app.py
```

The UI opens on the bundled audit (`examples/sjs.audit.json`) if it exists. Use
the sidebar to load another audit JSON, or to run a new audit from a PDF path or
upload. A new audit uses the same local pipeline and the same Ollama model as the
CLI; run limits for pages and candidates are exposed so a first run stays short.

The four tabs are:

1. **Summary & coverage** — scored findings, targets with disclosure, judged
   candidates, review status, per-goal and per-target coverage, and the
   five-dimension evidence coverage matrix.
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

## Semantic support (Stage 2, Phase 5)

Stage 1 verifies that the quote exists on the cited page. Stage 2 asks whether
the quote supports the claim. It is opt-in because the NLI model adds startup
and CPU cost:

```powershell
python -m prism17.pipeline data/reports/sjs_india_brsr_fy2023_24.pdf `
  --sdg 12 --max-candidates 12 --semantic --out outputs/sjs.semantic.json
```

Choose the method with `--semantic-method nli` (default) or
`--semantic-method llm`. Set `verification.semantic_enabled: true` to enable it
for every run, or pass `--no-semantic` to override the config.

A deterministic guard first rejects claims that share no content words with the
quote. The NLI or Ollama method then assigns:

- `supported`: the quote supports the claim; the finding can be scored;
- `uncertain`: ambiguous or the model was unavailable; needs human review;
- `not_supported`: the quote does not support the claim; not scored;
- `not_checked`: Stage 2 was disabled.

`uncertain` and `not_supported` findings keep `level: null` and are never
scored.

Stage 2 was checked on five quote-verified S.J.S. findings. Three direct
paraphrases were `supported` (entailment 0.982-0.996). Two findings were
`not_supported`: one claim added a conclusion absent from the quote (0.001),
and one faithful paraphrase scored low (0.235). The second case is a
conservative false negative, so the finding stays unscored for human review
rather than being accepted. Because the small NLI model is conservative on
some paraphrases, Stage 2 remains opt-in and disabled by default. Treat its
labels as a review aid, not as ground truth.

## Multi-year tracking (Phase 6)

Compare two audit JSON files for the same company:

```powershell
python track.py outputs/alpha_2022.audit.json `
  outputs/alpha_2023.audit.json `
  --earlier-label "Alpha 2022" --later-label "Alpha 2023" `
  --out outputs/alpha.tracking.json --markdown outputs/alpha.tracking.md
```

`python -m prism17.tracking ...` is equivalent.

Tracking is deliberately narrow:

- only scored findings at or above the configured rubric level are compared
  (Level 3 by default; change with `--min-level` or `tracking.min_level`);
- one representative finding per target is chosen deterministically: highest
  rubric level, then judge confidence, then page and passage id;
- targets are matched by exact SDG target ID; reworded indicators are not
  matched;
- values are compared only when a recognised unit was extracted from both
  reports and the units match, including multipliers such as `million`; when
  both reports record a baseline, those baselines must also agree;
- a missing unit, mismatched unit, or changed baseline is reported as
  `not comparable`;
- a missing later target is labelled `not found in later report (needs review)`
  and is **never** described as discontinued;
- direction comes from the target wording; if the favourable direction cannot
  be inferred, the result is `changed`, not `progressing`.

Statuses are `progressing`, `unchanged`, `changed`, `not_comparable`,
`not_found_in_later_report`, and `new_in_later_report`.

## Cross-company comparison (Phase 6)

```powershell
python compare.py outputs/alpha.audit.json outputs/beta.audit.json `
  --labels "Alpha" "Beta" --sdg 13 `
  --out outputs/compare.json --markdown outputs/compare.md
```

`python -m prism17.compare ...` is equivalent.

The comparison is descriptive and is not a ranking. It reports scored findings,
targets with scored disclosure, rubric-level distribution, review status, and a
configurable high-level count (`compare.min_level`, `--min-level`, and
`compare.high_level_label`; an explicit override renders its own `L<n>+` label).
It does not normalize for report length, page coverage, sector, or retrieval
settings.

## Human-annotation benchmark (Phase 5 tooling)

Export a stratified sample directly from the DEV report PDFs:

```powershell
python eval/export_candidates.py --judge --sample-size 175 --per-report 25
```

The script reads `data/reports/manifest.csv`, opens only rows whose `split` is
`dev`, and never parses, judges, or tunes on `test` or `excluded` reports. It
resolves `<report_id>.pdf` (also accepting `.pdf.pdf`) in
`data/dataset_prism17/` or `data/reports/`, runs retrieval, optionally judges a
page-diverse subset, and samples scored findings across rubric levels plus
rejected, table, and review passages. Reviewer columns are blank by design.
When judged findings cannot fill the requested sample, remaining slots are
filled with retrieval-only table and body passages interleaved across DEV
reports. `eval/annotate_template.provenance.json` records the manifest digest,
source PDFs, split, seed, sample counts, and strata. Use `--no-judge` to export
retrieval candidates without model columns.

Generate a blank template from one or more audit files:

```powershell
python eval/make_gold_template.py outputs/alpha.audit.json `
  --labels "Alpha 2022" --out data/gold/annotations.csv
```

The reviewer columns are intentionally blank. Fill them independently from the
PDF, following [`docs/ANNOTATION_GUIDELINES.md`](docs/ANNOTATION_GUIDELINES.md).
Then score:

```powershell
python eval/score_benchmark.py data/gold/annotations.csv `
  --out eval/benchmark.json
```

Scoring is paired and reports accuracy, precision, recall, F1, confusion
matrices, rubric exact/within-one agreement, mean absolute difference, Cohen's
kappa, quadratic weighted kappa, semantic-support agreement, and the number of
annotated pairs. With blank reviewer columns it writes
`status: "no_annotations"` and null metrics; it never fills in placeholder
numbers. Invalid reviewer values yield `status: "invalid_annotations"`, and
annotations without a model counterpart yield `status: "no_pairs"`, again with
null metrics. No human annotations exist in this repository, so no benchmark
numbers are reported.

## OSDG retrieval check

`eval/osdg_check.py` runs a retrieval-only relevance check against the OSDG
community dataset (version 2024.04, Zenodo DOI `10.5281/zenodo.11441197`,
CC-BY-4.0). The dataset is tab-delimited despite its `.csv` extension. The
script records the dataset version and file digest in the provenance block. A
full sample run has not been executed in this checkout; no OSDG metrics are
claimed.

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

## Current limitations (Phases 2-6)

- Quote verification checks that the quote exists on the source page. Stage 2
  semantic support is optional and off by default; when it is enabled, the NLI
  model may be unavailable and the finding is then marked `uncertain`.
- Stage 2 was checked on five S.J.S. findings, but that is a small DEV sample.
  The NLI model rejected one faithful paraphrase (entailment 0.235), so its
  labels are a review aid, not a validated classifier.
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
  listed disclosure gaps should be read alongside the quote.
- Judging every candidate on a laptop is slow. Use `--sdg`, `--max-pages`, and
  `--max-candidates` for quick runs.
- Coverage is limited to the focus goals, targets, and candidates that were
  retrieved and judged in a run. It is not a complete inventory of the report.
- Report guidance is deterministic template text keyed to rubric gaps. It is
  not tailored to the company's sector, strategy, or prior commitments.
- Human-review decisions annotate the report; they do not change the rubric
  level, coverage counts, or the audit JSON. Treat them as an audit trail for a
  reviewer, not as a second scoring system.
- The Streamlit UI is single-user and local, with no authentication. Do not
  expose it on a network. A review file must be kept with its audit JSON because
  decisions are keyed to that run's passages and pages.
- Multi-year tracking matches exact SDG target IDs and comparable units only. It
  does not match reworded indicators, adjust for restatements, or establish a
  trend. A target absent from a later report may have moved section or may not
  have been retrieved; the result needs review.
- Cross-company comparison is descriptive. It does not normalize for report
  length, page coverage, sector, or retrieval settings, so it must not be used
  as a ranking or as a performance judgment.
- No human gold annotations exist, so the benchmark reports null metrics and a
  status. No benchmark accuracy, kappa, or agreement number is claimed.
- A full OSDG retrieval/relevance evaluation has not been run in this checkout.
- Reports are Markdown and HTML; there is no PDF export.

## Responsible use

- A disclosure gap is not proof of poor performance.
- A high level reflects the presence of measurable, time-bound, baseline-
  anchored disclosure, not the quality of the underlying activity.
- Quotes are checked against the source page, but semantic support and rubric
  decisions should be reviewed by a human.
- The tool must not be used to make automated accusations about a company.

## Sample document

`data/reports/sjs_india_brsr_fy2023_24.pdf` is a publicly filed BRSR used for
local testing. It remains the property of its publisher and is included only as
a development fixture.
