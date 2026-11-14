# SDG Alignment Report — sjs_india_brsr_fy2023_24

> This report assesses **disclosure quality**, not real-world sustainability performance. It is decision support: a human reviewer should confirm or override every finding before it is used.

## Report details

| Field | Value |
| --- | --- |
| Document | `sjs_india_brsr_fy2023_24.pdf` |
| Pages parsed | 59 |
| Focus goals | 12 |
| Local judge | qwen2.5:7b-instruct |
| Embedding model | BAAI/bge-small-en-v1.5 |
| Generated | 2026-11-13 11:37 UTC |
| Tool | Prism17 0.1.0 |

## Summary

| Measure | Value |
| --- | --- |
| Scored findings | 2 |
| Level distribution | L1 1 · L2 1 · L3 0 · L4 0 |
| Targets with scored disclosure | 2 of 11 focus targets |
| Judged candidates | 12 |
| Not relevant | 10 |
| Needs human review | 0 |
| Unreadable pages | 0 |
| Human review | Not reviewed (2 pending) |

Coverage counts are limited to the goals, targets, and candidates that were retrieved and judged in this run. A target without a scored finding is reported as not identified, not as evidence that the company does nothing.

## Coverage by goal

| Goal | Focus area | Targets with scored disclosure | Scored findings | Best level | Status |
| --- | --- | --- | --- | --- | --- |
| Goal 12 | Responsible Consumption and Production | 2 of 11 | 2 | Level 2 | partial disclosure |

## Evidence coverage matrix

Counts show how many scored findings disclose each of the five rubric dimensions. The flags come from the judge and describe disclosure in this run, not real-world performance. A low count is a disclosure observation, not evidence that the company does nothing.

| Goal | Scored findings | Specificity | Measurable indicator | Deadline | Baseline | Reported progress |
| --- | --- | --- | --- | --- | --- | --- |
| Goal 12 | 2 | 1/2 (50%) | 1/2 (50%) | 0/2 (0%) | 0/2 (0%) | 0/2 (0%) |
| All focus goals | 2 | 1/2 (50%) | 1/2 (50%) | 0/2 (0%) | 0/2 (0%) | 0/2 (0%) |

## Scored findings

### Goal 12 — Responsible Consumption and Production

Ensure sustainable consumption and production patterns.

**Goal status:** partial disclosure (2 scored findings; 2 of 11 targets with scored disclosure).

**Goal guidance:** Disclosures for this goal are specific but incomplete. Where material, add a target date, a baseline, and reported progress.

#### Target 12.5 — Level 2

*Target:* By 2030, substantially reduce waste generation through prevention, reduction, recycling and reuse

**Page 49 — quote verified (page_fuzzy 96.8)**

> "FY 2022-2023 (Previous Financial Year) Total Waste generated (in metric tonnes) Plastic waste (A) 698.860 654.9 E-waste (B) 0 0.48 Bio-medical waste (C) 0.0022 0.0025 Construction and demolition waste (D) 0 0 Battery waste (E) 0 0 Radioactive waste (F) 0 0 Other Hazardous waste (H): 1.52 3.48 Total (A+ B + C + D + E + F + G + H) 700.3842 658.8625"

- **Assessment:** Level 2: specific disclosure with at least one of a measurable indicator, deadline, baseline, or reported progress.
- **Disclosure gaps:**
  - No target date or deadline is disclosed.
  - No baseline or prior-year comparison is disclosed.
  - No reported progress or result is disclosed.
- **Suggested improvements:**
  - State a target date or the reporting period for the commitment.
  - State a baseline year and baseline value for comparison.
  - Report progress against the commitment, including the current result.

#### Target 12.6 — Level 1

*Target:* Encourage companies, especially large and transnational companies, to adopt sustainable practices and to integrate sustainability information into their reporting cycle

**Page 25 — quote verified (page_exact 100.0)**

> "The Company remains committed to upholding this principle and will continue to seek ways to improve our sustainability and safety practices."

- **Assessment:** Level 1: relevant but general disclosure without a concrete action, commitment, or measurable element.
- **Disclosure gaps:**
  - The disclosure is general rather than specific.
  - No measurable indicator is disclosed.
  - No target date or deadline is disclosed.
  - No baseline or prior-year comparison is disclosed.
  - No reported progress or result is disclosed.
- **Suggested improvements:**
  - Describe the specific action, policy, or commitment instead of a general statement.
  - Add a measurable indicator, such as a quantity, percentage, or intensity, with its unit and scope.
  - State a target date or the reporting period for the commitment.
  - State a baseline year and baseline value for comparison.
  - Report progress against the commitment, including the current result.

## Candidates excluded as not relevant

10 candidates were judged not relevant to the retrieved target and were not scored. This count describes retrieval and judging output, not the company.

## Method and limitations

- **Scope.** The report describes the presence and specificity of disclosure. It does not verify real-world impact, performance, or intent.
- **Parsing.** Text and tables are extracted page by page. Unreadable pages are recorded and never treated as evidence of no disclosure.
- **Retrieval.** Passages are compared with the UN SDG targets using a local sentence-embedding model and FAISS on CPU. Mapping a passage to a target is an approximation.
- **Judging.** A local Ollama model returns strict JSON. Malformed output is retained as needing human review, not scored.
- **Verification.** Stage 1 checks that each quote exists on the cited page, after whitespace and hyphenation normalisation, with a fuzzy fallback. Stage 2 semantic support is optional and off by default; when it is enabled, an ambiguous result keeps the finding unscored.
- **Rubric.** Levels 1-4 are assigned by deterministic rules over five disclosure dimensions. A high level means the disclosure is measurable, time-bound, baseline-anchored, and reports progress.
- **Human review.** Unverified quotes, malformed judge output, and ambiguous mappings require human review before use.

## Responsible use

- A disclosure gap is not proof of poor performance.
- A high level reflects measurable, time-bound, baseline-anchored disclosure, not the quality of the underlying activity.
- Quotes are checked against the source page; semantic support and rubric decisions should be reviewed by a human.
- This report must not be used to make automated accusations about a company.
