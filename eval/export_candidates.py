#!/usr/bin/env python
"""Export a stratified, DEV-only annotation sample from report PDFs.

Only reports whose ``manifest.csv`` row has ``split == "dev"`` are opened.
Reports marked ``test`` or ``excluded`` are never parsed, retrieved, judged, or
used to tune anything. The sample covers scored rubrics, passages the judge
rejected, and table passages, and is written to a blank reviewer template.

Run:
    python eval/export_candidates.py --judge
    python eval/export_candidates.py --no-judge --sample-size 60
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(PROJECT_ROOT))

from prism17.benchmark import BENCHMARK_FIELDS, write_template  # noqa: E402
from prism17.config import load_config  # noqa: E402
from prism17.evaluation import file_info, utc_now_iso  # noqa: E402
from prism17.judge import judge_candidates  # noqa: E402
from prism17.models import (  # noqa: E402
    Candidate,
    VerifiedFinding,
    is_not_relevant,
    needs_human_review,
)
from prism17.parse import ParsedDocument, parse_pdf  # noqa: E402
from prism17.retrieve import retrieve_candidates  # noqa: E402
from prism17.rubric import grade_findings  # noqa: E402
from prism17.verify import verify_findings  # noqa: E402

DEFAULT_MANIFEST = "data/reports/manifest.csv"
DEFAULT_DATASET_DIR = "data/dataset_prism17"
DEFAULT_REPORTS_DIR = "data/reports"
DEFAULT_OUT = "eval/annotate_template.csv"
DEFAULT_SAMPLE_SIZE = 175
DEFAULT_PER_REPORT = 60
DEFAULT_SEED = 17

# Target mix for the sample. Quotas are approximate; remaining slots are filled
# from any judged finding. If too few findings exist, table and body passages
# are added with blank model columns instead.
STRATUM_SHARES = {
    "scored": 0.40,
    "rejected": 0.30,
    "table": 0.20,
    "review": 0.10,
}


@dataclass
class ReportSource:
    """One DEV report and the model output produced for it."""

    report_id: str
    company: str
    year: str
    split: str
    path: Path
    document: ParsedDocument
    findings: list[VerifiedFinding] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    top_by_passage: dict[str, Candidate] = field(default_factory=dict)


def _relative(path: str | Path) -> str:
    """Store repository-relative paths so saved JSON has no absolute paths."""
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _relative_file_info(path: str | Path) -> dict[str, Any]:
    info = file_info(Path(path).expanduser().resolve())
    info["path"] = _relative(path)
    return info


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read and validate a manifest with ``report_id`` and ``split``."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest.csv has no header")
        missing = {"report_id", "split"} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                "manifest.csv is missing required columns: "
                + ", ".join(sorted(missing))
            )
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]

    if not rows:
        raise ValueError("manifest.csv has no data rows")
    seen: set[str] = set()
    for row in rows:
        report_id = row["report_id"]
        if not report_id:
            raise ValueError("manifest row has an empty report_id")
        if report_id in seen:
            raise ValueError(f"duplicate report_id in manifest: {report_id}")
        seen.add(report_id)
    return rows


def dev_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """Return only rows explicitly marked as DEV, preserving manifest order."""
    dev = [row for row in rows if row["split"].lower() == "dev"]
    if not dev:
        raise ValueError("manifest has no reports with split == 'dev'")
    return dev


def resolve_report_pdf(
    report_id: str,
    *,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> Path:
    """Resolve ``<report_id>.pdf`` without ever scanning non-DEV files."""
    dataset = Path(dataset_dir)
    reports = Path(reports_dir)
    candidates = [
        dataset / f"{report_id}.pdf",
        dataset / f"{report_id}.pdf.pdf",
        reports / f"{report_id}.pdf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"no PDF found for DEV report {report_id!r}; looked for "
        f"{report_id}.pdf in {dataset} and {reports}"
    )


def _dedupe_top_candidate(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Keep the highest-scoring target for each passage."""
    best: dict[str, Candidate] = {}
    for candidate in candidates:
        current = best.get(candidate.passage_id)
        if current is None or candidate.score > current.score:
            best[candidate.passage_id] = candidate
    return list(best.values())


def select_candidates_for_judging(
    candidates: Sequence[Candidate],
    limit: int,
    rng: random.Random,
) -> list[Candidate]:
    """Select a page-diverse candidate set with explicit table coverage."""
    if limit <= 0:
        return []
    unique = _dedupe_top_candidate(candidates)
    if len(unique) <= limit:
        return sorted(unique, key=lambda item: (item.page, item.passage_id))

    tables = [item for item in unique if item.is_table]
    body = [item for item in unique if not item.is_table]
    table_limit = min(len(tables), max(1, int(limit * 0.30)))
    selected = sorted(
        rng.sample(tables, table_limit) if len(tables) > table_limit else tables,
        key=lambda item: (item.page, item.passage_id),
    )

    remaining = limit - len(selected)
    if remaining <= 0:
        return selected[:limit]

    # Round-robin across pages so a single dense section does not dominate.
    by_page: dict[int, list[Candidate]] = {}
    for item in body:
        by_page.setdefault(item.page, []).append(item)
    for page_items in by_page.values():
        page_items.sort(key=lambda item: item.score, reverse=True)
    page_order = sorted(
        by_page,
        key=lambda page: (-max(item.score for item in by_page[page]), page),
    )
    while remaining > 0 and page_order:
        next_order: list[int] = []
        for page in page_order:
            if remaining <= 0:
                break
            page_items = by_page.get(page) or []
            if page_items:
                selected.append(page_items.pop(0))
                remaining -= 1
            if page_items:
                next_order.append(page)
        page_order = next_order

    return selected


def _balanced_scored(
    findings: Sequence[VerifiedFinding],
    limit: int,
    rng: random.Random,
) -> list[VerifiedFinding]:
    """Take scored findings round-robin across rubric levels."""
    by_level: dict[int, list[VerifiedFinding]] = {1: [], 2: [], 3: [], 4: []}
    for finding in findings:
        if finding.level is not None:
            by_level.setdefault(int(finding.level), []).append(finding)
    for level_findings in by_level.values():
        rng.shuffle(level_findings)

    selected: list[VerifiedFinding] = []
    while len(selected) < limit and any(by_level.values()):
        for level in sorted(by_level):
            if len(selected) >= limit:
                break
            if by_level[level]:
                selected.append(by_level[level].pop())
    return selected


def _take_unique(
    pool: Sequence[VerifiedFinding],
    limit: int,
    selected: list[VerifiedFinding],
    seen: set[tuple[str, str]],
    total: int,
    rng: random.Random,
) -> None:
    shuffled = list(pool)
    rng.shuffle(shuffled)
    taken = 0
    for finding in shuffled:
        if len(selected) >= total or taken >= limit:
            return
        candidate = finding.judged.candidate
        key = (candidate.passage_id, finding.judged.sdg_target)
        if key in seen:
            continue
        selected.append(finding)
        seen.add(key)
        taken += 1


def sample_findings(
    findings: Sequence[VerifiedFinding],
    size: int,
    *,
    seed: int = DEFAULT_SEED,
) -> list[VerifiedFinding]:
    """Stratify by rubric level, rejected status, table layout, review state."""
    if size <= 0 or not findings:
        return []
    rng = random.Random(seed)
    total = min(size, len(findings))
    selected: list[VerifiedFinding] = []
    seen: set[tuple[str, str]] = set()

    scored = [finding for finding in findings if finding.is_scored]
    scored_limit = max(1, int(total * STRATUM_SHARES["scored"]))
    for finding in _balanced_scored(scored, scored_limit, rng):
        candidate = finding.judged.candidate
        key = (candidate.passage_id, finding.judged.sdg_target)
        if key not in seen:
            selected.append(finding)
            seen.add(key)

    _take_unique(
        [finding for finding in findings if is_not_relevant(finding)],
        max(1, int(total * STRATUM_SHARES["rejected"])),
        selected,
        seen,
        total,
        rng,
    )
    _take_unique(
        [finding for finding in findings if finding.judged.candidate.is_table],
        max(1, int(total * STRATUM_SHARES["table"])),
        selected,
        seen,
        total,
        rng,
    )
    _take_unique(
        [finding for finding in findings if needs_human_review(finding)],
        max(1, int(total * STRATUM_SHARES["review"])),
        selected,
        seen,
        total,
        rng,
    )
    _take_unique(findings, total, selected, seen, total, rng)
    return selected[:total]


def _finding_row(
    finding: VerifiedFinding,
    source: ReportSource,
    index: int,
) -> dict[str, Any]:
    candidate = finding.judged.candidate
    return {
        "annotation_id": f"ann-{source.report_id}-{index:04d}",
        "company": source.company,
        "year": source.year,
        "source_path": _relative(source.path),
        "title": source.document.title,
        "target_id": finding.judged.sdg_target,
        "page": finding.source_page,
        "passage_id": candidate.passage_id,
        "claim": finding.judged.claim,
        "quote": finding.judged.supporting_quote,
        "model_relevant": int(bool(finding.judged.relevant)),
        "model_level": "" if finding.level is None else int(finding.level),
        "model_quote_ok": int(bool(finding.quote_verified)),
        "model_semantic_support": finding.semantic_support,
        "model_review_reason": finding.review_reason or "",
        "reviewer_relevant": "",
        "reviewer_level": "",
        "reviewer_quote_ok": "",
        "reviewer_semantic_support": "",
        "reviewer_notes": "",
        "annotator": "",
        "annotated_at_utc": "",
    }


def _passage_row(
    source: ReportSource,
    passage_id: str,
    index: int,
) -> dict[str, Any] | None:
    candidate = source.top_by_passage.get(passage_id)
    if candidate is None:
        return None
    passage = next(
        (
            item
            for item in source.document.passages
            if item.passage_id == passage_id
        ),
        None,
    )
    if passage is None or not passage.text.strip():
        return None
    return {
        "annotation_id": f"ann-{source.report_id}-u{index:04d}",
        "company": source.company,
        "year": source.year,
        "source_path": _relative(source.path),
        "title": source.document.title,
        "target_id": candidate.target_id,
        "page": passage.page,
        "passage_id": passage.passage_id,
        "claim": "",
        "quote": passage.text,
        "model_relevant": "",
        "model_level": "",
        "model_quote_ok": "",
        "model_semantic_support": "",
        "model_review_reason": "not judged; retrieval candidate only",
        "reviewer_relevant": "",
        "reviewer_level": "",
        "reviewer_quote_ok": "",
        "reviewer_semantic_support": "",
        "reviewer_notes": "",
        "annotator": "",
        "annotated_at_utc": "",
    }


def _fill_with_passages(
    rows: list[dict[str, Any]],
    sources: Sequence[ReportSource],
    selected: Sequence[VerifiedFinding],
    size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Add unjudged table/body passages, round-robin across reports.

    Interleaving sources keeps one long report from consuming every remaining
    slot, so the sample still covers each DEV report.
    """
    if len(rows) >= size:
        return rows
    used = {
        (row["source_path"], str(row["passage_id"])) for row in rows
    }
    queues: list[tuple[ReportSource, list[Any]]] = []
    for source in sources:
        source_path = _relative(source.path)
        source_finding_ids = {id(finding) for finding in source.findings}
        selected_passages = {
            finding.judged.candidate.passage_id
            for finding in selected
            if id(finding) in source_finding_ids
        }
        # Table passages first, then a random body sample.
        ordered = sorted(
            source.document.passages,
            key=lambda passage: (
                not passage.is_table,
                passage.page,
                passage.passage_id,
            ),
        )
        table = [passage for passage in ordered if passage.is_table]
        body = [passage for passage in ordered if not passage.is_table]
        rng.shuffle(body)
        queue = [
            passage
            for passage in [*table, *body]
            if passage.passage_id not in selected_passages
            and (source_path, str(passage.passage_id)) not in used
        ]
        queues.append((source, queue))

    while len(rows) < size and any(queue for _, queue in queues):
        for source, queue in queues:
            if len(rows) >= size:
                break
            while queue:
                passage = queue.pop(0)
                source_path = _relative(source.path)
                if (source_path, str(passage.passage_id)) in used:
                    continue
                row = _passage_row(
                    source, passage.passage_id, index=len(rows) + 1
                )
                if row is not None:
                    rows.append(row)
                    used.add((source_path, str(row["passage_id"])))
                    break
    return rows


def _retrieve_source(
    source: ReportSource, config: dict[str, Any]
) -> None:
    candidates = retrieve_candidates(source.document, config)
    source.candidates = candidates
    source.top_by_passage = {
        item.passage_id: item
        for item in _dedupe_top_candidate(candidates)
    }


def _judge_source(
    source: ReportSource,
    config: dict[str, Any],
    *,
    per_report: int,
    seed: int,
) -> None:
    selected = select_candidates_for_judging(
        source.candidates, per_report, random.Random(seed)
    )

    def progress(index: int, total: int, finding: Any) -> None:
        state = "relevant" if finding.relevant else "not relevant"
        print(
            f"  judged {index}/{total} (page {finding.candidate.page}, "
            f"target {finding.candidate.target_id}: {state})",
            file=sys.stderr,
        )

    judged = judge_candidates(
        selected, config, progress_callback=progress
    )
    source.findings = grade_findings(
        verify_findings(judged, source.document, config), config
    )


def _stratum_counts(
    findings: Sequence[VerifiedFinding],
) -> dict[str, Any]:
    return {
        "scored": sum(1 for finding in findings if finding.is_scored),
        "rejected": sum(1 for finding in findings if is_not_relevant(finding)),
        "review": sum(1 for finding in findings if needs_human_review(finding)),
        "table": sum(
            1 for finding in findings if finding.judged.candidate.is_table
        ),
        "levels": {
            str(level): sum(
                1 for finding in findings if finding.level == level
            )
            for level in (1, 2, 3, 4)
        },
    }


def _provenance(
    *,
    manifest_path: str | Path,
    sources: Sequence[ReportSource],
    rows: Sequence[dict[str, Any]],
    sample_size: int,
    seed: int,
    judged: bool,
    per_report: int,
) -> dict[str, Any]:
    return {
        "created_at_utc": utc_now_iso(),
        "dataset": {
            "name": "Prism17 DEV annotation sample",
            "version": "0.1",
            "split": "dev",
            "manifest": _relative_file_info(manifest_path),
        },
        "reports": [
            {
                "report_id": source.report_id,
                "company": source.company,
                "year": source.year,
                "split": source.split,
                "path": _relative(source.path),
                "sha256": file_info(source.path)["sha256"],
                "pages": source.document.page_count,
                "passages": len(source.document.passages),
                "table_passages": source.document.table_count,
                "judged": len(source.findings),
                "strata": _stratum_counts(source.findings),
            }
            for source in sources
        ],
        "sample": {
            "requested": sample_size,
            "rows": len(rows),
            "seed": seed,
            "judged_model_columns": judged,
            "per_report_judge_limit": per_report,
        },
        "notes": [
            "Only manifest rows with split == 'dev' were opened.",
            "Test and excluded reports were not parsed, judged, or used.",
            "Reviewer columns are blank by design; fill them from the PDF.",
            "Rows with blank model columns are retrieval candidates, not "
            "judged predictions, and cannot be paired in benchmark scoring.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export a DEV-only stratified annotation sample from report PDFs."
        )
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--provenance",
        default="eval/annotate_template.provenance.json",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE
    )
    parser.add_argument(
        "--per-report",
        type=int,
        default=DEFAULT_PER_REPORT,
        help="Passages judged per DEV report when --judge is used",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    judge_group = parser.add_mutually_exclusive_group()
    judge_group.add_argument(
        "--judge",
        action="store_true",
        help="Run the local judge (default)",
    )
    judge_group.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the local judge and export retrieval candidates only",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Development limit per report; never use to tune on test data",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        rows = dev_rows(load_manifest(args.manifest))
        config = load_config(args.config)
        sources: list[ReportSource] = []
        for row in rows:
            report_id = row["report_id"]
            path = resolve_report_pdf(
                report_id,
                dataset_dir=args.dataset_dir,
                reports_dir=args.reports_dir,
            )
            document = parse_pdf(path, config, max_pages=args.max_pages)
            source = ReportSource(
                report_id=report_id,
                company=row.get("company") or report_id,
                year=row.get("year") or "",
                split=row["split"],
                path=path,
                document=document,
            )
            print(
                f"DEV report: {report_id} ({len(document.passages)} passages, "
                f"{document.table_count} tables)",
                file=sys.stderr,
            )
            _retrieve_source(source, config)
            if not args.no_judge:
                _judge_source(
                    source,
                    config,
                    per_report=args.per_report,
                    seed=args.seed,
                )
            sources.append(source)

        all_findings = [
            finding for source in sources for finding in source.findings
        ]
        if all_findings:
            selected = sample_findings(
                all_findings, args.sample_size, seed=args.seed
            )
            finding_source = {
                id(finding): source
                for source in sources
                for finding in source.findings
            }
            template_rows: list[dict[str, Any]] = []
            for index, finding in enumerate(selected, start=1):
                source = finding_source[id(finding)]
                template_rows.append(_finding_row(finding, source, index))
        else:
            selected = []
            template_rows = []

        template_rows = _fill_with_passages(
            template_rows,
            sources,
            selected,
            args.sample_size,
            random.Random(args.seed),
        )
        template_rows.sort(
            key=lambda row: (
                str(row["company"]),
                int(row["page"]),
                str(row["passage_id"]),
            )
        )
        path = write_template(args.out, template_rows, force=args.force)
        provenance = _provenance(
            manifest_path=args.manifest,
            sources=sources,
            rows=template_rows,
            sample_size=args.sample_size,
            seed=args.seed,
            judged=not args.no_judge,
            per_report=args.per_report,
        )
        provenance_path = Path(args.provenance)
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(template_rows)} annotation rows to {path}")
    print(f"Wrote provenance to {provenance_path}")
    print(f"Columns: {', '.join(BENCHMARK_FIELDS)}")
    print(
        "Fill only the reviewer_* columns, annotator, and annotated_at_utc "
        "from the source PDF."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
