"""Tests for the DEV-only annotation candidate export."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

from prism17.models import Candidate, JudgedFinding, VerifiedFinding
from prism17.parse import ParsedDocument, Passage

ROOT = Path(__file__).resolve().parent.parent


def _load_export():
    spec = importlib.util.spec_from_file_location(
        "prism17_eval_export_candidates",
        ROOT / "eval" / "export_candidates.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXPORT = _load_export()


def _finding(
    *,
    target_id: str = "12.5",
    goal: int = 12,
    level: int | None = 2,
    relevant: bool = True,
    page: int = 10,
    is_table: bool = False,
    review_reason: str | None = None,
) -> VerifiedFinding:
    quote = "Example quote with enough words to verify against the page."
    candidate = Candidate(
        passage_id=f"p{page:04d}-{target_id.replace('.', '')}",
        page=page,
        target_id=target_id,
        target_goal=goal,
        target_text=f"Target {target_id}",
        score=0.8,
        passage_text=quote,
        is_table=is_table,
    )
    judged = JudgedFinding(
        candidate=candidate,
        relevant=relevant,
        sdg_goal=goal,
        sdg_target=target_id,
        claim="Example claim.",
        supporting_quote=quote,
        dimensions={"specificity": True},
        scored=True,
    )
    return VerifiedFinding(
        judged=judged,
        quote_verified=True,
        quote_method="page_exact",
        quote_score=100.0,
        source_page=page,
        semantic_support="not_checked",
        level=level if relevant else None,
        review_reason=review_reason,
    )


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.csv"
    path.write_text(
        "report_id,company,year,split\n"
        "alpha,Alpha,FY2024,dev\n"
        "beta,Beta,FY2024,test\n"
        "gamma,Gamma,FY2024,excluded\n"
        "delta,Delta,FY2024,dev\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Manifest and path handling
# ---------------------------------------------------------------------------


def test_manifest_requires_split_column(tmp_path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text("report_id,company\nalpha,Alpha\n", encoding="utf-8")

    with pytest.raises(ValueError, match="split"):
        EXPORT.load_manifest(path)


def test_dev_rows_keep_only_the_dev_split(tmp_path) -> None:
    rows = EXPORT.dev_rows(EXPORT.load_manifest(_manifest(tmp_path)))

    assert [row["report_id"] for row in rows] == ["alpha", "delta"]


def test_dev_rows_require_at_least_one_dev_report(tmp_path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text(
        "report_id,company,year,split\nbeta,Beta,FY2024,test\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no reports"):
        EXPORT.dev_rows(EXPORT.load_manifest(path))


def test_manifest_rejects_duplicate_report_ids(tmp_path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text(
        "report_id,company,year,split\n"
        "alpha,Alpha,FY2024,dev\n"
        "alpha,Alpha again,FY2024,dev\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        EXPORT.load_manifest(path)


def test_resolve_report_pdf_prefers_exact_name(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    reports = tmp_path / "reports"
    dataset.mkdir()
    reports.mkdir()
    exact = dataset / "alpha.pdf"
    exact.write_bytes(b"%PDF-1.4")

    resolved = EXPORT.resolve_report_pdf(
        "alpha", dataset_dir=dataset, reports_dir=reports
    )

    assert resolved == exact.resolve()


def test_resolve_report_pdf_accepts_double_pdf_suffix(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    reports = tmp_path / "reports"
    dataset.mkdir()
    reports.mkdir()
    doubled = dataset / "beta.pdf.pdf"
    doubled.write_bytes(b"%PDF-1.4")

    resolved = EXPORT.resolve_report_pdf(
        "beta", dataset_dir=dataset, reports_dir=reports
    )

    assert resolved == doubled.resolve()


def test_resolve_report_pdf_never_scans_unlisted_files(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    reports = tmp_path / "reports"
    dataset.mkdir()
    reports.mkdir()
    (dataset / "held_out.pdf").write_bytes(b"%PDF-1.4")

    with pytest.raises(FileNotFoundError):
        EXPORT.resolve_report_pdf(
            "missing", dataset_dir=dataset, reports_dir=reports
        )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_select_candidates_covers_tables_and_stays_within_limit() -> None:
    candidates = [
        Candidate(
            passage_id=f"b{index:04d}",
            page=index,
            target_id="13.2",
            target_goal=13,
            target_text="Target",
            score=0.9 - index * 0.01,
            passage_text="Body passage.",
            is_table=False,
        )
        for index in range(20)
    ] + [
        Candidate(
            passage_id=f"t{index:04d}",
            page=100 + index,
            target_id="12.5",
            target_goal=12,
            target_text="Target",
            score=0.6,
            passage_text="Table passage.",
            is_table=True,
        )
        for index in range(5)
    ]

    selected = EXPORT.select_candidates_for_judging(
        candidates, 8, random.Random(7)
    )

    assert len(selected) == 8
    assert any(candidate.is_table for candidate in selected)
    assert len({candidate.passage_id for candidate in selected}) == 8


def test_sample_findings_covers_levels_rejected_and_tables() -> None:
    findings = [
        _finding(target_id="12.5", level=1, page=1),
        _finding(target_id="12.6", level=2, page=2),
        _finding(target_id="12.7", level=3, page=3),
        _finding(target_id="12.8", level=4, page=4, is_table=True),
        _finding(
            target_id="13.1",
            goal=13,
            level=None,
            relevant=False,
            page=5,
        ),
        _finding(
            target_id="13.2",
            goal=13,
            level=None,
            relevant=False,
            page=6,
            is_table=True,
        ),
    ]

    sample = EXPORT.sample_findings(findings, 5, seed=3)

    assert len(sample) == 5
    assert any(finding.is_scored for finding in sample)
    assert any(EXPORT.is_not_relevant(finding) for finding in sample)
    assert any(finding.judged.candidate.is_table for finding in sample)
    assert {finding.level for finding in sample if finding.is_scored}


def test_sample_findings_never_exceeds_available_rows() -> None:
    findings = [_finding(page=index) for index in range(1, 4)]

    assert len(EXPORT.sample_findings(findings, 10)) == 3
    assert EXPORT.sample_findings([], 10) == []


def test_sample_findings_is_deterministic() -> None:
    findings = [_finding(page=index) for index in range(1, 20)]

    first = EXPORT.sample_findings(findings, 8, seed=11)
    second = EXPORT.sample_findings(findings, 8, seed=11)

    assert [finding.judged.candidate.passage_id for finding in first] == [
        finding.judged.candidate.passage_id for finding in second
    ]


def _source(tmp_path: Path, report_id: str, count: int) -> object:
    passages = [
        Passage(
            passage_id=f"p{index:04d}-001",
            page=index,
            section_heading=None,
            text=f"{report_id} passage {index} with enough words.",
            is_table=False,
        )
        for index in range(1, count + 1)
    ]
    document = ParsedDocument(
        pdf_path=f"data/reports/{report_id}.pdf",
        title=report_id,
        page_count=count,
        passages=passages,
    )
    top = {
        passage.passage_id: Candidate(
            passage_id=passage.passage_id,
            page=passage.page,
            target_id="13.2",
            target_goal=13,
            target_text="Target 13.2",
            score=0.8,
            passage_text=passage.text,
        )
        for passage in passages
    }
    return EXPORT.ReportSource(
        report_id=report_id,
        company=report_id,
        year="FY2024",
        split="dev",
        path=tmp_path / f"{report_id}.pdf",
        document=document,
        top_by_passage=top,
    )


def test_fill_with_passages_interleaves_reports(tmp_path) -> None:
    sources = [
        _source(tmp_path, "alpha", 5),
        _source(tmp_path, "beta", 5),
    ]

    rows = EXPORT._fill_with_passages(
        [], sources, [], 6, random.Random(3)
    )

    assert len(rows) == 6
    assert [row["company"] for row in rows] == [
        "alpha",
        "beta",
        "alpha",
        "beta",
        "alpha",
        "beta",
    ]
