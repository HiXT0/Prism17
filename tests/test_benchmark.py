"""Tests for annotation templates and benchmark scoring."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from prism17.benchmark import (
    ANNOTATION_FIELDS,
    build_annotation_rows,
    read_annotation_csv,
    score_annotations,
    write_annotation_csv,
)
from prism17.models import Candidate, JudgedFinding, VerifiedFinding

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _finding(
    target_id: str = "13.2",
    *,
    level: int | None = 3,
    relevant: bool = True,
    quote_verified: bool = True,
    semantic_support: str = "not_checked",
    review_reason: str | None = None,
    page: int = 10,
) -> VerifiedFinding:
    candidate = Candidate(
        passage_id=f"p{page:04d}-001",
        page=page,
        target_id=target_id,
        target_goal=int(target_id.split(".", 1)[0]),
        target_text="Target text",
        score=0.8,
        passage_text="Passage text",
    )
    judged = JudgedFinding(
        candidate=candidate,
        relevant=relevant,
        sdg_goal=int(target_id.split(".", 1)[0]),
        sdg_target=target_id,
        claim="Example claim.",
        supporting_quote="Example quote.",
        dimensions={"specificity": True},
        scored=True,
    )
    return VerifiedFinding(
        judged=judged,
        quote_verified=quote_verified,
        quote_method="page_exact" if quote_verified else None,
        quote_score=100.0 if quote_verified else 0.0,
        source_page=page,
        semantic_support=semantic_support,
        level=level if quote_verified else None,
        review_reason=review_reason,
    )


def _payload(findings: list[VerifiedFinding]) -> dict:
    return {
        "document": {
            "title": "Example Report FY2023-24",
            "pdf_path": "reports/example.pdf",
            "page_count": 59,
        },
        "findings": [finding.to_dict() for finding in findings],
    }


def _row(
    *,
    reviewer_relevant: str = "",
    model_relevant: str = "1",
    reviewer_level: str = "",
    model_level: str = "3",
    reviewer_quote_ok: str = "",
    model_quote_verified: str = "1",
    reviewer_semantic_support: str = "",
    model_semantic_support: str = "not_checked",
) -> dict[str, str]:
    return {
        "annotation_id": "ann-0001",
        "page": "10",
        "target_id": "13.2",
        "claim": "Example claim.",
        "quote": "Example quote.",
        "model_relevant": model_relevant,
        "model_level": model_level,
        "model_quote_verified": model_quote_verified,
        "model_semantic_support": model_semantic_support,
        "reviewer_relevant": reviewer_relevant,
        "reviewer_level": reviewer_level,
        "reviewer_quote_ok": reviewer_quote_ok,
        "reviewer_semantic_support": reviewer_semantic_support,
    }


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------


def test_template_includes_scored_and_reviewable_findings_only() -> None:
    payload = _payload(
        [
            _finding(page=10),
            _finding(quote_verified=False, page=11),
            _finding(relevant=False, level=None, page=12),
        ]
    )
    rows = build_annotation_rows(
        payload, audit_path="outputs/audit.json", prefix="sjs"
    )

    assert len(rows) == 2
    assert [row["page"] for row in rows] == ["10", "11"]
    assert rows[0]["model_level"] == "3"
    assert rows[1]["model_level"] == ""
    assert rows[0]["model_relevant"] == "1"
    assert rows[0]["reviewer_relevant"] == ""
    assert rows[0]["reviewer_level"] == ""
    assert rows[0]["reviewer_quote_ok"] == ""
    assert rows[0]["model_review_reason"] == ""
    assert rows[0]["company"] == "Example Report FY2023-24"
    assert rows[0]["reporting_year"] == "2024"
    assert rows[0]["audit_path"] == "outputs/audit.json"
    assert rows[0]["annotation_id"] == "sjs-0001"
    assert set(rows[0]) == set(ANNOTATION_FIELDS)


def test_template_can_exclude_unscored_or_include_not_relevant() -> None:
    payload = _payload(
        [
            _finding(page=10),
            _finding(quote_verified=False, page=11),
            _finding(relevant=False, level=None, page=12),
        ]
    )

    scored_only = build_annotation_rows(payload, include_unscored=False)
    assert [row["page"] for row in scored_only] == ["10"]

    with_negatives = build_annotation_rows(
        payload, include_not_relevant=True
    )
    assert [row["page"] for row in with_negatives] == ["10", "11", "12"]
    assert with_negatives[2]["model_relevant"] == "0"


def test_template_requires_a_findings_list() -> None:
    with pytest.raises(ValueError):
        build_annotation_rows({"document": {}})


def test_csv_round_trip_and_missing_column_validation(tmp_path) -> None:
    rows = build_annotation_rows(_payload([_finding()]), prefix="test")
    path = write_annotation_csv(rows, tmp_path / "gold.csv")
    read_back = read_annotation_csv(path)

    assert read_back[0]["annotation_id"] == "test-0001"
    assert list(read_back[0]) == ANNOTATION_FIELDS

    bad = tmp_path / "bad.csv"
    bad.write_text("annotation_id,page\nx,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        read_annotation_csv(bad)


# ---------------------------------------------------------------------------
# Benchmark scoring
# ---------------------------------------------------------------------------


def test_blank_template_has_no_metrics() -> None:
    rows = build_annotation_rows(_payload([_finding()]))
    metrics = score_annotations(rows)

    assert metrics["status"] == "no_annotations"
    assert metrics["rows_total"] == 1
    assert metrics["reviewer_values_present"] == 0
    assert metrics["pairs"] == {
        "relevance": 0,
        "rubric": 0,
        "quote_verification": 0,
        "semantic_support": 0,
    }
    assert metrics["relevance"] is None
    assert metrics["rubric"] is None
    assert metrics["quote_verification"] is None
    assert metrics["semantic_support"] is None
    assert metrics["skipped"]["missing_relevance"] == 1
    assert metrics["skipped"]["missing_levels"] == 1
    assert metrics["skipped"]["missing_quote"] == 1


def test_filled_annotations_produce_relevance_and_rubric_metrics() -> None:
    rows = [
        _row(reviewer_relevant="1", reviewer_level="4", model_level="4",
             reviewer_quote_ok="1"),
        _row(reviewer_relevant="1", reviewer_level="3", model_level="3",
             reviewer_quote_ok="1"),
        _row(reviewer_relevant="0", reviewer_level="2", model_level="2",
             reviewer_quote_ok="1"),
        _row(reviewer_relevant="0", reviewer_level="", model_level="",
             reviewer_quote_ok="0", model_quote_verified="0",
             model_relevant="0"),
    ]
    metrics = score_annotations(rows)

    assert metrics["status"] == "ok"
    assert metrics["pairs"] == {
        "relevance": 4,
        "rubric": 3,
        "quote_verification": 4,
        "semantic_support": 0,
    }

    relevance = metrics["relevance"]
    assert relevance["tp"] == 2
    assert relevance["fp"] == 1
    assert relevance["tn"] == 1
    assert relevance["fn"] == 0
    assert relevance["accuracy"] == 0.75

    rubric = metrics["rubric"]
    assert rubric["exact_agreement"] == 1.0
    assert rubric["within_one_agreement"] == 1.0
    assert rubric["quadratic_weighted_kappa"] == 1.0
    assert metrics["level_counts"]["reviewer"]["4"] == 1
    assert metrics["level_counts"]["reviewer"]["3"] == 1
    assert metrics["level_counts"]["reviewer"]["2"] == 1
    assert metrics["level_counts"]["model"]["4"] == 1

    assert metrics["quote_verification"]["accuracy"] == 1.0


def test_partial_annotations_report_only_available_metrics() -> None:
    metrics = score_annotations([_row(reviewer_relevant="1")])

    assert metrics["status"] == "ok"
    assert metrics["relevance"] is not None
    assert metrics["rubric"] is None
    assert metrics["quote_verification"] is None


def test_semantic_support_agreement_uses_categorical_labels() -> None:
    rows = [
        _row(
            reviewer_semantic_support="supported",
            model_semantic_support="supported",
        ),
        _row(
            reviewer_semantic_support="uncertain",
            model_semantic_support="supported",
        ),
        _row(
            reviewer_semantic_support="not_supported",
            model_semantic_support="not_checked",
        ),
    ]
    metrics = score_annotations(rows)

    assert metrics["status"] == "ok"
    assert metrics["pairs"]["semantic_support"] == 2
    assert metrics["skipped"]["missing_semantic"] == 1
    assert metrics["relevance"] is None
    assert metrics["semantic_support"]["n"] == 2
    assert metrics["semantic_support"]["observed_agreement"] == 0.5
    assert metrics["semantic_support"]["labels"] == [
        "supported",
        "uncertain",
        "not_supported",
    ]
    assert metrics["semantic_support"]["confusion_matrix"] == [
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ]


def test_invalid_values_are_skipped_not_coerced() -> None:
    metrics = score_annotations(
        [
            _row(
                reviewer_relevant="maybe",
                reviewer_level="9",
                reviewer_quote_ok="2",
                reviewer_semantic_support="probably",
            )
        ]
    )

    assert metrics["status"] == "invalid_annotations"
    assert metrics["skipped"]["invalid_values"] == 1
    assert metrics["relevance"] is None
    assert metrics["rubric"] is None


def test_reviewer_labels_without_model_values_report_no_pairs() -> None:
    metrics = score_annotations(
        [
            _row(
                reviewer_semantic_support="supported",
                model_semantic_support="not_checked",
            )
        ]
    )

    assert metrics["status"] == "no_pairs"
    assert metrics["reviewer_values_present"] == 1
    assert metrics["pairs"]["semantic_support"] == 0
    assert metrics["semantic_support"] is None


def test_yes_no_labels_are_accepted() -> None:
    metrics = score_annotations(
        [_row(reviewer_relevant="yes", reviewer_quote_ok="no")]
    )
    assert metrics["relevance"]["n"] == 1
    assert metrics["quote_verification"]["fp"] == 1


def test_fractional_rubric_level_is_not_truncated() -> None:
    metrics = score_annotations(
        [
            _row(
                reviewer_relevant="1",
                reviewer_level="3.7",
                reviewer_quote_ok="1",
            )
        ]
    )

    assert metrics["skipped"]["invalid_values"] == 1
    assert metrics["rubric"] is None
    assert metrics["pairs"]["rubric"] == 0


# ---------------------------------------------------------------------------
# Command-line scripts
# ---------------------------------------------------------------------------


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "eval" / script), *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_make_template_script_and_overwrite_guard(tmp_path) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(_payload([_finding()])), encoding="utf-8")
    out = tmp_path / "template.csv"

    first = _run("make_gold_template.py", str(audit), "--out", str(out))
    assert first.returncode == 0, first.stderr
    assert "contains no gold labels" in first.stdout
    assert out.exists()

    blocked = _run("make_gold_template.py", str(audit), "--out", str(out))
    assert blocked.returncode == 2
    assert "already exists" in blocked.stderr

    forced = _run(
        "make_gold_template.py", str(audit), "--out", str(out), "--force"
    )
    assert forced.returncode == 0


def test_score_benchmark_script_reports_no_annotations_then_metrics(
    tmp_path,
) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(_payload([_finding()])), encoding="utf-8")
    template = tmp_path / "annotations.csv"
    assert (
        _run("make_gold_template.py", str(audit), "--out", str(template))
        .returncode
        == 0
    )

    report_path = tmp_path / "benchmark.json"
    blank = _run(
        "score_benchmark.py",
        str(template),
        "--out",
        str(report_path),
    )
    assert blank.returncode == 0, blank.stderr
    assert "no metrics were computed" in blank.stdout
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["status"] == "no_annotations"
    assert payload["metrics"]["rubric"] is None
    assert payload["provenance"]["files"]["gold"]["sha256"]

    with template.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["reviewer_relevant"] = "1"
    rows[0]["reviewer_level"] = "3"
    rows[0]["reviewer_quote_ok"] = "1"
    with template.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    filled = _run(
        "score_benchmark.py",
        str(template),
        "--out",
        str(report_path),
        "--notes",
        "single annotator smoke test",
    )
    assert filled.returncode == 0, filled.stderr
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["status"] == "ok"
    assert payload["metrics"]["rubric"]["exact_agreement"] == 1.0
    assert "single annotator smoke test" in payload["provenance"]["notes"]
