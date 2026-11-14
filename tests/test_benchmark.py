"""Tests for human-annotation benchmark tooling."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from prism17.benchmark import (
    BENCHMARK_FIELDS,
    parse_bool,
    parse_level,
    parse_semantic,
    read_rows,
    score_annotations,
    score_csv,
    template_rows,
    write_report,
    write_template,
)
from prism17.models import Candidate, JudgedFinding, VerifiedFinding

ROOT = Path(__file__).resolve().parent.parent


def _finding(
    *,
    target_id: str = "13.2",
    goal: int = 13,
    level: int | None = 3,
    relevant: bool = True,
    quote_verified: bool = True,
    semantic_support: str = "not_checked",
    page: int = 10,
) -> VerifiedFinding:
    quote = "Example quote with enough words to verify against the page."
    candidate = Candidate(
        passage_id=f"p{page:04d}-001",
        page=page,
        target_id=target_id,
        target_goal=goal,
        target_text=f"Target {target_id}",
        score=0.8,
        passage_text=quote,
    )
    judged = JudgedFinding(
        candidate=candidate,
        relevant=relevant,
        sdg_goal=goal,
        sdg_target=target_id,
        claim="Example claim.",
        supporting_quote=quote,
        dimensions={"specificity": True, "measurable_indicator": True},
        scored=True,
    )
    return VerifiedFinding(
        judged=judged,
        quote_verified=quote_verified,
        quote_method="page_exact" if quote_verified else None,
        quote_score=100.0 if quote_verified else 0.0,
        source_page=page,
        semantic_support=semantic_support,
        level=level,
    )


def _payload(
    findings: list[VerifiedFinding], *, title: str = "Alpha 2022"
) -> dict:
    return {
        "schema_version": "0.1",
        "document": {"title": title, "year": 2022},
        "findings": [finding.to_dict() for finding in findings],
    }


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "annotation_id": "ann-0001",
        "company": "Alpha",
        "year": "2022",
        "source_path": "alpha.json",
        "title": "Alpha 2022",
        "target_id": "13.2",
        "page": "10",
        "passage_id": "p0010-001",
        "claim": "Example claim.",
        "quote": "Example quote.",
        "model_relevant": "1",
        "model_level": "3",
        "model_quote_ok": "1",
        "model_semantic_support": "not_checked",
        "model_review_reason": "",
        "reviewer_relevant": "",
        "reviewer_level": "",
        "reviewer_quote_ok": "",
        "reviewer_semantic_support": "",
        "reviewer_notes": "",
        "annotator": "",
        "annotated_at_utc": "",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_bool_accepts_common_forms_and_blank() -> None:
    assert parse_bool("1") is True
    assert parse_bool("yes") is True
    assert parse_bool("TRUE") is True
    assert parse_bool("0") is False
    assert parse_bool("no") is False
    assert parse_bool("") is None
    assert parse_bool(None) is None


def test_parse_bool_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        parse_bool("maybe")


def test_parse_level_accepts_strict_integers_only() -> None:
    assert parse_level("3") == 3
    assert parse_level(" 4 ") == 4
    assert parse_level("") is None
    assert parse_level(None) is None

    for invalid in ("3.7", "5", "0", "abc", "-1"):
        with pytest.raises(ValueError):
            parse_level(invalid)


def test_parse_semantic_normalises_and_rejects_unknown_labels() -> None:
    assert parse_semantic("supported") == "supported"
    assert parse_semantic("Not-Supported") == "not_supported"
    assert parse_semantic("not checked") is None
    assert parse_semantic("") is None
    assert parse_semantic(None) is None

    with pytest.raises(ValueError):
        parse_semantic("partly supported")


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def test_template_rows_expose_model_values_and_blank_reviewer_columns() -> None:
    payload = _payload(
        [
            _finding(level=3),
            _finding(target_id="13.1", level=None, relevant=False, page=11),
        ]
    )
    rows = template_rows([payload], labels=["Alpha"], paths=["alpha.json"])

    assert len(rows) == 2
    first = rows[0]
    assert first["company"] == "Alpha"
    assert first["year"] == 2022
    assert first["target_id"] == "13.2"
    assert first["page"] == 10
    assert first["model_relevant"] == 1
    assert first["model_level"] == 3
    assert first["model_quote_ok"] == 1
    assert first["model_semantic_support"] == "not_checked"
    for field in (
        "reviewer_relevant",
        "reviewer_level",
        "reviewer_quote_ok",
        "reviewer_semantic_support",
        "reviewer_notes",
        "annotator",
        "annotated_at_utc",
    ):
        assert first[field] == ""

    second = rows[1]
    assert second["model_level"] == ""
    assert second["model_relevant"] == 0


def test_template_rows_require_matching_labels_and_paths() -> None:
    payload = _payload([_finding()])

    with pytest.raises(ValueError):
        template_rows([payload], labels=["Alpha", "Beta"])
    with pytest.raises(ValueError):
        template_rows([payload], paths=["a.json", "b.json"])


def test_template_rows_use_path_stem_when_label_is_omitted() -> None:
    rows = template_rows(
        [_payload([_finding()])], paths=["outputs/alpha.audit.json"]
    )

    assert rows[0]["company"] == "alpha.audit"
    assert rows[0]["annotation_id"] == "ann-alpha.audit-0001"


def test_write_template_refuses_overwrite_unless_forced(tmp_path) -> None:
    path = tmp_path / "gold.csv"
    rows = template_rows([_payload([_finding()])], labels=["Alpha"])

    write_template(path, rows)
    assert read_rows(path)[0]["target_id"] == "13.2"

    with pytest.raises(FileExistsError):
        write_template(path, rows)

    write_template(path, rows, force=True)
    assert path.exists()


def test_read_rows_rejects_missing_columns(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "annotation_id,company,claim,page,quote\nann-1,Alpha,Claim,10,Quote\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns"):
        read_rows(path)


def test_write_report_writes_json(tmp_path) -> None:
    path = tmp_path / "nested" / "benchmark.json"
    written = write_report(path, {"status": "no_annotations"})

    assert written == path
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "status": "no_annotations"
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_blank_reviewer_columns_report_no_annotations_and_null_metrics() -> None:
    metrics = score_annotations([_row()])

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


def test_invalid_reviewer_values_report_invalid_annotations() -> None:
    metrics = score_annotations(
        [_row(reviewer_relevant="maybe", reviewer_level="3.7")]
    )

    assert metrics["status"] == "invalid_annotations"
    assert metrics["skipped"]["invalid_values"] == 1
    assert metrics["reviewer_values_present"] == 0
    assert metrics["relevance"] is None
    assert metrics["rubric"] is None


def test_fractional_rubric_level_is_not_truncated() -> None:
    metrics = score_annotations(
        [_row(reviewer_relevant="1", reviewer_level="3.7", reviewer_quote_ok="1")]
    )

    assert metrics["skipped"]["invalid_values"] == 1
    assert metrics["rubric"] is None
    assert metrics["pairs"]["rubric"] == 0


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


def test_partial_annotation_pairs_only_available_metrics() -> None:
    metrics = score_annotations([_row(reviewer_relevant="1")])

    assert metrics["status"] == "ok"
    assert metrics["pairs"]["relevance"] == 1
    assert metrics["pairs"]["rubric"] == 0
    assert metrics["skipped"]["missing_levels"] == 0
    assert metrics["relevance"]["accuracy"] == 1.0
    assert metrics["rubric"] is None


def test_full_annotations_report_agreement_metrics() -> None:
    rows = [
        _row(
            reviewer_relevant="1",
            reviewer_level="3",
            reviewer_quote_ok="1",
            reviewer_semantic_support="supported",
        ),
        _row(
            annotation_id="ann-0002",
            reviewer_relevant="0",
            reviewer_level="2",
            reviewer_quote_ok="0",
            reviewer_semantic_support="uncertain",
            model_relevant="0",
            model_level="3",
            model_quote_ok="1",
            model_semantic_support="uncertain",
        ),
        _row(
            annotation_id="ann-0003",
            reviewer_relevant="1",
            reviewer_level="4",
            reviewer_quote_ok="1",
            reviewer_semantic_support="supported",
            model_relevant="0",
            model_level="4",
            model_quote_ok="0",
            model_semantic_support="uncertain",
        ),
    ]
    metrics = score_annotations(rows)

    assert metrics["status"] == "ok"
    assert metrics["reviewer_values_present"] == 12
    assert metrics["pairs"] == {
        "relevance": 3,
        "rubric": 3,
        "quote_verification": 3,
        "semantic_support": 2,
    }

    relevance = metrics["relevance"]
    assert relevance["tp"] == 1
    assert relevance["tn"] == 1
    assert relevance["fn"] == 1
    assert relevance["fp"] == 0
    assert relevance["precision"] == 1.0
    assert relevance["recall"] == 0.5
    assert relevance["accuracy"] == pytest.approx(2 / 3)
    assert relevance["confusion"]["matrix"] == [[1, 0], [1, 1]]

    quote = metrics["quote_verification"]
    assert quote["accuracy"] == pytest.approx(1 / 3)
    assert quote["precision"] == 0.5
    assert quote["recall"] == 0.5

    rubric = metrics["rubric"]
    assert rubric["n"] == 3
    assert rubric["exact_agreement"] == pytest.approx(2 / 3)
    assert rubric["within_one"] == 1.0

    semantic = metrics["semantic_support"]
    assert semantic["n"] == 2
    assert semantic["agreement"] == 0.5
    assert semantic["confusion"]["labels"] == [
        "supported",
        "uncertain",
        "not_supported",
    ]


def test_invalid_model_values_are_counted_separately() -> None:
    metrics = score_annotations(
        [
            _row(
                reviewer_level="3",
                model_level="not-a-level",
                reviewer_relevant="1",
            )
        ]
    )

    assert metrics["skipped"]["invalid_model_values"] == 1
    assert metrics["skipped"]["missing_levels"] == 1
    assert metrics["rubric"] is None
    assert metrics["pairs"]["relevance"] == 1


def test_score_csv_reads_and_scores_a_file(tmp_path) -> None:
    path = tmp_path / "gold.csv"
    rows = [
        _row(
            reviewer_relevant="1",
            reviewer_level="3",
            reviewer_quote_ok="1",
            reviewer_semantic_support="supported",
            model_semantic_support="supported",
        )
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    metrics = score_csv(path)

    assert metrics["status"] == "ok"
    assert metrics["relevance"]["accuracy"] == 1.0
    assert metrics["semantic_support"]["agreement"] == 1.0


# ---------------------------------------------------------------------------
# Command-line scripts
# ---------------------------------------------------------------------------


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"prism17_eval_{name}", ROOT / "eval" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_template_and_score_benchmark_clis(tmp_path, capsys) -> None:
    audit = tmp_path / "alpha.audit.json"
    audit.write_text(
        json.dumps(_payload([_finding(level=3)])), encoding="utf-8"
    )
    gold = tmp_path / "gold.csv"
    report = tmp_path / "benchmark.json"

    template_cli = _load_script("make_gold_template")
    assert template_cli.main([str(audit), "--out", str(gold)]) == 0
    assert read_rows(gold)[0]["reviewer_relevant"] == ""

    # The guard prevents an accidental overwrite of an existing gold file.
    assert template_cli.main([str(audit), "--out", str(gold)]) == 1

    score_cli = _load_script("score_benchmark")
    assert score_cli.main([str(gold), "--out", str(report)]) == 0

    document = json.loads(report.read_text(encoding="utf-8"))
    assert document["metrics"]["status"] == "no_annotations"
    assert document["metrics"]["relevance"] is None
    assert document["provenance"]["dataset"]["version"] == "0.1"

    output = capsys.readouterr().out
    assert "No reviewer annotations found" in output
