"""Tests for Phase 4 human-review decisions and persistence."""

from __future__ import annotations

import json

from prism17.models import Candidate, JudgedFinding, VerifiedFinding
from prism17.review import (
    ReviewDecision,
    export_rows,
    finding_key,
    load_reviews,
    make_decision,
    save_reviews,
    summarise_reviews,
)


def _finding(
    *,
    target_id: str = "12.5",
    page: int = 49,
    passage_id: str = "p0049-001",
    scored: bool = True,
) -> VerifiedFinding:
    candidate = Candidate(
        passage_id=passage_id,
        page=page,
        target_id=target_id,
        target_goal=12,
        target_text="Target text",
        score=0.9,
        passage_text="Passage text",
    )
    judged = JudgedFinding(
        candidate=candidate,
        relevant=scored,
        sdg_goal=12,
        sdg_target=target_id,
        claim="Example claim.",
        supporting_quote="Example quote with enough words to verify.",
        dimensions={
            "specificity": True,
            "measurable_indicator": True,
            "deadline": False,
            "baseline": False,
            "reported_progress": False,
        },
        scored=True,
    )
    return VerifiedFinding(
        judged=judged,
        quote_verified=scored,
        quote_method="page_exact" if scored else None,
        quote_score=100.0 if scored else 0.0,
        source_page=page,
        level=2 if scored else None,
    )


def test_finding_key_is_stable_and_page_specific() -> None:
    finding = _finding()

    assert finding_key(finding) == "p0049-001|12.5|49"
    assert finding_key(finding) == finding_key(_finding())
    assert finding_key(finding) != finding_key(_finding(page=50))


def test_make_decision_normalises_unknown_status_to_pending() -> None:
    decision = make_decision("key", status="not-a-status", level=3)

    assert decision.status == "pending"
    assert decision.level is None


def test_override_keeps_level_and_other_statuses_drop_it() -> None:
    override = make_decision("key", status="overridden", level=3)
    confirmed = make_decision("key", status="confirmed", level=3)

    assert override.level == 3
    assert override.is_valid_override()
    assert confirmed.level is None


def test_invalid_override_level_is_flagged() -> None:
    decision = ReviewDecision(finding_key="key", status="overridden", level=None)
    normalised = decision.normalised()

    assert normalised.level is None
    assert not normalised.is_valid_override()


def test_save_and_load_roundtrip(tmp_path) -> None:
    decisions = {
        "a|12.5|49": make_decision(
            "a|12.5|49",
            status="overridden",
            level=4,
            note="Baseline is in the table.",
            reviewer="Reviewer A",
        ),
        "b|12.6|25": make_decision(
            "b|12.6|25", status="confirmed", reviewer="Reviewer B"
        ),
    }

    path = save_reviews(tmp_path / "reviews.json", decisions)
    loaded = load_reviews(path)

    assert set(loaded) == set(decisions)
    assert loaded["a|12.5|49"].level == 4
    assert loaded["a|12.5|49"].note == "Baseline is in the table."
    assert loaded["b|12.6|25"].status == "confirmed"
    assert loaded["b|12.6|25"].level is None


def test_saved_file_is_stable_and_sorted(tmp_path) -> None:
    decisions = {
        "z|12.6|25": make_decision("z|12.6|25", status="confirmed"),
        "a|12.5|49": make_decision("a|12.5|49", status="flagged"),
    }

    path = save_reviews(tmp_path / "reviews.json", decisions)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "0.1"
    assert [item["finding_key"] for item in payload["reviews"]] == [
        "a|12.5|49",
        "z|12.6|25",
    ]


def test_missing_review_file_returns_empty_mapping(tmp_path) -> None:
    assert load_reviews(tmp_path / "missing.json") == {}


def test_malformed_review_file_returns_empty_mapping(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    assert load_reviews(path) == {}


def test_load_accepts_plain_list_and_normalises(tmp_path) -> None:
    path = tmp_path / "reviews.json"
    path.write_text(
        json.dumps(
            [
                {"finding_key": "a|12.5|49", "status": "overridden", "level": 3},
                {"finding_key": "b|12.6|25", "status": "bogus", "level": 4},
                {"not": "a decision"},
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_reviews(path)

    assert loaded["a|12.5|49"].level == 3
    assert loaded["b|12.6|25"].status == "pending"
    assert loaded["b|12.6|25"].level is None


def test_summarise_reviews_counts_only_scored_findings() -> None:
    scored_a = _finding(page=49)
    scored_b = _finding(page=50, passage_id="p0050-001")
    unscored = _finding(page=51, passage_id="p0051-001", scored=False)
    decisions = {
        finding_key(scored_a): make_decision(
            finding_key(scored_a), status="confirmed"
        ),
        finding_key(scored_b): make_decision(
            finding_key(scored_b), status="overridden", level=4
        ),
    }

    summary = summarise_reviews([scored_a, scored_b, unscored], decisions)

    assert summary.total == 2
    assert summary.confirmed == 1
    assert summary.overridden == 1
    assert summary.pending == 0
    assert summary.reviewed == 2


def test_summarise_reviews_without_decisions_is_pending() -> None:
    summary = summarise_reviews([_finding()], {})

    assert summary.total == 1
    assert summary.pending == 1
    assert summary.reviewed == 0


def test_export_rows_merges_findings_and_decisions() -> None:
    finding = _finding()
    key = finding_key(finding)
    decisions = {
        key: make_decision(
            key,
            status="overridden",
            level=3,
            note="Prior-year table shows a baseline.",
            reviewer="Reviewer A",
        )
    }

    rows = export_rows([finding, _finding(scored=False)], decisions)

    assert len(rows) == 1
    row = rows[0]
    assert row["finding_key"] == key
    assert row["target"] == "12.5"
    assert row["page"] == 49
    assert row["rubric_level"] == 2
    assert row["review_status"] == "overridden"
    assert row["review_level"] == 3
    assert row["reviewer"] == "Reviewer A"
    assert "Prior-year" in row["note"]
