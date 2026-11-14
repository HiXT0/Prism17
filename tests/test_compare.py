"""Tests for descriptive cross-company comparison."""

from __future__ import annotations

import json

import pytest

from prism17.compare import (
    build_company_profile,
    compare_payloads,
    main,
    render_markdown,
)
from prism17.models import Candidate, JudgedFinding, VerifiedFinding

CONFIG = {
    "compare": {
        "default_sdg": 13,
        "min_level": 3,
        "high_level_label": "L3+",
    }
}


def _finding(
    target_id: str,
    *,
    goal: int,
    level: int | None,
    page: int = 10,
    relevant: bool = True,
    quote_verified: bool = True,
    semantic_support: str = "not_checked",
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
        review_reason=review_reason,
    )


def _payload(
    title: str,
    findings: list[VerifiedFinding],
    *,
    year: int | None = None,
) -> dict:
    document: dict = {"title": title, "pages": 20}
    if year is not None:
        document["year"] = year
    return {
        "schema_version": "0.1",
        "document": document,
        "findings": [finding.to_dict() for finding in findings],
    }


def _alpha_payload() -> dict:
    return _payload(
        "Alpha Sustainability 2022",
        [
            _finding("13.2", goal=13, level=3, page=10),
            _finding("13.2", goal=13, level=2, page=11),
            _finding("13.1", goal=13, level=4, page=12),
            _finding("12.5", goal=12, level=1, page=49),
            _finding("13.3", goal=13, level=None, relevant=False, page=13),
            _finding("13.4", goal=13, level=None, quote_verified=False, page=14),
        ],
        year=2022,
    )


def _beta_payload() -> dict:
    return _payload(
        "Beta Sustainability 2022",
        [
            _finding("13.2", goal=13, level=1, page=20),
            _finding("7.2", goal=7, level=3, page=30),
        ],
        year=2022,
    )


# ---------------------------------------------------------------------------
# Company profiles
# ---------------------------------------------------------------------------


def test_company_profile_counts_scored_findings_and_levels() -> None:
    profile = build_company_profile(
        _alpha_payload(), label="Alpha", path="alpha.json", min_level=3
    )

    assert profile.label == "Alpha"
    assert profile.title == "Alpha Sustainability 2022"
    assert profile.year == 2022
    assert profile.findings_total == 6
    assert profile.scored_total == 4
    assert profile.high_level_count == 2
    assert profile.best_level == 4
    assert profile.level_counts == {1: 1, 2: 1, 3: 1, 4: 1}
    assert profile.review_counts["scored"] == 4
    assert profile.review_counts["not_relevant"] == 1
    assert profile.review_counts["needs_review"] == 1
    assert profile.review_counts["unscored"] == 2


def test_company_profile_min_level_changes_high_level_counts() -> None:
    profile = build_company_profile(
        _alpha_payload(), label="Alpha", min_level=2
    )

    assert profile.high_level_count == 3
    assert [goal.high_level_count for goal in profile.goals] == [0, 3]


def test_goal_profile_counts_targets_and_best_level() -> None:
    profile = build_company_profile(
        _alpha_payload(), label="Alpha", min_level=3
    )
    goals = {goal.goal: goal for goal in profile.goals}

    assert set(goals) == {12, 13}
    assert goals[13].scored_findings == 3
    assert goals[13].targets_with_scored_disclosure == 2
    assert goals[13].best_level == 4
    assert goals[13].high_level_count == 2
    assert goals[12].scored_findings == 1
    assert goals[12].best_level == 1
    assert goals[12].high_level_count == 0


def test_focus_goals_restrict_goal_profiles_only() -> None:
    profile = build_company_profile(
        _alpha_payload(),
        label="Alpha",
        min_level=3,
        focus_goals=[12],
    )

    assert [goal.goal for goal in profile.goals] == [12]
    assert profile.scored_total == 4
    assert profile.high_level_count == 2


def test_profile_uses_path_stem_when_title_is_missing() -> None:
    payload = _alpha_payload()
    payload["document"] = {}
    profile = build_company_profile(
        payload, label="Alpha", path="outputs/alpha.audit.json"
    )

    assert profile.title == "alpha.audit"
    assert profile.year is None


def test_compare_paths_are_saved_without_absolute_directories(tmp_path) -> None:
    audit = tmp_path / "alpha.audit.json"

    report = compare_payloads([_alpha_payload()], paths=[str(audit)])

    assert report.companies[0].path == "alpha.audit.json"
    assert str(tmp_path) not in report.companies[0].path


def test_invalid_company_min_level_is_rejected() -> None:
    for invalid in (0, 5, 2.5, True):
        with pytest.raises(ValueError):
            build_company_profile(
                _alpha_payload(), label="Alpha", min_level=invalid
            )


def test_missing_findings_list_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_company_profile({"document": {}}, label="Alpha")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def test_compare_payloads_uses_configured_default_goal() -> None:
    report = compare_payloads(
        [_alpha_payload(), _beta_payload()],
        labels=["Alpha", "Beta"],
        config=CONFIG,
    )

    assert report.goals == [13]
    assert report.high_level_label == "L3+"
    assert report.min_level == 3
    assert [company.label for company in report.companies] == ["Alpha", "Beta"]


def test_min_level_override_changes_label_and_counts() -> None:
    report = compare_payloads(
        [_alpha_payload()],
        labels=["Alpha"],
        config=CONFIG,
        min_level=2,
    )

    assert report.min_level == 2
    assert report.high_level_label == "L2+"
    assert report.companies[0].high_level_count == 3
    assert report.companies[0].goals[0].high_level_count == 3


def test_configured_label_is_kept_when_it_matches_min_level() -> None:
    config = {
        "compare": {
            "default_sdg": 13,
            "min_level": 2,
            "high_level_label": "L2+",
        }
    }
    report = compare_payloads([_alpha_payload()], config=config)

    assert report.min_level == 2
    assert report.high_level_label == "L2+"


def test_explicit_focus_goals_override_default() -> None:
    report = compare_payloads(
        [_alpha_payload(), _beta_payload()],
        config=CONFIG,
        focus_goals=[7, 13],
    )

    assert report.goals == [7, 13]


def test_labels_and_paths_must_match_payload_count() -> None:
    with pytest.raises(ValueError):
        compare_payloads([_alpha_payload()], labels=["Alpha", "Beta"])
    with pytest.raises(ValueError):
        compare_payloads([_alpha_payload()], paths=["a.json", "b.json"])


def test_invalid_compare_min_level_is_rejected() -> None:
    with pytest.raises(ValueError):
        compare_payloads([_alpha_payload()], config=CONFIG, min_level=0)


def test_json_payload_round_trips_core_metrics() -> None:
    report = compare_payloads(
        [_alpha_payload()], labels=["Alpha"], config=CONFIG
    )
    data = report.to_dict()

    assert data["schema_version"] == "0.1"
    assert data["high_level_label"] == "L3+"
    assert data["min_level"] == 3
    assert data["goals"] == [13]
    assert data["companies"][0]["label"] == "Alpha"
    assert data["companies"][0]["scored_total"] == 4
    assert data["companies"][0]["high_level_count"] == 2
    assert data["companies"][0]["level_counts"] == {
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 1,
    }
    assert data["companies"][0]["goals"][0]["goal"] == 13
    assert data["notes"]

    text = json.dumps(data)
    assert "level_3_plus" not in text


def test_markdown_is_descriptive_and_not_a_ranking() -> None:
    report = compare_payloads(
        [_alpha_payload(), _beta_payload()],
        labels=["Alpha", "Beta"],
        config=CONFIG,
    )
    markdown = render_markdown(report)

    assert "not a ranking" in markdown
    assert "L3+" in markdown
    assert "| Alpha | 2022 | 6 | 4 | 2 | 4 | 1/1 |" in markdown
    assert markdown.isascii()


def test_markdown_without_goals_omits_per_goal_table() -> None:
    config = {"compare": {"min_level": 3}}
    report = compare_payloads([_alpha_payload()], config=config)

    assert report.goals == []
    assert "Per-goal metrics" not in render_markdown(report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_writes_json_and_markdown(tmp_path) -> None:
    alpha = tmp_path / "alpha.json"
    beta = tmp_path / "beta.json"
    alpha.write_text(json.dumps(_alpha_payload()), encoding="utf-8")
    beta.write_text(json.dumps(_beta_payload()), encoding="utf-8")
    out_json = tmp_path / "compare.json"
    out_md = tmp_path / "compare.md"

    code = main(
        [
            str(alpha),
            str(beta),
            "--labels",
            "Alpha",
            "Beta",
            "--sdg",
            "13",
            "--out",
            str(out_json),
            "--markdown",
            str(out_md),
        ]
    )

    assert code == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert [company["label"] for company in data["companies"]] == [
        "Alpha",
        "Beta",
    ]
    assert "high_level_count" not in data
    assert data["companies"][0]["high_level_count"] == 2
    assert "not a ranking" in out_md.read_text(encoding="utf-8")


def test_cli_returns_two_for_label_mismatch(tmp_path) -> None:
    alpha = tmp_path / "alpha.json"
    alpha.write_text(json.dumps(_alpha_payload()), encoding="utf-8")

    assert main([str(alpha), "--labels", "Alpha", "Beta"]) == 2


def test_cli_returns_two_for_invalid_sdg_or_level(tmp_path) -> None:
    alpha = tmp_path / "alpha.json"
    alpha.write_text(json.dumps(_alpha_payload()), encoding="utf-8")

    assert main([str(alpha), "--sdg", "18"]) == 2
    assert main([str(alpha), "--min-level", "5"]) == 2


def test_cli_returns_two_for_malformed_audit(tmp_path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"document": {}}), encoding="utf-8")

    assert main([str(broken)]) == 2
