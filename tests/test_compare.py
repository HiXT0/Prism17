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


def _finding(
    target_id: str = "13.2",
    *,
    goal: int | None = None,
    level: int | None = 3,
    relevant: bool = True,
    quote_verified: bool = True,
    semantic_support: str = "not_checked",
    review_reason: str | None = None,
    page: int = 10,
) -> VerifiedFinding:
    actual_goal = goal if goal is not None else int(target_id.split(".", 1)[0])
    candidate = Candidate(
        passage_id=f"p{page:04d}-001",
        page=page,
        target_id=target_id,
        target_goal=actual_goal,
        target_text="Target text",
        score=0.8,
        passage_text="Passage text",
    )
    judged = JudgedFinding(
        candidate=candidate,
        relevant=relevant,
        sdg_goal=actual_goal,
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


def _payload(
    findings: list[VerifiedFinding],
    *,
    title: str = "Example Report",
    year_title: str | None = None,
    model: str = "qwen2.5:7b-instruct",
    pages: int = 59,
    pdf_path: str = "reports/example.pdf",
) -> dict:
    return {
        "schema_version": "0.2",
        "document": {
            "title": year_title if year_title is not None else title,
            "pdf_path": pdf_path,
            "page_count": pages,
        },
        "settings": {"model": model},
        "findings": [finding.to_dict() for finding in findings],
    }


def test_company_profile_counts_scored_disclosure_and_review_status() -> None:
    payload = _payload(
        [
            _finding("13.2", level=4, page=10),
            _finding("13.2", level=3, page=11),
            _finding("13.1", level=2, page=12),
            _finding("13.1", quote_verified=False, page=13),
            _finding("13.1", relevant=False, level=None, page=14),
            _finding("13.1", semantic_support="uncertain", page=15),
        ],
        year_title="Annual Report FY2023-24",
        model="llama3.1:8b",
        pages=120,
    )

    profile = build_company_profile(payload)

    assert profile.label == "Annual Report FY2023-24"
    assert profile.reporting_year == 2024
    assert profile.model == "llama3.1:8b"
    assert profile.pages == 120
    assert profile.findings_total == 6
    assert profile.scored_total == 3
    assert profile.level_counts == {1: 0, 2: 1, 3: 1, 4: 1}
    assert profile.best_level == 4
    assert profile.high_level_count == 2
    assert profile.needs_review_total == 2
    assert profile.not_relevant_total == 1
    assert profile.goals_with_scored == [13]

    goal = profile.goals[13]
    assert goal.scored_findings == 3
    assert goal.targets == ["13.1", "13.2"]
    assert goal.targets_with_scored == 2
    assert goal.best_level == 4


def test_company_profile_can_filter_to_one_goal() -> None:
    payload = _payload(
        [_finding("13.2"), _finding("12.5"), _finding("8.5")]
    )
    profile = build_company_profile(payload, goals=[12])

    assert profile.scored_total == 1
    assert profile.goals_with_scored == [12]
    assert 13 not in profile.goals


def test_missing_findings_list_is_an_error() -> None:
    with pytest.raises(ValueError):
        build_company_profile({"document": {}})


def test_default_labels_fall_back_to_file_stem_and_index() -> None:
    first = _payload([_finding()], title="", pdf_path="reports/company_a.pdf")
    second = _payload([_finding()], title="", pdf_path="")
    report = compare_payloads([first, second], {})

    assert report.profiles[0].label == "company_a"
    assert report.profiles[1].label == "company 2"


def test_duplicate_labels_are_made_unique() -> None:
    payload = _payload([_finding()], title="Same Name")
    report = compare_payloads([payload, payload], {})

    assert [profile.label for profile in report.profiles] == [
        "Same Name",
        "Same Name (2)",
    ]


def test_label_count_must_match_payload_count() -> None:
    with pytest.raises(ValueError):
        compare_payloads([_payload([_finding()])], {}, labels=["A", "B"])


def test_empty_comparison_is_rejected() -> None:
    with pytest.raises(ValueError):
        compare_payloads([])


def test_comparison_uses_union_of_goals_and_custom_level_label() -> None:
    first = _payload([_finding("13.2", level=4), _finding("12.5", level=3)])
    second = _payload([_finding("8.5", level=2), _finding("12.5", level=4)])
    report = compare_payloads(
        [first, second],
        {"compare": {"min_level": 3, "high_level_label": "L3+"}},
        labels=["First", "Second"],
    )

    assert report.goals == [8, 12, 13]
    assert report.high_level_label == "L3+"
    assert report.profiles[0].high_level_count == 2
    assert report.profiles[1].high_level_count == 1


def test_custom_min_level_updates_the_label() -> None:
    report = compare_payloads(
        [_payload([_finding(level=2)])],
        {"compare": {"min_level": 3, "high_level_label": "L3+"}},
        min_level=2,
    )
    assert report.min_level == 2
    assert report.high_level_label == "L2+"


def test_custom_min_level_counts_level_twos_at_company_and_goal_level() -> None:
    payload = _payload(
        [
            _finding("13.2", level=4),
            _finding("13.2", level=2),
            _finding("12.5", level=2),
            _finding("12.5", level=1),
        ]
    )
    default = compare_payloads([payload], {"compare": {"min_level": 3}})
    lowered = compare_payloads([payload], {"compare": {"min_level": 3}}, min_level=2)

    assert default.profiles[0].high_level_count == 1
    assert default.profiles[0].goals[12].high_level_count == 0
    assert default.profiles[0].goals[13].high_level_count == 1

    assert lowered.profiles[0].high_level_count == 3
    assert lowered.profiles[0].min_level == 2
    assert lowered.profiles[0].goals[12].high_level_count == 1
    assert lowered.profiles[0].goals[13].high_level_count == 2
    assert lowered.profiles[0].goals[12].min_level == 2
    assert lowered.to_dict()["companies"][0]["high_level_count"] == 3
    assert lowered.to_dict()["companies"][0]["goals"][0]["high_level_count"] == 1


def test_invalid_min_level_is_rejected() -> None:
    with pytest.raises(ValueError):
        compare_payloads([_payload([_finding()])], {}, min_level=5)
    with pytest.raises(ValueError):
        compare_payloads([_payload([_finding()])], {}, min_level=0)


def test_markdown_comparison_is_neutral_and_contains_all_companies() -> None:
    report = compare_payloads(
        [
            _payload([_finding("13.2", level=4)], title="Alpha Ltd"),
            _payload([_finding("13.2", level=2)], title="Beta Ltd"),
        ],
        {"compare": {"min_level": 3, "high_level_label": "L3+"}},
    )
    markdown = render_markdown(report)

    assert "Alpha Ltd" in markdown
    assert "Beta Ltd" in markdown
    assert "not a ranking" in markdown
    assert "does not verify real-world performance" in markdown
    assert "| Alpha Ltd | 2024 |" not in markdown  # no invented year
    assert "| 13 | 1 | 1 | 4 | 1 |" in markdown
    assert "| 13 | 1 | 1 | 2 | 0 |" in markdown
    lower = markdown.lower()
    for word in ("leader", "worst", "better than"):
        assert word not in lower


def test_json_payload_round_trips_core_metrics() -> None:
    report = compare_payloads(
        [_payload([_finding("13.2", level=4)])], {}, labels=["Alpha"]
    )
    data = report.to_dict()

    assert data["schema_version"] == "0.1"
    assert data["high_level_label"] == "L3+"
    assert data["companies"][0]["label"] == "Alpha"
    assert data["companies"][0]["scored_total"] == 1
    assert data["companies"][0]["high_level_count"] == 1
    assert data["companies"][0]["goals"][0]["goal"] == 13
    assert data["goals"] == [13]
    assert data["notes"]


def test_cli_writes_comparison_artifacts(tmp_path, capsys) -> None:
    first_path = tmp_path / "alpha.json"
    second_path = tmp_path / "beta.json"
    first_path.write_text(
        json.dumps(_payload([_finding("13.2", level=4)], title="Alpha")),
        encoding="utf-8",
    )
    second_path.write_text(
        json.dumps(_payload([_finding("13.2", level=2)], title="Beta")),
        encoding="utf-8",
    )
    out_path = tmp_path / "compare.json"
    markdown_path = tmp_path / "compare.md"

    code = main(
        [
            str(first_path),
            str(second_path),
            "--labels",
            "Alpha",
            "Beta",
            "--sdg",
            "13",
            "--out",
            str(out_path),
            "--markdown",
            str(markdown_path),
        ]
    )

    assert code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(payload["companies"]) == 2
    assert payload["goals"] == [13]
    assert "Cross-company" in markdown_path.read_text(encoding="utf-8")
    assert "Wrote comparison JSON" in capsys.readouterr().out


def test_cli_returns_error_for_missing_input(tmp_path) -> None:
    assert main([str(tmp_path / "missing.json")]) == 2


def test_cli_returns_error_for_label_mismatch(tmp_path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(_payload([_finding()])), encoding="utf-8")
    assert main([str(path), "--labels", "A", "B"]) == 2
