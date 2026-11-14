"""Tests for narrow multi-year tracking."""

from __future__ import annotations

import json

import pytest

from prism17.models import Candidate, JudgedFinding, VerifiedFinding
from prism17.tracking import (
    TRACKING_STATUS_TEXT,
    build_tracking,
    main,
    normalise_unit,
    render_markdown,
    track_payloads,
)

CONFIG = {
    "tracking": {
        "min_level": 3,
        "require_same_unit": True,
    }
}


def _finding(
    *,
    target_id: str = "13.2",
    goal: int = 13,
    target_text: str = "Reduce emissions by 2030.",
    value: str = "100",
    unit: str = "tCO2e",
    quote: str | None = None,
    level: int | None = 3,
    page: int = 10,
    confidence: float = 0.9,
    baseline: str = "",
    relevant: bool = True,
    quote_verified: bool = True,
    semantic_support: str = "not_checked",
    review_reason: str | None = None,
    passage_id: str | None = None,
) -> VerifiedFinding:
    if quote is None:
        suffix = f" {unit}" if unit else ""
        quote = f"Emissions were reported at {value}{suffix} this year."
    candidate = Candidate(
        passage_id=passage_id or f"p{page:04d}-001",
        page=page,
        target_id=target_id,
        target_goal=goal,
        target_text=target_text,
        score=0.9,
        passage_text=quote,
    )
    judged = JudgedFinding(
        candidate=candidate,
        relevant=relevant,
        sdg_goal=goal,
        sdg_target=target_id,
        claim=f"Emissions were {value} {unit}.".strip(),
        supporting_quote=quote,
        dimensions={"specificity": True, "measurable_indicator": True},
        extracted={
            "value": value,
            "unit": unit,
            "year": "2022",
            "status": "reported",
            "baseline": baseline,
        },
        confidence=confidence,
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


def _payload(title: str, findings: list[VerifiedFinding]) -> dict:
    return {
        "schema_version": "0.1",
        "document": {"title": title, "pages": 20},
        "findings": [finding.to_dict() for finding in findings],
    }


# ---------------------------------------------------------------------------
# Value and unit extraction
# ---------------------------------------------------------------------------


def test_unit_normalisation_handles_common_aliases() -> None:
    assert normalise_unit("MTCO2e") == "mtco2e"
    assert normalise_unit("ktCO2e") == "ktco2e"
    assert normalise_unit("Tonnes") == "t"
    assert normalise_unit(" per cent ") == "%"
    assert normalise_unit("million tonnes") == "million t"
    assert normalise_unit("") == ""


# ---------------------------------------------------------------------------
# Comparison outcomes
# ---------------------------------------------------------------------------


def test_progressing_when_value_falls_for_reduction_target() -> None:
    report = build_tracking(
        [_finding(value="100")], [_finding(value="80")], CONFIG
    )

    record = report.records[0]
    assert record.status == "progressing"
    assert record.change == -20.0
    assert record.change_pct == -20.0
    assert record.direction == "decrease"
    assert report.counts()["progressing"] == 1


def test_progressing_when_value_rises_for_increase_target() -> None:
    report = build_tracking(
        [
            _finding(
                target_id="7.2",
                goal=7,
                target_text="Increase the share of renewable energy.",
                value="10",
                unit="%",
            )
        ],
        [
            _finding(
                target_id="7.2",
                goal=7,
                target_text="Increase the share of renewable energy.",
                value="20",
                unit="%",
            )
        ],
        CONFIG,
    )

    assert report.records[0].status == "progressing"
    assert report.records[0].direction == "increase"


def test_unchanged_when_values_are_equal() -> None:
    report = build_tracking(
        [_finding(value="100")], [_finding(value="100")], CONFIG
    )

    record = report.records[0]
    assert record.status == "unchanged"
    assert record.change == 0.0
    assert record.direction is None


def test_changed_when_direction_is_unclear_or_unfavourable() -> None:
    unclear = build_tracking(
        [
            _finding(
                target_text="Integrate climate measures into policy.",
                value="100",
            )
        ],
        [
            _finding(
                target_text="Integrate climate measures into policy.",
                value="80",
            )
        ],
        CONFIG,
    )
    assert unclear.records[0].status == "changed"
    assert "clear favourable direction" in unclear.records[0].reason

    unfavourable = build_tracking(
        [
            _finding(
                target_text="Increase the share of renewable energy.",
                value="10",
                unit="%",
            )
        ],
        [
            _finding(
                target_text="Increase the share of renewable energy.",
                value="5",
                unit="%",
            )
        ],
        CONFIG,
    )
    assert unfavourable.records[0].status == "changed"
    assert "not the direction" in unfavourable.records[0].reason


def test_missing_unit_is_not_comparable() -> None:
    earlier = _finding(value="100", unit="", quote="We planted 100 saplings.")
    later = _finding(value="80", unit="", quote="We planted 80 saplings.")

    report = build_tracking([earlier], [later], CONFIG)

    assert report.records[0].status == "not_comparable"
    assert "missing recognized unit" in report.records[0].reason


def test_missing_value_is_not_comparable() -> None:
    earlier = _finding(value="", unit="", quote="We will reduce waste.")
    later = _finding(value="", unit="", quote="We will reduce waste further.")

    report = build_tracking([earlier], [later], CONFIG)

    assert report.records[0].status == "not_comparable"
    assert "missing numeric value" in report.records[0].reason


def test_changed_unit_is_not_comparable() -> None:
    report = build_tracking(
        [_finding(value="100", unit="tCO2e")],
        [_finding(value="80", unit="MWh")],
        CONFIG,
    )

    record = report.records[0]
    assert record.status == "not_comparable"
    assert "unit changed" in record.reason


def test_multiplier_and_kilotonne_units_are_not_collapsed() -> None:
    million_tonnes = _finding(
        value="",
        unit="",
        quote="Emissions were 1 million tonnes in FY2023.",
    )
    plain_tonnes = _finding(
        value="900000", unit="t", quote="Emissions were 900,000 tonnes."
    )
    multiplier_report = build_tracking([million_tonnes], [plain_tonnes], CONFIG)
    assert multiplier_report.records[0].earlier is not None
    assert multiplier_report.records[0].earlier.unit == "million t"
    assert multiplier_report.records[0].status == "not_comparable"

    kilotonnes = _finding(value="1", unit="ktCO2e")
    tonnes = _finding(value="1000", unit="tCO2e")
    unit_report = build_tracking([kilotonnes], [tonnes], CONFIG)
    assert unit_report.records[0].status == "not_comparable"
    assert "unit changed" in unit_report.records[0].reason


def test_year_tokens_are_not_used_as_values() -> None:
    year_only = _finding(
        value="",
        unit="",
        quote="In FY2023 we reduced waste at our sites.",
        target_text="Reduce waste.",
    )
    report = build_tracking([year_only], [], CONFIG)
    assert report.records[0].earlier is not None
    assert report.records[0].earlier.value is None

    extracted_year = _finding(
        value="FY2023",
        unit="",
        quote="In FY2023 we reduced waste at our sites.",
        target_text="Reduce waste.",
    )
    report = build_tracking([extracted_year], [], CONFIG)
    assert report.records[0].earlier is not None
    assert report.records[0].earlier.value is None

    with_number = _finding(
        value="",
        unit="",
        quote="In FY2023 we reduced waste by 10 tonnes.",
        target_text="Reduce waste.",
    )
    later = _finding(value="8", unit="t", target_text="Reduce waste.")
    report = build_tracking([with_number], [later], CONFIG)
    assert report.records[0].earlier is not None
    assert report.records[0].earlier.value == 10.0
    assert report.records[0].status == "progressing"


def test_number_is_bound_to_the_recognised_unit() -> None:
    earlier = _finding(
        value="",
        unit="",
        quote="We planted 1 million trees and emitted 50 tCO2e.",
    )
    later = _finding(value="40", unit="tCO2e")

    report = build_tracking([earlier], [later], CONFIG)

    assert report.records[0].earlier is not None
    assert report.records[0].earlier.value == 50.0
    assert report.records[0].status == "progressing"


def test_baseline_change_is_not_comparable() -> None:
    report = build_tracking(
        [_finding(value="100", baseline="2020")],
        [_finding(value="80", baseline="2021")],
        CONFIG,
    )

    record = report.records[0]
    assert record.status == "not_comparable"
    assert "baseline changed" in record.reason


def test_matching_baseline_is_comparable() -> None:
    report = build_tracking(
        [_finding(value="100", baseline="FY2020")],
        [_finding(value="80", baseline="fy2020")],
        CONFIG,
    )

    assert report.records[0].status == "progressing"


def test_unit_check_can_be_disabled_by_config() -> None:
    config = {"tracking": {"min_level": 3, "require_same_unit": False}}
    report = build_tracking(
        [_finding(value="100", unit="tCO2e")],
        [_finding(value="80", unit="MWh")],
        config,
    )

    assert report.records[0].status == "progressing"


# ---------------------------------------------------------------------------
# Target presence and representative selection
# ---------------------------------------------------------------------------


def test_target_missing_later_is_not_found_not_discontinued() -> None:
    report = build_tracking([_finding(value="100")], [], CONFIG)

    record = report.records[0]
    assert record.status == "not_found_in_later_report"
    assert "needs review" in record.status_text
    assert "discontinued" not in " ".join(report.notes).lower().replace(
        "never discontinued", ""
    )
    assert "never discontinued" in " ".join(report.notes).lower()


def test_new_target_in_later_report() -> None:
    report = build_tracking([], [_finding(value="100")], CONFIG)

    record = report.records[0]
    assert record.status == "new_in_later_report"
    assert record.earlier is None
    assert record.later is not None


def test_unscored_and_below_level_findings_are_ignored() -> None:
    unverified = _finding(value="100", quote_verified=False)
    low_level = _finding(value="90", level=2, page=11)
    report = build_tracking([unverified], [low_level], CONFIG)

    assert report.records == []


def test_flagged_findings_are_ignored() -> None:
    flagged = _finding(value="100", review_reason="needs human review")
    report = build_tracking([flagged], [_finding(value="80")], CONFIG)

    assert [record.status for record in report.records] == [
        "new_in_later_report"
    ]


def test_semantically_unsupported_findings_are_ignored() -> None:
    unsupported = _finding(value="100", semantic_support="not_supported")
    report = build_tracking([unsupported], [_finding(value="80")], CONFIG)

    assert report.records[0].status == "new_in_later_report"


def test_highest_level_representative_is_chosen_deterministically() -> None:
    earlier = [
        _finding(value="100", level=3, page=10),
        _finding(value="90", level=4, page=11),
    ]
    report = build_tracking(earlier, [_finding(value="80")], CONFIG)

    record = report.records[0]
    assert record.earlier is not None
    assert record.earlier.level == 4
    assert record.earlier.value == 90.0


def test_highest_confidence_breaks_a_level_tie() -> None:
    earlier = [
        _finding(value="100", level=3, page=10, confidence=0.5),
        _finding(value="90", level=3, page=11, confidence=0.95),
    ]
    report = build_tracking(earlier, [_finding(value="80")], CONFIG)

    assert report.records[0].earlier is not None
    assert report.records[0].earlier.value == 90.0


def test_records_are_sorted_by_goal_and_target() -> None:
    earlier = [
        _finding(target_id="13.2", goal=13),
        _finding(target_id="6.4", goal=6),
        _finding(target_id="13.1", goal=13),
    ]
    report = build_tracking(earlier, [], CONFIG)

    assert [record.target_id for record in report.records] == [
        "6.4",
        "13.1",
        "13.2",
    ]


# ---------------------------------------------------------------------------
# Configuration and rendering
# ---------------------------------------------------------------------------


def test_min_level_override_allows_lower_levels() -> None:
    earlier = _finding(value="100", level=3)
    later = _finding(value="90", unit="tCO2e", level=2)
    report = build_tracking([earlier], [later], CONFIG, min_level=2)

    assert report.records[0].status == "progressing"
    assert report.min_level == 2
    assert report.to_dict()["min_level"] == 2
    assert "Only Level 2 and above scored findings" in report.notes[0]
    assert "Only Level 2 and above scored findings" in render_markdown(report)


def test_default_min_level_is_level_three() -> None:
    report = build_tracking([], [], {})
    assert report.min_level == 3


def test_invalid_min_level_is_rejected() -> None:
    for invalid in (0, 5, 2.5, True, "three"):
        with pytest.raises(ValueError):
            build_tracking([], [], CONFIG, min_level=invalid)


def test_empty_report_message_uses_configured_min_level() -> None:
    markdown = render_markdown(build_tracking([], [], CONFIG, min_level=2))
    assert "No quantified findings at Level 2 or above" in markdown


def test_markdown_and_json_include_statuses_and_counts() -> None:
    report = build_tracking(
        [_finding(value="100")],
        [_finding(value="80")],
        CONFIG,
        earlier_label="Alpha 2022",
        later_label="Alpha 2023",
    )
    markdown = render_markdown(report)
    data = report.to_dict()

    assert "Alpha 2022" in markdown
    assert "Alpha 2023" in markdown
    assert "| 13.2 | progressing |" in markdown
    assert data["counts"]["progressing"] == 1
    assert data["records"][0]["status_text"] == "progressing"
    assert set(data["counts"]) == set(TRACKING_STATUS_TEXT)


def test_markdown_uses_ascii_placeholders() -> None:
    markdown = render_markdown(build_tracking([], [], CONFIG))
    assert markdown.isascii()


# ---------------------------------------------------------------------------
# Payload loading and CLI
# ---------------------------------------------------------------------------


def test_missing_findings_list_is_an_error() -> None:
    with pytest.raises(ValueError):
        track_payloads(
            {"document": {"title": "Earlier"}},
            {"document": {"title": "Later"}, "findings": []},
            CONFIG,
        )


def test_track_payloads_round_trips_serialized_findings() -> None:
    report = track_payloads(
        _payload("Earlier", [_finding(value="100")]),
        _payload("Later", [_finding(value="80")]),
        CONFIG,
    )

    assert report.records[0].status == "progressing"


def test_cli_writes_json_and_markdown(tmp_path) -> None:
    earlier = tmp_path / "earlier.json"
    later = tmp_path / "later.json"
    earlier.write_text(
        json.dumps(_payload("Earlier", [_finding(value="100")])),
        encoding="utf-8",
    )
    later.write_text(
        json.dumps(_payload("Later", [_finding(value="80")])),
        encoding="utf-8",
    )
    out_json = tmp_path / "tracking.json"
    out_md = tmp_path / "tracking.md"

    code = main(
        [
            str(earlier),
            str(later),
            "--out",
            str(out_json),
            "--markdown",
            str(out_md),
            "--earlier-label",
            "Alpha 2022",
            "--later-label",
            "Alpha 2023",
        ]
    )

    assert code == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["earlier_label"] == "Alpha 2022"
    assert data["records"][0]["status"] == "progressing"
    assert "progressing" in out_md.read_text(encoding="utf-8")


def test_cli_returns_two_for_invalid_payload(tmp_path) -> None:
    earlier = tmp_path / "earlier.json"
    later = tmp_path / "later.json"
    earlier.write_text(json.dumps({"document": {}}), encoding="utf-8")
    later.write_text(json.dumps(_payload("Later", [])), encoding="utf-8")

    assert main([str(earlier), str(later)]) == 2


def test_cli_returns_two_for_invalid_min_level(tmp_path) -> None:
    earlier = tmp_path / "earlier.json"
    later = tmp_path / "later.json"
    earlier.write_text(json.dumps(_payload("Earlier", [])), encoding="utf-8")
    later.write_text(json.dumps(_payload("Later", [])), encoding="utf-8")

    assert main([str(earlier), str(later), "--min-level", "5"]) == 2
