"""Tests for narrow multi-year target tracking."""

from __future__ import annotations

import json

import pytest

from prism17.models import Candidate, JudgedFinding, VerifiedFinding
from prism17.tracking import (
    TRACKING_STATUS_TEXT,
    build_tracking,
    infer_direction,
    infer_reporting_year,
    main,
    normalise_unit,
    render_markdown,
    track_payloads,
)

CONFIG = {"tracking": {"min_level": 3, "require_same_unit": True}}


def _finding(
    target_id: str = "13.2",
    *,
    level: int = 3,
    value: str = "",
    unit: str = "",
    year: str = "",
    status: str = "reported",
    baseline: str = "",
    page: int = 10,
    quote: str = "Example quantified disclosure text.",
    claim: str = "Example claim.",
    target_text: str = "Reduce emissions.",
    confidence: float = 0.9,
    passage_id: str | None = None,
    quote_verified: bool = True,
    semantic_support: str = "not_checked",
) -> VerifiedFinding:
    candidate = Candidate(
        passage_id=passage_id or f"p{page:04d}-001",
        page=page,
        target_id=target_id,
        target_goal=int(target_id.split(".", 1)[0]),
        target_text=target_text,
        score=0.8,
        passage_text=quote,
    )
    judged = JudgedFinding(
        candidate=candidate,
        relevant=True,
        sdg_goal=int(target_id.split(".", 1)[0]),
        sdg_target=target_id,
        claim=claim,
        supporting_quote=quote,
        dimensions={"specificity": True},
        extracted={
            "value": value,
            "unit": unit,
            "year": year,
            "status": status,
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
        level=level if quote_verified else None,
        semantic_support=semantic_support,
    )


def _record(report, status: str):
    return next(record for record in report.records if record.status == status)


# ---------------------------------------------------------------------------
# Value and unit extraction
# ---------------------------------------------------------------------------


def test_unit_normalisation_handles_common_aliases() -> None:
    assert normalise_unit("MTCO2e") == "tco2e"
    assert normalise_unit("ktCO2e") == "ktco2e"
    assert normalise_unit("Tonnes") == "t"
    assert normalise_unit(" per cent ") == "%"
    assert normalise_unit("") == ""


def test_value_falls_back_to_the_verified_quote() -> None:
    earlier = _finding(
        value="",
        unit="",
        quote="We cut waste to 25 tonnes in the reporting year.",
        target_text="Reduce waste.",
    )
    later = _finding(value="20", unit="t", quote="Waste was 20 tonnes.")
    report = build_tracking([earlier], [later], CONFIG)

    record = report.records[0]
    assert record.earlier is not None
    assert record.earlier.value == 25.0
    assert record.earlier.unit == "t"
    assert record.status == "progressing"
    assert record.change == -5.0


# ---------------------------------------------------------------------------
# Comparison outcomes
# ---------------------------------------------------------------------------


def test_progressing_when_value_moves_in_favourable_direction() -> None:
    earlier = _finding(value="100", unit="tCO2e", target_text="Reduce emissions.")
    later = _finding(value="80", unit="tCO2e", target_text="Reduce emissions.")
    report = build_tracking([earlier], [later], CONFIG)

    record = report.records[0]
    assert record.status == "progressing"
    assert record.direction == "decrease"
    assert record.change == -20.0
    assert record.change_percent == -20.0
    assert "favourable direction" in record.reason


def test_changed_when_value_moves_against_favourable_direction() -> None:
    earlier = _finding(value="100", unit="tCO2e", target_text="Reduce emissions.")
    later = _finding(value="120", unit="tCO2e", target_text="Reduce emissions.")
    report = build_tracking([earlier], [later], CONFIG)

    record = report.records[0]
    assert record.status == "changed"
    assert record.change == 20.0
    assert "against the favourable direction" in record.reason


def test_unchanged_when_value_is_the_same_within_tolerance() -> None:
    earlier = _finding(value="100", unit="tCO2e")
    later = _finding(value="100.0", unit="tCO2e")
    report = build_tracking([earlier], [later], CONFIG)

    record = report.records[0]
    assert record.status == "unchanged"
    assert record.change == 0.0


def test_changed_when_favourable_direction_is_unknown() -> None:
    earlier = _finding(
        value="10",
        unit="units",
        target_text="Our approach to water.",
        claim="Our approach to water.",
        quote="Our approach to water.",
    )
    later = _finding(
        value="20",
        unit="units",
        target_text="Our approach to water.",
        claim="Our approach to water.",
        quote="Our approach to water.",
    )
    report = build_tracking([earlier], [later], CONFIG)

    record = report.records[0]
    assert record.status == "changed"
    assert record.direction == "unknown"
    assert "could not be inferred" in record.reason


def test_unit_change_is_not_comparable() -> None:
    earlier = _finding(value="100", unit="tCO2e")
    later = _finding(value="80", unit="MWh")
    report = build_tracking([earlier], [later], CONFIG)

    record = report.records[0]
    assert record.status == "not_comparable"
    assert record.change is None
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


def test_unit_change_can_be_ignored_by_config() -> None:
    config = {"tracking": {"min_level": 3, "require_same_unit": False}}
    earlier = _finding(value="100", unit="tCO2e")
    later = _finding(value="80", unit="MWh")
    report = build_tracking([earlier], [later], config)

    assert report.records[0].status == "progressing"


def test_missing_unit_is_not_comparable() -> None:
    earlier = _finding(
        value="100",
        unit="",
        quote="Example quantified disclosure text without a unit.",
    )
    later = _finding(value="80", unit="tCO2e")
    report = build_tracking([earlier], [later], CONFIG)

    record = report.records[0]
    assert record.status == "not_comparable"
    assert "unit was not extracted" in record.reason


def test_baseline_change_is_not_comparable() -> None:
    earlier = _finding(value="100", unit="tCO2e", baseline="FY2020")
    later = _finding(value="80", unit="tCO2e", baseline="FY2021")
    report = build_tracking([earlier], [later], CONFIG)

    record = report.records[0]
    assert record.status == "not_comparable"
    assert "baseline changed" in record.reason


def test_missing_numeric_value_is_not_comparable() -> None:
    earlier = _finding(value="", unit="", quote="No numeric value disclosed here.")
    later = _finding(value="80", unit="tCO2e")
    report = build_tracking([earlier], [later], CONFIG)

    record = report.records[0]
    assert record.status == "not_comparable"
    assert "numeric value was not extracted" in record.reason


def test_missing_later_target_is_never_called_discontinued() -> None:
    earlier = _finding(value="100", unit="tCO2e")
    report = build_tracking([earlier], [], CONFIG)

    record = report.records[0]
    assert record.status == "not_found_in_later_report"
    assert record.status_text == "not found in later report (needs review)"
    assert "not evidence that the target was discontinued" in record.reason


def test_new_later_target_is_reported_separately() -> None:
    later = _finding(value="100", unit="tCO2e")
    report = build_tracking([], [later], CONFIG)

    record = report.records[0]
    assert record.status == "new_in_later_report"
    assert record.earlier is None


def test_only_scored_findings_at_or_above_min_level_are_compared() -> None:
    low_level = _finding(value="100", unit="tCO2e", level=2)
    unverified = _finding(value="100", unit="tCO2e", quote_verified=False)
    blocked = _finding(
        value="100", unit="tCO2e", semantic_support="uncertain"
    )
    report = build_tracking([low_level, unverified, blocked], [], CONFIG)

    assert report.records == []


def test_min_level_override_allows_lower_levels() -> None:
    earlier = _finding(value="100", unit="tCO2e", level=2)
    later = _finding(value="90", unit="tCO2e", level=2)
    report = build_tracking([earlier], [later], CONFIG, min_level=2)

    assert report.records[0].status == "progressing"
    assert report.min_level == 2
    assert report.to_dict()["min_level"] == 2
    assert "Only Level 2 and above" in report.notes[0]
    assert "Only Level 2 and above scored findings" in render_markdown(report)


def test_invalid_min_level_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_tracking([], [], CONFIG, min_level=5)
    with pytest.raises(ValueError):
        build_tracking([], [], CONFIG, min_level=0)


def test_empty_report_message_uses_configured_min_level() -> None:
    markdown = render_markdown(build_tracking([], [], CONFIG, min_level=2))
    assert "No quantified findings at Level 2 or above" in markdown


def test_missing_findings_list_is_an_error() -> None:
    with pytest.raises(ValueError):
        track_payloads(
            {"document": {"title": "Earlier"}},
            {"document": {"title": "Later"}, "findings": []},
            CONFIG,
        )


def test_highest_level_representative_is_chosen_deterministically() -> None:
    earlier = [
        _finding(value="100", unit="tCO2e", level=3, page=10),
        _finding(value="90", unit="tCO2e", level=4, page=11),
    ]
    later = _finding(value="90", unit="tCO2e", level=4, page=12)
    report = build_tracking(earlier, [later], CONFIG)

    record = report.records[0]
    assert record.earlier is not None
    assert record.earlier.value == 90.0
    assert record.earlier.level == 4
    assert record.status == "unchanged"


# ---------------------------------------------------------------------------
# Direction and year inference
# ---------------------------------------------------------------------------


def test_direction_inference_uses_target_wording() -> None:
    decrease = _finding(value="1", target_text="Reduce emissions.")
    increase = _finding(value="1", target_text="Increase renewable capacity.")
    unknown = _finding(value="1", target_text="Our approach to water.")

    assert infer_direction(
        next(iter(build_tracking([decrease], [], CONFIG).records)).earlier
    ) == "decrease"
    assert infer_direction(
        next(iter(build_tracking([increase], [], CONFIG).records)).earlier
    ) == "increase"
    assert infer_direction(
        next(iter(build_tracking([unknown], [], CONFIG).records)).earlier
    ) == "unknown"


def test_reporting_year_prefers_financial_year_end() -> None:
    assert infer_reporting_year({"title": "Annual Report FY2023-24"}) == 2024
    assert infer_reporting_year({"title": "brsr_fy2023_24"}) == 2024
    assert infer_reporting_year({"title": "Sustainability Report 2022"}) == 2022
    assert infer_reporting_year({"pdf_path": "reports/annual_2021.pdf"}) == 2021
    assert infer_reporting_year({}) is None
    assert infer_reporting_year(None) is None


def test_track_payloads_uses_document_labels_and_years() -> None:
    earlier = _finding(value="100", unit="tCO2e")
    later = _finding(value="80", unit="tCO2e")
    earlier_payload = {
        "document": {"title": "Earlier Report FY2022-23"},
        "findings": [earlier.to_dict()],
    }
    later_payload = {
        "document": {"title": "Later Report FY2023-24"},
        "findings": [later.to_dict()],
    }

    report = track_payloads(earlier_payload, later_payload, CONFIG)

    assert report.earlier_label == "Earlier Report FY2022-23"
    assert report.later_label == "Later Report FY2023-24"
    assert report.records[0].earlier is not None
    assert report.records[0].earlier.year == "2023"
    assert report.records[0].later is not None
    assert report.records[0].later.year == "2024"


# ---------------------------------------------------------------------------
# Rendering and CLI
# ---------------------------------------------------------------------------


def test_status_labels_are_neutral() -> None:
    for label in TRACKING_STATUS_TEXT.values():
        assert "discontinued" not in label.lower()


def test_markdown_report_contains_neutral_caveat_and_records() -> None:
    earlier = _finding(value="100", unit="tCO2e")
    later = _finding(value="80", unit="tCO2e")
    markdown = render_markdown(build_tracking([earlier], [later], CONFIG))

    assert "does not verify real-world performance" in markdown
    assert "| progressing |" in markdown
    assert "not evidence that the target was discontinued" in markdown


def test_cli_writes_json_and_markdown(tmp_path, capsys) -> None:
    earlier = _finding(value="100", unit="tCO2e")
    later = _finding(value="80", unit="tCO2e")
    earlier_path = tmp_path / "earlier.json"
    later_path = tmp_path / "later.json"
    out_path = tmp_path / "tracking.json"
    markdown_path = tmp_path / "tracking.md"
    earlier_path.write_text(
        json.dumps(
            {
                "document": {"title": "Earlier Report"},
                "findings": [earlier.to_dict()],
            }
        ),
        encoding="utf-8",
    )
    later_path.write_text(
        json.dumps(
            {
                "document": {"title": "Later Report"},
                "findings": [later.to_dict()],
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            str(earlier_path),
            str(later_path),
            "--out",
            str(out_path),
            "--markdown",
            str(markdown_path),
        ]
    )

    assert code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "0.1"
    assert payload["records"][0]["status"] == "progressing"
    assert "progressing" in markdown_path.read_text(encoding="utf-8")
    assert "Wrote tracking JSON" in capsys.readouterr().out


def test_cli_returns_error_for_missing_input(tmp_path) -> None:
    code = main(
        [str(tmp_path / "missing-a.json"), str(tmp_path / "missing-b.json")]
    )
    assert code == 2
