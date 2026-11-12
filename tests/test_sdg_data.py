"""Validate the bundled UN SDG target dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SDG_PATH = PROJECT_ROOT / "data" / "sdg_targets.json"

EXPECTED_TARGETS_PER_GOAL = {
    1: 7,
    2: 8,
    3: 13,
    4: 10,
    5: 9,
    6: 8,
    7: 5,
    8: 12,
    9: 8,
    10: 10,
    11: 10,
    12: 11,
    13: 5,
    14: 10,
    15: 12,
    16: 12,
    17: 19,
}


def _load() -> dict:
    with SDG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_dataset_has_all_goals_and_targets() -> None:
    data = _load()
    assert len(data["goals"]) == 17
    assert len(data["targets"]) == 169
    assert sum(EXPECTED_TARGETS_PER_GOAL.values()) == 169


def test_target_counts_match_un_framework() -> None:
    data = _load()
    counts = Counter(target["goal"] for target in data["targets"])
    assert dict(counts) == EXPECTED_TARGETS_PER_GOAL


def test_target_ids_are_unique_and_well_formed() -> None:
    data = _load()
    ids = [target["id"] for target in data["targets"]]
    assert len(ids) == len(set(ids))
    assert all("." in target_id for target_id in ids)
    assert all(target["text"].strip() for target in data["targets"])


def test_every_goal_has_targets() -> None:
    data = _load()
    by_goal = {goal["goal"]: 0 for goal in data["goals"]}
    for target in data["targets"]:
        by_goal[target["goal"]] += 1
    assert all(count > 0 for count in by_goal.values())
