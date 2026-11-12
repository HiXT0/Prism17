"""Tests for target loading and embedding-cache behaviour."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prism17.config import load_config
from prism17.retrieve import (
    embed_texts,
    load_targets,
    rank_targets_for_texts,
)


def _config(tmp_path: Path) -> dict:
    config = load_config()
    config["retrieval"] = {
        **config["retrieval"],
        "cache_dir": str(tmp_path / "embeddings"),
    }
    return config


def test_load_targets_returns_all_goals_by_default() -> None:
    targets = load_targets(load_config())
    assert len(targets) == 169


def test_load_targets_filters_focus_goals() -> None:
    targets = load_targets(load_config(), [13])
    assert targets
    assert all(int(target["goal"]) == 13 for target in targets)


def test_load_targets_rejects_unknown_focus() -> None:
    with pytest.raises(ValueError):
        load_targets(load_config(), [99])


def test_embed_texts_empty_returns_empty_matrix() -> None:
    result = embed_texts([], load_config(), cache_prefix="test")
    assert result.shape == (0, 0)


def test_embed_texts_caches_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, texts, **kwargs):  # noqa: ANN001, ANN003, ANN201
            self.calls += 1
            return np.ones((len(texts), 4), dtype="float32")

    fake = FakeModel()
    monkeypatch.setattr("prism17.retrieve._load_model", lambda _name: fake)

    config = _config(tmp_path)
    first = embed_texts(["alpha", "beta"], config, cache_prefix="test")
    second = embed_texts(["alpha", "beta"], config, cache_prefix="test")

    assert first.shape == (2, 4)
    assert np.array_equal(first, second)
    assert fake.calls == 1


def test_rank_targets_for_texts_returns_top_k_without_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeModel:
        def encode(self, texts, **kwargs):  # noqa: ANN001, ANN003, ANN201
            return np.array(
                [
                    [1.0, 0.0] if "climate" in text else [0.0, 1.0]
                    for text in texts
                ],
                dtype="float32",
            )

    targets = [
        {"id": "13.2", "goal": 13, "text": "climate action"},
        {"id": "6.1", "goal": 6, "text": "water supply"},
    ]
    monkeypatch.setattr(
        "prism17.retrieve._load_model", lambda _name: FakeModel()
    )
    monkeypatch.setattr(
        "prism17.retrieve.load_targets",
        lambda _config, focus_goals=None: targets,
    )

    ranked = rank_targets_for_texts(
        ["climate policy", "water policy"], _config(tmp_path), top_k=2
    )

    assert len(ranked) == 2
    assert ranked[0][0]["target_id"] == "13.2"
    assert ranked[0][0]["score"] == pytest.approx(1.0)
    assert ranked[1][0]["target_id"] == "6.1"
    assert len(ranked[0]) == 2


def test_rank_targets_for_texts_empty_input() -> None:
    assert rank_targets_for_texts([], load_config()) == []
