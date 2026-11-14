"""Tests for configuration path helpers."""

from __future__ import annotations

from prism17.config import display_path, project_root, resolve_path


def test_display_path_keeps_relative_paths_relative() -> None:
    assert display_path("data/reports/example.pdf") == "data/reports/example.pdf"


def test_display_path_strips_absolute_project_paths(tmp_path) -> None:
    config = {"_project_root": str(tmp_path)}
    absolute = tmp_path / "data" / "reports" / "example.pdf"

    assert display_path(absolute, config) == "data/reports/example.pdf"


def test_display_path_reduces_outside_paths_to_a_file_name(tmp_path) -> None:
    config = {"_project_root": str(tmp_path / "project")}
    outside = tmp_path / "elsewhere" / "example.pdf"

    assert display_path(outside, config) == "example.pdf"


def test_resolve_path_uses_the_configured_project_root(tmp_path) -> None:
    config = {"_project_root": str(tmp_path)}

    assert resolve_path("data/example.pdf", config) == (
        tmp_path / "data" / "example.pdf"
    ).resolve()
    assert project_root(config) == tmp_path
