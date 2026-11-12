"""Configuration loading for Prism17.

Paths in config.yaml are resolved relative to the project root (the directory
containing config.yaml), not the current working directory. This keeps the CLI
and Streamlit app consistent regardless of where they are launched from.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml and attach the resolved project root.

    Raises:
        FileNotFoundError: if the config file does not exist.
        ValueError: if the YAML document is not a mapping.
    """
    config_path = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    data["_config_path"] = str(config_path)
    data["_project_root"] = str(config_path.parent)
    return data


def project_root(config: dict[str, Any] | None = None) -> Path:
    """Return the project root associated with a loaded config."""
    if config and config.get("_project_root"):
        return Path(str(config["_project_root"]))
    return PROJECT_ROOT


def resolve_path(value: str | Path, config: dict[str, Any] | None = None) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root(config) / path
    return path.resolve()


def configured_path(
    config: dict[str, Any],
    *keys: str,
    default: str | Path | None = None,
) -> Path:
    """Read a nested config value and resolve it as a path.

    Example:
        configured_path(config, "paths", "reports_dir")
    """
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            if default is None:
                raise KeyError(f"Missing config key: {'.'.join(keys)}")
            value = default
            break
        value = value[key]
    return resolve_path(value, config)


def get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Read a nested config value with a default."""
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def coerce_rubric_level(
    value: Any,
    *,
    setting: str = "min_level",
    allowed: tuple[int, ...] = (1, 2, 3, 4),
) -> int:
    """Return a validated integer rubric level.

    Rejects booleans, non-numeric values, fractional values, and levels outside
    ``allowed`` rather than silently truncating them.
    """
    if isinstance(value, bool):
        raise ValueError(f"{setting} must be an integer, got {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{setting} must be an integer, got {value!r}") from exc
    if not numeric.is_integer():
        raise ValueError(f"{setting} must be an integer, got {value!r}")
    level = int(numeric)
    if level not in allowed:
        raise ValueError(f"{setting} must be one of {allowed}, got {level}")
    return level
