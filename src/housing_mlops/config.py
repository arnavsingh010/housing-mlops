"""Config loading (YAML + optional nested overrides)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def _deep_update(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Load YAML config and apply nested overrides (handy for tests)."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = _deep_update(copy.deepcopy(cfg), overrides)
    return cfg
