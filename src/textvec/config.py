"""Configuration loading and access.

A thin wrapper around a YAML file that exposes values both as a nested
mapping (for serialization / reproducibility) and via attribute access
(``cfg.preprocessing.lowercase``) for convenience.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Repository root = two levels above this file (src/textvec/config.py -> Project/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class ConfigNode:
    """Recursive attribute/dict access wrapper around a mapping."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        return _wrap(value)

    def __getitem__(self, key: str) -> Any:
        return _wrap(self._data[key])

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._data:
            return _wrap(self._data[key])
        return default

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ConfigNode({self._data!r})"


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigNode(value)
    return value


def load_config(path: str | Path | None = None) -> ConfigNode:
    """Load configuration from ``path`` (defaults to ``config/default.yaml``)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return ConfigNode(data)


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(relative)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
