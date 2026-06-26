"""Configuration loading and access.

A thin wrapper around a YAML file that exposes values both as a nested
mapping (for serialization / reproducibility) and via attribute access
(``cfg.preprocessing.lowercase``) for convenience.

Optional overlay: ``config/secrets.local.yaml`` (gitignored) is deep-merged
on top of the main config. Environment variables override sensitive fields:
  * ``NCBI_API_KEY``  -> corpus.sources.pubmed.api_key
  * ``NCBI_EMAIL``    -> corpus.sources.pubmed.email
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

# Repository root = two levels above this file (src/textvec/config.py -> Project/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
SECRETS_PATH = PROJECT_ROOT / "config" / "secrets.local.yaml"


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


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Inject secrets from environment variables when set."""
    out = copy.deepcopy(data)
    pubmed = out.setdefault("corpus", {}).setdefault("sources", {}).setdefault("pubmed", {})
    if os.environ.get("NCBI_API_KEY"):
        pubmed["api_key"] = os.environ["NCBI_API_KEY"]
    if os.environ.get("NCBI_EMAIL"):
        pubmed["email"] = os.environ["NCBI_EMAIL"]
    return out


def load_config(path: str | Path | None = None) -> ConfigNode:
    """Load configuration from ``path`` (defaults to ``config/default.yaml``)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if SECRETS_PATH.exists():
        with open(SECRETS_PATH, "r", encoding="utf-8") as fh:
            secrets = yaml.safe_load(fh) or {}
        data = _deep_merge(data, secrets)
    data = _apply_env_overrides(data)
    return ConfigNode(data)


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(relative)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
