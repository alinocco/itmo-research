"""ISO 639-1 language code normalization for corpus metadata."""
from __future__ import annotations

import ast
import re
from typing import Any

_BROKEN_LIST_LANG_RE = re.compile(r"""^\[['"]?([a-z]{2,3})""", re.IGNORECASE)

_ALIASES: dict[str, str] = {
    "eng": "en",
    "english": "en",
    "rus": "ru",
    "russian": "ru",
    "deu": "de",
    "ger": "de",
    "german": "de",
    "fra": "fr",
    "fre": "fr",
    "french": "fr",
    "spa": "es",
    "spanish": "es",
    "zho": "zh",
    "chi": "zh",
    "chinese": "zh",
    "jpn": "ja",
    "japanese": "ja",
    "por": "pt",
    "portuguese": "pt",
    "ita": "it",
    "italian": "it",
    "kor": "ko",
    "korean": "ko",
    "ara": "ar",
    "arabic": "ar",
}

# PubMed [Language] filter names -> ISO 639-1 (subset used in this project).
_PUBMED_TO_ISO: dict[str, str] = {
    "english": "en",
    "russian": "ru",
    "german": "de",
    "french": "fr",
    "spanish": "es",
    "chinese": "zh",
    "japanese": "ja",
    "portuguese": "pt",
    "italian": "it",
    "korean": "ko",
    "arabic": "ar",
}


def normalize_language_code(value: Any, *, default: str = "unknown") -> str:
    """Map PubMed / metadata language values to a short ISO 639-1 code.

    PubMed XML often stores ``Language`` as a list (e.g. ``['eng']``). Older
    code used ``str(value)[:2]``, which turned that into ``"['"`` in figures.
    """
    if value is None:
        return default

    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return normalize_language_code(value[0], default=default)

    text = str(value).strip().lower()
    if not text:
        return default

    if text.startswith("["):
        match = _BROKEN_LIST_LANG_RE.match(text)
        if match:
            return normalize_language_code(match.group(1), default=default)
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (list, tuple)) and parsed:
            return normalize_language_code(parsed[0], default=default)

    if text in _ALIASES:
        return _ALIASES[text]
    if text in _PUBMED_TO_ISO:
        return _PUBMED_TO_ISO[text]
    if len(text) == 2 and text.isalpha():
        return text

    return default


def detect_language_from_text(text: str, *, default: str = "unknown") -> str:
    """Guess ISO 639-1 code from document text (title + abstract)."""
    sample = str(text).strip()
    if len(sample) < 20:
        return default
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return normalize_language_code(detect(sample), default=default)
    except Exception:  # noqa: BLE001
        return default


def resolve_document_language(
    value: Any,
    *,
    text: str | None = None,
    query_language: str | None = None,
    default: str = "unknown",
) -> str:
    """Resolve language from metadata, with optional query/text fallbacks."""
    code = normalize_language_code(value, default="")
    if code and code != "unknown":
        return code

    query_code = normalize_language_code(query_language, default="") if query_language else ""
    if query_code and query_code != "unknown":
        return query_code

    if text:
        detected = detect_language_from_text(text, default="")
        if detected and detected != "unknown":
            return detected

    return default


def repair_language_series(metadata, text):
    """Normalize metadata language codes; detect from text when still unknown."""
    import pandas as pd

    meta = pd.Series(metadata)
    doc_text = pd.Series(text, index=meta.index)
    resolved = meta.map(lambda v: normalize_language_code(v, default="unknown"))
    unknown = resolved == "unknown"
    if unknown.any():
        resolved.loc[unknown] = doc_text.loc[unknown].map(
            lambda t: detect_language_from_text(t, default="unknown"),
        )
    return resolved
