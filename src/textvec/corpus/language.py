"""ISO 639-1 language code normalization for corpus metadata."""
from __future__ import annotations

import ast
from typing import Any

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
