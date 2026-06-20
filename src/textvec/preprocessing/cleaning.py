"""Low-level text cleaning and normalization (task.md 3.3 - 3.4).

These functions operate on raw strings and are independent of spaCy so they
can be unit-tested in isolation.
"""
from __future__ import annotations

import re
import unicodedata

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_MULTISPACE_RE = re.compile(r"\s+")
# Keep letters/digits/whitespace/basic apostrophe-hyphen; drop the rest.
_NONTEXT_RE = re.compile(r"[^\w\s'-]", flags=re.UNICODE)
_DIGIT_RE = re.compile(r"\b\d+\b")
# Line-break hyphenation: "repre-\nsentation" -> "representation".
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
# A heading that marks the start of the reference list (last occurrence wins).
_REFERENCES_RE = re.compile(r"\n\s*(references|bibliography|references and notes)\s*\n", flags=re.IGNORECASE)


def strip_html(text: str) -> str:
    """Remove HTML/XML markup. Falls back to a regex if bs4 is unavailable."""
    if not text or "<" not in text:
        return text or ""
    try:
        from bs4 import BeautifulSoup

        return BeautifulSoup(text, "lxml").get_text(separator=" ")
    except Exception:  # noqa: BLE001
        return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


def remove_emails(text: str) -> str:
    return _EMAIL_RE.sub(" ", text)


def remove_numbers(text: str) -> str:
    return _DIGIT_RE.sub(" ", text)


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def remove_punctuation(text: str) -> str:
    return _NONTEXT_RE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    return _MULTISPACE_RE.sub(" ", text).strip()


def dehyphenate(text: str) -> str:
    """Join words split across line breaks by a hyphen."""
    return _HYPHEN_BREAK_RE.sub(r"\1\2", text)


def cut_references(text: str) -> str:
    """Drop everything from the last 'References'/'Bibliography' heading onward."""
    matches = list(_REFERENCES_RE.finditer(text))
    if not matches:
        return text
    # Only cut if the heading sits in the back half (avoids cutting a mention early on).
    last = matches[-1]
    if last.start() > len(text) * 0.5:
        return text[: last.start()]
    return text


def clean_scientific_fulltext(text: str) -> str:
    """Structural cleanup applied to raw full text before tokenization.

    Removes the reference list, fixes line-break hyphenation and normalizes
    whitespace, but keeps case/punctuation for the main preprocessing stage.
    """
    if not isinstance(text, str) or not text:
        return ""
    text = dehyphenate(text)
    text = cut_references(text)
    return normalize_whitespace(text)


def clean_text(
    text: str,
    *,
    remove_html: bool = True,
    drop_urls: bool = True,
    drop_emails: bool = True,
    lowercase: bool = True,
    strip_punct: bool = True,
    drop_numbers: bool = False,
) -> str:
    """Apply the configured cleaning steps in a sensible order."""
    if not isinstance(text, str):
        return ""
    if remove_html:
        text = strip_html(text)
    if drop_urls:
        text = remove_urls(text)
    if drop_emails:
        text = remove_emails(text)
    if lowercase:
        text = text.lower()
    if drop_numbers:
        text = remove_numbers(text)
    if strip_punct:
        text = remove_punctuation(text)
    return normalize_whitespace(text)
