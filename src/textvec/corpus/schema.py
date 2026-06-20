"""Unified document schema shared by every corpus source.

Keeping a single, source-agnostic record makes it trivial to merge ArXiv,
PubMed (and future sources) into one CSV/JSON corpus with consistent
metadata (task.md, stage 2.5).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Order of columns in the exported CSV / JSON.
DOCUMENT_FIELDS = [
    "doc_id",
    "source",
    "topic",
    "title",
    "abstract",
    "authors",
    "published",
    "categories",
    "url",
    "language",
]


@dataclass
class Document:
    """A single article with normalized metadata."""

    doc_id: str
    source: str               # "arxiv" | "pubmed"
    topic: str                # human-readable topical label from the query
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    published: str = ""       # ISO date string (YYYY-MM-DD) when available
    categories: list[str] = field(default_factory=list)
    url: str = ""
    language: str = "en"

    def to_row(self) -> dict[str, str]:
        """Flatten to a CSV-friendly dict (lists joined by '; ')."""
        row = asdict(self)
        row["authors"] = "; ".join(self.authors)
        row["categories"] = "; ".join(self.categories)
        return row

    def to_json_obj(self) -> dict:
        return asdict(self)
