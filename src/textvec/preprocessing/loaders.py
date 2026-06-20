"""Load raw texts from multiple formats (task.md 3.2): csv, json, txt, html, pdf."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..utils import get_logger
from .cleaning import strip_html

logger = get_logger("textvec.preprocessing.loaders")


def load_text_file(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm", ".xml"}:
        return strip_html(path.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".pdf":
        return _load_pdf(path)
    raise ValueError(f"Unsupported text format: {suffix}")


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a tabular corpus from csv or json."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path).fillna("")
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return pd.DataFrame(data).fillna("")
    raise ValueError(f"Unsupported tabular format: {suffix}")


def load_directory(directory: str | Path, pattern: str = "*") -> pd.DataFrame:
    """Build a minimal corpus DataFrame from a directory of text/html/pdf files."""
    directory = Path(directory)
    rows = []
    for fp in sorted(directory.glob(pattern)):
        if fp.suffix.lower() not in {".txt", ".md", ".html", ".htm", ".xml", ".pdf"}:
            continue
        try:
            rows.append({"doc_id": fp.stem, "source": "file", "topic": directory.name,
                         "title": fp.stem, "abstract": load_text_file(fp)})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", fp, exc)
    logger.info("Loaded %d files from %s", len(rows), directory)
    return pd.DataFrame(rows)
