"""Primary corpus analysis: volume and text characteristics (task.md 2.6)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..corpus.language import normalize_language_code
from ..utils import get_logger

logger = get_logger("textvec.analysis.stats")


def _detect_language(text: str) -> str:
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(text) if text.strip() else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def corpus_statistics(df: pd.DataFrame, sample_lang: int = 200) -> dict:
    """Compute volume and length characteristics of the corpus.

    Language is detected on a sample (``sample_lang`` docs) to keep it fast.
    When a ``language`` metadata column is present it is also summarized directly.
    """
    text = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).str.strip()
    word_counts = text.str.split().map(len)
    char_counts = text.str.len()

    lang_sample = text.head(sample_lang).map(_detect_language)

    summary = {
        "n_documents": int(len(df)),
        "by_source": df["source"].value_counts().to_dict(),
        "by_topic": df["topic"].value_counts().to_dict(),
        "language_sample": lang_sample.value_counts().to_dict(),
        "words": {
            "min": int(word_counts.min()) if len(df) else 0,
            "max": int(word_counts.max()) if len(df) else 0,
            "mean": round(float(word_counts.mean()), 2) if len(df) else 0.0,
            "median": float(word_counts.median()) if len(df) else 0.0,
        },
        "chars": {
            "min": int(char_counts.min()) if len(df) else 0,
            "max": int(char_counts.max()) if len(df) else 0,
            "mean": round(float(char_counts.mean()), 2) if len(df) else 0.0,
        },
    }
    if "language" in df.columns:
        langs = df["language"].map(lambda v: normalize_language_code(v, default="unknown"))
        summary["by_language"] = langs.value_counts().to_dict()
    return summary


def save_statistics(summary: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    logger.info("Saved corpus statistics -> %s", out_path)
    return out_path


def print_statistics(summary: dict) -> None:
    logger.info("Corpus: %d documents", summary["n_documents"])
    logger.info("  by source: %s", summary["by_source"])
    logger.info("  by topic : %s", summary["by_topic"])
    logger.info("  words/doc: mean=%s median=%s (min=%s, max=%s)",
                summary["words"]["mean"], summary["words"]["median"],
                summary["words"]["min"], summary["words"]["max"])
    logger.info("  language (sample): %s", summary["language_sample"])
    if "by_language" in summary:
        logger.info("  by language (metadata): %s", summary["by_language"])
