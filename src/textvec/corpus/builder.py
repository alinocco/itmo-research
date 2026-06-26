"""Build a unified corpus from configured sources and persist it (csv/json)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import ConfigNode, resolve_path
from ..utils import get_logger
from .arxiv_loader import fetch_arxiv
from .pubmed_loader import fetch_pubmed
from .schema import DOCUMENT_FIELDS, Document

logger = get_logger("textvec.corpus.builder")


def build_corpus(cfg: ConfigNode) -> pd.DataFrame:
    """Collect documents from all enabled sources, unify and save them."""
    corpus_cfg = cfg.corpus
    documents: list[Document] = []

    sources = corpus_cfg.sources
    if "arxiv" in sources and sources.arxiv.get("enabled", False):
        for q in sources.arxiv.queries:
            documents += fetch_arxiv(
                topic=q["topic"], search=q["search"], max_results=q.get("max_results", 100)
            )

    if "pubmed" in sources and sources.pubmed.get("enabled", False):
        email = sources.pubmed.get("email", "research@example.com")
        api_key = sources.pubmed.get("api_key")
        for q in sources.pubmed.queries:
            documents += fetch_pubmed(
                topic=q["topic"],
                term=q["term"],
                max_results=q.get("max_results", 100),
                email=email,
                api_key=api_key,
                language=q.get("language"),
            )

    df = _to_dataframe(documents)
    _save(df, documents, corpus_cfg)
    return df


def _to_dataframe(documents: list[Document]) -> pd.DataFrame:
    rows = [d.to_row() for d in documents]
    df = pd.DataFrame(rows, columns=DOCUMENT_FIELDS)
    before = len(df)
    # Deduplicate by id, then by (title + abstract) to drop cross-source repeats.
    df = df.drop_duplicates(subset=["doc_id"])
    df = df.drop_duplicates(subset=["title", "abstract"])
    df = df[df["abstract"].str.len() > 0].reset_index(drop=True)
    logger.info("Unified corpus: %d documents (dropped %d duplicates/empty)", len(df), before - len(df))
    return df


def _save(df: pd.DataFrame, documents: list[Document], corpus_cfg: ConfigNode) -> None:
    csv_path = resolve_path(corpus_cfg.corpus_csv)
    json_path = resolve_path(corpus_cfg.corpus_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False, encoding="utf-8")

    kept_ids = set(df["doc_id"])
    json_objs = [d.to_json_obj() for d in documents if d.doc_id in kept_ids]
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(json_objs, fh, ensure_ascii=False, indent=2)

    logger.info("Saved corpus -> %s | %s", csv_path, json_path)


def load_corpus(csv_path: str | Path) -> pd.DataFrame:
    """Load a previously saved corpus CSV."""
    return pd.read_csv(resolve_path(csv_path)).fillna("")
