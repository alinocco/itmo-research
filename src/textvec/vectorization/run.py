"""Run vectorization for a set of methods and persist embeddings + params."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ConfigNode, resolve_path
from ..utils import get_logger, set_seed
from .base import VectorizationResult
from .registry import build_vectorizer

logger = get_logger("textvec.vectorization.run")


def run_vectorization(
    df: pd.DataFrame,
    cfg: ConfigNode,
    methods: list[str] | None = None,
) -> dict[str, VectorizationResult]:
    """Vectorize the (preprocessed) corpus with each requested method."""
    set_seed(cfg.project.get("seed", 42))
    methods = methods or list(cfg.vectorization.get("methods", []))
    out_dir = resolve_path(cfg.vectorization.get("output_dir", "results/embeddings"))
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, VectorizationResult] = {}
    # Merge with any previously saved manifest so repeated runs accumulate methods.
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    else:
        manifest = {}

    for method in methods:
        logger.info("=== Vectorizing with '%s' ===", method)
        try:
            vec = build_vectorizer(method, cfg)
            result = vec.fit_transform(df)
        except Exception as exc:  # noqa: BLE001 - keep going if one model fails
            logger.error("Method '%s' failed: %s", method, exc)
            manifest[method] = {"status": "failed", "error": str(exc)}
            continue

        _save_result(result, out_dir)
        results[method] = result
        manifest[method] = {
            "status": "ok",
            "dim": result.dim,
            "n_docs": len(result.doc_ids),
            "elapsed_sec": result.elapsed_sec,
            "params": _jsonable(result.params),
        }
        logger.info("  -> dim=%d, %.1fs", result.dim, result.elapsed_sec)

    _save_doc_ids(df, out_dir)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    logger.info("Saved embeddings + manifest -> %s", out_dir)
    return results


def _save_result(result: VectorizationResult, out_dir: Path) -> None:
    np.save(out_dir / f"{result.method}.npy", result.embeddings)
    with open(out_dir / f"{result.method}.meta.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "method": result.method,
                "dim": result.dim,
                "elapsed_sec": result.elapsed_sec,
                "params": _jsonable(result.params),
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )


def _save_doc_ids(df: pd.DataFrame, out_dir: Path) -> None:
    meta_cols = [c for c in ("doc_id", "topic", "source", "title") if c in df.columns]
    df[meta_cols].to_csv(out_dir / "doc_index.csv", index=False, encoding="utf-8")


def _jsonable(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if isinstance(v, (tuple, set)):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def load_embeddings(method: str, embeddings_dir: str | Path) -> np.ndarray:
    return np.load(resolve_path(Path(embeddings_dir) / f"{method}.npy"))
