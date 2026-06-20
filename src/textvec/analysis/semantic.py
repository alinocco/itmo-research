"""Semantic-space analysis: distribution metrics + 2D projections (task.md 5)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ConfigNode, resolve_path
from ..utils import get_logger
from .reduce import reduce
from .visualize import scatter_2d

logger = get_logger("textvec.analysis.semantic")


def topic_separability(embeddings: np.ndarray, labels: list[str]) -> dict:
    """Descriptive (not clustering) metrics of how topics sit in the space.

    * silhouette (cosine) using the known topic labels as a reference grouping;
    * k-NN topic purity: share of nearest neighbours sharing a document's topic.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    labels_arr = np.asarray(labels)
    n_topics = len(set(labels))
    metrics: dict[str, float] = {"n_topics": n_topics}

    if n_topics > 1 and len(labels) > n_topics:
        try:
            metrics["silhouette_cosine"] = round(
                float(silhouette_score(embeddings, labels_arr, metric="cosine")), 4
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("silhouette failed: %s", exc)
            metrics["silhouette_cosine"] = None

    k = min(10, len(labels) - 1)
    if k >= 1:
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
        _, idx = nn.kneighbors(embeddings)
        neighbours = idx[:, 1:]  # drop self
        purity = np.mean([
            np.mean(labels_arr[neighbours[i]] == labels_arr[i]) for i in range(len(labels))
        ])
        metrics["knn_topic_purity"] = round(float(purity), 4)
        metrics["knn_k"] = k
    return metrics


def run_semantic_analysis(
    df: pd.DataFrame,
    embeddings_by_method: dict[str, np.ndarray],
    cfg: ConfigNode,
) -> dict:
    """For every method: compute metrics, build PCA/UMAP projections, save figures."""
    an = cfg.analysis
    seed = cfg.project.get("seed", 42)
    figures_dir = resolve_path(an.get("figures_dir", "results/figures"))
    reports_dir = resolve_path(an.get("reports_dir", "results/reports"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    labels = df["topic"].astype(str).tolist() if "topic" in df.columns else ["all"] * len(df)
    reducers = list(an.get("reducers", ["pca", "umap"]))
    summary: dict[str, dict] = {}

    for method, emb in embeddings_by_method.items():
        logger.info("=== Semantic analysis: '%s' (dim=%d) ===", method, emb.shape[1])
        metrics = topic_separability(emb, labels)
        figures: dict[str, str] = {}

        for r in reducers:
            params = _reducer_params(an, r, seed)
            try:
                coords = reduce(emb, r, **params)
            except Exception as exc:  # noqa: BLE001
                logger.error("Reducer '%s' failed for '%s': %s", r, method, exc)
                continue
            fig_path = figures_dir / f"{method}_{r}.png"
            scatter_2d(coords, labels, f"{method.upper()} - {r.upper()} projection", fig_path)
            figures[r] = str(fig_path)

        summary[method] = {"metrics": metrics, "figures": figures}

    _save_summary(summary, reports_dir)
    return summary


def _reducer_params(an: ConfigNode, reducer: str, seed: int) -> dict:
    if reducer == "pca":
        return {"n_components": an.pca.get("n_components", 2), "seed": seed}
    if reducer == "umap":
        u = an.umap
        return {
            "n_components": u.get("n_components", 2),
            "n_neighbors": u.get("n_neighbors", 15),
            "min_dist": u.get("min_dist", 0.1),
            "metric": u.get("metric", "cosine"),
            "seed": seed,
        }
    return {"seed": seed}


def _save_summary(summary: dict, reports_dir: Path) -> None:
    with open(reports_dir / "semantic_analysis.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    lines = [
        "# Semantic-space analysis",
        "",
        "Descriptive metrics computed against known topic labels "
        "(higher silhouette / purity = topics are better separated in the embedding space).",
        "",
        "| Method | Silhouette (cosine) | kNN topic purity | #topics |",
        "|--------|---------------------|------------------|---------|",
    ]
    # Sort by silhouette desc when available.
    def _sil(item):
        v = item[1]["metrics"].get("silhouette_cosine")
        return v if isinstance(v, (int, float)) else -1.0

    for method, data in sorted(summary.items(), key=_sil, reverse=True):
        m = data["metrics"]
        sil = m.get("silhouette_cosine", "-")
        pur = m.get("knn_topic_purity", "-")
        lines.append(f"| {method} | {sil} | {pur} | {m.get('n_topics', '-')} |")

    report_path = reports_dir / "semantic_analysis.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved semantic-analysis report -> %s", report_path)
