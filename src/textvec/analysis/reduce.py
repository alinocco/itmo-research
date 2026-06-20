"""Dimensionality reduction for semantic-space visualization (task.md 5.3)."""
from __future__ import annotations

import numpy as np

from ..utils import get_logger

logger = get_logger("textvec.analysis.reduce")


def reduce_pca(embeddings: np.ndarray, n_components: int = 2, seed: int = 42) -> np.ndarray:
    from sklearn.decomposition import PCA

    n_components = min(n_components, *embeddings.shape)
    return PCA(n_components=n_components, random_state=seed).fit_transform(embeddings)


def reduce_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    seed: int = 42,
) -> np.ndarray:
    import umap

    # UMAP requires n_neighbors < n_samples.
    n_neighbors = min(n_neighbors, max(2, embeddings.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def reduce(embeddings: np.ndarray, method: str, **params) -> np.ndarray:
    method = method.lower()
    if method == "pca":
        return reduce_pca(embeddings, **params)
    if method == "umap":
        return reduce_umap(embeddings, **params)
    raise ValueError(f"Unknown reducer: {method}")
