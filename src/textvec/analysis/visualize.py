"""2D visualizations of the semantic space (task.md 5.4)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils import get_logger

logger = get_logger("textvec.analysis.visualize")


def scatter_2d(
    coords: np.ndarray,
    labels: list[str],
    title: str,
    out_path: str | Path,
    *,
    figsize: tuple[int, int] = (9, 7),
    legend_title: str = "topic",
) -> Path:
    """Save a 2D scatter plot colored by ``labels`` (e.g. topic)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(labels)
    unique = sorted(set(labels))
    cmap = plt.get_cmap("tab10" if len(unique) <= 10 else "tab20")
    color_of = {lab: cmap(i % cmap.N) for i, lab in enumerate(unique)}

    fig, ax = plt.subplots(figsize=figsize)
    for lab in unique:
        idx = [i for i, l in enumerate(labels) if l == lab]
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            s=18, alpha=0.7, color=color_of[lab], label=str(lab),
        )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(title=legend_title, fontsize=8, markerscale=1.4, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure -> %s", out_path)
    return out_path


def plot_corpus_lengths(word_counts, out_path: str | Path) -> Path:
    """Histogram of document lengths for the primary corpus analysis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(word_counts, bins=40, color="#4C72B0", alpha=0.85)
    ax.set_title("Document length distribution (words)")
    ax.set_xlabel("words per document")
    ax.set_ylabel("number of documents")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure -> %s", out_path)
    return out_path
