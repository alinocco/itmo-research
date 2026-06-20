"""Command-line interface tying the four stages together.

Examples
--------
    python -m textvec collect
    python -m textvec stats
    python -m textvec preprocess
    python -m textvec vectorize --methods tfidf sbert
    python -m textvec analyze
    python -m textvec all --methods tfidf doc2vec sbert
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis.semantic import run_semantic_analysis
from .analysis.stats import corpus_statistics, print_statistics, save_statistics
from .analysis.visualize import plot_corpus_lengths
from .config import load_config, resolve_path
from .preprocessing import preprocess_corpus
from .utils import get_logger, set_seed
from .vectorization.run import run_vectorization

logger = get_logger("textvec.cli")


def _load_corpus(cfg) -> pd.DataFrame:
    path = resolve_path(cfg.corpus.corpus_csv)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found at {path}. Run 'collect' first.")
    return pd.read_csv(path).fillna("")


def _load_clean_corpus(cfg) -> pd.DataFrame:
    path = resolve_path(cfg.preprocessing.output_csv)
    if not path.exists():
        raise FileNotFoundError(f"Clean corpus not found at {path}. Run 'preprocess' first.")
    return pd.read_csv(path).fillna("")


def cmd_collect(cfg, args) -> pd.DataFrame:
    from .corpus.builder import build_corpus

    df = build_corpus(cfg)
    logger.info("Collected %d documents.", len(df))
    return df


def cmd_stats(cfg, args) -> dict:
    df = _load_corpus(cfg)
    summary = corpus_statistics(df)
    print_statistics(summary)
    reports_dir = resolve_path(cfg.analysis.get("reports_dir", "results/reports"))
    save_statistics(summary, reports_dir / "corpus_stats.json")
    text = (df["title"].fillna("") + ". " + df["abstract"].fillna(""))
    plot_corpus_lengths(
        text.str.split().map(len),
        resolve_path(cfg.analysis.get("figures_dir", "results/figures")) / "corpus_lengths.png",
    )
    return summary


def cmd_preprocess(cfg, args) -> pd.DataFrame:
    df = _load_corpus(cfg)
    return preprocess_corpus(df, cfg)


def cmd_vectorize(cfg, args):
    df = _load_clean_corpus(cfg)
    methods = args.methods or list(cfg.vectorization.get("methods", []))
    return run_vectorization(df, cfg, methods=methods)


def cmd_analyze(cfg, args) -> dict:
    df = _load_clean_corpus(cfg)
    emb_dir = resolve_path(cfg.vectorization.get("output_dir", "results/embeddings"))
    methods = args.methods or [p.stem for p in sorted(emb_dir.glob("*.npy"))]
    embeddings = {}
    for m in methods:
        npy = emb_dir / f"{m}.npy"
        if npy.exists():
            embeddings[m] = np.load(npy)
        else:
            logger.warning("No embeddings for '%s' at %s", m, npy)
    if not embeddings:
        raise FileNotFoundError("No embeddings found. Run 'vectorize' first.")
    return run_semantic_analysis(df, embeddings, cfg)


def cmd_all(cfg, args):
    cmd_collect(cfg, args)
    cmd_stats(cfg, args)
    cmd_preprocess(cfg, args)
    cmd_vectorize(cfg, args)
    cmd_analyze(cfg, args)
    logger.info("Full pipeline finished.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="textvec", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, func in [
        ("collect", cmd_collect),
        ("stats", cmd_stats),
        ("preprocess", cmd_preprocess),
        ("vectorize", cmd_vectorize),
        ("analyze", cmd_analyze),
        ("all", cmd_all),
    ]:
        p = sub.add_parser(name, help=func.__doc__)
        if name in {"vectorize", "analyze", "all"}:
            p.add_argument("--methods", nargs="*", default=None,
                           help="Subset of vectorization methods to run.")
        p.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg.project.get("seed", 42))
    args.func(cfg, args)


if __name__ == "__main__":
    main()
