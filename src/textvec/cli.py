"""Command-line interface tying the stages together.

Experiment variants (abstract / full_text / abstract_subset) keep their own
clean corpus and output folders, so results never overwrite each other.

Examples
--------
    python -m textvec collect
    python -m textvec stats
    python -m textvec fulltext                          # build the full-text subset
    python -m textvec preprocess --variant abstract
    python -m textvec vectorize  --variant full_text --methods sbert bge-m3
    python -m textvec analyze    --variant full_text
    python -m textvec all        --variant abstract --methods tfidf doc2vec sbert
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

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


def _variant_paths(cfg, variant: str) -> SimpleNamespace:
    """Resolve per-variant source CSV, text fields and isolated output folders."""
    variants = cfg.experiment.variants
    if variant not in variants:
        available = list(variants.to_dict().keys())
        raise ValueError(f"Unknown variant '{variant}'. Available: {available}")
    v = variants[variant]
    return SimpleNamespace(
        name=variant,
        source_csv=v.source_csv,
        text_fields=list(v.text_fields),
        clean_csv=f"data/processed/clean_{variant}.csv",
        emb_dir=f"results/embeddings/{variant}",
        fig_dir=f"results/figures/{variant}",
        rep_dir=f"results/reports/{variant}",
    )


def _resolve_variant(cfg, args) -> str:
    return getattr(args, "variant", None) or cfg.experiment.get("variant", "abstract")


def _load_csv(path) -> pd.DataFrame:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    return pd.read_csv(p).fillna("")


def cmd_collect(cfg, args) -> pd.DataFrame:
    from .corpus.builder import build_corpus

    df = build_corpus(cfg)
    logger.info("Collected %d documents.", len(df))
    return df


def cmd_stats(cfg, args) -> dict:
    df = _load_csv(cfg.corpus.corpus_csv)
    summary = corpus_statistics(df)
    print_statistics(summary)
    reports_dir = resolve_path(cfg.analysis.get("reports_dir", "results/reports"))
    save_statistics(summary, reports_dir / "corpus_stats.json")
    text = df["title"].fillna("") + ". " + df["abstract"].fillna("")
    plot_corpus_lengths(
        text.str.split().map(len),
        resolve_path(cfg.analysis.get("figures_dir", "results/figures")) / "corpus_lengths.png",
    )
    return summary


def cmd_fulltext(cfg, args) -> pd.DataFrame:
    from .corpus.fulltext import augment_with_fulltext

    df = _load_csv(cfg.corpus.corpus_csv)
    ft = cfg.experiment.get("fulltext", None)
    per_topic = args.per_topic or (ft.get("per_topic", 150) if ft else 150)
    email = cfg.corpus.sources.pubmed.get("email", "research@example.com")
    api_key = cfg.corpus.sources.pubmed.get("api_key")

    subset = augment_with_fulltext(df, per_topic=per_topic, email=email, api_key=api_key)
    out = resolve_path(cfg.experiment.variants["full_text"].source_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out, index=False, encoding="utf-8")
    logger.info("Saved full-text subset (%d docs) -> %s", len(subset), out)
    return subset


def cmd_preprocess(cfg, args) -> pd.DataFrame:
    vp = _variant_paths(cfg, _resolve_variant(cfg, args))
    df = _load_csv(vp.source_csv)
    return preprocess_corpus(df, cfg, text_fields=vp.text_fields, output_csv=vp.clean_csv)


def cmd_vectorize(cfg, args):
    vp = _variant_paths(cfg, _resolve_variant(cfg, args))
    df = _load_csv(vp.clean_csv)
    methods = args.methods or list(cfg.vectorization.get("methods", []))
    return run_vectorization(df, cfg, methods=methods, output_dir=vp.emb_dir)


def cmd_analyze(cfg, args) -> dict:
    vp = _variant_paths(cfg, _resolve_variant(cfg, args))
    df = _load_csv(vp.clean_csv)
    emb_dir = resolve_path(vp.emb_dir)
    methods = args.methods or [p.stem for p in sorted(emb_dir.glob("*.npy"))]
    embeddings = {}
    for m in methods:
        npy = emb_dir / f"{m}.npy"
        if npy.exists():
            embeddings[m] = np.load(npy)
        else:
            logger.warning("No embeddings for '%s' at %s", m, npy)
    if not embeddings:
        raise FileNotFoundError(f"No embeddings in {emb_dir}. Run 'vectorize' first.")
    return run_semantic_analysis(df, embeddings, cfg, figures_dir=vp.fig_dir, reports_dir=vp.rep_dir)


def cmd_all(cfg, args):
    cmd_collect(cfg, args)
    cmd_stats(cfg, args)
    cmd_preprocess(cfg, args)
    cmd_vectorize(cfg, args)
    cmd_analyze(cfg, args)
    logger.info("Full pipeline finished for variant '%s'.", _resolve_variant(cfg, args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="textvec", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    sub = parser.add_subparsers(dest="command", required=True)

    specs = {
        "collect": cmd_collect,
        "stats": cmd_stats,
        "fulltext": cmd_fulltext,
        "preprocess": cmd_preprocess,
        "vectorize": cmd_vectorize,
        "analyze": cmd_analyze,
        "all": cmd_all,
    }
    for name, func in specs.items():
        p = sub.add_parser(name, help=(func.__doc__ or "").strip().splitlines()[0] if func.__doc__ else name)
        if name in {"preprocess", "vectorize", "analyze", "all"}:
            p.add_argument("--variant", type=str, default=None,
                           help="Experiment variant (abstract | full_text | abstract_subset).")
        if name in {"vectorize", "analyze", "all"}:
            p.add_argument("--methods", nargs="*", default=None,
                           help="Subset of vectorization methods to run.")
        if name == "fulltext":
            p.add_argument("--per-topic", type=int, default=None, dest="per_topic",
                           help="Documents per topic to fetch full text for.")
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
