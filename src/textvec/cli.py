"""Command-line interface tying the stages together.

Experiment variants (abstract / full_text / abstract_subset) keep their own
clean corpus and output folders, so results never overwrite each other.

Every invocation is automatically tracked by :mod:`textvec.run_registry`:
a versioned snapshot is created under ``results/runs/<RUN_ID>/`` containing
the run manifest, copies of figures and reports, and references to embeddings.

Examples
--------
    python -m textvec collect
    python -m textvec stats
    python -m textvec fulltext                          # build the full-text subset
    python -m textvec preprocess --variant abstract
    python -m textvec vectorize  --variant full_text --methods sbert bge-m3
    python -m textvec analyze    --variant full_text
    python -m textvec all        --variant abstract --methods tfidf doc2vec sbert
    python -m textvec runs                              # list all recorded runs
"""
from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .analysis.semantic import run_semantic_analysis
from .analysis.stats import corpus_statistics, print_statistics, save_statistics
from .analysis.visualize import plot_corpus_lengths
from .config import load_config, resolve_path
from .preprocessing import preprocess_corpus
from .run_registry import RunRegistry, list_runs
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
    registry: RunRegistry | None = getattr(args, "_registry", None)
    if registry is not None:
        registry.record_data("n_corpus_docs", len(df))
    return df


def cmd_stats(cfg, args) -> dict:
    df = _load_csv(cfg.corpus.corpus_csv)
    sample_lang = int(cfg.analysis.get("language_sample", 200))
    summary = corpus_statistics(df, sample_lang=sample_lang)
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
    result = preprocess_corpus(df, cfg, text_fields=vp.text_fields, output_csv=vp.clean_csv)
    registry: RunRegistry | None = getattr(args, "_registry", None)
    if registry is not None:
        registry.record_data(f"n_clean_{vp.name}", len(result))
    return result


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
    registry: RunRegistry | None = getattr(args, "_registry", None)

    stages = [
        ("collect", cmd_collect),
        ("stats", cmd_stats),
        ("preprocess", cmd_preprocess),
        ("vectorize", cmd_vectorize),
        ("analyze", cmd_analyze),
    ]
    for stage_name, stage_fn in stages:
        t0 = time.monotonic()
        try:
            stage_fn(cfg, args)
            elapsed = time.monotonic() - t0
            if registry is not None:
                registry.record_stage(stage_name, "completed", elapsed_sec=elapsed)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            if registry is not None:
                registry.record_stage(
                    stage_name, "failed", elapsed_sec=elapsed,
                    extra={"error": str(exc)[:300]},
                )
            raise

    logger.info("Full pipeline finished for variant '%s'.", _resolve_variant(cfg, args))


def cmd_runs(_cfg, _args) -> None:
    """Print a table of all recorded pipeline runs."""
    runs = list_runs()
    if not runs:
        print("No runs recorded yet. Run the pipeline first.")
        return

    col_w = {"run_id": 17, "label": 12, "status": 10, "variant": 18, "started_at": 25, "data": 30}
    header = (
        f"{'RUN ID':<{col_w['run_id']}}  "
        f"{'LABEL':<{col_w['label']}}  "
        f"{'STATUS':<{col_w['status']}}  "
        f"{'VARIANT':<{col_w['variant']}}  "
        f"{'STARTED':<{col_w['started_at']}}  "
        f"DATA VOLUMES"
    )
    print(header)
    print("-" * len(header))
    for r in runs:
        data = r.get("data", {})
        data_str = "  ".join(f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}"
                             for k, v in data.items()) or "—"
        print(
            f"{r['run_id']:<{col_w['run_id']}}  "
            f"{str(r.get('label') or ''):<{col_w['label']}}  "
            f"{r.get('status', '?'):<{col_w['status']}}  "
            f"{str(r.get('variant', '?')):<{col_w['variant']}}  "
            f"{str(r.get('started_at', '?'))[:24]:<{col_w['started_at']}}  "
            f"{data_str}"
        )

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
        "runs": cmd_runs,
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

    # ``runs`` is a read-only listing command – skip run tracking for it.
    if args.command == "runs":
        args.func(cfg, args)
        return

    registry = RunRegistry()
    variant = getattr(args, "variant", None) or cfg.experiment.get("variant", "abstract")    registry.set_config(args.config or "config/default.yaml", variant=variant)
    args._registry = registry

    t0 = time.monotonic()
    run_status = "completed"
    try:
        args.func(cfg, args)
        # For single-stage commands (not "all"), record the stage timing here.
        if args.command != "all":
            registry.record_stage(
                args.command, "completed", elapsed_sec=time.monotonic() - t0
            )
    except Exception as exc:
        run_status = "failed"
        if args.command != "all":
            registry.record_stage(
                args.command, "failed",
                elapsed_sec=time.monotonic() - t0,
                extra={"error": str(exc)[:300]},
            )
        raise
    finally:
        _finalize_registry(registry, cfg, variant, run_status)


def _finalize_registry(
    registry: RunRegistry, cfg, variant: str, status: str
) -> None:
    """Collect data volumes from filesystem, snapshot artifacts, finalize."""
    try:
        # Data volumes — read from CSV headers (cheap, no full load).
        corpus_csv = resolve_path(cfg.corpus.corpus_csv)
        if corpus_csv.exists():
            with corpus_csv.open(encoding="utf-8") as fh:
                n = sum(1 for _ in fh) - 1
            registry.record_data("n_corpus_docs", max(0, n))

        clean_csv = resolve_path(f"data/processed/clean_{variant}.csv")
        if clean_csv.exists():
            with clean_csv.open(encoding="utf-8") as fh:
                n = sum(1 for _ in fh) - 1
            registry.record_data(f"n_clean_{variant}", max(0, n))

        # Snapshot per-variant artifacts.
        try:
            vp = _variant_paths(cfg, variant)
            fig_dir = resolve_path(vp.fig_dir)
            rep_dir = resolve_path(vp.rep_dir)
            emb_dir = resolve_path(vp.emb_dir)
            registry.snapshot_artifacts(
                figures_dir=fig_dir if fig_dir.exists() else None,
                reports_dir=rep_dir if rep_dir.exists() else None,
                emb_dir=emb_dir if emb_dir.exists() else None,
            )
        except Exception as exc:
            logger.warning("Artifact snapshot skipped: %s", exc)

        # Also snapshot root-level stats report if present.
        root_rep = resolve_path(cfg.analysis.get("reports_dir", "results/reports"))
        stats_json = root_rep / "corpus_stats.json"
        if stats_json.exists():
            import json as _json
            try:
                s = _json.loads(stats_json.read_text(encoding="utf-8"))
                if "n_documents" in s:
                    registry.record_data("n_documents_stats", s["n_documents"])
            except Exception:
                pass

    except Exception as exc:
        logger.warning("Registry finalization error: %s", exc)
    finally:
        registry.finalize(status)


if __name__ == "__main__":
    main()

