"""Run registry – tracks every pipeline execution as a versioned snapshot.

Each run is stored under ``results/runs/<RUN_ID>/`` and contains:

  * ``manifest.json``       – full metadata (timestamps, config, variant,
                              data volumes, per-stage status + duration)
  * ``logs/``               – copy of the shell-level run log
  * ``reports/``            – copies of JSON / Markdown analysis reports
  * ``figures/``            – copies of all PNG figures produced by this run
  * ``embeddings.ref.json`` – lightweight references to .npy files
                              (files are NOT copied – they are large)
  * ``README.md``           – human-readable one-page run summary

``results/runs/index.json`` is updated with a one-line summary of every run.
``results/runs/latest``     is a symlink to the most recent run directory.

Shell scripts pass ``TEXTVEC_RUN_ID`` via environment so that Python and bash
use the same identifier for the same invocation.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .utils import get_logger

logger = get_logger("textvec.run_registry")

_INDEX_PATH = PROJECT_ROOT / "results" / "runs" / "index.json"


class RunRegistry:
    """Manages a single pipeline run: creates its directory, records metadata,
    snapshots lightweight artifacts, and writes a manifest on completion."""

    def __init__(self, run_id: str | None = None, label: str | None = None):
        import os

        self.run_id = run_id or os.environ.get("TEXTVEC_RUN_ID") or _make_run_id()
        self.run_dir = PROJECT_ROOT / "results" / "runs" / self.run_id

        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "figures").mkdir(parents=True, exist_ok=True)

        self._manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "label": label or os.environ.get("TEXTVEC_RUN_LABEL"),
            "started_at": _now_iso(),
            "finished_at": None,
            "status": "running",
            "config": None,
            "variant": None,
            "stages": {},
            "data": {},
            "artifacts": {"reports": [], "figures": [], "embeddings": []},
        }

        self._write_manifest()
        logger.info("Run started  run_id=%s  label=%s  dir=%s",
                    self.run_id, self._manifest["label"], self.run_dir)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def set_config(self, config_path: str | Path, variant: str | None = None) -> None:
        self._manifest["config"] = str(config_path)
        self._manifest["variant"] = variant
        self._write_manifest()

    def set_label(self, label: str) -> None:
        """Set or update the human-readable label for this run (e.g. '15k', '100k-v2')."""
        self._manifest["label"] = label
        self._write_manifest()

    def record_data(self, label: str, value: Any) -> None:
        """Record a data-volume metric.

        Examples::

            registry.record_data("n_corpus_docs", 98_432)
            registry.record_data("n_clean_abstract", 97_111)
        """
        self._manifest["data"][label] = value
        self._write_manifest()

    def record_stage(
        self,
        stage: str,
        status: str,
        elapsed_sec: float | None = None,
        extra: dict | None = None,
    ) -> None:
        """Record outcome of a pipeline stage."""
        entry: dict[str, Any] = {"status": status}
        if elapsed_sec is not None:
            entry["elapsed_sec"] = round(elapsed_sec, 1)
        if extra:
            entry.update(extra)
        self._manifest["stages"][stage] = entry
        self._write_manifest()

    # ------------------------------------------------------------------
    # Artifact snapshot
    # ------------------------------------------------------------------

    def snapshot_artifacts(
        self,
        figures_dir: Path | None = None,
        reports_dir: Path | None = None,
        emb_dir: Path | None = None,
    ) -> None:
        """Copy figures and reports into the run directory; reference embeddings.

        Safe to call even when directories do not yet exist (e.g. the stage
        that produces them did not run this invocation).
        """
        if figures_dir and figures_dir.exists():
            dst_fig = self.run_dir / "figures"
            copied = []
            for png in sorted(figures_dir.rglob("*.png")):
                dst = dst_fig / png.name
                shutil.copy2(png, dst)
                copied.append(str(png.relative_to(PROJECT_ROOT)))
            self._manifest["artifacts"]["figures"] = copied
            if copied:
                logger.info("Snapshotted %d figure(s) -> %s", len(copied), dst_fig)

        if reports_dir and reports_dir.exists():
            dst_rep = self.run_dir / "reports"
            copied = []
            for f in sorted(reports_dir.rglob("*")):
                if f.is_file() and f.suffix in {".json", ".md", ".txt"}:
                    dst = dst_rep / f.name
                    shutil.copy2(f, dst)
                    copied.append(str(f.relative_to(PROJECT_ROOT)))
            self._manifest["artifacts"]["reports"] = copied
            if copied:
                logger.info("Snapshotted %d report file(s) -> %s", len(copied), dst_rep)

        if emb_dir and emb_dir.exists():
            refs = []
            for npy in sorted(emb_dir.glob("*.npy")):
                size_mb = round(npy.stat().st_size / 1_048_576, 1)
                refs.append(
                    {
                        "method": npy.stem,
                        "path": str(npy.relative_to(PROJECT_ROOT)),
                        "size_mb": size_mb,
                    }
                )
            self._manifest["artifacts"]["embeddings"] = refs
            ref_path = self.run_dir / "embeddings.ref.json"
            ref_path.write_text(
                json.dumps(refs, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if refs:
                logger.info("Recorded refs to %d embedding file(s).", len(refs))

        self._write_manifest()

    def link_log(self, log_path: str | Path) -> None:
        """Copy the shell-level log file into the run directory."""
        p = Path(log_path)
        if p.exists():
            dst = self.run_dir / "logs" / p.name
            shutil.copy2(p, dst)
            logger.info("Run log copied -> %s", dst)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(self, status: str = "completed") -> None:
        self._manifest["finished_at"] = _now_iso()
        self._manifest["status"] = status
        self._write_manifest()
        self._write_readme()
        self._update_index()
        self._update_latest_symlink()
        logger.info(
            "Run finalised  run_id=%s  status=%s  dir=%s",
            self.run_id, status, self.run_dir,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_manifest(self) -> None:
        path = self.run_dir / "manifest.json"
        path.write_text(
            json.dumps(self._manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _write_readme(self) -> None:
        m = self._manifest
        data = m.get("data", {})
        stages = m.get("stages", {})
        arts = m.get("artifacts", {})

        started = m.get("started_at", "—")
        finished = m.get("finished_at", "—")
        elapsed_total = _elapsed_label(started, finished)

        lines = [
            f"# Run `{m['run_id']}`",
            "",
            "## Overview",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| **Status** | {m['status']} |",
            f"| **Label** | {m.get('label') or '—'} |",
            f"| **Started** | {started} |",
            f"| **Finished** | {finished} |",
            f"| **Total elapsed** | {elapsed_total} |",
            f"| **Config** | `{m.get('config', '—')}` |",
            f"| **Variant** | `{m.get('variant', '—')}` |",
            "",
            "## Data volumes",
            "",
        ]

        if data:
            lines += ["| Metric | Value |", "|---|---|"]
            for k, v in data.items():
                fmt = f"{v:,}" if isinstance(v, int) else str(v)
                lines.append(f"| {k} | {fmt} |")
        else:
            lines.append("_(not recorded for this invocation)_")

        lines += ["", "## Stages", ""]
        if stages:
            lines += ["| Stage | Status | Elapsed |", "|---|---|---|"]
            for stage, info in stages.items():
                elapsed = f"{info.get('elapsed_sec', '—')} s"
                lines.append(f"| {stage} | {info.get('status', '?')} | {elapsed} |")
        else:
            lines.append("_(none recorded)_")

        lines += ["", "## Artifacts", ""]
        n_fig = len(arts.get("figures", []))
        n_rep = len(arts.get("reports", []))
        embs = arts.get("embeddings", [])
        n_emb = len(embs)

        lines += [
            f"- **Figures**: {n_fig} PNG file(s)  →  `{self.run_dir.name}/figures/`",
            f"- **Reports**: {n_rep} file(s)  →  `{self.run_dir.name}/reports/`",
            f"- **Embeddings**: {n_emb} .npy file(s) (not copied – referenced in "
            f"`embeddings.ref.json`)",
        ]

        if embs:
            lines += ["", "### Embedding files", ""]
            lines += ["| Method | Size |", "|---|---|"]
            for e in embs:
                lines.append(f"| {e['method']} | {e['size_mb']} MB |")

        readme = self.run_dir / "README.md"
        readme.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _update_index(self) -> None:
        _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        runs: list[dict] = []
        if _INDEX_PATH.exists():
            try:
                runs = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
            except Exception:
                runs = []

        m = self._manifest
        entry = {
            "run_id": m["run_id"],
            "label": m.get("label"),
            "started_at": m.get("started_at"),
            "finished_at": m.get("finished_at"),
            "status": m.get("status"),
            "variant": m.get("variant"),
            "config": m.get("config"),
            "data": m.get("data", {}),
            "stages": {k: v.get("status") for k, v in m.get("stages", {}).items()},
        }
        runs = [r for r in runs if r.get("run_id") != m["run_id"]]
        runs.append(entry)
        runs.sort(key=lambda r: r["run_id"])
        _INDEX_PATH.write_text(
            json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _update_latest_symlink(self) -> None:
        latest = PROJECT_ROOT / "results" / "runs" / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(self.run_id)
        except Exception as exc:
            logger.warning("Could not update 'latest' symlink: %s", exc)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _make_run_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _elapsed_label(started: str, finished: str) -> str:
    try:
        from datetime import timedelta

        t0 = datetime.fromisoformat(started)
        t1 = datetime.fromisoformat(finished)
        secs = int((t1 - t0).total_seconds())
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h {m}m {s}s"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"
    except Exception:
        return "—"


def list_runs() -> list[dict]:
    """Return all recorded runs from the index (newest first)."""
    if not _INDEX_PATH.exists():
        return []
    try:
        runs = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
        return list(reversed(runs))
    except Exception:
        return []
