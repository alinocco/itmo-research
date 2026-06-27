#!/usr/bin/env bash
# Full pipeline for the 100k multilingual corpus (abstract variant).
#
# Usage:
#   ./scripts/run_100k_multilingual.sh setup     # install deps + spaCy models (once)
#   ./scripts/run_100k_multilingual.sh all       # collect → stats → preprocess → vectorize → analyze
#   ./scripts/run_100k_multilingual.sh collect   # single stage
#   ./scripts/run_100k_multilingual.sh vectorize # embedding stages only (batched for GPU)
#   ./scripts/run_100k_multilingual.sh status    # show corpus / embedding progress
#
# Requires: config/secrets.local.yaml with NCBI email + api_key (see secrets.local.yaml.example)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-config/corpus_100k_multilingual.yaml}"
VARIANT="${VARIANT:-abstract}"
VENV="${VENV:-$ROOT/.venv}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
export TEXTVEC_RUN_ID="${TEXTVEC_RUN_ID:-$TIMESTAMP}"
export TEXTVEC_RUN_LABEL="${TEXTVEC_RUN_LABEL:-}"
LOG_DIR="${LOG_DIR:-$ROOT/results/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/run_${TEXTVEC_RUN_ID}.log}"

SPACY_MODELS=(
  en_core_web_sm
  ru_core_news_sm
  de_core_news_sm
  fr_core_news_sm
  es_core_news_sm
  pt_core_news_sm
)

# Transformer batches — one model at a time to stay within 12 GB VRAM.
TRANSFORMER_BATCHES=(
  "sbert e5 gte"
  "bge-m3"
  "bert"
)
CLASSICAL_METHODS="tfidf fasttext doc2vec"

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg" | tee -a "$LOG_FILE"
}

die() {
  log "ERROR: $*"
  exit 1
}

activate_venv() {
  if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$VENV/bin/activate"
  else
    die "Virtualenv not found at $VENV. Run: python3 -m venv .venv && ./scripts/run_100k_multilingual.sh setup"
  fi
}

run_textvec() {
  log ">>> python -m textvec --config $CONFIG $*"
  python -m textvec --config "$CONFIG" "$@" 2>&1 | tee -a "$LOG_FILE"
}

check_secrets() {
  if [[ ! -f "$ROOT/config/secrets.local.yaml" ]]; then
    die "Missing config/secrets.local.yaml — copy secrets.local.yaml.example and fill in NCBI email + api_key"
  fi
  python - <<'PY' || die "NCBI api_key not loaded — check config/secrets.local.yaml"
from textvec.config import load_config
cfg = load_config("config/corpus_100k_multilingual.yaml")
key = cfg.corpus.sources.pubmed.get("api_key")
email = cfg.corpus.sources.pubmed.get("email")
assert key, "api_key is empty"
assert email and "example.com" not in email, "set a real email in secrets.local.yaml"
print(f"OK: email={email}, api_key={str(key)[:8]}...")
PY
}

cmd_setup() {
  mkdir -p "$LOG_DIR"
  log "=== SETUP ==="

  if [[ ! -f "$VENV/bin/activate" ]]; then
    log "Creating virtualenv at $VENV"
    python3 -m venv "$VENV"
  fi
  activate_venv

  log "Installing Python dependencies"
  pip install -U pip wheel
  pip install -r requirements.txt
  pip install -e .

  # CUDA PyTorch — skip if already installed with CUDA support.
  if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    log "PyTorch with CUDA already available"
  else
    log "Installing PyTorch (CUDA 12.4). Adjust index URL if your driver needs another build."
    pip install torch --index-url https://download.pytorch.org/whl/cu124
  fi

  python -c "import nltk; nltk.download('stopwords', quiet=True)"

  for model in "${SPACY_MODELS[@]}"; do
    if python -c "import spacy; spacy.load('$model')" 2>/dev/null; then
      log "spaCy model '$model' already installed"
    else
      log "Downloading spaCy model '$model'"
      python -m spacy download "$model"
    fi
  done

  if [[ ! -f "$ROOT/config/secrets.local.yaml" ]]; then
    log "WARNING: config/secrets.local.yaml not found — copy secrets.local.yaml.example"
  fi

  log "=== SETUP DONE ==="
}

cmd_collect() {
  mkdir -p "$LOG_DIR"
  activate_venv
  check_secrets
  log "=== COLLECT (may take several hours) ==="
  run_textvec collect
  log "=== COLLECT DONE ==="
}

cmd_stats() {
  mkdir -p "$LOG_DIR"
  activate_venv
  log "=== STATS ==="
  run_textvec stats
  log "=== STATS DONE ==="
}

cmd_preprocess() {
  mkdir -p "$LOG_DIR"
  activate_venv
  log "=== PREPROCESS (variant=$VARIANT) ==="
  run_textvec preprocess --variant "$VARIANT"
  log "=== PREPROCESS DONE ==="
}

cmd_vectorize() {
  mkdir -p "$LOG_DIR"
  activate_venv
  log "=== VECTORIZE (variant=$VARIANT, batched for GPU) ==="

  for methods in "${TRANSFORMER_BATCHES[@]}"; do
    log "--- transformers: $methods ---"
    # shellcheck disable=SC2086
    run_textvec vectorize --variant "$VARIANT" --methods $methods
  done

  log "--- classical: $CLASSICAL_METHODS ---"
  # shellcheck disable=SC2086
  run_textvec vectorize --variant "$VARIANT" --methods $CLASSICAL_METHODS

  log "=== VECTORIZE DONE ==="
}

cmd_analyze() {
  mkdir -p "$LOG_DIR"
  activate_venv
  log "=== ANALYZE (variant=$VARIANT) ==="
  run_textvec analyze --variant "$VARIANT"
  log "=== ANALYZE DONE ==="
}

cmd_all() {
  mkdir -p "$LOG_DIR"
  log "=== FULL PIPELINE START (config=$CONFIG, variant=$VARIANT) ==="
  log "Run ID    : $TEXTVEC_RUN_ID"
  log "Log file  : $LOG_FILE"
  cmd_collect
  cmd_stats
  cmd_preprocess
  cmd_vectorize
  cmd_analyze
  log "=== FULL PIPELINE COMPLETE ==="
  log "Figures   : results/figures/$VARIANT/"
  log "Reports   : results/reports/$VARIANT/"
  log "Embeddings: results/embeddings/$VARIANT/"
  _copy_log_to_run_dir
  log "Run dir   : results/runs/$TEXTVEC_RUN_ID/"
}

_copy_log_to_run_dir() {
  local run_dir="$ROOT/results/runs/$TEXTVEC_RUN_ID/logs"
  if [[ -f "$LOG_FILE" ]]; then
    mkdir -p "$run_dir"
    cp "$LOG_FILE" "$run_dir/$(basename "$LOG_FILE")"
    log "Log copied : results/runs/$TEXTVEC_RUN_ID/logs/"
  fi
}

cmd_runs() {
  activate_venv
  python -m textvec --config "$CONFIG" runs
}

cmd_status() {
  activate_venv
  python - <<PY
from pathlib import Path
import json

root = Path("$ROOT")
cfg_path = root / "$CONFIG"
corpus_csv = root / "data/processed/corpus_100k.csv"
clean_csv = root / "data/processed/clean_${VARIANT}.csv"
emb_dir = root / "results/embeddings/${VARIANT}"
stats_json = root / "results/reports/corpus_stats.json"

def count_lines(p):
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return sum(1 for _ in f) - 1  # minus header

def fmt_docs(p):
    n = count_lines(p)
    if n is None:
        return "not built"
    return f"{n} docs"

print("Config     :", cfg_path)
print("Corpus CSV :", corpus_csv, "→", fmt_docs(corpus_csv))
print("Clean CSV  :", clean_csv, "→", fmt_docs(clean_csv))

if stats_json.exists():
    s = json.loads(stats_json.read_text(encoding="utf-8"))
    print("Stats      :", s.get("n_documents"), "docs")
    if "by_language" in s:
        print("Languages  :", s["by_language"])

if emb_dir.exists():
    methods = sorted(p.stem for p in emb_dir.glob("*.npy"))
    print("Embeddings :", ", ".join(methods) if methods else "none")
    manifest = emb_dir / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        for name, info in sorted(m.items()):
            status = info.get("status", "?")
            extra = f"dim={info.get('dim')}, {info.get('elapsed_sec')}s" if status == "ok" else info.get("error", "")
            print(f"  - {name}: {status} {extra}")
else:
    print("Embeddings : not built")
PY
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  setup       Create venv, install deps, download spaCy models
  collect     Fetch ~100k articles from ArXiv + PubMed
  stats       Corpus statistics + length histogram
  preprocess  Clean / lemmatize (multilingual)
  vectorize   All embedding methods (GPU-batched)
  analyze     PCA/UMAP projections + metrics
  all         Run everything in order
  runs        List all recorded pipeline runs
  status      Show current progress

Environment overrides:
  CONFIG=config/corpus_100k_multilingual.yaml
  VARIANT=abstract
  VENV=$ROOT/.venv
  TEXTVEC_RUN_ID=YYYYMMDD_HHMMSS   (auto-generated if not set)
  LOG_FILE=path/to/log.log

Examples:
  ./scripts/run_100k_multilingual.sh setup
  ./scripts/run_100k_multilingual.sh all
  ./scripts/run_100k_multilingual.sh vectorize   # resume embeddings only
  ./scripts/run_100k_multilingual.sh runs        # show run history
  ./scripts/run_100k_multilingual.sh status
EOF
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    setup)      cmd_setup ;;
    collect)    cmd_collect ;;
    stats)      cmd_stats ;;
    preprocess) cmd_preprocess ;;
    vectorize)  cmd_vectorize ;;
    analyze)    cmd_analyze ;;
    all)        cmd_all ;;
    runs)       cmd_runs ;;
    status)     cmd_status ;;
    -h|--help|help|"") usage ;;
    *) die "Unknown command: $cmd. Run with --help." ;;
  esac
}

main "$@"
