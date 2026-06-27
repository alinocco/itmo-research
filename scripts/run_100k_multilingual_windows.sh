#!/usr/bin/env bash
# Full pipeline for the 100k multilingual corpus (abstract variant).
#
# Linux / macOS:
#   ./scripts/run_100k_multilingual.sh setup
#   ./scripts/run_100k_multilingual.sh all
#
# Windows Git Bash (venv: .venv/Scripts/activate):
#   ./scripts/run_100k_multilingual.sh setup
#   ./scripts/run_100k_multilingual.sh all
#
# Requires: config/secrets.local.yaml with NCBI email + api_key

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

TRANSFORMER_BATCHES=(
  "sbert e5 gte"
  "bge-m3"
  "bert"
)
CLASSICAL_METHODS="tfidf fasttext doc2vec"

# --- OS detection (Git Bash / MSYS / Cygwin on Windows) -----------------------
is_windows() {
  case "$(uname -s 2>/dev/null)" in
    MINGW*|MSYS*|CYGWIN*|Windows_NT) return 0 ;;
    *) return 1 ;;
  esac
}

# Path to ``source .../activate`` (Unix: bin/activate, Windows: Scripts/activate).
venv_activate() {
  if [[ -f "$VENV/Scripts/activate" ]]; then
    echo "$VENV/Scripts/activate"
  elif [[ -f "$VENV/bin/activate" ]]; then
    echo "$VENV/bin/activate"
  else
    echo ""
  fi
}

venv_exists() {
  [[ -n "$(venv_activate)" ]]
}

# Python launcher for creating venv / running before activation.
python_bin() {
  if command -v python &>/dev/null; then
    echo python
  elif command -v python3 &>/dev/null; then
    echo python3
  else
    echo ""
  fi
}

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg" | tee -a "$LOG_FILE"
}

die() {
  log "ERROR: $*"
  exit 1
}

activate_venv() {
  local act
  act="$(venv_activate)"
  if [[ -z "$act" ]]; then
    die "Virtualenv not found at $VENV. Run: ./scripts/run_100k_multilingual.sh setup"
  fi
  # shellcheck disable=SC1090
  source "$act"
}

run_textvec() {
  log ">>> python -m textvec --config $CONFIG $*"
  python -m textvec --config "$CONFIG" "$@" 2>&1 | tee -a "$LOG_FILE"
}

check_secrets() {
  if [[ ! -f "$ROOT/config/secrets.local.yaml" ]]; then
    die "Missing config/secrets.local.yaml — copy secrets.local.yaml.example and fill in NCBI email + api_key"
  fi
  CONFIG="$CONFIG" python - <<'PY' || die "NCBI api_key not loaded — check config/secrets.local.yaml"
import os
from textvec.config import load_config
cfg = load_config(os.environ["CONFIG"])
key = cfg.corpus.sources.pubmed.get("api_key")
email = cfg.corpus.sources.pubmed.get("email")
assert key, "api_key is empty"
assert email and "example.com" not in email, "set a real email in secrets.local.yaml"
print(f"OK: email={email}, api_key={str(key)[:8]}...")
PY
}

install_pytorch_cuda() {
  log "Installing PyTorch with CUDA 12.4 (removes CPU-only build from PyPI)..."
  pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
  pip install --force-reinstall torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
  python - <<'PY' || return 1
import torch
print(f"torch {torch.__version__}, cuda build={torch.version.cuda}, available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("ERROR: CUDA still not available. Check: nvidia-smi")
    raise SystemExit(1)
PY
}

cmd_setup_gpu() {
  mkdir -p "$LOG_DIR"
  activate_venv
  log "=== SETUP-GPU ==="
  install_pytorch_cuda
  log "=== SETUP-GPU DONE ==="
}

cmd_gpu() {
  activate_venv
  python - <<'PY'
import torch
print("torch version :", torch.__version__)
print("cuda compiled :", torch.version.cuda or "NO (CPU-only build)")
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu name      :", torch.cuda.get_device_name(0))
    print("gpu memory    :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
else:
    print()
    print("Fix: ./scripts/run_100k_multilingual_windows.sh setup-gpu")
PY
}

cmd_setup() {
  mkdir -p "$LOG_DIR"
  log "=== SETUP ==="
  if is_windows; then
    log "Platform: Windows (Git Bash) — venv uses Scripts/activate"
  else
    log "Platform: Unix — venv uses bin/activate"
  fi

  if ! venv_exists; then
    local py
    py="$(python_bin)"
    [[ -n "$py" ]] || die "Python not found. Install Python 3.10+ and add it to PATH."
    log "Creating virtualenv at $VENV"
    "$py" -m venv "$VENV"
  fi
  activate_venv

  log "Installing Python dependencies"
  pip install -U pip wheel
  pip install -r requirements.txt
  pip install -e .

  install_pytorch_cuda || log "WARNING: CUDA PyTorch install failed — run: ./scripts/run_100k_multilingual_windows.sh setup-gpu"

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
  setup-gpu   Reinstall PyTorch with CUDA (fix "not compiled with CUDA")
  gpu         Check GPU / PyTorch CUDA status
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
  # Git Bash on Windows (from the Project folder):
  cd /c/Users/Aidar/Alina/itmo-research
  ./scripts/run_100k_multilingual_windows.sh setup-gpu
  ./scripts/run_100k_multilingual_windows.sh gpu
  ./scripts/run_100k_multilingual_windows.sh vectorize
  ./scripts/run_100k_multilingual_windows.sh runs
EOF
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    setup)      cmd_setup ;;
    setup-gpu)  cmd_setup_gpu ;;
    gpu)        cmd_gpu ;;
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
