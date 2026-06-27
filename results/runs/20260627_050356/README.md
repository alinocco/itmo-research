# Run `20260627_050356`

> **Platform:** Windows (Git Bash) — `./scripts/run_100k_multilingual_windows.sh all`

## Overview

| Field | Value |
|---|---|
| **Status** | completed_partial |
| **Label** | — |
| **Started** | 2026-06-27 03:07:08 |
| **Finished** | 2026-06-27 03:46:32 |
| **Total elapsed** | ~39m 24s |
| **Config** | `config/corpus_100k_multilingual.yaml` |
| **Variant** | `abstract` |

## Data volumes

| Metric | Value |
|---|---|
| n_corpus_docs | 45,516 |
| n_corpus_arxiv | 22,312 |
| n_corpus_pubmed | 23,204 |
| n_duplicates_dropped | 22,776 |
| n_clean_abstract | 45,516 |
| n_methods_ok | 3 |
| n_methods_failed | 5 |

## Topics

| Topic | Docs |
|---|---|
| oncology | 9,607 |
| neurology | 4,999 |
| cardiology | 3,873 |
| endocrinology | 2,755 |
| machine_learning | 2,000 |
| astrophysics | 2,000 |
| quantum_computing | 1,999 |
| information_theory | 1,871 |
| cryptography_security | 1,851 |
| software_engineering | 1,781 |
| computation_language | 1,776 |
| statistics | 1,775 |
| robotics | 1,741 |
| computer_vision | 1,717 |

## Stages

| Stage | Status | Elapsed | Note |
|---|---|---|---|
| collect | ✅ completed | ~20m 45s | 45 516 docs, 22 776 dropped |
| stats | ✅ completed | ~14s | |
| preprocess | ✅ completed | ~1m 30s | 45 516 docs kept |
| vectorize / tfidf | ✅ completed | 41.9s | dim=300 |
| vectorize / fasttext | ✅ completed | 351.5s | dim=200 |
| vectorize / doc2vec | ✅ completed | 122.9s | dim=200 |
| vectorize / sbert | ❌ failed | — | Torch not compiled with CUDA |
| vectorize / e5 | ❌ failed | — | Torch not compiled with CUDA |
| vectorize / gte | ❌ failed | — | trust_remote_code=True required |
| vectorize / bge-m3 | ❌ failed | — | Torch not compiled with CUDA |
| vectorize / bert | ❌ failed | — | Torch not compiled with CUDA |
| analyze | ✅ completed | ~3m 43s | doc2vec, fasttext, tfidf — 12 figures |

## Issues

### ❌ Трансформеры не запустились (CUDA)
`sbert`, `e5`, `bge-m3`, `bert` упали с ошибкой **"Torch not compiled with CUDA enabled"**.

**Решение:** переустановить PyTorch с поддержкой CUDA:
```bash
./scripts/run_100k_multilingual_windows.sh setup-gpu
```

### ❌ GTE требует trust_remote_code
`Alibaba-NLP/gte-multilingual-base` требует `trust_remote_code=True`.

**Решение:** добавить параметр при загрузке модели.

### ⚠️ Язык PubMed-документов определился как `['`
23 204 документа получили языковой тег `['` вместо корректного кода.  
Это баг парсинга метаданных языка из PubMed.

## Artifacts

- **Figures:** 63 PNG (abstract / abstract_subset / full_text × методы × pca/umap/language)  →  `figures/`
- **Reports:** 10 файлов (corpus_stats, semantic_analysis × 3 варианта)  →  `reports/`
- **Embeddings:** 5 .npy файлов в `artefacts/` (doc2vec, e5, fasttext, sbert, tfidf)  →  см. `manifest.json`
- **Logs:** 14 файлов  →  `logs/`

> **Примечание:** `.npy` для `e5` и `sbert` присутствуют — они были получены в последующих прогонах (см. `logs/run_20260627_044022.log`, `run_20260627_044711.log`).
> Методы `bert`, `bge-m3`, `gte` в `.npy` отсутствуют.
