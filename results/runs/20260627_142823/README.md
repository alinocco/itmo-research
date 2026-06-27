# Run `20260627_142823`

> **Platform:** Windows (Git Bash) + CUDA GPU — `./scripts/run_100k_multilingual_windows.sh all`

## Overview

| Field | Value |
|---|---|
| **Status** | ✅ completed |
| **Label** | — |
| **Started** | 2026-06-27 05:04:53 |
| **Finished** | 2026-06-27 06:15:16 |
| **Total elapsed** | ~1h 10m 23s |
| **Config** | `config/corpus_100k_multilingual.yaml` |
| **Variant** | `abstract` |

## Data volumes

| Metric | Value |
|---|---|
| n_corpus_docs | 45,581 |
| n_corpus_arxiv | 22,312 |
| n_corpus_pubmed | 23,269 |
| n_clean_abstract | 45,581 |
| n_methods_ok | **8 / 8** |
| n_methods_failed | 0 |

## Languages

| Язык | Документов |
|---|---|
| en | 37,726 |
| ru | 1,984 |
| zh | 1,874 |
| ja | 1,529 |
| fr | 1,058 |
| es | 801 |
| de | 497 |
| pt | 112 |

## Topics

| Topic | Docs |
|---|---|
| oncology | 9,607 |
| neurology | 4,999 |
| cardiology | 3,874 |
| endocrinology | 2,819 |
| machine_learning | 2,000 |
| astrophysics | 1,998 |
| quantitative_biology | 1,984 |
| infectious | 1,970 |
| nlp | 1,911 |
| quantum_computing | 1,907 |
| information_theory | 1,871 |
| cryptography_security | 1,851 |
| software_engineering | 1,781 |
| computation_language | 1,776 |
| statistics | 1,775 |
| robotics | 1,741 |
| computer_vision | 1,717 |

## Stages

| Stage | Status | Elapsed | Dim |
|---|---|---|---|
| collect | ✅ completed | ~19m 23s | — |
| stats | ✅ completed | ~14s | — |
| preprocess | ✅ completed | ~1m 30s | — |
| vectorize / tfidf | ✅ completed | 36.0s | 300 |
| vectorize / doc2vec | ✅ completed | 109.2s | 200 |
| vectorize / fasttext | ✅ completed | 301.8s | 200 |
| vectorize / sbert | ✅ completed | 148.7s | 384 |
| vectorize / e5 | ✅ completed | 316.2s | 768 |
| vectorize / gte | ✅ completed | 190.8s | 768 |
| vectorize / bert | ✅ completed | 320.2s | 768 |
| vectorize / bge-m3 | ✅ completed | 967.7s | 1024 |
| analyze | ✅ completed | ~38m | 32 figures |

## Artifacts

- **Figures:** 73 PNG (abstract: 32 + language variants; abstract_subset: 16; full_text: 16; корневые: 9)  →  `figures/`
- **Reports:** 10 файлов (corpus_stats, semantic_analysis × 3 варианта)  →  `reports/`
- **Embeddings:** 8 .npy (все методы)  →  см. `manifest.json`
- **Logs:** 16 файлов  →  `logs/`
