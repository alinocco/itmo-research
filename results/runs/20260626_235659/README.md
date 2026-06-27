# Run `20260626_235659`  —  15k

> **Retrospective snapshot** — captured from existing artifacts on 2026-06-27.

## Overview

| Field | Value |
|---|---|
| **Status** | completed |
| **Label** | 15k |
| **Started** | 2026-06-20T19:24:03+00:00 |
| **Finished** | 2026-06-26T17:56:59+00:00 |
| **Config** | `config/default.yaml` |
| **Variants** | abstract, abstract_subset, full_text |

## Data volumes

| Metric | Value |
|---|---|
| n_corpus_docs | 15,431 |
| n_corpus_arxiv | 9,618 |
| n_corpus_pubmed | 5,813 |
| n_clean_abstract | 15,431 |
| n_clean_abstract_subset | 300 |
| n_clean_full_text | 300 |

## Topics (corpus_stats)

| Topic | Docs |
|---|---|
| machine_learning | 2,000 |
| astrophysics | 2,000 |
| quantitative_biology | 1,984 |
| cardiology | 1,951 |
| oncology | 1,938 |
| neurology | 1,924 |
| cryptography_security | 1,873 |
| computation_language | 1,761 |

## Artifacts

- **Figures**: 57 PNG files  →  `20260626_235659/figures/`
- **Reports**: 10 files  →  `20260626_235659/reports/`
- **Embeddings**: 24 .npy files (8 methods × 3 variants)  →  see `embeddings.ref.json`

### Embedding files by variant

| Variant | Method | Size |
|---|---|---|
| abstract | bert | 45.2 MB |
| abstract | bge-m3 | 60.3 MB |
| abstract | doc2vec | 11.8 MB |
| abstract | e5 | 45.2 MB |
| abstract | fasttext | 11.8 MB |
| abstract | gte | 45.2 MB |
| abstract | sbert | 22.6 MB |
| abstract | tfidf | 17.7 MB |
| abstract_subset | bert | 0.9 MB |
| abstract_subset | bge-m3 | 1.2 MB |
| abstract_subset | doc2vec | 0.2 MB |
| abstract_subset | e5 | 0.9 MB |
| abstract_subset | fasttext | 0.2 MB |
| abstract_subset | gte | 0.9 MB |
| abstract_subset | sbert | 0.4 MB |
| abstract_subset | tfidf | 0.3 MB |
| full_text | bert | 0.9 MB |
| full_text | bge-m3 | 1.2 MB |
| full_text | doc2vec | 0.2 MB |
| full_text | e5 | 0.9 MB |
| full_text | fasttext | 0.2 MB |
| full_text | gte | 0.9 MB |
| full_text | sbert | 0.4 MB |
| full_text | tfidf | 0.3 MB |
