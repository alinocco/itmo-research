"""Tests for multilingual corpus collection and preprocessing."""
import pandas as pd
import pytest

from textvec.corpus.pubmed_loader import _build_search_term, _normalize_language
from textvec.preprocessing.pipeline import _fallback_tokens, _is_cjk_heavy, preprocess_corpus


def test_pubmed_language_filter():
    assert _build_search_term("cancer", "ru") == "(cancer) AND russian[Language]"
    assert _build_search_term("cancer", "en") == "(cancer) AND english[Language]"
    assert _build_search_term("cancer", None) == "cancer"


def test_normalize_language():
    assert _normalize_language("Russian") == "ru"
    assert _normalize_language("eng") == "en"
    assert _normalize_language("chinese") == "zh"
    assert _normalize_language(["eng"]) == "en"
    assert _normalize_language("['eng']") == "en"
    assert _normalize_language(["rus", "eng"]) == "ru"
    assert _normalize_language("") == "unknown"


def test_normalize_language_no_bracket_artifact():
    from textvec.corpus.language import normalize_language_code

    assert normalize_language_code(["eng"]) == "en"
    assert normalize_language_code("['eng']") == "en"
    assert normalize_language_code("['") == "unknown"


def test_cjk_tokenization():
    text = "癌症免疫治疗的新方法"
    assert _is_cjk_heavy(text)
    toks = _fallback_tokens(text, min_token_len=1)
    assert len(toks) > 5
    assert all(len(t) == 1 for t in toks)


def test_multilingual_preprocess_smoke(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
project:
  seed: 42
  language: multilingual
preprocessing:
  strategy: multilingual
  spacy_models:
    en: en_core_web_sm
  lowercase: true
  remove_html: true
  remove_urls: true
  remove_emails: true
  remove_numbers: false
  strip_punctuation: true
  min_token_len: 1
  lemmatize: false
  remove_stopwords: false
  text_fields: [title, abstract]
  output_csv: clean.csv
""",
        encoding="utf-8",
    )
    from textvec.config import load_config

    cfg = load_config(cfg_path)
    df = pd.DataFrame([
        {"doc_id": "1", "topic": "bio", "language": "en",
         "title": "Cancer study", "abstract": "Immune therapy works."},
        {"doc_id": "2", "topic": "bio", "language": "zh",
         "title": "癌症研究", "abstract": "免疫治疗有效。"},
    ])
    out = preprocess_corpus(df, cfg, output_csv=str(tmp_path / "clean.csv"))
    assert len(out) == 2
    assert out.loc[out["doc_id"] == "1", "n_tokens"].iloc[0] > 0
    assert out.loc[out["doc_id"] == "2", "n_tokens"].iloc[0] > 0
