"""End-to-end smoke test on a tiny synthetic corpus (no network, no big models)."""
import numpy as np
import pandas as pd
import pytest

from textvec.analysis.semantic import topic_separability
from textvec.config import load_config
from textvec.preprocessing import preprocess_corpus
from textvec.vectorization.registry import _needs_trust_remote_code, build_vectorizer


@pytest.fixture
def tiny_corpus():
    rows = [
        ("a1", "ml", "Deep learning models", "Neural networks learn representations from data."),
        ("a2", "ml", "Gradient descent", "Optimization trains neural network weights efficiently."),
        ("a3", "ml", "Transformers", "Attention mechanisms power modern language models."),
        ("b1", "bio", "Cancer cells", "Tumor immunology studies immune response to cancer."),
        ("b2", "bio", "Gene therapy", "DNA editing repairs genetic mutations in cells."),
        ("b3", "bio", "Protein folding", "Proteins fold into structures determining biological function."),
    ]
    return pd.DataFrame(rows, columns=["doc_id", "topic", "title", "abstract"])


def test_preprocess_and_classical_vectorizers(tiny_corpus, tmp_path):
    cfg = load_config()
    clean = preprocess_corpus(tiny_corpus, cfg, output_csv=str(tmp_path / "clean.csv"))
    assert "clean_text" in clean.columns
    assert (clean["n_tokens"] > 0).all()

    for method in ["tfidf", "fasttext", "doc2vec"]:
        vec = build_vectorizer(method, cfg)
        result = vec.fit_transform(clean)
        assert result.embeddings.shape[0] == len(clean)
        assert result.dim > 0
        assert np.isfinite(result.embeddings).all()


def test_topic_separability_metrics(tiny_corpus, tmp_path):
    cfg = load_config()
    clean = preprocess_corpus(tiny_corpus, cfg, output_csv=str(tmp_path / "clean.csv"))
    vec = build_vectorizer("tfidf", cfg)
    result = vec.fit_transform(clean)
    metrics = topic_separability(result.embeddings, clean["topic"].tolist())
    assert metrics["n_topics"] == 2
    assert "knn_topic_purity" in metrics


def test_gte_trust_remote_code_for_alibaba_model():
    assert _needs_trust_remote_code("Alibaba-NLP/gte-multilingual-base")
    assert not _needs_trust_remote_code("thenlper/gte-base")

    cfg = load_config("config/corpus_100k_multilingual.yaml")
    vec = build_vectorizer("gte", cfg)
    assert vec.params.get("trust_remote_code") is True
