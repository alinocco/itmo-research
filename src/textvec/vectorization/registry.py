"""Factory that maps a method name + config to a configured vectorizer."""
from __future__ import annotations

from ..config import ConfigNode
from .base import BaseVectorizer
from .classical import Doc2VecVectorizer, FastTextVectorizer, TfidfVectorizerWrapper
from .transformer import TransformerVectorizer

CLASSICAL_METHODS = {"tfidf", "fasttext", "doc2vec"}
TRANSFORMER_METHODS = {"bert", "sbert", "bge-m3", "e5", "gte"}
AVAILABLE_METHODS = sorted(CLASSICAL_METHODS | TRANSFORMER_METHODS)


def build_vectorizer(method: str, cfg: ConfigNode) -> BaseVectorizer:
    """Instantiate the vectorizer for ``method`` using the project config."""
    vc = cfg.vectorization
    seed = cfg.project.get("seed", 42)
    device = cfg.project.get("device", "auto")

    if method == "tfidf":
        return TfidfVectorizerWrapper(**vc.tfidf.to_dict())

    if method == "fasttext":
        return FastTextVectorizer(seed=seed, **vc.fasttext.to_dict())

    if method == "doc2vec":
        return Doc2VecVectorizer(seed=seed, **vc.doc2vec.to_dict())

    if method in TRANSFORMER_METHODS:
        tr = vc.transformers
        models = tr.models.to_dict()
        if method not in models:
            raise ValueError(f"No model configured for transformer method '{method}'")
        return TransformerVectorizer(
            name=method,
            model_name=models[method],
            device=device,
            batch_size=tr.get("batch_size", 16),
            max_seq_length=tr.get("max_seq_length", 512),
            normalize_embeddings=tr.get("normalize_embeddings", True),
        )

    raise ValueError(f"Unknown vectorization method: '{method}'. Available: {AVAILABLE_METHODS}")
