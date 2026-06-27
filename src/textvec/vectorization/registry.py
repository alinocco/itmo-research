"""Factory that maps a method name + config to a configured vectorizer."""
from __future__ import annotations

from ..config import ConfigNode
from .base import BaseVectorizer
from .classical import Doc2VecVectorizer, FastTextVectorizer, TfidfVectorizerWrapper
from .transformer import TransformerVectorizer

CLASSICAL_METHODS = {"tfidf", "fasttext", "doc2vec"}
TRANSFORMER_METHODS = {"bert", "sbert", "bge-m3", "e5", "gte"}
AVAILABLE_METHODS = sorted(CLASSICAL_METHODS | TRANSFORMER_METHODS)


def _needs_trust_remote_code(model_name: str) -> bool:
    """Some HuggingFace checkpoints (e.g. Alibaba-NLP/gte-multilingual-base) ship custom code."""
    name = model_name.lower()
    return "alibaba-nlp" in name or "gte-multilingual" in name


def build_vectorizer(method: str, cfg: ConfigNode) -> BaseVectorizer:
    """Instantiate the vectorizer for ``method`` using the project config."""
    vc = cfg.vectorization
    seed = cfg.project.get("seed", 42)
    device = cfg.project.get("device", "auto")

    if method == "tfidf":
        return TfidfVectorizerWrapper(seed=seed, **vc.tfidf.to_dict())

    if method == "fasttext":
        return FastTextVectorizer(seed=seed, **vc.fasttext.to_dict())

    if method == "doc2vec":
        return Doc2VecVectorizer(seed=seed, **vc.doc2vec.to_dict())

    if method in TRANSFORMER_METHODS:
        tr = vc.transformers
        models = tr.models.to_dict()
        if method not in models:
            raise ValueError(f"No model configured for transformer method '{method}'")
        model_name = models[method]
        trust_remote_code = tr.get("trust_remote_code")
        if trust_remote_code is None:
            trust_remote_code = _needs_trust_remote_code(model_name)
        return TransformerVectorizer(
            name=method,
            model_name=model_name,
            device=device,
            batch_size=tr.get("batch_size", 16),
            max_seq_length=tr.get("max_seq_length", 512),
            normalize_embeddings=tr.get("normalize_embeddings", True),
            chunk_long_docs=tr.get("chunk_long_docs", False),
            chunk_size_words=tr.get("chunk_size_words", 350),
            trust_remote_code=bool(trust_remote_code),
        )

    raise ValueError(f"Unknown vectorization method: '{method}'. Available: {AVAILABLE_METHODS}")
