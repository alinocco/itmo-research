"""Transformer-based vectorizers (BERT, SBERT, BGE-M3, E5, GTE).

All are loaded through ``sentence_transformers.SentenceTransformer``. A plain
``bert-base-uncased`` checkpoint is wrapped with mean pooling automatically,
giving a standard mean-pooled BERT baseline.
"""
from __future__ import annotations

import numpy as np

from ..utils import get_logger, resolve_device
from .base import BaseVectorizer

logger = get_logger("textvec.vectorization.transformer")


class TransformerVectorizer(BaseVectorizer):
    """Sentence/document embeddings from a HuggingFace model."""

    input_kind = "raw"

    def __init__(self, name: str, model_name: str, **params):
        super().__init__(**params)
        self.name = name
        self.model_name = model_name

    def _query_prefix(self) -> str:
        """Some models (E5) expect an instruction prefix on inputs."""
        lname = self.model_name.lower()
        if "e5" in lname:
            return "query: "
        return ""

    def _fit_transform(self, data) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        device = resolve_device(self.params.get("device", "auto"))
        trust_remote_code = bool(self.params.get("trust_remote_code", False))
        logger.info("Loading '%s' on %s ...", self.model_name, device)
        model = SentenceTransformer(
            self.model_name,
            device=device,
            trust_remote_code=trust_remote_code,
        )

        max_seq = self.params.get("max_seq_length")
        if max_seq:
            model.max_seq_length = int(max_seq)

        prefix = self._query_prefix()

        if self.params.get("chunk_long_docs"):
            embeddings = self._encode_chunked(model, data, prefix)
        else:
            texts = [prefix + t for t in data] if prefix else list(data)
            embeddings = model.encode(
                texts,
                batch_size=self.params.get("batch_size", 16),
                convert_to_numpy=True,
                normalize_embeddings=self.params.get("normalize_embeddings", True),
                show_progress_bar=True,
            )
        self.params["actual_dim"] = int(embeddings.shape[1])
        self.params["resolved_model"] = self.model_name
        return np.asarray(embeddings, dtype=np.float32)

    def _encode_chunked(self, model, data, prefix: str) -> np.ndarray:
        """Split long documents into word windows, encode all chunks at once,
        then mean-pool chunk vectors back into one embedding per document."""
        chunk_words = int(self.params.get("chunk_size_words", 350))
        all_chunks: list[str] = []
        owner: list[int] = []  # chunk index -> document index
        for doc_idx, text in enumerate(data):
            words = str(text).split()
            if not words:
                words = [""]
            for start in range(0, len(words), chunk_words):
                chunk = " ".join(words[start:start + chunk_words])
                all_chunks.append(prefix + chunk if prefix else chunk)
                owner.append(doc_idx)

        chunk_emb = model.encode(
            all_chunks,
            batch_size=self.params.get("batch_size", 16),
            convert_to_numpy=True,
            normalize_embeddings=False,  # normalize after pooling
            show_progress_bar=True,
        )

        dim = chunk_emb.shape[1]
        doc_emb = np.zeros((len(data), dim), dtype=np.float32)
        counts = np.zeros(len(data), dtype=np.int64)
        for vec, doc_idx in zip(chunk_emb, owner):
            doc_emb[doc_idx] += vec
            counts[doc_idx] += 1
        doc_emb /= np.maximum(counts[:, None], 1)

        if self.params.get("normalize_embeddings", True):
            norms = np.linalg.norm(doc_emb, axis=1, keepdims=True)
            doc_emb = doc_emb / np.maximum(norms, 1e-12)
        self.params["n_chunks"] = len(all_chunks)
        return doc_emb
