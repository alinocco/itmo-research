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
        logger.info("Loading '%s' on %s ...", self.model_name, device)
        model = SentenceTransformer(self.model_name, device=device)

        max_seq = self.params.get("max_seq_length")
        if max_seq:
            model.max_seq_length = int(max_seq)

        prefix = self._query_prefix()
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
