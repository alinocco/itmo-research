"""Common interface for all vectorizers."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class VectorizationResult:
    """Embeddings plus reproducibility metadata for one method."""

    method: str
    embeddings: np.ndarray            # shape (n_docs, dim)
    doc_ids: list[str]
    params: dict = field(default_factory=dict)
    elapsed_sec: float = 0.0

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0


class BaseVectorizer:
    """Base class. Subclasses implement ``_fit_transform``.

    ``input_kind`` selects which document representation is fed to the model:
      * "clean"  -> lemmatized, stopword-filtered text (classical methods)
      * "tokens" -> list of tokens (gensim methods)
      * "raw"    -> lightly-cleaned natural text (transformer methods)
    """

    name: str = "base"
    input_kind: str = "clean"

    def __init__(self, **params):
        self.params = params

    def _select_input(self, df: pd.DataFrame):
        if self.input_kind == "tokens":
            return [t.split() if isinstance(t, str) else list(t) for t in df["clean_text"]]
        if self.input_kind == "raw":
            col = "text_raw" if "text_raw" in df.columns else "clean_text"
            return df[col].fillna("").astype(str).tolist()
        return df["clean_text"].fillna("").astype(str).tolist()

    def _fit_transform(self, data) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> VectorizationResult:
        data = self._select_input(df)
        start = time.perf_counter()
        embeddings = np.asarray(self._fit_transform(data), dtype=np.float32)
        elapsed = time.perf_counter() - start
        return VectorizationResult(
            method=self.name,
            embeddings=embeddings,
            doc_ids=df["doc_id"].astype(str).tolist(),
            params=self.params,
            elapsed_sec=round(elapsed, 3),
        )
