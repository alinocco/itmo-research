"""Classical vectorizers: TF-IDF, FastText, Doc2Vec."""
from __future__ import annotations

import numpy as np

from ..utils import get_logger
from .base import BaseVectorizer

logger = get_logger("textvec.vectorization.classical")


class TfidfVectorizerWrapper(BaseVectorizer):
    """TF-IDF kept sparse; optionally reduced with TruncatedSVD (LSA).

    Materializing the full dense matrix does not scale (50k x 20k float32 ~ 4 GB),
    so when ``svd_components`` is set we go through ``TruncatedSVD`` and never call
    ``.toarray()`` on the large sparse matrix.
    """

    name = "tfidf"
    input_kind = "clean"

    def _fit_transform(self, data) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(
            max_features=self.params.get("max_features", 20000),
            ngram_range=tuple(self.params.get("ngram_range", (1, 2))),
            min_df=self.params.get("min_df", 2),
        )
        matrix = vec.fit_transform(data)
        self.params["vocab_size"] = matrix.shape[1]

        svd_components = self.params.get("svd_components")
        if svd_components:
            from sklearn.decomposition import TruncatedSVD

            n_components = min(int(svd_components), matrix.shape[1] - 1, matrix.shape[0] - 1)
            svd = TruncatedSVD(n_components=n_components, random_state=self.params.get("seed", 42))
            reduced = svd.fit_transform(matrix)
            self.params["actual_dim"] = int(reduced.shape[1])
            self.params["svd_explained_variance"] = round(
                float(svd.explained_variance_ratio_.sum()), 4
            )
            return reduced

        # No SVD requested: only safe to densify for small vocabularies / corpora.
        self.params["actual_dim"] = matrix.shape[1]
        return matrix.toarray()


class FastTextVectorizer(BaseVectorizer):
    """Average of FastText word vectors over each document."""

    name = "fasttext"
    input_kind = "tokens"

    def _fit_transform(self, data) -> np.ndarray:
        from gensim.models import FastText

        vector_size = self.params.get("vector_size", 200)
        model = FastText(
            sentences=data,
            vector_size=vector_size,
            window=self.params.get("window", 5),
            min_count=self.params.get("min_count", 2),
            epochs=self.params.get("epochs", 20),
            workers=self.params.get("workers", 4),
            seed=self.params.get("seed", 42),
        )
        return np.vstack([self._doc_vector(model, toks, vector_size) for toks in data])

    @staticmethod
    def _doc_vector(model, tokens, dim) -> np.ndarray:
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        if not vecs:
            # FastText can still vectorize OOV via subwords; fall back to that.
            vecs = [model.wv[t] for t in tokens] if tokens else []
        return np.mean(vecs, axis=0) if vecs else np.zeros(dim, dtype=np.float32)


class Doc2VecVectorizer(BaseVectorizer):
    name = "doc2vec"
    input_kind = "tokens"

    def _fit_transform(self, data) -> np.ndarray:
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument

        documents = [TaggedDocument(words=toks, tags=[i]) for i, toks in enumerate(data)]
        model = Doc2Vec(
            documents=documents,
            vector_size=self.params.get("vector_size", 200),
            window=self.params.get("window", 5),
            min_count=self.params.get("min_count", 2),
            epochs=self.params.get("epochs", 40),
            workers=self.params.get("workers", 4),
            seed=self.params.get("seed", 42),
        )
        return np.vstack([model.dv[i] for i in range(len(documents))])
