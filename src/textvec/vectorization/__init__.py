"""Text vectorization methods (Stage 4 of task.md).

Classical: TF-IDF, FastText, Doc2Vec.
Neural   : BERT, Sentence-BERT, BGE-M3, E5, GTE (via sentence-transformers).
"""

from .base import BaseVectorizer, VectorizationResult
from .registry import build_vectorizer, AVAILABLE_METHODS

__all__ = ["BaseVectorizer", "VectorizationResult", "build_vectorizer", "AVAILABLE_METHODS"]
