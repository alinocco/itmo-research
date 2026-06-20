"""Text preprocessing pipeline (Stage 3 of task.md).

cleaning -> normalization -> tokenization -> lemmatization -> stopword removal.
"""

from .pipeline import TextPreprocessor, preprocess_corpus

__all__ = ["TextPreprocessor", "preprocess_corpus"]
