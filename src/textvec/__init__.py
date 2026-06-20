"""textvec - research pipeline for text vectorization and semantic-space analysis.

Stages (see context/task.md, Semester II):
    1. corpus        - collect & unify a corpus of scientific articles (ArXiv / PubMed)
    2. preprocessing - clean, normalize, tokenize, lemmatize, remove stopwords
    3. vectorization - build document embeddings with several methods
    4. analysis      - reduce dimensionality (PCA / UMAP) and visualize the semantic space
"""

__version__ = "0.1.0"

# Keep matplotlib / HF caches inside the project so the pipeline works even when
# the user's home directory is not writable.
import os as _os
from pathlib import Path as _Path

_CACHE = _Path(__file__).resolve().parents[2] / ".cache"
_os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
(_CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
