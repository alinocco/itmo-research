"""End-to-end preprocessing orchestrator built on spaCy (task.md 3.5 - 3.7)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import ConfigNode, resolve_path
from ..utils import get_logger
from .cleaning import clean_text

logger = get_logger("textvec.preprocessing.pipeline")


class TextPreprocessor:
    """Clean -> tokenize -> lemmatize -> remove stopwords.

    The heavy spaCy model is loaded lazily on first use.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        language: str = "en",
        *,
        lowercase: bool = True,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_numbers: bool = False,
        strip_punctuation: bool = True,
        min_token_len: int = 2,
        lemmatize: bool = True,
        remove_stopwords: bool = True,
        extra_stopwords: list[str] | None = None,
    ):
        self.spacy_model = spacy_model
        self.language = language
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_numbers = remove_numbers
        self.strip_punctuation = strip_punctuation
        self.min_token_len = min_token_len
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.extra_stopwords = set(extra_stopwords or [])
        self._nlp = None
        self._stopwords: set[str] = set()

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = self._load_spacy()
        return self._nlp

    def _load_spacy(self):
        import spacy

        try:
            # parser/ner not needed for lemmatization; disabling them is much faster.
            nlp = spacy.load(self.spacy_model, disable=["parser", "ner"])
        except OSError as exc:
            raise OSError(
                f"spaCy model '{self.spacy_model}' is not installed. "
                f"Run: python -m spacy download {self.spacy_model}"
            ) from exc
        self._stopwords = set(nlp.Defaults.stop_words) | self.extra_stopwords
        return nlp

    def _clean(self, text: str) -> str:
        return clean_text(
            text,
            remove_html=self.remove_html,
            drop_urls=self.remove_urls,
            drop_emails=self.remove_emails,
            lowercase=self.lowercase,
            strip_punct=self.strip_punctuation,
            drop_numbers=self.remove_numbers,
        )

    def _keep(self, token) -> bool:
        if token.is_space or token.is_punct:
            return False
        if self.remove_numbers and token.like_num:
            return False
        lemma = (token.lemma_ if self.lemmatize else token.text).lower().strip()
        if len(lemma) < self.min_token_len:
            return False
        if self.remove_stopwords and (lemma in self._stopwords or token.is_stop):
            return False
        return True

    def process(self, text: str) -> list[str]:
        """Return the list of cleaned lemmas/tokens for a single text."""
        cleaned = self._clean(text)
        if not cleaned:
            return []
        doc = self.nlp(cleaned)
        out = []
        for tok in doc:
            if self._keep(tok):
                out.append((tok.lemma_ if self.lemmatize else tok.text).lower().strip())
        return out

    def process_batch(self, texts: list[str], batch_size: int = 64, n_process: int = 1) -> list[list[str]]:
        """Vectorized processing using spaCy's nlp.pipe for speed."""
        cleaned = [self._clean(t) for t in texts]
        self.nlp  # ensure stopwords initialized
        results: list[list[str]] = []
        for doc in self.nlp.pipe(cleaned, batch_size=batch_size, n_process=n_process):
            results.append([
                (tok.lemma_ if self.lemmatize else tok.text).lower().strip()
                for tok in doc if self._keep(tok)
            ])
        return results


def build_document_text(df: pd.DataFrame, text_fields: list[str]) -> pd.Series:
    """Concatenate the configured fields into one raw document text."""
    available = [f for f in text_fields if f in df.columns]
    if not available:
        raise ValueError(f"None of text_fields {text_fields} present in columns {list(df.columns)}")
    text = df[available[0]].fillna("").astype(str)
    for f in available[1:]:
        text = text.str.cat(df[f].fillna("").astype(str), sep=". ")
    return text


def preprocess_corpus(
    df: pd.DataFrame,
    cfg: ConfigNode,
    text_fields: list[str] | None = None,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """Run the full pipeline over a corpus DataFrame and persist the result.

    ``text_fields`` / ``output_csv`` override the config defaults so the same
    code serves different experiment variants (abstract vs full_text).
    """
    pp = cfg.preprocessing
    preprocessor = TextPreprocessor(
        spacy_model=pp.get("spacy_model", "en_core_web_sm"),
        language=cfg.project.get("language", "en"),
        lowercase=pp.get("lowercase", True),
        remove_html=pp.get("remove_html", True),
        remove_urls=pp.get("remove_urls", True),
        remove_emails=pp.get("remove_emails", True),
        remove_numbers=pp.get("remove_numbers", False),
        strip_punctuation=pp.get("strip_punctuation", True),
        min_token_len=pp.get("min_token_len", 2),
        lemmatize=pp.get("lemmatize", True),
        remove_stopwords=pp.get("remove_stopwords", True),
        extra_stopwords=list(pp.get("extra_stopwords", []) or []),
    )

    fields = text_fields or list(pp.get("text_fields", ["title", "abstract"]))
    raw_text = build_document_text(df, fields)
    logger.info("Preprocessing %d documents (fields=%s)...", len(df), fields)
    tokens = preprocessor.process_batch(raw_text.tolist())

    out = df.copy()
    out["text_raw"] = raw_text.values
    out["tokens"] = tokens
    out["clean_text"] = [" ".join(t) for t in tokens]
    out["n_tokens"] = [len(t) for t in tokens]

    before = len(out)
    out = out[out["n_tokens"] > 0].reset_index(drop=True)
    logger.info("Preprocessing done: %d documents kept (%d empty removed)", len(out), before - len(out))

    out_path = resolve_path(output_csv or pp.get("output_csv", "data/processed/corpus_clean.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist tokens as space-joined string for CSV friendliness.
    save_df = out.copy()
    save_df["tokens"] = save_df["clean_text"]
    save_df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Saved cleaned corpus -> %s", out_path)
    return out
