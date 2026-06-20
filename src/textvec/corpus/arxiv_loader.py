"""ArXiv corpus loader (uses the public arXiv API via the `arxiv` package)."""
from __future__ import annotations

from ..utils import get_logger
from .schema import Document

logger = get_logger("textvec.corpus.arxiv")


def fetch_arxiv(topic: str, search: str, max_results: int = 100) -> list[Document]:
    """Fetch up to ``max_results`` articles for an arXiv ``search`` query.

    Parameters
    ----------
    topic:    human-readable label stored as metadata (e.g. "machine_learning").
    search:   arXiv query string, e.g. "cat:cs.LG" or "all:transformer".
    """
    import arxiv  # imported lazily so the package is optional at import time

    client = arxiv.Client(page_size=min(max_results, 100), delay_seconds=3, num_retries=3)
    query = arxiv.Search(
        query=search,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    docs: list[Document] = []
    for result in client.results(query):
        short_id = result.get_short_id()
        docs.append(
            Document(
                doc_id=f"arxiv:{short_id}",
                source="arxiv",
                topic=topic,
                title=_clean(result.title),
                abstract=_clean(result.summary),
                authors=[a.name for a in result.authors],
                published=result.published.date().isoformat() if result.published else "",
                categories=list(result.categories or []),
                url=result.entry_id,
                language="en",
            )
        )
    logger.info("arXiv [%s] '%s': fetched %d documents", topic, search, len(docs))
    return docs


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())
