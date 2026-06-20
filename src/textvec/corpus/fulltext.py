"""On-demand full-text fetching for a corpus subset.

Full text is NOT available from the metadata APIs:
  * ArXiv  -> download the PDF and extract text (pypdf);
  * PubMed -> only the PMC Open Access subset has free full text (JATS XML),
              reached via elink (PMID -> PMCID) + efetch(db="pmc").

Documents without freely available full text are dropped from the subset so
that the abstract-vs-full_text comparison runs on the same set of articles.
"""
from __future__ import annotations

import io
import time

import pandas as pd
import requests

from ..preprocessing.cleaning import clean_scientific_fulltext
from ..utils import get_logger

logger = get_logger("textvec.corpus.fulltext")

_HEADERS = {"User-Agent": "textvec-research/0.1 (academic use)"}


def _fetch_arxiv_fulltext(short_id: str, client=None) -> str:
    """Download the arXiv PDF directly and extract text with pypdf."""
    from pypdf import PdfReader

    # Strip version suffix is unnecessary; arxiv.org/pdf/<id> resolves either way.
    pdf_url = f"https://arxiv.org/pdf/{short_id}"
    resp = requests.get(pdf_url, headers=_HEADERS, timeout=60)
    resp.raise_for_status()
    reader = PdfReader(io.BytesIO(resp.content))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return clean_scientific_fulltext(text)


def _fetch_pmc_fulltext(pmid: str, email: str, api_key: str | None = None) -> str:
    from Bio import Entrez
    from bs4 import BeautifulSoup

    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    # PMID -> PMCID (only present for PMC-archived articles).
    with Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid) as handle:
        links = Entrez.read(handle)
    pmcids = [
        link["Id"]
        for ls in links
        for linkset in ls.get("LinkSetDb", [])
        for link in linkset.get("Link", [])
    ]
    if not pmcids:
        return ""

    with Entrez.efetch(db="pmc", id=pmcids[0], rettype="full", retmode="xml") as handle:
        xml = handle.read()
    soup = BeautifulSoup(xml, "lxml-xml")
    body = soup.find("body")
    if body is None:
        return ""
    paragraphs = [p.get_text(" ", strip=True) for p in body.find_all("p")]
    return clean_scientific_fulltext("\n".join(paragraphs))


def _fetch_one(row, email: str, api_key: str | None) -> str:
    """Fetch full text for a single corpus row based on its source."""
    if row.source == "arxiv":
        short_id = str(row.doc_id).replace("arxiv:", "")
        text = _fetch_arxiv_fulltext(short_id)
        time.sleep(0.5)  # be gentle with arxiv.org
        return text
    if row.source == "pubmed":
        pmid = str(row.doc_id).replace("pubmed:", "")
        text = _fetch_pmc_fulltext(pmid, email, api_key)
        time.sleep(0.4)  # be gentle with NCBI
        return text
    return ""


def augment_with_fulltext(
    df: pd.DataFrame,
    per_topic: int = 150,
    email: str = "research@example.com",
    api_key: str | None = None,
    min_chars: int = 500,
    max_attempts_factor: int = 3,
) -> pd.DataFrame:
    """Return a balanced subset of ``df`` enriched with a ``full_text`` column.

    For each topic we iterate over candidate documents and keep fetching until
    ``per_topic`` articles with usable full text are collected (or a per-topic
    attempt budget is exhausted). This is necessary because not every PubMed
    article is in the PMC Open Access subset (recent ones are often embargoed),
    whereas arXiv PDFs are almost always available.
    """
    collected: list[dict] = []

    for topic, group in df.groupby("topic"):
        got, attempts = 0, 0
        budget = per_topic * max_attempts_factor
        for row in group.itertuples(index=False):
            if got >= per_topic or attempts >= budget:
                break
            # Give up early on a topic whose slice is fully embargoed (e.g. recent PubMed).
            if attempts >= 15 and got == 0:
                logger.info("[%s] no full text in first %d attempts; skipping topic", topic, attempts)
                break
            attempts += 1
            try:
                text = _fetch_one(row, email, api_key)
            except Exception as exc:  # noqa: BLE001 - skip failures, keep going
                logger.debug("Full text failed for %s: %s", row.doc_id, exc)
                continue
            if text and len(text) >= min_chars:
                record = row._asdict()
                record["full_text"] = text
                collected.append(record)
                got += 1
                if got % 25 == 0:
                    logger.info("  [%s] %d/%d collected (%d attempts)", topic, got, per_topic, attempts)
        logger.info("[%s] collected %d full texts in %d attempts", topic, got, attempts)

    subset = pd.DataFrame(collected)
    logger.info("Full-text subset: %d documents", len(subset))
    if not subset.empty:
        logger.info("  by topic: %s", subset["topic"].value_counts().to_dict())
    return subset
