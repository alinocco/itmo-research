"""PubMed corpus loader (uses NCBI Entrez E-utilities via Biopython)."""
from __future__ import annotations

from ..utils import get_logger
from .schema import Document

logger = get_logger("textvec.corpus.pubmed")


def fetch_pubmed(
    topic: str,
    term: str,
    max_results: int = 100,
    email: str = "research@example.com",
    api_key: str | None = None,
) -> list[Document]:
    """Fetch up to ``max_results`` PubMed records matching ``term``.

    NCBI requires a contact ``email``; an ``api_key`` raises the rate limit.
    """
    from Bio import Entrez  # lazy import

    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    # 1) esearch -> list of PMIDs
    with Entrez.esearch(db="pubmed", term=term, retmax=max_results, sort="date") as handle:
        search_results = Entrez.read(handle)
    id_list = search_results.get("IdList", [])
    if not id_list:
        logger.warning("PubMed [%s] '%s': no results", topic, term)
        return []

    # 2) efetch -> full records (MEDLINE XML)
    with Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml") as handle:
        records = Entrez.read(handle)

    docs: list[Document] = []
    for article in records.get("PubmedArticle", []):
        try:
            docs.append(_parse_article(article, topic))
        except Exception as exc:  # noqa: BLE001 - skip malformed records, keep going
            logger.debug("Skipping a PubMed record: %s", exc)
    logger.info("PubMed [%s] '%s': fetched %d documents", topic, term, len(docs))
    return docs


def _parse_article(article: dict, topic: str) -> Document:
    medline = article["MedlineCitation"]
    pmid = str(medline["PMID"])
    art = medline["Article"]

    title = _to_text(art.get("ArticleTitle", ""))
    abstract = _extract_abstract(art)
    authors = _extract_authors(art)
    published = _extract_date(art)
    categories = [_to_text(mh["DescriptorName"]) for mh in medline.get("MeshHeadingList", [])]

    return Document(
        doc_id=f"pubmed:{pmid}",
        source="pubmed",
        topic=topic,
        title=title,
        abstract=abstract,
        authors=authors,
        published=published,
        categories=categories,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        language="en",
    )


def _to_text(value) -> str:
    return " ".join(str(value).split())


def _extract_abstract(art: dict) -> str:
    abstract_obj = art.get("Abstract", {})
    parts = abstract_obj.get("AbstractText", [])
    if isinstance(parts, list):
        return " ".join(_to_text(p) for p in parts)
    return _to_text(parts)


def _extract_authors(art: dict) -> list[str]:
    authors = []
    for a in art.get("AuthorList", []):
        last = a.get("LastName")
        fore = a.get("ForeName")
        if last and fore:
            authors.append(f"{fore} {last}")
        elif a.get("CollectiveName"):
            authors.append(_to_text(a["CollectiveName"]))
    return authors


def _extract_date(art: dict) -> str:
    issue = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
    year = issue.get("Year", "")
    month = issue.get("Month", "")
    day = issue.get("Day", "")
    if not year:
        return ""
    months = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
        "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
    }
    mm = months.get(str(month)[:3], str(month).zfill(2) if month else "01")
    dd = str(day).zfill(2) if day else "01"
    return f"{year}-{mm}-{dd}"
