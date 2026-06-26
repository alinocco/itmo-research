"""PubMed corpus loader (uses NCBI Entrez E-utilities via Biopython)."""
from __future__ import annotations

from ..utils import get_logger
from .language import normalize_language_code, resolve_document_language
from .schema import Document

logger = get_logger("textvec.corpus.pubmed")

# NCBI PubMed [Language] filter values (https://www.ncbi.nlm.nih.gov/books/NBK3827/)
_PUBMED_LANGUAGE = {
    "en": "english",
    "ru": "russian",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "pt": "portuguese",
    "it": "italian",
    "ko": "korean",
    "ar": "arabic",
}


def _build_search_term(term: str, language: str | None) -> str:
    """Append a PubMed language filter when requested."""
    if not language:
        return term
    lang = language.lower().strip()
    pubmed_lang = _PUBMED_LANGUAGE.get(lang, lang)
    return f"({term}) AND {pubmed_lang}[Language]"


def fetch_pubmed(
    topic: str,
    term: str,
    max_results: int = 100,
    email: str = "research@example.com",
    api_key: str | None = None,
    language: str | None = None,
) -> list[Document]:
    """Fetch up to ``max_results`` PubMed records matching ``term``.

    NCBI requires a contact ``email``; an ``api_key`` raises the rate limit.
    When ``language`` is set (e.g. ``"ru"``), results are restricted via
  ``{language}[Language]`` in the Entrez query.
    """
    from Bio import Entrez  # lazy import

    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    search_term = _build_search_term(term, language)

    # 1) esearch -> list of PMIDs
    with Entrez.esearch(db="pubmed", term=search_term, retmax=max_results, sort="date") as handle:
        search_results = Entrez.read(handle)
    id_list = search_results.get("IdList", [])
    if not id_list:
        logger.warning("PubMed [%s] '%s' (lang=%s): no results", topic, term, language or "any")
        return []

    # 2) efetch -> full records (MEDLINE XML), batched to stay within request limits.
    docs: list[Document] = []
    batch_size = 200
    for start in range(0, len(id_list), batch_size):
        batch = id_list[start:start + batch_size]
        try:
            with Entrez.efetch(db="pubmed", id=batch, rettype="abstract", retmode="xml") as handle:
                records = Entrez.read(handle)
        except Exception as exc:  # noqa: BLE001 - skip a failed batch, keep going
            logger.warning("PubMed efetch batch failed (%d-%d): %s", start, start + len(batch), exc)
            continue
        for article in records.get("PubmedArticle", []):
            try:
                docs.append(_parse_article(article, topic, query_language=language))
            except Exception as exc:  # noqa: BLE001 - skip malformed records
                logger.debug("Skipping a PubMed record: %s", exc)

    logger.info(
        "PubMed [%s] '%s' (lang=%s): fetched %d documents",
        topic, term, language or "any", len(docs),
    )
    return docs


def _parse_article(article: dict, topic: str, query_language: str | None = None) -> Document:
    medline = article["MedlineCitation"]
    pmid = str(medline["PMID"])
    art = medline["Article"]

    title_en = _to_text(art.get("ArticleTitle", ""))
    abstract_en = _extract_abstract_text(art.get("Abstract", {}))
    vernacular_title = _to_text(art.get("VernacularTitle", ""))
    other_abstracts = _extract_other_abstracts(medline)

    title, abstract = _select_localized_text(
        title_en,
        abstract_en,
        vernacular_title,
        other_abstracts,
        query_language=query_language,
    )

    authors = _extract_authors(art)
    published = _extract_date(art)
    categories = [_to_text(mh["DescriptorName"]) for mh in medline.get("MeshHeadingList", [])]

    lang_raw = art.get("Language", "")
    doc_lang = resolve_document_language(
        lang_raw,
        text=f"{title}. {abstract}",
        query_language=query_language,
        default="en",
    )

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
        language=doc_lang,
    )


def _select_localized_text(
    title_en: str,
    abstract_en: str,
    vernacular_title: str,
    other_abstracts: list[tuple[str, str]],
    *,
    query_language: str | None,
) -> tuple[str, str]:
    """Prefer VernacularTitle / non-English OtherAbstract for non-English queries."""
    query_code = normalize_language_code(query_language, default="") if query_language else ""

    if not query_code or query_code == "en":
        return title_en, abstract_en

    title = vernacular_title or title_en
    abstract = abstract_en
    for lang, text in other_abstracts:
        if lang == query_code:
            abstract = text
            break
    else:
        for lang, text in other_abstracts:
            if lang != "en":
                abstract = text
                break

    return title, abstract


def _extract_other_abstracts(medline: dict) -> list[tuple[str, str]]:
    """Return ``(iso639_1, abstract_text)`` pairs from Medline ``OtherAbstract``."""
    raw = medline.get("OtherAbstract", [])
    if isinstance(raw, dict):
        raw = [raw]

    out: list[tuple[str, str]] = []
    for item in raw or []:
        lang = normalize_language_code(item.get("Language", ""), default="")
        text = _extract_abstract_text(item)
        if lang and text:
            out.append((lang, text))
    return out



def _normalize_language(value) -> str:
    """Backward-compatible alias for :func:`normalize_language_code`."""
    return normalize_language_code(value)


def _to_text(value) -> str:
    return " ".join(str(value).split())


def _extract_abstract(art: dict) -> str:
    """Extract the primary (usually English) abstract from an Article."""
    return _extract_abstract_text(art.get("Abstract", {}))


def _extract_abstract_text(abstract_obj: dict | None) -> str:
    if not abstract_obj:
        return ""
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
