"""Tests for PubMed XML parsing and localized text selection."""
from textvec.corpus.pubmed_loader import (
    _extract_other_abstracts,
    _parse_article,
    _select_localized_text,
)


def _sample_article(
    *,
    title_en: str = "English title",
    abstract_en: str = "English abstract.",
    vernacular: str = "",
    other_lang: str = "",
    other_text: str = "",
    medline_lang: str | list = "eng",
) -> dict:
    medline: dict = {
        "PMID": 12345678,
        "Article": {
            "ArticleTitle": title_en,
            "Abstract": {"AbstractText": abstract_en},
            "AuthorList": [],
            "Language": medline_lang,
            "Journal": {"JournalIssue": {"PubDate": {"Year": "2024", "Month": "Jan", "Day": "1"}}},
        },
    }
    if vernacular:
        medline["Article"]["VernacularTitle"] = vernacular
    if other_lang and other_text:
        medline["OtherAbstract"] = {
            "Language": other_lang,
            "AbstractText": other_text,
        }
    return {"PubmedArticle": {"MedlineCitation": medline}}


def test_select_localized_text_english_query_keeps_english():
    title, abstract = _select_localized_text(
        "English title",
        "English abstract.",
        "Deutscher Titel",
        [("de", "Deutsche Zusammenfassung.")],
        query_language="en",
    )
    assert title == "English title"
    assert abstract == "English abstract."


def test_select_localized_text_german_query_uses_vernacular_and_other_abstract():
    title, abstract = _select_localized_text(
        "Myocardial infarction treatment.",
        "English abstract.",
        "Behandlung des Myokardinfarkts.",
        [("de", "Deutsche Zusammenfassung.")],
        query_language="de",
    )
    assert title == "Behandlung des Myokardinfarkts."
    assert abstract == "Deutsche Zusammenfassung."


def test_select_localized_text_german_query_falls_back_to_english_without_localized_fields():
    title, abstract = _select_localized_text(
        "Myocardial infarction treatment.",
        "English abstract.",
        "",
        [],
        query_language="de",
    )
    assert title == "Myocardial infarction treatment."
    assert abstract == "English abstract."


def test_extract_other_abstracts_normalizes_language_codes():
    medline = {
        "OtherAbstract": [
            {"Language": "ger", "AbstractText": "Deutscher Text."},
            {"Language": "fre", "AbstractText": "Texte français."},
        ],
    }
    assert _extract_other_abstracts(medline) == [
        ("de", "Deutscher Text."),
        ("fr", "Texte français."),
    ]


def test_parse_article_german_query_with_localized_fields():
    article = _sample_article(
        title_en="Heart failure management.",
        abstract_en="English abstract.",
        vernacular="Herzinsuffizienz-Management.",
        other_lang="ger",
        other_text="Deutsche Zusammenfassung.",
        medline_lang=["ger"],
    )["PubmedArticle"]

    doc = _parse_article(article, topic="cardiology", query_language="de")
    assert doc.title == "Herzinsuffizienz-Management."
    assert doc.abstract == "Deutsche Zusammenfassung."
    assert doc.language == "de"


def test_parse_article_english_query_ignores_vernacular():
    article = _sample_article(
        title_en="Heart failure management.",
        abstract_en="English abstract.",
        vernacular="Herzinsuffizienz-Management.",
        other_lang="ger",
        other_text="Deutsche Zusammenfassung.",
        medline_lang=["ger"],
    )["PubmedArticle"]

    doc = _parse_article(article, topic="cardiology", query_language="en")
    assert doc.title == "Heart failure management."
    assert doc.abstract == "English abstract."
