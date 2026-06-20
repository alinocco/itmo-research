"""Unit tests for low-level text cleaning."""
from textvec.preprocessing.cleaning import (
    clean_scientific_fulltext,
    clean_text,
    cut_references,
    dehyphenate,
    normalize_whitespace,
    remove_urls,
    strip_html,
)


def test_strip_html():
    assert strip_html("<p>Hello <b>world</b></p>").split() == ["Hello", "world"]


def test_remove_urls():
    assert "http" not in remove_urls("see https://example.com/page for info")


def test_normalize_whitespace():
    assert normalize_whitespace("  a\t b\n c ") == "a b c"


def test_clean_text_pipeline():
    raw = "<div>Deep Learning!! Visit http://x.io for code@mail.com</div> 123"
    out = clean_text(raw, drop_numbers=True)
    assert "<" not in out and "http" not in out and "@" not in out
    assert out == out.lower()
    assert "123" not in out.split()


def test_clean_text_keeps_words():
    out = clean_text("Neural Networks for NLP")
    assert "neural" in out.split()
    assert "networks" in out.split()


def test_dehyphenate():
    assert dehyphenate("repre-\nsentation learning") == "representation learning"


def test_cut_references_removes_tail():
    body = "Intro. " * 40 + "\nReferences\n[1] Smith et al. 2020. [2] Doe 2019."
    out = cut_references(body)
    assert "Smith et al" not in out
    assert "Intro" in out


def test_cut_references_keeps_early_mention():
    # A 'references' word early in the text must not truncate the document.
    text = "We add references\n to prior work. " + "Body content. " * 40
    assert "Body content" in cut_references(text)


def test_clean_scientific_fulltext():
    body = "Deep lear-\nning works well in many tasks. " * 10
    raw = body + "\nReferences\n[1] X 2021. [2] Y 2020."
    out = clean_scientific_fulltext(raw)
    assert "learning" in out          # de-hyphenation worked
    assert "lear-" not in out
    assert "[1]" not in out           # reference list cut
