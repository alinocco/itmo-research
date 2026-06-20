"""Unit tests for low-level text cleaning."""
from textvec.preprocessing.cleaning import (
    clean_text,
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
