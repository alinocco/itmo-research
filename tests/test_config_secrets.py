"""Config loading with secrets overlay."""
from textvec.config import load_config


def test_secrets_overlay_merges_pubmed_key():
    cfg = load_config("config/corpus_100k_multilingual.yaml")
    api_key = cfg.corpus.sources.pubmed.get("api_key")
    assert api_key, "expected api_key from config/secrets.local.yaml"
