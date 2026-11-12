"""Remember environment configuration."""

from remember_api.config import RememberConfig


def test_remember_side_embeddings_settings_are_recognized(monkeypatch):
    monkeypatch.setenv("REMEMBER_API_QUERY_EMBEDDINGS_PROVIDER", "openai")
    monkeypatch.setenv("REMEMBER_API_QUERY_EMBEDDINGS_OPENAI_API_KEY", "test-key")
    config = RememberConfig.from_env()
    assert config.query_embeddings is not None
    assert config.query_embeddings.embeddings_provider == "openai"
