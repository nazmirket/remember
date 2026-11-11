"""Configuration failures propagate through provider construction."""

import pytest


def test_llm_provider_from_env_surfaces_invalid_configuration(monkeypatch):
    """Config-level validation now reaches this factory instead of being bypassed.

    A retain budget smaller than the chunk size is rejected when the config is built.
    Because the factory goes through that config, the misconfiguration is reported
    here too rather than only once some later code path happened to load it.
    """
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.llm_wrapper import LLMProvider

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "fake-key")
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS", "1000")
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_CHUNK_SIZE", "2000")
    clear_config_cache()

    with pytest.raises(ValueError, match="HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS"):
        LLMProvider.from_env()
    clear_config_cache()
