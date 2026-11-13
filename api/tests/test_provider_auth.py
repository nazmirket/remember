"""Authentication requirements for the supported local and hosted providers."""

import pytest

from remember_api.provider_auth import requires_api_key


@pytest.mark.parametrize("provider", ["openai-codex", "claude-code", "ollama", "lmstudio", "llamacpp"])
def test_local_and_subscription_providers_do_not_require_an_api_key(provider):
    assert not requires_api_key(provider)
    assert not requires_api_key(provider.upper())


@pytest.mark.parametrize("provider", ["openai", "openai-responses", "anthropic"])
def test_hosted_api_providers_require_an_api_key_without_federation(provider, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_FEDERATION_RULE_ID", raising=False)
    assert requires_api_key(provider)
