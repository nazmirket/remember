"""
Tests for the ``llm_extra_body`` knob (env: ``REMEMBER_API_LLM_EXTRA_BODY``).

The same JSON dict of extra request-body params is threaded into every API
provider. Each provider merges it in its own native parameter space:

- OpenAI-compatible / Fireworks: OpenAI SDK ``extra_body`` (already covered
  elsewhere; the wiring predates this change).
- Anthropic: the Anthropic SDK ``extra_body`` kwarg.
- Gemini / VertexAI: seeded into ``GenerateContentConfig`` (the SDK's native
  generation-param space — Gemini nests these in the request body).
- LiteLLM (+ bedrock alias + router): merged as top-level ``acompletion`` kwargs
  so LiteLLM normalizes/drops them per-provider.

These are deterministic unit tests: the SDK client is mocked and we assert the
params actually reach the call.
"""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

EXTRA_BODY = {"temperature": 0.2, "top_p": 0.9}
DEFAULT_HEADERS = {"X-Component-Id": "remember", "X-Trace": "abc"}


# ─── config / env parsing ─────────────────────────────────────────────────────


def test_extra_body_parsed_from_env():
    """The JSON env var is parsed into ``RememberConfig.llm_extra_body``."""
    import json

    from remember_api.config import ENV_LLM_EXTRA_BODY, RememberConfig, clear_config_cache

    with patch.dict(os.environ, {ENV_LLM_EXTRA_BODY: json.dumps(EXTRA_BODY)}, clear=False):
        clear_config_cache()
        config = RememberConfig.from_env()
        assert config.llm_extra_body == EXTRA_BODY
        clear_config_cache()


def test_extra_body_default_is_none():
    """When the env var is unset, ``llm_extra_body`` defaults to None."""
    from remember_api.config import ENV_LLM_EXTRA_BODY, RememberConfig, clear_config_cache

    env = {k: v for k, v in os.environ.items() if k != ENV_LLM_EXTRA_BODY}
    with patch.dict(os.environ, env, clear=True):
        clear_config_cache()
        config = RememberConfig.from_env()
        assert config.llm_extra_body is None
        clear_config_cache()


# ─── Anthropic ────────────────────────────────────────────────────────────────


def _make_anthropic_provider(extra_body=None):
    with patch("anthropic.AsyncAnthropic") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        from remember_api.engine.providers.anthropic_llm import AnthropicLLM

        provider = AnthropicLLM(
            provider="anthropic",
            api_key="fake-key",
            base_url="",
            model="claude-sonnet-4-20250514",
            extra_body=extra_body,
        )
    provider._client = MagicMock()
    return provider


def _fake_anthropic_response():
    block = MagicMock()
    block.type = "text"
    block.text = "ok"
    resp = MagicMock()
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=5, output_tokens=2, cache_read_input_tokens=0)
    resp.stop_reason = "end_turn"
    return resp


def test_anthropic_stores_extra_body():
    provider = _make_anthropic_provider(extra_body=EXTRA_BODY)
    assert provider._extra_body == EXTRA_BODY


def test_anthropic_empty_extra_body_defaults_to_dict():
    provider = _make_anthropic_provider(extra_body=None)
    assert provider._extra_body == {}


@pytest.mark.asyncio
async def test_anthropic_call_passes_extra_body():
    """``call()`` forwards extra_body via the Anthropic SDK ``extra_body`` kwarg."""
    provider = _make_anthropic_provider(extra_body=EXTRA_BODY)
    provider._client.messages.create = AsyncMock(return_value=_fake_anthropic_response())

    with patch("remember_api.engine.providers.anthropic_llm.get_metrics_collector"):
        await provider.call(messages=[{"role": "user", "content": "hi"}], scope="test", max_retries=0)

    kwargs = provider._client.messages.create.call_args.kwargs
    assert kwargs.get("extra_body") == EXTRA_BODY


@pytest.mark.asyncio
async def test_anthropic_no_extra_body_omits_key():
    """``call()`` does not pass ``extra_body`` when none is configured."""
    provider = _make_anthropic_provider(extra_body=None)
    provider._client.messages.create = AsyncMock(return_value=_fake_anthropic_response())

    with patch("remember_api.engine.providers.anthropic_llm.get_metrics_collector"):
        await provider.call(messages=[{"role": "user", "content": "hi"}], scope="test", max_retries=0)

    assert "extra_body" not in provider._client.messages.create.call_args.kwargs


# ─── Gemini ───────────────────────────────────────────────────────────────────


# ─── LiteLLM ──────────────────────────────────────────────────────────────────
