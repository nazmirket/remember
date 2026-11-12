"""Retry handling for empty responses from OpenAI-compatible servers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from remember_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM, ProviderResponseError


class _Response(BaseModel):
    answer: str


def _make_llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="openai",
        api_key="sk-test",
        base_url="http://localhost:1234/v1",
        model="local-model",
    )


def _make_chat_response(content: str | None) -> MagicMock:
    """Build a mock that matches the shape expected by _first_choice_or_error.

    Key fields that must be explicitly set (not left as auto-MagicMock):
    - response.error = None          (otherwise truthy MagicMock triggers error path)
    - response.model_dump()          (returns dict without 'error' key)
    - choice.message.tool_calls/refusal  (otherwise truthy MagicMock in error msg)
    - usage.completion_tokens_details  (otherwise reasoning-token math crashes, #2378)
    - usage.cached_tokens / usage.prompt_tokens_details  (otherwise the cached-
      token extraction reads an auto-MagicMock and metrics' `cached_input_tokens
      > 0` raises "'>' not supported between MagicMock and int" — only when the
      metrics path runs, which makes it an intermittent xdist failure)
    """
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = content
    choice.message.tool_calls = None
    choice.message.refusal = None

    response = MagicMock()
    response.error = None
    response.model_dump.return_value = {}
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 0 if content is None else 5
    response.usage.total_tokens = 10 if content is None else 15
    response.usage.completion_tokens_details = None
    # Cover both cached-token extraction paths (response_usage.cached_tokens and
    # usage.prompt_tokens_details.cached_tokens) so neither leaks a MagicMock.
    response.usage.cached_tokens = 0
    response.usage.prompt_tokens_details = None
    response.choices = [choice]
    return response


@pytest.mark.asyncio
async def test_null_content_raises_after_retries_exhausted():
    """All retries return null content -> ProviderResponseError, not TypeError."""
    llm = _make_llm()

    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _make_chat_response(None)

        with pytest.raises(ProviderResponseError, match="empty message content"):
            await llm.call(
                messages=[{"role": "user", "content": "extract facts"}],
                response_format=_Response,
                max_retries=2,
                initial_backoff=0.0,
                max_backoff=0.0,
            )

    # 3 attempts = max_retries (2) + 1 initial
    assert mock_create.call_count == 3


@pytest.mark.asyncio
async def test_null_content_recovers_on_retry():
    """Provider returns null on first call, valid JSON on second -> request succeeds."""
    llm = _make_llm()

    responses = [
        _make_chat_response(None),
        _make_chat_response('{"answer": "ok"}'),
    ]

    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = responses

        result = (
            await llm.call(
                messages=[{"role": "user", "content": "extract facts"}],
                response_format=_Response,
                max_retries=2,
                initial_backoff=0.0,
                max_backoff=0.0,
            )
        ).content

    assert isinstance(result, _Response)
    assert result.answer == "ok"
    assert mock_create.call_count == 2
