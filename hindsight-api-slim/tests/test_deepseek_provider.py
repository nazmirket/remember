"""DeepSeek as a first-class OpenAI-compatible provider: its defaults and request shape."""

import pytest

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

# These tests assert the default json_object behaviour, not a job's JSON-mode override.
pytestmark = pytest.mark.usefixtures("default_json_mode")


def _make_deepseek_llm(model: str = "deepseek-v4-flash") -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="openai",
        api_key="sk-test",
        base_url="http://localhost:1234/v1",
        model=model,
    )


def test_deepseek_flash_is_not_treated_as_reasoning_model():
    llm = _make_deepseek_llm("deepseek-v4-flash")

    assert llm._supports_reasoning_model() is False


def test_deepseek_reasoning_models_still_use_reasoning_parameters():
    llm = _make_deepseek_llm("deepseek-v4-pro")

    assert llm._supports_reasoning_model() is True


@pytest.mark.asyncio
async def test_strict_schema_reaches_openai_compatible_local_server():
    """A local OpenAI-compatible endpoint receives the configured strict schema."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from hindsight_api.engine.providers.tools_via_schema import ResponseSchema, schema_format

    llm = _make_deepseek_llm()
    reply = MagicMock()
    reply.choices[0].message.content = '{"query": "x"}'
    reply.choices[0].finish_reason = "stop"
    reply.usage.prompt_tokens, reply.usage.completion_tokens, reply.usage.total_tokens = 1, 1, 2
    reply.usage.completion_tokens_details = None
    reply.error = None
    reply.model_dump.return_value = {}
    fmt = schema_format(
        ResponseSchema(name="recall", schema={"type": "object", "properties": {"query": {"type": "string"}}})
    )
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock, return_value=reply) as create:
        result = await llm.call(
            [{"role": "user", "content": "q"}],
            response_format=fmt,
            strict_schema=True,
            skip_validation=True,
            max_retries=0,
        )
    sent = create.await_args.kwargs
    assert sent["response_format"]["type"] == "json_schema"
    assert "query" in sent["response_format"]["json_schema"]["schema"]["properties"]
    assert result.content == {"query": "x"}
