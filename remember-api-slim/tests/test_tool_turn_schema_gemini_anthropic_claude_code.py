"""A tool-using turn reaches Gemini, Anthropic and Claude Code as a response schema.

``LLMProvider.call_with_tools`` hands each provider's ``call`` a raw JSON schema
(``tools_via_schema.schema_format``), not a pydantic model, and no tools. Each provider must
send that schema through its own structured-output path unchanged and hand the reply back
as a dict.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from remember_api.engine.llm_interface import LLM_TOOL_CHOICE_REQUIRED
from remember_api.engine.providers import tools_via_schema

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search memories.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    }
]
_FORMAT = tools_via_schema.schema_format(tools_via_schema.response_schema(_TOOLS, LLM_TOOL_CHOICE_REQUIRED))
_REPLY = {"call": {"name": "recall", "arguments": {"query": "Q2 billing errors"}}}
_MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]


async def _call(provider: Any) -> Any:
    return (
        await provider.call(
            messages=_MESSAGES,
            response_format=_FORMAT,
            strict_schema=True,
            skip_validation=True,
            scope="test",
            max_retries=0,
        )
    ).content


@pytest.mark.asyncio
async def test_anthropic_sends_the_schema_as_its_structured_output():
    with patch("anthropic.AsyncAnthropic") as client_cls:
        client_cls.return_value = MagicMock()
        from remember_api.engine.providers.anthropic_llm import AnthropicLLM

        provider = AnthropicLLM(provider="anthropic", api_key="k", base_url="", model="claude-sonnet-4-20250514")
    block = MagicMock(type="text", text=json.dumps(_REPLY))
    response = MagicMock(content=[block], stop_reason="end_turn")
    response.usage = MagicMock(
        input_tokens=1, output_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    provider._client = MagicMock()
    provider._client.messages.create = AsyncMock(return_value=response)

    with patch("remember_api.engine.providers.anthropic_llm.get_metrics_collector"):
        assert await _call(provider) == _REPLY
    kwargs = provider._client.messages.create.call_args.kwargs
    # Native structured output carries exactly the turn's schema; no tools are sent.
    assert "tools" not in kwargs
    assert (
        kwargs["output_config"]["format"]["schema"]
        == tools_via_schema.response_schema(_TOOLS, LLM_TOOL_CHOICE_REQUIRED).schema
    )


@pytest.mark.asyncio
async def test_claude_code_puts_the_schema_in_the_prompt(monkeypatch):
    import claude_agent_sdk

    prompts: list[str] = []

    class _Text:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Assistant:
        def __init__(self, content: list[Any]) -> None:
            self.content = content

    async def fake_query(prompt: str, options: Any):
        prompts.append(prompt)
        yield _Assistant([_Text(json.dumps(_REPLY))])

    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", lambda **kw: MagicMock(**kw))
    monkeypatch.setattr(claude_agent_sdk, "AssistantMessage", _Assistant)
    monkeypatch.setattr(claude_agent_sdk, "TextBlock", _Text)
    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)
    from remember_api.engine.providers.claude_code_llm import ClaudeCodeLLM

    provider = ClaudeCodeLLM(provider="claude-code", api_key="", base_url="", model="claude-haiku-4-5")
    assert await _call(provider) == _REPLY
    assert '"anyOf"' in prompts[0] and '"recall"' in prompts[0]
