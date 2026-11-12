"""A configured reasoning effort is honoured, or reported — in every provider (#3449).

Issue #3449 was filed against the OpenAI-compatible lane, where the setting was gated on
a model-name match and silently dropped. The same setting was dead weight in three more
lanes for a different reason: they accepted `reasoning_effort` into the constructor and
never looked at it again. From the operator's seat the symptom is identical — the
variable is set, documented and visible in the environment, and nothing happens.

Two outcomes are acceptable, and nothing else:
  * **honoured** — LiteLLM (and its Router subclass) forwards it, and LiteLLM translates
    it per target provider (Anthropic thinking budgets, Gemini thinking config, OpenAI's
    flat parameter). `litellm.drop_params = True` discards it for models with no
    reasoning knob rather than raising.
  * **reported** — providers with no reasoning control at all (Gemini and Anthropic
    native SDKs, Claude Code) log a WARNING at startup naming the ignored value.
"""

import logging

import pytest


class TestProvidersWithoutReasoningControlSaySo:
    """These providers cannot act on the setting; they must not swallow it in silence."""

    def test_anthropic_warns(self, caplog):
        from remember_api.engine.providers.anthropic_llm import AnthropicLLM

        with caplog.at_level(logging.WARNING):
            AnthropicLLM(
                provider="anthropic",
                api_key="k",
                base_url="",
                model="claude-sonnet-4-20250514",
                reasoning_effort="high",
            )
        assert any("reasoning_effort" in r.message and "ignored" in r.message for r in caplog.records)
