"""Provider usage adapters preserve visible output and reasoning separately."""

from types import SimpleNamespace

import pytest

from remember_api.engine.providers.openai_compatible_llm import _usage_from_openai_response


@pytest.mark.parametrize("adapter", [_usage_from_openai_response])
@pytest.mark.parametrize(
    ("completion_tokens", "total_tokens"),
    [(100, 110), (40, 110)],
    ids=["reasoning-folded-into-completion", "reasoning-outside-completion"],
)
def test_openai_shaped_usage_splits_reasoning(adapter, completion_tokens, total_tokens):
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=5),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=60),
        )
    )

    usage = adapter(response)

    assert usage.input_tokens == 10
    assert usage.output_tokens == 40
    assert usage.cached_tokens == 5
    assert usage.thoughts_tokens == 60
