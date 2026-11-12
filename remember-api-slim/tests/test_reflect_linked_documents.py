"""Reflection does not query removed knowledge banks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from remember_api.models import RequestContext
from tests.test_reflect_decision_model_switch import engine  # noqa: F401  (fixture)


@pytest.mark.asyncio
@pytest.mark.parametrize("label", ["reflect", "refresh_mental_model", "dry_run_refresh_mental_model"])
async def test_reflect_does_not_read_knowledge_banks(engine, monkeypatch, label):  # noqa: F811
    from remember_api.engine.reflect.models import ReflectAgentResult

    engine._config_resolver = SimpleNamespace(
        resolve_full_config=AsyncMock(
            return_value=SimpleNamespace(
                reflect_use_decision_model=False,
                mental_model_refresh_use_decision_model=False,
            )
        ),
        get_bank_config=AsyncMock(return_value={}),
    )
    recalled: list[bool] = []

    async def fake_tool_recall(*args, include_documents: bool = False, **kwargs):
        recalled.append(include_documents)
        return {}

    async def fake_run_reflect_agent(**kwargs):
        await kwargs["recall_fn"]("q", 100, 100)
        return ReflectAgentResult(text="ok")

    monkeypatch.setattr("remember_api.engine.memory_engine.tool_recall", fake_tool_recall)
    monkeypatch.setattr("remember_api.engine.memory_engine.run_reflect_agent", fake_run_reflect_agent)
    await engine.reflect_async(
        bank_id="bank-1",
        query="test",
        request_context=RequestContext(),
        exclude_mental_models=True,
        _operation_label=label,
    )
    assert recalled == [False]
