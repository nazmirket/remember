"""Embedding model changes return a retryable response through HTTP routes."""

import httpx
import pytest

from remember_api.api.http import _internal_error, _store_unavailable
from remember_api.engine.reflect.agent import ReflectToolExecutionError


class _Shed(Exception):
    sqlstate = "HS503"


def _wrapped() -> RuntimeError:
    """The shape a route sees: the store's refusal, wrapped by the engine on the way up."""
    try:
        try:
            raise _Shed("embedding model changed; retry later")
        except _Shed as inner:
            raise RuntimeError("retain failed") from inner
    except RuntimeError as outer:
        return outer


def test_a_wrapped_reembed_guard_is_a_503_with_retry_after():
    err = _store_unavailable(_wrapped())
    assert err is not None
    assert err.status_code == 503
    assert err.headers == {"Retry-After": "30"}
    assert "retry later" in err.detail


def test_the_shared_catch_all_answers_503_for_it():
    try:
        raise _wrapped()
    except RuntimeError as e:
        assert _internal_error(e, "test").status_code == 503


def test_any_other_failure_is_still_a_500():
    assert _store_unavailable(RuntimeError("boom")) is None
    try:
        raise RuntimeError("boom")
    except RuntimeError as e:
        assert _internal_error(e, "test").status_code == 500


@pytest.mark.asyncio
async def test_retain_recall_and_reflect_answer_503_for_reembed_guard(memory, monkeypatch):
    """Through the real routes: each has its own catch-all rather than the shared one."""
    from remember_api.api import create_app

    async def shed(*args, **kwargs):
        raise _wrapped()

    monkeypatch.setattr(memory, "retain_batch_async", shed)

    async def reflect_tool_shed(*args, **kwargs):
        # Reflect wraps a failed retrieval tool in its own error, chaining the store's refusal.
        raise ReflectToolExecutionError("Reflect tool 'recall' failed on iteration 1") from _wrapped()

    monkeypatch.setattr(memory, "recall_async", shed)
    monkeypatch.setattr(memory, "reflect_async", reflect_tool_shed)
    app = create_app(memory, initialize_memory=False)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        retain = await client.post(
            "/v1/default/banks/shed-test/memories",
            json={"items": [{"content": "a fact"}], "async": False},
        )
        recall = await client.post(
            "/v1/default/banks/shed-test/memories/recall",
            json={"query": "anything"},
        )
        reflect = await client.post(
            "/v1/default/banks/shed-test/reflect",
            json={"query": "anything"},
        )

    for response in (retain, recall, reflect):
        assert response.status_code == 503, response.text
        assert response.headers["retry-after"] == "30"
