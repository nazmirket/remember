"""Memory-store reads use the local connection pool."""

from contextlib import asynccontextmanager

import pytest

import remember_api.engine.memory_engine as engine_mod
from remember_api.engine.memory_engine import MemoryEngine


@pytest.mark.asyncio
async def test_memory_read_acquires_and_releases_the_local_connection(monkeypatch):
    connection = object()
    events = []

    @asynccontextmanager
    async def fake_acquire(backend):
        events.append(("acquired", backend))
        try:
            yield connection
        finally:
            events.append(("released", backend))

    monkeypatch.setattr(engine_mod, "acquire_with_retry", fake_acquire)
    engine = object.__new__(MemoryEngine)
    engine._initialized = True
    engine._backend = "pool"
    async with engine._store_read_conn() as actual:
        assert actual is connection
        assert events == [("acquired", "pool")]
    assert events == [("acquired", "pool"), ("released", "pool")]
