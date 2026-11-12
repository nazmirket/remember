"""Startup migrates exactly the configured local schema."""

from types import SimpleNamespace

import pytest

from remember_api import migrations
from remember_api.engine import memory_engine
from remember_api.engine.memory_engine import MemoryEngine


@pytest.mark.asyncio
async def test_startup_migrates_only_the_configured_schema(monkeypatch):
    config = SimpleNamespace(
        database_schema="local_memory",
        migration_database_url=None,
        vector_extension="pgvector",
        text_search_extension="native",
    )
    calls = []
    monkeypatch.setattr(memory_engine, "get_config", lambda: config)
    monkeypatch.setattr(migrations, "run_local_migrations", lambda *a, **kw: calls.append((a, kw)))
    engine = SimpleNamespace(
        embeddings=SimpleNamespace(dimension=384),
        _backend=SimpleNamespace(supports_bm25=True),
    )

    await MemoryEngine._run_pg_migrations(engine, "postgresql://local")

    assert len(calls) == 1
    args, options = calls[0]
    assert args == ("postgresql://local", "local_memory")
    assert options["embedding_dimension"] == 384
