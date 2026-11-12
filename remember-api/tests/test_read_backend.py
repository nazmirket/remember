"""Tests for the local database used by reads and writes."""

from __future__ import annotations

import pytest
import pytest_asyncio

from remember_api import MemoryEngine
from remember_api.engine.task_backend import SyncTaskBackend


def _make_engine(pg0_db_url: str, embeddings, cross_encoder, query_analyzer) -> MemoryEngine:
    """Build a MemoryEngine for a single test. Tiny pool, no migrations,
    SyncTaskBackend so async tasks resolve inline. Mirrors conftest's
    ``memory`` fixture but lets each test build its own with custom env.
    """
    return MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="none",  # No LLM calls in these tests
        memory_llm_api_key="unused",
        memory_llm_model="unused",
        embeddings=embeddings,
        cross_encoder=cross_encoder,
        query_analyzer=query_analyzer,
        pool_min_size=1,
        pool_max_size=2,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
    )


@pytest_asyncio.fixture
async def engine_no_read_url(monkeypatch, pg0_db_url, embeddings, cross_encoder, query_analyzer):
    """Engine built with READ_DATABASE_URL explicitly unset."""
    from remember_api.config import clear_config_cache

    monkeypatch.delenv("REMEMBER_API_READ_DATABASE_URL", raising=False)
    clear_config_cache()

    mem = _make_engine(pg0_db_url, embeddings, cross_encoder, query_analyzer)
    await mem.initialize()
    yield mem
    try:
        await mem.close()
    except Exception:
        pass
    clear_config_cache()


@pytest.mark.asyncio
async def test_local_engine_uses_one_database_backend(engine_no_read_url):
    """Reads and writes use the same initialized local pool."""
    engine = engine_no_read_url
    assert await engine._get_backend() is engine._backend
    assert engine._backend.get_pool() is not None


def test_separate_read_database_is_rejected(monkeypatch, pg0_db_url, embeddings, cross_encoder, query_analyzer):
    """An unsupported read-replica setting must not silently redirect or go unused."""
    from remember_api.config import clear_config_cache

    monkeypatch.setenv("REMEMBER_API_READ_DATABASE_URL", pg0_db_url)
    clear_config_cache()
    try:
        with pytest.raises(ValueError, match="separate read database"):
            _make_engine(pg0_db_url, embeddings, cross_encoder, query_analyzer)
    finally:
        clear_config_cache()
