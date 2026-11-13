"""The local memory store is PostgreSQL-backed."""

from remember_api.engine.memories import get_memories
from remember_api.engine.memories.postgres import PostgresMemories


def test_memory_store_is_postgres():
    assert isinstance(get_memories(), PostgresMemories)
