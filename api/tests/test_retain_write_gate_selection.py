"""Local retains share the configured database write gate."""

import asyncio

from remember_api.engine.memory_engine import MemoryEngine


def test_local_banks_share_the_database_write_gate():
    engine = MemoryEngine.__new__(MemoryEngine)
    engine._put_semaphore = asyncio.Semaphore(2)
    assert engine._db_semaphore_for("first") is engine._put_semaphore
    assert engine._db_semaphore_for("second") is engine._put_semaphore
