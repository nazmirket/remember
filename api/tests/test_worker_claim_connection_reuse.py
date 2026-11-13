"""The local claim cycle uses one connection and one transaction."""

import json
import uuid
from contextlib import asynccontextmanager

import pytest

from remember_api.worker import WorkerPoller


class _StubConnection:
    """Records every statement and transaction opened on it."""

    def __init__(self, recorder: "_StubBackend") -> None:
        self._recorder = recorder

    @asynccontextmanager
    async def transaction(self):
        self._recorder.transactions += 1
        yield self

    async def fetchval(self, query: str, *args, **kwargs):
        self._recorder.statements.append(query)
        # Per-schema EXISTS probe: every schema has work.
        return True

    async def fetch(self, query: str, *args, **kwargs):
        self._recorder.statements.append(query)
        return []


class _StubOps:
    def __init__(self, recorder: "_StubBackend") -> None:
        self._recorder = recorder

    async def claim_tasks(self, conn, table, worker_id, reserved_limits, shared_limit, **kwargs):
        from remember_api.engine.db.ops import ClaimedOperations

        self._recorder.claimed_tables.append(table)
        self._recorder.claim_conns.append(conn)
        rows = [
            {
                "operation_id": uuid.uuid4(),
                "operation_type": "test",
                "task_payload": json.dumps({"type": "test_task"}),
                "retry_count": 0,
                # Retain-fold only applies to 'retain' rows with a key; keep this
                # stub out of that path.
                "serialization_key": None,
                "bank_id": "test-bank",
            }
        ]
        # The bank rotation's cursor rides back with the rows (#3861); this stub
        # claims for one bank, so it reports that bank as where it got to.
        return ClaimedOperations(rows=rows, next_bank_cursor="test-bank")


class _StubBackend:
    """Minimal DatabaseBackend surface the poller's claim path touches.

    Uses a stub backend to record connection and transaction use.
    """

    backend_type = "stub"

    def __init__(self) -> None:
        self.acquires = 0
        self.transactions = 0
        self.statements: list[str] = []
        self.claimed_tables: list[str] = []
        self.claim_conns: list[object] = []
        self.ops = _StubOps(self)

    @asynccontextmanager
    async def acquire(self):
        self.acquires += 1
        yield _StubConnection(self)


@pytest.mark.asyncio
async def test_claim_uses_one_connection_and_transaction_for_the_local_schema():
    backend = _StubBackend()
    poller = WorkerPoller(
        backend=backend,
        worker_id="test-worker",
        executor=lambda task: None,
        schema="local_schema",
        max_slots=3,
        slot_reservations={},
    )

    claimed = await poller.claim_batch()

    assert backend.acquires == 1
    assert backend.transactions == 1
    assert len(claimed) == 1
    assert backend.claimed_tables == ['"local_schema".async_operations']
    assert len(backend.claim_conns) == 1
