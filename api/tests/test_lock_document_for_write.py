"""Regression tests for PostgreSQLOps.lock_document_for_write.

Covers vectorize-io/remember#1944: the retain document-ownership gate used a
single ``INSERT ... ON CONFLICT DO UPDATE ... RETURNING`` upsert. PostgreSQL
runs it as-is, but the Oracle adapter rewrites ``ON CONFLICT DO UPDATE`` to a
``MERGE``, which cannot carry a ``RETURNING`` clause — so every retain 500'd
with ``DPY-1003: the executed statement does not return rows``.

The lock-and-read step now lives behind ``ops.lock_document_for_write`` so each
backend implements it natively (PG: one upsert; Oracle: idempotent insert +
``SELECT ... FOR UPDATE``). These tests pin the contract on PG (run in the
default suite) and the Oracle rewrite limitation that motivated the split.
"""

from datetime import datetime, timezone

import pytest

from remember_api.engine.schema import fq_store_table


def _ts() -> float:
    return datetime.now(timezone.utc).timestamp()


async def _seed_bank(conn, bank_id: str) -> None:
    await conn.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )


@pytest.mark.asyncio
async def test_lock_document_for_write_returns_pending_then_existing_hash(memory):
    """Fresh row → '__pending__'; existing row → its stored hash."""
    bank_id = f"test_lock_doc_{_ts()}"
    doc_id = "doc-lock-regression"
    documents = fq_store_table("documents")

    backend = await memory._get_backend()
    ops = backend.ops
    async with backend.acquire() as conn:
        await _seed_bank(conn, bank_id)

        # First call creates the row and reports the placeholder hash.
        async with conn.transaction():
            first = await ops.lock_document_for_write(conn, documents, doc_id, bank_id)
        assert first == "__pending__"

        # Promote the placeholder to a real content hash, as the real retain
        # flow does immediately after taking the lock.
        await conn.execute(
            f"UPDATE {documents} SET content_hash = $1 WHERE id = $2 AND bank_id = $3",
            "real-hash-v1",
            doc_id,
            bank_id,
        )

        # A subsequent writer sees the committed hash (and re-takes the lock).
        async with conn.transaction():
            second = await ops.lock_document_for_write(conn, documents, doc_id, bank_id)
        assert second == "real-hash-v1"


@pytest.mark.asyncio
async def test_lock_document_for_write_isolates_by_bank(memory):
    """The lock/read is scoped to (id, bank_id) — same doc_id in another bank
    is a distinct row and starts at '__pending__'."""
    suffix = _ts()
    bank_a = f"test_lock_doc_a_{suffix}"
    bank_b = f"test_lock_doc_b_{suffix}"
    doc_id = "shared-doc-id"
    documents = fq_store_table("documents")

    backend = await memory._get_backend()
    ops = backend.ops
    async with backend.acquire() as conn:
        await _seed_bank(conn, bank_a)
        await _seed_bank(conn, bank_b)

        async with conn.transaction():
            assert await ops.lock_document_for_write(conn, documents, doc_id, bank_a) == "__pending__"
        await conn.execute(
            f"UPDATE {documents} SET content_hash = $1 WHERE id = $2 AND bank_id = $3",
            "bank-a-hash",
            doc_id,
            bank_a,
        )

        async with conn.transaction():
            assert await ops.lock_document_for_write(conn, documents, doc_id, bank_b) == "__pending__"
