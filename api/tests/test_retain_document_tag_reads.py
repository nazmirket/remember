"""A scoped retain's write check reads existing documents' tags in one go, with the same answer
as reading them one at a time (#6327)."""

import random

import asyncpg

from remember_api.engine.memories import postgres


async def test_batched_read_matches_per_document_read(pg0_db_url, monkeypatch):
    monkeypatch.setattr(postgres, "fq_store_table", lambda name: f"pg_temp.{name}")
    store = postgres.PostgresMemories()
    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(
            "CREATE TEMP TABLE documents (id text, bank_id text, tags varchar[], PRIMARY KEY (id, bank_id))"
        )
        rng = random.Random(6327)
        rows = []
        for i in range(2500):
            kind = rng.choice(["tagged", "untagged", "empty", "other-bank", "missing"])
            if kind == "missing":
                continue
            tags = {"tagged": ["user:dan", "kind:rule"], "untagged": None, "empty": []}.get(kind, ["user:kate"])
            rows.append((f"d{i}", "other" if kind == "other-bank" else "bank", tags))
        await conn.executemany("INSERT INTO documents VALUES ($1, $2, $3)", rows)
        ids = sorted(f"d{i}" for i in range(2500))

        per_doc = {}
        for did in ids:
            cur = await store.current_document_tags(conn=conn, fq_table=None, bank_id="bank", document_id=did)
            if cur.found:
                per_doc[did] = cur.tags
        batched = await store.existing_documents_tags(conn=conn, fq_table=None, bank_id="bank", document_ids=ids)
        assert batched == per_doc
    finally:
        await conn.close()
