"""Local bank lists retain write-time ordering and search."""

from datetime import datetime, timedelta, timezone

import pytest


@pytest.mark.asyncio
async def test_bank_last_write_includes_fact_updates(memory, request_context):
    bank_id = f"lastfact_{datetime.now(timezone.utc).timestamp()}"
    pool = await memory._get_pool()
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "xyzabc123 !@# a slice"}],
            document_id="session-1",
            request_context=request_context,
        )
        # Consolidation can update a fact without updating its source document.
        forged = datetime.now(timezone.utc) + timedelta(hours=1)
        async with pool.acquire() as conn:
            await conn.execute("UPDATE memory_units SET updated_at = $1 WHERE bank_id = $2", forged, bank_id)
        page = await memory.list_banks(limit=50, offset=0, request_context=request_context)
        entry = next(bank for bank in page["banks"] if bank["bank_id"] == bank_id)
        assert datetime.fromisoformat(entry["last_write_at"]) == forged
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_bank_search_filters_the_ranked_list(memory, request_context):
    names = ["find_me_alpha", "find_me_beta", "unrelated_gamma"]
    try:
        for name in names:
            await memory.ensure_bank_profile(name, request_context=request_context)
        page = await memory.list_banks(search_query="find_me", limit=50, offset=0, request_context=request_context)
        assert sorted(bank["bank_id"] for bank in page["banks"]) == names[:2]
    finally:
        for name in names:
            await memory.delete_bank(name, request_context=request_context)
