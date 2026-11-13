"""Explicit tag filters preserve source-text boundaries in local recall."""

import uuid
import pytest
from remember_api.engine.memories import get_memories
from remember_api.engine.schema import fq_store_table_explicit
from remember_api.engine.search.tags import TagGroupLeaf, tags_satisfy_groups, tags_writable
from remember_api.models import RequestContext

DAN_SCOPE = [TagGroupLeaf(tags=["user:dan", "kind:rule"], match="any_strict")]
KATE_TEXT = "Nobody deploys to production on Fridays. Kate is interviewing at another company next week."
DAN_TEXT = "Dan is working on the search latency regression."


def test_tags_satisfy_groups():
    assert tags_satisfy_groups(["user:dan"], DAN_SCOPE)
    assert tags_satisfy_groups(["user:kate", "kind:rule"], DAN_SCOPE)
    assert not tags_satisfy_groups(["user:kate"], DAN_SCOPE)
    assert not tags_satisfy_groups([], DAN_SCOPE), "strict: untagged rows are outside the scope"
    assert tags_satisfy_groups(["anything"], None)


def test_tags_writable():
    assert tags_writable(["user:dan"], ["user:dan"])
    assert not tags_writable(["user:dan", "kind:rule"], ["user:dan"]), "every tag must be writable"
    assert tags_writable(["user:kate", "kind:rule"], ["user:kate", "kind:*"])
    assert not tags_writable([], ["user:dan"]), "an untagged item belongs to everyone"
    assert tags_writable([], None)


@pytest.fixture
async def scoped_bank(memory):
    """A bank holding Kate's note (one fact forged into the shared ``kind:rule`` scope) and Dan's."""
    bank_id = f"tag-scope-{uuid.uuid4().hex[:8]}"
    admin = RequestContext()
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": KATE_TEXT, "document_id": "kate-sync", "tags": ["user:kate"]}],
        request_context=admin,
    )
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": DAN_TEXT, "document_id": "dan-notes", "tags": ["user:dan"]}],
        request_context=admin,
    )
    # In production the rule tag comes from an entity label with `tag: true`, which needs a real
    # LLM to extract; the mock cannot. Forging it on the stored fact is the only way to put a
    # memory and its document in different scopes here, which is exactly the case under test.
    # Written wherever this bank's memories live. Postgres's `update_memories` is a no-op (its
    # callers write the row themselves), so for a SQL bank the UPDATE is that write.
    get_memories()
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        await conn.execute(
            f"UPDATE {fq_store_table_explicit('memory_units')} SET tags = ARRAY['user:kate', 'kind:rule'] "
            "WHERE bank_id = $1 AND fact_type = 'world' AND text ILIKE '%Fridays%'",
            bank_id,
        )
    try:
        yield bank_id
    finally:
        await memory.delete_bank(bank_id, request_context=admin)


@pytest.mark.asyncio
async def test_recall_chunks_follow_the_document_without_a_validator(memory, scoped_bank):
    """#5030: with no extension at all, a reader's own tag filter gates source text too."""
    filtered = await memory.recall_async(
        bank_id=scoped_bank,
        query="Can I deploy on Fridays?",
        tags=["user:dan", "kind:rule"],
        tags_match="any_strict",
        include_chunks=True,
        request_context=RequestContext(),
    )
    assert any("Fridays" in r.text for r in filtered.results), "the shared rule is still recalled"
    filtered_chunks = " | ".join(c.chunk_text for c in (filtered.chunks or {}).values())
    assert "interviewing" not in filtered_chunks

    unfiltered = await memory.recall_async(
        bank_id=scoped_bank, query="Can I deploy on Fridays?", include_chunks=True, request_context=RequestContext()
    )
    assert "interviewing" in " | ".join(c.chunk_text for c in (unfiltered.chunks or {}).values())
