"""Tag edits made after recall must survive a prepared consolidation UPDATE."""

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from remember_api import RequestContext
from remember_api.config import _get_raw_config
from remember_api.engine.cross_encoder import RRFPassthroughCrossEncoder
from remember_api.engine.memory_engine import MemoryEngine
from remember_api.engine.response_models import MemoryFact
from tests.consolidation_actions import execute_create_action, execute_update_action
from tests.test_retain_same_document_concurrency import _StubEmbeddings


@pytest_asyncio.fixture
async def tag_memory(pg0_db_url: str) -> AsyncIterator[MemoryEngine]:
    # This regression concerns storage, not model quality. Neither component calls a model.
    memory = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="mock",
        memory_llm_api_key="",
        memory_llm_model="mock",
        embeddings=_StubEmbeddings(),
        cross_encoder=RRFPassthroughCrossEncoder(),
        pool_min_size=1,
        pool_max_size=3,
        run_migrations=False,
    )
    await memory.initialize()
    try:
        yield memory
    finally:
        await memory.close()


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
@pytest.mark.parametrize("current_tags", [["scope", "status:active", "knowledge:decision"], ["scope"], []])
async def test_update_uses_current_tags_and_records_them_in_history(
    tag_memory: MemoryEngine, current_tags: list[str]
) -> None:
    memory = tag_memory
    context = RequestContext()
    bank_id = f"test-current-tags-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=context)
    source_id = uuid.uuid4()
    snapshot_tags = ["scope", "status:obsolete"]
    config = _get_raw_config()
    config.enable_observation_history = True
    try:
        # Seed a source directly so this storage race has no extraction/background job.
        # The observation itself is created by the production consolidation action.
        async with memory._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO memory_units (id, bank_id, text, fact_type, tags) VALUES ($1, $2, $3, 'world', $4)",
                source_id,
                bank_id,
                "Alice plays cello.",
                ["source:new"],
            )
        await execute_create_action(
            pool=await memory._get_backend(),
            memory_engine=memory,
            bank_id=bank_id,
            source_memory_ids=[source_id],
            text="Alice plays cello.",
            source_fact_tags=snapshot_tags,
        )
        result = await memory.list_memory_units(bank_id=bank_id, fact_type="observation", request_context=context)
        assert result["total"] == 1
        observation_id = result["items"][0]["id"]
        snapshot = MemoryFact(id=observation_id, text="Alice plays cello.", fact_type="observation", tags=snapshot_tags)
        # Force the stale-recall window discussed in #4831: an edit commits after recall, before apply.
        # There is no public API for directly replacing an observation's tags.
        async with memory._pool.acquire() as conn:
            await conn.execute(
                "UPDATE memory_units SET tags = $2 WHERE id = $1", uuid.UUID(observation_id), current_tags
            )
        await execute_update_action(
            pool=await memory._get_backend(),
            memory_engine=memory,
            bank_id=bank_id,
            source_memory_ids=[source_id],
            observation_id=observation_id,
            new_text="Alice still plays cello.",
            observations=[snapshot],
            source_fact_tags=["source:new"],
        )
        updated = await memory.get_memory_unit(bank_id=bank_id, memory_id=observation_id, request_context=context)
        assert updated["text"] == "Alice still plays cello."
        assert set(updated["tags"]) == set(current_tags) | {"source:new"}
        history = await memory.get_observation_history(bank_id, observation_id, context)
        assert set(history[0]["previous_tags"]) == set(current_tags)
    finally:
        await memory.delete_bank(bank_id, request_context=context)
