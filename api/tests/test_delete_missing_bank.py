"""Bank deletion is safe for missing banks and removes existing banks."""

import pytest

from remember_api.models import RequestContext


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, {"memory_units_deleted": 0, "entities_deleted": 0, "documents_deleted": 0}),
        ({"delete_bank_profile": False}, {"memory_units_deleted": 0, "entities_deleted": 0, "documents_deleted": 0}),
        ({"fact_type": "world"}, {"memory_units_deleted": 0, "entities_deleted": 0}),
    ],
    ids=["delete_bank", "clear_memories", "clear_memories_by_type"],
)
async def test_delete_missing_bank_is_a_no_op(memory, kwargs, expected):
    result = await memory.delete_bank("delete_missing_bank_never_created", request_context=RequestContext(), **kwargs)
    for key, value in expected.items():
        assert result[key] == value


@pytest.mark.asyncio
async def test_delete_existing_bank(memory):
    bank_id = "delete_missing_bank_control"
    context = RequestContext()
    await memory.ensure_bank_profile(bank_id, request_context=context)
    result = await memory.delete_bank(bank_id, request_context=context)
    assert result["bank_deleted"] is True
