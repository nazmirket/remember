"""Local bank vector index creation and repair."""

import uuid
from types import SimpleNamespace

import pytest

from remember_api.admin import cli
from remember_api.engine import memory_engine as memory_engine_module
from remember_api.engine import vector_index_health
from remember_api.engine.retain import bank_utils
from remember_api.engine.vector_index_health import BankIndexResult

pytestmark = pytest.mark.xdist_group("vector_index_reconcile")
_TEST_SCHEMA = "public"


async def _returns(value):
    """Await-able stand-in, so a monkeypatched coroutine can be written as a lambda."""
    return value


@pytest.fixture(autouse=True)
def eager(monkeypatch):
    """Pin eager mode on for every test here, rather than inheriting the default.

    These tests are about WHICH banks get indexes, not about the size threshold, and
    the two policies are alternatives: with ``REMEMBER_API_VECTOR_INDEX_MIN_ROWS``
    set, creation builds nothing for *any* bank — which would fail the SQL-owned
    assertions and, worse, make the store-owned ones pass vacuously. Patched on every
    module that imported the helper by name; a missed one silently restores the
    ambient default (the same trap ``test_hnsw_indexes.py::threshold_set`` documents).
    """
    for module in (vector_index_health, bank_utils, memory_engine_module):
        monkeypatch.setattr(module, "per_bank_indexes_are_eager", lambda: True)


async def _bank_indexes(pool, bank_id: str) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = $1
              AND tablename = 'memory_units'
              AND indexname LIKE 'idx_mu_emb_%'
              -- strpos, not LIKE: every bank id here contains underscores, which LIKE
              -- reads as single-char wildcards. Production (_index_health) matches the
              -- fuller `(bank_id = 'x'::text)` rendering; this needle is the part of it
              -- that identifies the bank, which is all this helper needs.
              AND strpos(indexdef, $2) > 0
            ORDER BY indexname
            """,
            _TEST_SCHEMA,
            f"bank_id = '{bank_id}'",
        )
    return [row["indexname"] for row in rows]


@pytest.mark.memory_backend_incompatible
async def test_a_sql_owned_bank_still_gets_all_three(memory, request_context):
    """The silent half: inverting the condition strips ANN from every ordinary bank.

    Nothing fails when that happens — recall falls back to the exact ``(bank_id,
    fact_type)`` scan and returns the same rows, more slowly — so only an explicit
    assertion catches it. Overlaps ``test_repair_bank_vector_indexes.py``'s
    ``test_bank_creation_builds_all_three_indexes`` on purpose: it is the in-file
    control that gives the store-owned assertion above its meaning.
    """
    bank_id = f"test_so_sql_{uuid.uuid4().hex[:8]}"
    try:
        await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)

        assert len(await _bank_indexes(memory._pool, bank_id)) == 3
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_a_schema_that_fails_partway_still_reports_the_banks_it_did_repair(monkeypatch, pg0_db_url, capsys):
    """Work that already happened must survive the failure that stopped the rest.

    ``CREATE INDEX CONCURRENTLY`` is not rolled back by a later exception, so banks
    reconciled before the blip are real. Building them as a list comprehension threw
    all of them away — the comprehension binds nothing until it finishes — taking the
    failed-index names with them. Three commits of this PR have fixed a version of
    that bug at a different scope; this is the one that pins it.
    """
    calls: list[str] = []

    async def _flaky_reconcile(conn, schema, bid, index_clause, dry_run=False):
        calls.append(bid)
        if len(calls) == 3:
            raise RuntimeError("router blipped on the third bank")
        return BankIndexResult(bank_id=bid, created=1, failed=1, failed_indexes=[f"idx_dead_{bid}"])

    monkeypatch.setattr(cli, "list_bank_ids", lambda conn, schema: _returns(["b1", "b2", "b3", "b4"]))
    monkeypatch.setattr(cli, "reconcile_bank_vector_indexes", _flaky_reconcile)

    sweep = await cli._run_repair_bank(
        pg0_db_url, base_schema=_TEST_SCHEMA, schema=_TEST_SCHEMA, bank_id=None, dry_run=False
    )

    assert [r.bank_id for r in sweep.banks] == ["b1", "b2"], f"completed reconciles were discarded: {sweep.banks}"
    assert sweep.skipped_schema == _TEST_SCHEMA, "the schema still needs a re-run for b3/b4"
    # The names are the point: nothing else in the output carries them.
    assert [n for r in sweep.banks for n in r.failed_indexes] == ["idx_dead_b1", "idx_dead_b2"]
    # And the operator is told how far it got, not just that it stopped.
    assert "2 of 4 bank(s) done" in capsys.readouterr().err


async def test_a_dead_connection_reports_completed_banks(monkeypatch, pg0_db_url, capsys):
    import asyncpg

    async def reconcile(conn, schema, bid, index_clause, dry_run=False):
        if bid == "b2":
            raise asyncpg.InterfaceError("connection was closed")
        return BankIndexResult(bank_id=bid, created=1)

    monkeypatch.setattr(cli, "list_bank_ids", lambda conn, schema: _returns(["b1", "b2", "b3"]))
    monkeypatch.setattr(cli, "reconcile_bank_vector_indexes", reconcile)
    monkeypatch.setattr(
        cli,
        "sql_memories",
        lambda: SimpleNamespace(
            admin_sweep_orphan_observations=lambda **kwargs: _returns(0),
            admin_rebuild_observation_search_vectors=lambda **kwargs: _returns(0),
        ),
    )
    sweep = await cli._run_repair_bank(pg0_db_url, base_schema=_TEST_SCHEMA, schema=None, bank_id=None, dry_run=False)
    assert sweep.skipped_schema == _TEST_SCHEMA
    assert [result.bank_id for result in sweep.banks] == ["b1"]
    assert "1 of 3 bank(s) done" in capsys.readouterr().err
