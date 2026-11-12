"""
Tests for EntityResolver edge cases.
"""

import itertools
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from remember_api.engine.db import create_database_backend
from remember_api.engine.db.result import DictResultRow as ResultRow
from remember_api.engine.memories.pg.entity_resolver import EntityResolver, _canonical_cooccurrence_pairs
from remember_api.engine.retain.types import ResolvedEntity
from remember_api.pg0 import resolve_database_url

# ---------------------------------------------------------------------------
# Unit tests for discard_pending_stats() — no database required
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discard_pending_stats_clears_both_dicts():
    """discard_pending_stats() must remove entries for the current task from
    both _pending_stats and _pending_cooccurrences."""
    resolver = EntityResolver(pool=None)  # type: ignore[arg-type]
    key = resolver._task_key()

    resolver._pending_stats[key] = [object()]  # type: ignore[list-item]
    resolver._pending_cooccurrences[key] = [object()]  # type: ignore[list-item]

    resolver.discard_pending_stats()

    assert key not in resolver._pending_stats
    assert key not in resolver._pending_cooccurrences


@pytest.mark.asyncio
async def test_discard_pending_stats_is_idempotent():
    """Calling discard_pending_stats() when nothing is pending must not raise."""
    resolver = EntityResolver(pool=None)  # type: ignore[arg-type]
    resolver.discard_pending_stats()
    resolver.discard_pending_stats()  # second call — still safe


@pytest.mark.asyncio
async def test_discard_pending_stats_does_not_affect_other_task_keys():
    """discard_pending_stats() must only remove the current task's entries,
    leaving entries keyed under other task IDs untouched."""
    resolver = EntityResolver(pool=None)  # type: ignore[arg-type]
    other_key = -1  # A fake key that can never be a real task id

    resolver._pending_stats[other_key] = [object()]  # type: ignore[list-item]
    resolver._pending_cooccurrences[other_key] = [object()]  # type: ignore[list-item]

    resolver.discard_pending_stats()  # discards current task's key only

    assert other_key in resolver._pending_stats, "other task's stats must be preserved"
    assert other_key in resolver._pending_cooccurrences, "other task's cooccurrences must be preserved"


# ---------------------------------------------------------------------------
# Unit tests for _canonical_cooccurrence_pairs() — no database required
# ---------------------------------------------------------------------------


def test_canonical_cooccurrence_pairs_keeps_every_pair_when_input_is_unsorted():
    """Every distinct pair must survive, whatever order the entities arrive in.

    ["C", "A", "B"] is the minimal witness: ordering the second pair used to
    swap the outer loop variable, so ("A", "C") was silently dropped.
    """
    pairs = list(_canonical_cooccurrence_pairs(["C", "A", "B"]))

    assert set(pairs) == {("A", "B"), ("A", "C"), ("B", "C")}
    assert len(pairs) == 3


def test_canonical_cooccurrence_pairs_is_input_order_invariant():
    """entity_list is built from a set, so its order is arbitrary per run: the
    emitted pairs must not depend on it."""
    expected = {("e1", "e2"), ("e1", "e3"), ("e1", "e4"), ("e2", "e3"), ("e2", "e4"), ("e3", "e4")}

    for order in itertools.permutations(["e1", "e2", "e3", "e4"]):
        assert set(_canonical_cooccurrence_pairs(list(order))) == expected, f"lost a pair for input order {order}"


def test_canonical_cooccurrence_pairs_emits_canonical_order():
    """Each pair must come out as (a, b) with a < b, the ordering the
    entity_cooccurrences PK and check constraint require."""
    for order in itertools.permutations(["e1", "e2", "e3"]):
        for a, b in _canonical_cooccurrence_pairs(list(order)):
            assert a < b, f"non-canonical pair {(a, b)} for input order {order}"


def test_canonical_cooccurrence_pairs_skips_duplicates_and_short_input():
    assert list(_canonical_cooccurrence_pairs(["e1", "e1"])) == []
    assert list(_canonical_cooccurrence_pairs(["e1"])) == []
    assert list(_canonical_cooccurrence_pairs([])) == []


@pytest.mark.asyncio
async def test_resolve_entities_batch_handles_unicode_lower_conflicts(pg0_db_url):
    """
    Existing entities with PostgreSQL/Python lowercase mismatches should resolve
    to the conflicted row instead of leaving a missing entity_id.
    """
    resolved_url = await resolve_database_url(pg0_db_url)
    backend = create_database_backend("postgresql")
    await backend.initialize(resolved_url, min_size=1, max_size=2, command_timeout=30)
    bank_id = f"test-entity-resolver-{uuid.uuid4().hex[:8]}"
    event_date = datetime(2024, 1, 15, tzinfo=timezone.utc)
    resolver = EntityResolver(pool=backend, entity_lookup="full")

    try:
        async with backend.acquire() as conn:
            existing_entity_id = await conn.fetchval(
                """
                INSERT INTO entities (bank_id, canonical_name, first_seen, last_seen, mention_count)
                VALUES ($1, $2, $3, $3, 1)
                RETURNING id
                """,
                bank_id,
                "İstanbul",
                event_date,
            )

            resolved_entities = await resolver.resolve_entities_batch(
                bank_id=bank_id,
                entities_data=[
                    {
                        "text": "istanbul",
                        "nearby_entities": [],
                        "event_date": event_date,
                    }
                ],
                context="unicode case mismatch",
                unit_event_date=event_date,
                conn=conn,
            )

            entity_rows = await conn.fetch(
                """
                SELECT id, canonical_name
                FROM entities
                WHERE bank_id = $1
                ORDER BY canonical_name
                """,
                bank_id,
            )

        assert resolved_entities == [ResolvedEntity(entity_id=existing_entity_id, canonical_name="İstanbul")]
        assert len(entity_rows) == 1
        assert entity_rows[0]["id"] == existing_entity_id
        assert entity_rows[0]["canonical_name"] == "İstanbul"
    finally:
        async with backend.acquire() as conn:
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)
        await backend.shutdown()


@pytest.mark.asyncio
async def test_label_entities_stored_and_reasserted_with_label_kind(pg0_db_url):
    """Label entities are created with entity_kind='label' and keep it through
    the Phase-2 reassert.

    The kind decides membership in the partial trigram index (#3208): a label
    row resurrected as 'regular' would silently re-enter the index, so the
    resurrect path is the interesting one to pin, not just the insert.
    """
    resolved_url = await resolve_database_url(pg0_db_url)
    backend = create_database_backend("postgresql")
    await backend.initialize(resolved_url, min_size=1, max_size=2, command_timeout=30)
    bank_id = f"test-entity-kind-{uuid.uuid4().hex[:8]}"
    # Free-text label group: any "topic:..." text classifies as a label.
    entity_labels = [{"key": "topic", "type": "text", "description": "session topic"}]
    resolver = EntityResolver(pool=backend, entity_lookup="trigram")

    try:
        async with backend.acquire() as conn:
            resolved = await resolver.resolve_entities_batch(
                bank_id=bank_id,
                entities_data=[
                    {"text": "topic:algebra", "nearby_entities": []},
                    {"text": "Alice", "nearby_entities": []},
                ],
                context="entity kind storage",
                unit_event_date=None,
                conn=conn,
                entity_labels=entity_labels,
            )

            async def _kinds_by_name():
                rows = await conn.fetch("SELECT canonical_name, entity_kind FROM entities WHERE bank_id = $1", bank_id)
                return {row["canonical_name"]: row["entity_kind"] for row in rows}

            assert await _kinds_by_name() == {"topic:algebra": "label", "Alice": "regular"}
            kinds_resolved = {e.canonical_name: e.entity_kind for e in resolved}
            assert kinds_resolved == {"topic:algebra": "label", "Alice": "regular"}

            # Simulate the prune-between-phases race: rows vanish after Phase-1
            # resolution, then the Phase-2 reassert re-creates them.
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)
            await resolver.reassert_entities_batch(bank_id, resolved, conn)
            assert await _kinds_by_name() == {"topic:algebra": "label", "Alice": "regular"}
    finally:
        async with backend.acquire() as conn:
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)
        await backend.shutdown()


@pytest.mark.asyncio
async def test_fuzzy_scoring_never_merges_regular_text_into_label_row():
    """A regular text must not fuzzy-merge into a label row (#1558).

    The trigram/UTL_MATCH probes exclude label rows in SQL (entity_kind
    filter), but the "full" fallback strategy loads every bank entity, so the
    Python scoring loop must skip label-classified candidates itself. The
    candidate here would clear the 0.6 merge threshold on name similarity plus
    temporal proximity if it weren't skipped.
    """
    from types import SimpleNamespace

    from remember_api.engine.retain.entity_labels import build_labels_lookup, parse_entity_labels

    now = datetime.now(timezone.utc)
    labels_cfg = parse_entity_labels([{"key": "topic", "type": "text"}])
    ops = SimpleNamespace(
        bulk_insert_entities=AsyncMock(return_value={"topic empathy": "new-entity-id"}),
        fetch_missing_entity_ids=AsyncMock(return_value=[]),
    )
    resolver = EntityResolver(pool=SimpleNamespace(ops=ops), entity_lookup="full")

    resolved = await resolver._resolve_from_candidates(
        conn=AsyncMock(),
        bank_id="bank-1",
        # "topic empathy" (no colon) is NOT a label text, so it goes through
        # fuzzy scoring rather than the exact-match label branch.
        entities_data=[{"text": "topic empathy", "nearby_entities": [], "event_date": now}],
        unit_event_date=now,
        all_candidates={"topic empathy": [("label-row-id", "topic:empathy", {}, now, 5)]},
        cooccurrence_map={},
        taxonomy_lookup=build_labels_lookup(labels_cfg),
        labels_cfg=labels_cfg,
    )

    assert resolved == [ResolvedEntity(entity_id="new-entity-id", canonical_name="topic empathy")]
    ops.bulk_insert_entities.assert_awaited_once()


@pytest.mark.asyncio
async def test_unresolved_names_beat_a_winning_fuzzy_candidate():
    """A caller-authored name must not be re-resolved to a similar entity (#3479).

    "Dr Wall" here is the shape that broke curation: a typo entity that outscores the
    0.6 threshold on name similarity (0.41) plus co-occurrence with the other name in
    the same request (0.3), so resolution silently swaps it in for the corrected
    "Dr. Waller". With resolve=False on that mention the scoring is skipped entirely and
    the literal name is created/reused instead.
    """
    from types import SimpleNamespace

    now = datetime.now(timezone.utc)
    candidates = {"Dr. Waller": [("typo-entity-id", "Dr Wall", {}, now, 40)]}
    cooccurrence = {"typo-entity-id": {"careorg"}}
    entities_data = [
        {
            "text": "Dr. Waller",
            "nearby_entities": [{"text": "Dr. Waller"}, {"text": "CareOrg"}],
            "event_date": now,
        }
    ]

    # Fuzzy matching (retain's behaviour) picks the typo entity — the bug, pinned here so
    # the assertion below is measuring a real difference.
    fuzzy_resolver = EntityResolver(
        pool=SimpleNamespace(
            ops=SimpleNamespace(bulk_insert_entities=AsyncMock(), fetch_missing_entity_ids=AsyncMock())
        ),
        entity_lookup="full",
    )
    fuzzy = await fuzzy_resolver._resolve_from_candidates(
        conn=AsyncMock(),
        bank_id="bank-1",
        entities_data=entities_data,
        unit_event_date=now,
        all_candidates=candidates,
        cooccurrence_map=cooccurrence,
    )
    assert fuzzy[0].canonical_name == "Dr Wall", "precondition: fuzzy scoring picks the typo entity"

    ops = SimpleNamespace(
        bulk_insert_entities=AsyncMock(return_value={"dr. waller": "correct-entity-id"}),
        fetch_missing_entity_ids=AsyncMock(return_value=[]),
    )
    exact_resolver = EntityResolver(pool=SimpleNamespace(ops=ops), entity_lookup="full")
    exact = await exact_resolver._resolve_from_candidates(
        conn=AsyncMock(),
        bank_id="bank-1",
        entities_data=[{**entities_data[0], "resolve": False}],
        unit_event_date=now,
        all_candidates=candidates,
        cooccurrence_map=cooccurrence,
    )
    assert exact == [ResolvedEntity(entity_id="correct-entity-id", canonical_name="Dr. Waller")]


@pytest.mark.asyncio
async def test_per_mention_resolve_flag_only_spares_the_names_that_opt_out():
    """Retain mixes caller-supplied and extracted names in one batch (#3479).

    The flag is per mention precisely so a caller can have their own names taken literally
    without turning off resolution for the extractor's — which would silently fill the bank
    with near-duplicate entities. Here the caller's "Dr. Waller" must be created as written
    while the extractor's "Dr Waler" still resolves onto the existing "Dr Wall".
    """
    from types import SimpleNamespace

    now = datetime.now(timezone.utc)
    typo = ("typo-entity-id", "Dr Wall", {}, now, 40)
    ops = SimpleNamespace(
        bulk_insert_entities=AsyncMock(return_value={"dr. waller": "literal-id"}),
        fetch_missing_entity_ids=AsyncMock(return_value=[]),
    )
    resolver = EntityResolver(pool=SimpleNamespace(ops=ops), entity_lookup="full")

    resolved = await resolver._resolve_from_candidates(
        conn=AsyncMock(),
        bank_id="bank-1",
        entities_data=[
            {"text": "Dr. Waller", "nearby_entities": [], "event_date": now, "resolve": False},
            {"text": "Dr Waler", "nearby_entities": [], "event_date": now},
        ],
        unit_event_date=now,
        all_candidates={"Dr. Waller": [typo], "Dr Waler": [typo]},
        cooccurrence_map={},
    )

    assert resolved[0] == ResolvedEntity(entity_id="literal-id", canonical_name="Dr. Waller"), (
        "the opted-out name is created as written"
    )
    assert resolved[1].canonical_name == "Dr Wall", "the extracted name still resolves onto the existing entity"


@pytest.mark.asyncio
async def test_intrabatch_clustering_skips_names_that_opted_out():
    """A literal name must not be folded into a same-batch variant, or pull one into itself."""
    from types import SimpleNamespace

    ops = SimpleNamespace(
        bulk_insert_entities=AsyncMock(return_value={"alice": "alice-id", "alice smith": "alice-smith-id"}),
        fetch_missing_entity_ids=AsyncMock(return_value=[]),
    )
    resolver = EntityResolver(pool=SimpleNamespace(ops=ops), entity_lookup="full")

    resolved = await resolver._resolve_from_candidates(
        conn=AsyncMock(),
        bank_id="bank-1",
        entities_data=[
            {"text": "Alice", "nearby_entities": [], "resolve": False},
            {"text": "Alice Smith", "nearby_entities": [], "resolve": False},
        ],
        unit_event_date=None,
        all_candidates={},
        cooccurrence_map={},
    )

    assert [e.canonical_name for e in resolved] == ["Alice", "Alice Smith"]
    assert [e.entity_id for e in resolved] == ["alice-id", "alice-smith-id"]


# ---------------------------------------------------------------------------
# Oracle fuzzy entity resolution — unit tests (mock conn, no live DB)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_link_units_carries_event_date_into_cooccurrences(pg0_db_url):
    """
    `link_units_to_entities_batch` must propagate each unit's event_date onto the
    accumulated _CooccurrencePair entries, so flush_pending_stats() stamps
    entity_cooccurrences.last_cooccurred with the event time instead of "now".

    This protects banks that were backfilled from another memory system —
    without it, every pair collapses to the import moment and the UI's entity
    graph recency heat loses the underlying knowledge timeline.
    """
    resolved_url = await resolve_database_url(pg0_db_url)
    backend = create_database_backend("postgresql")
    await backend.initialize(resolved_url, min_size=1, max_size=2, command_timeout=30)
    bank_id = f"test-cooccurrence-evt-{uuid.uuid4().hex[:8]}"
    resolver = EntityResolver(pool=backend, entity_lookup="full")

    historical = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    try:
        async with backend.acquire() as conn:
            unit_id = await conn.fetchval(
                """
                INSERT INTO memory_units
                    (bank_id, text, fact_type, mentioned_at, occurred_start, created_at)
                VALUES ($1, 'paired-entity unit', 'experience', $2, $2, now())
                RETURNING id
                """,
                bank_id,
                historical,
            )
            e1 = await conn.fetchval(
                "INSERT INTO entities (bank_id, canonical_name, first_seen, last_seen, mention_count) "
                "VALUES ($1, 'alpha', $2, $2, 1) RETURNING id",
                bank_id,
                historical,
            )
            e2 = await conn.fetchval(
                "INSERT INTO entities (bank_id, canonical_name, first_seen, last_seen, mention_count) "
                "VALUES ($1, 'beta', $2, $2, 1) RETURNING id",
                bank_id,
                historical,
            )

            await resolver.link_units_to_entities_batch(
                [(str(unit_id), str(e1), historical), (str(unit_id), str(e2), historical)],
                conn=conn,
            )

        # Flush accumulates to entity_cooccurrences on a fresh connection, as
        # the production flush runs post-transaction to avoid lock contention.
        await resolver.flush_pending_stats()

        async with backend.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT last_cooccurred
                FROM entity_cooccurrences
                WHERE entity_id_1 = LEAST($1::uuid, $2::uuid)
                  AND entity_id_2 = GREATEST($1::uuid, $2::uuid)
                """,
                e1,
                e2,
            )
        assert row is not None, "co-occurrence row should exist"
        assert row["last_cooccurred"] == historical, (
            f"expected last_cooccurred == {historical}, got {row['last_cooccurred']}"
        )
    finally:
        async with backend.acquire() as conn:
            await conn.execute(
                "DELETE FROM unit_entities WHERE unit_id IN (SELECT id FROM memory_units WHERE bank_id = $1)", bank_id
            )
            await conn.execute(
                "DELETE FROM entity_cooccurrences WHERE entity_id_1 IN "
                "(SELECT id FROM entities WHERE bank_id = $1) "
                "OR entity_id_2 IN "
                "(SELECT id FROM entities WHERE bank_id = $1)",
                bank_id,
            )
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)
        await backend.shutdown()
