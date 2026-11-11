"""Tests for the database abstraction layer (db + sql modules).

Unit tests that verify the abstraction interfaces work correctly
without requiring a live database connection.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from hindsight_api.engine.db import DatabaseBackend, DatabaseConnection, create_database_backend
from hindsight_api.engine.db import postgresql as pg_backend
from hindsight_api.engine.db.ops import UpdatedWindow
from hindsight_api.engine.db.postgresql import PostgreSQLBackend, apply_session_settings
from hindsight_api.engine.db.result import DictResultRow as ResultRow
from hindsight_api.engine.sql import SQLDialect, create_sql_dialect
from hindsight_api.engine.sql.postgresql import PostgreSQLDialect

# A recall with no created_after/created_before — the graph-expansion CTEs must
# then render exactly as they did before the window existed.
_UNBOUNDED_WINDOW = UpdatedWindow(after=None, before=None, first_param_index=4)

# ---------------------------------------------------------------------------
# ResultRow tests
# ---------------------------------------------------------------------------


class TestResultRow:
    def test_dict_access(self):
        row = ResultRow({"id": 1, "name": "test"})
        assert row["id"] == 1
        assert row["name"] == "test"

    def test_attr_access(self):
        row = ResultRow({"id": 1, "name": "test"})
        assert row.id == 1
        assert row.name == "test"

    def test_get_with_default(self):
        row = ResultRow({"id": 1})
        assert row.get("id") == 1
        assert row.get("missing") is None
        assert row.get("missing", "default") == "default"

    def test_keys(self):
        row = ResultRow({"a": 1, "b": 2})
        assert set(row.keys()) == {"a", "b"}

    def test_values(self):
        row = ResultRow({"a": 1, "b": 2})
        assert set(row.values()) == {1, 2}

    def test_items(self):
        row = ResultRow({"a": 1, "b": 2})
        assert set(row.items()) == {("a", 1), ("b", 2)}

    def test_contains(self):
        row = ResultRow({"id": 1})
        assert "id" in row
        assert "missing" not in row

    def test_len(self):
        row = ResultRow({"a": 1, "b": 2, "c": 3})
        assert len(row) == 3

    def test_bool_delegates_to_data(self):
        row = ResultRow({"id": 1})
        assert bool(row)
        empty_row = ResultRow({})
        assert not bool(empty_row)

    def test_repr(self):
        row = ResultRow({"id": 1})
        assert "ResultRow" in repr(row)

    def test_missing_attr_raises(self):
        row = ResultRow({"id": 1})
        with pytest.raises(AttributeError):
            _ = row.missing


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFactories:
    def test_create_postgresql_backend(self):
        backend = create_database_backend("postgresql")
        assert isinstance(backend, PostgreSQLBackend)
        assert isinstance(backend, DatabaseBackend)

    def test_create_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown database backend"):
            create_database_backend("mysql")

    def test_create_postgresql_dialect(self):
        dialect = create_sql_dialect("postgresql")
        assert isinstance(dialect, PostgreSQLDialect)
        assert isinstance(dialect, SQLDialect)

    def test_create_unknown_dialect_raises(self):
        with pytest.raises(ValueError, match="Unknown SQL dialect"):
            create_sql_dialect("mysql")


# ---------------------------------------------------------------------------
# PostgreSQLDialect tests
# ---------------------------------------------------------------------------


class TestPostgreSQLDialect:
    @pytest.fixture()
    def d(self):
        return PostgreSQLDialect()

    def test_param(self, d):
        assert d.param(1) == "$1"
        assert d.param(3) == "$3"

    def test_cast(self, d):
        assert d.cast("$1", "jsonb") == "$1::jsonb"
        assert d.cast("$2", "uuid[]") == "$2::uuid[]"

    def test_vector_distance(self, d):
        assert d.vector_distance("embedding", "$1") == "embedding <=> $1::vector"

    def test_vector_similarity(self, d):
        assert d.vector_similarity("embedding", "$1") == "1 - (embedding <=> $1::vector)"

    def test_json_extract_text(self, d):
        assert d.json_extract_text("col", "key") == "col ->> 'key'"

    def test_json_contains(self, d):
        assert d.json_contains("col", "$1") == "col @> $1::jsonb"

    def test_json_merge(self, d):
        assert d.json_merge("col", "$1") == "col || $1::jsonb"

    def test_text_search_score_bm25(self, d):
        result = d.text_search_score("text", "$1", index_name="idx_test")
        assert "to_bm25query" in result

    def test_text_search_score_tsvector(self, d):
        result = d.text_search_score("text", "$1")
        assert "ts_rank_cd" in result

    def test_similarity(self, d):
        assert d.similarity("col", "$1") == "similarity(col, $1)"

    def test_upsert_do_nothing(self, d):
        sql = d.upsert("t", ["a", "b"], ["a"], [])
        assert "ON CONFLICT (a) DO NOTHING" in sql

    def test_upsert_do_update(self, d):
        sql = d.upsert("t", ["a", "b"], ["a"], ["b"])
        assert "ON CONFLICT (a) DO UPDATE SET b = EXCLUDED.b" in sql

    def test_bulk_unnest(self, d):
        result = d.bulk_unnest([("$1", "text[]"), ("$2", "uuid[]")])
        assert result == "unnest($1::text[], $2::uuid[])"

    def test_limit_offset(self, d):
        assert d.limit_offset("$1", "$2") == "LIMIT $1 OFFSET $2"

    def test_returning(self, d):
        assert d.returning(["id", "name"]) == "RETURNING id, name"

    def test_ilike(self, d):
        assert d.ilike("col", "$1") == "col ILIKE $1"

    def test_array_any(self, d):
        assert d.array_any("$1") == "= ANY($1)"

    def test_array_all(self, d):
        assert d.array_all("$1") == "!= ALL($1)"

    def test_array_contains(self, d):
        assert d.array_contains("tags", "$1") == "tags @> $1::varchar[]"

    def test_for_update_skip_locked(self, d):
        assert d.for_update_skip_locked() == "FOR UPDATE SKIP LOCKED"

    def test_generate_uuid(self, d):
        assert d.generate_uuid() == "gen_random_uuid()"

    def test_greatest(self, d):
        assert d.greatest("a", "b") == "GREATEST(a, b)"

    def test_current_timestamp(self, d):
        assert d.current_timestamp() == "now()"

    def test_array_agg(self, d):
        assert d.array_agg("col") == "array_agg(col)"

    def test_build_semantic_arm(self, d):
        arm = d.build_semantic_arm(
            table="schema.memory_units",
            cols="id, text",
            fact_type="world",
            embedding_param="$1",
            bank_id_param="$2",
            fetch_limit=100,
            min_similarity=0.58,
        )
        assert "1 - (embedding <=> $1::vector)" in arm
        assert ">= 0.58" in arm
        assert "fact_type = 'world'" in arm
        assert "LIMIT 100" in arm
        assert "'semantic' AS source" in arm

    def test_build_bm25_arm_native(self, d):
        arm = d.build_bm25_arm(
            table="schema.memory_units",
            cols="id, text",
            fact_type="world",
            bank_id_param="$2",
            limit_param="$3",
            text_param="$4",
        )
        assert "ts_rank_cd" in arm
        assert "to_tsquery" in arm
        assert "'bm25' AS source" in arm
        assert "LIMIT $3" in arm
        # Ties at the LIMIT cut break on id, so repeated recalls keep the same rows (#6056)
        assert "ORDER BY ts_rank_cd(search_vector, to_tsquery('english', $4)) DESC, id LIMIT $3" in arm
        # Default language is english when bm25_language is not specified
        assert "to_tsquery('english', $4)" in arm

    def test_build_bm25_arm_native_uses_configured_language(self, d):
        arm = d.build_bm25_arm(
            table="schema.memory_units",
            cols="id, text",
            fact_type="world",
            bank_id_param="$2",
            limit_param="$3",
            text_param="$4",
            bm25_language="french",
        )
        # Both the score and the WHERE filter must use the configured dictionary
        assert "to_tsquery('french', $4)" in arm
        assert "to_tsquery('english'" not in arm

    def test_build_bm25_arm_vchord(self, d):
        arm = d.build_bm25_arm(
            table="t",
            cols="id",
            fact_type="world",
            bank_id_param="$2",
            limit_param="$3",
            text_param="$4",
            text_search_extension="vchord",
        )
        assert "to_bm25query" in arm
        assert "tokenize" in arm

    def test_build_bm25_arm_vchord_gates_zero_score_by_default(self, d):
        """VectorChord ranks every doc, so a score gate must filter non-matches.

        The negated `<&>` score is BM25 (>= 0); the default 0 floor keeps only
        rows with a genuine query-term match, mirroring native tsvector's `@@`.
        """
        arm = d.build_bm25_arm(
            table="t",
            cols="id",
            fact_type="world",
            bank_id_param="$2",
            limit_param="$3",
            text_param="$4",
            text_search_extension="vchord",
        )
        assert (
            "-(search_vector <&> (SELECT to_bm25query('idx_memory_units_text_search', tokenize($4, 'llmlingua2')))) > 0"
            in arm
        )

    def test_build_bm25_arm_vchord_honors_custom_min_score(self, d):
        arm = d.build_bm25_arm(
            table="t",
            cols="id",
            fact_type="world",
            bank_id_param="$2",
            limit_param="$3",
            text_param="$4",
            text_search_extension="vchord",
            bm25_min_score=2.5,
        )
        # Inclusive (`>=`), matching the documented `min_scores` contract and the
        # semantic arm's `>= min_similarity`. This was `>` until #3882: the same
        # parameter served as both vchord's structural match gate and the caller's
        # floor, and the gate's `>` leaked into the public contract.
        assert ">= 2.5" in arm

    def test_build_bm25_arm_pg_textsearch_scores_each_row(self, d):
        arm = d.build_bm25_arm(
            table="schema.memory_units",
            cols="id, text",
            fact_type="world",
            bank_id_param="$2",
            limit_param="$3",
            text_param="$4",
            text_search_extension="pg_textsearch",
        )
        expected_distance = "text <@> to_bm25query($4, 'idx_memory_units_text_search')"
        assert f"-({expected_distance}) AS bm25_score" in arm
        assert f"ORDER BY {expected_distance} ASC" in arm
        assert "$4 <@>" not in arm

    def test_build_bm25_arm_pgroonga(self, d):
        arm = d.build_bm25_arm(
            table="schema.memory_units",
            cols="id, text",
            fact_type="world",
            bank_id_param="$2",
            limit_param="$3",
            text_param="$4",
            text_search_extension="pgroonga",
        )
        # pgroonga uses the &@~ operator + pgroonga_score for ranking with inline
        # pgroonga_tokenize aggregation.
        assert "&@~ (SELECT string_agg(pgroonga_query_escape(elem->>'value'), ' OR ')" in arm
        assert "pgroonga_tokenize($4, 'tokenizer', 'TokenBigram', 'normalizer', 'NormalizerNFKC150')" in arm
        assert "pgroonga_score(tableoid, ctid)" in arm
        assert "to_tsquery" not in arm

    def test_build_bm25_arm_pgroonga_ignores_bm25_language(self, d):
        """pgroonga's tokenizer is fixed at index creation; bm25_language must not leak in."""
        arm = d.build_bm25_arm(
            table="t",
            cols="id",
            fact_type="world",
            bank_id_param="$2",
            limit_param="$3",
            text_param="$4",
            text_search_extension="pgroonga",
            bm25_language="french",
        )
        assert "french" not in arm

    def test_build_bm25_arm_pg_search(self, d):
        arm = d.build_bm25_arm(
            table="schema.memory_units",
            cols="id, text",
            fact_type="world",
            bank_id_param="$2",
            limit_param="$3",
            text_param="$4",
            text_search_extension="pg_search",
        )
        assert "paradedb.score(id)" in arm
        # @@@ on the key_field requires a field-qualified query, so we
        # fan the bind param out across all indexed text fields.
        assert "id @@@ paradedb.boolean(should =>" in arm
        assert "paradedb.match('text', $4)" in arm
        assert "paradedb.match('text_signals', $4)" in arm
        # `context` multiplies the postings scanned for little signal (#4313).
        assert "'context'" not in arm
        assert "paradedb.score(id) DESC" in arm
        assert "'bm25' AS source" in arm
        assert "LIMIT $3" in arm

    def test_build_bm25_arm_pg_search_tokenizer_prunes_to_terms(self, d):
        """With a configured tokenizer the query becomes capped exact term queries (#4313)."""
        arm = d.build_bm25_arm(
            table="schema.memory_units",
            cols="id, text",
            fact_type="world",
            bank_id_param="$2",
            limit_param="$3",
            text_param="$4",
            text_search_extension="pg_search",
            pg_search_tokenizer="jieba",
            max_query_terms=16,
        )
        assert "unnest($4::text::pdb.jieba::text[])" in arm
        assert "paradedb.term(f, t)" in arm
        assert "LIMIT 16" in arm
        assert "paradedb.match(" not in arm

        uncapped = d.build_bm25_arm(
            table="schema.memory_units",
            cols="id, text",
            fact_type="world",
            bank_id_param="$2",
            limit_param="$3",
            text_param="$4",
            text_search_extension="pg_search",
            pg_search_tokenizer="lindera(chinese)",
        )
        assert "pdb.lindera(chinese)::text[]" in uncapped
        assert "min(o) LIMIT" not in uncapped

    def test_build_bm25_arm_pg_search_ngram_keeps_raw_match(self, d):
        arm = d.build_bm25_arm(
            table="schema.memory_units",
            cols="id, text",
            fact_type="world",
            bank_id_param="$2",
            limit_param="$3",
            text_param="$4",
            text_search_extension="pg_search",
            pg_search_tokenizer="ngram(2,3)",
            max_query_terms=16,
        )
        assert "paradedb.match('text', $4)" in arm
        assert "term(" not in arm

    def test_build_bm25_arm_pg_search_custom_schema(self, d):
        arm = d.build_bm25_arm(
            table="schema.memory_units",
            cols="id, text",
            fact_type="world",
            bank_id_param="$2",
            limit_param="$3",
            text_param="$4",
            text_search_extension="pg_search",
            pg_search_function_schema="pgsearch",
        )
        assert "pgsearch.score(id)" in arm
        assert "id @@@ pgsearch.boolean(should =>" in arm
        assert "pgsearch.match('text', $4)" in arm
        assert "pgsearch.match('text_signals', $4)" in arm
        assert "pgsearch.score(id) DESC" in arm
        assert "'bm25' AS source" in arm
        assert "LIMIT $3" in arm

    def test_prepare_bm25_text_native(self, d):
        result = d.prepare_bm25_text(["hello", "world"], "hello world")
        assert result == "hello | world"

    def test_prepare_bm25_text_vchord(self, d):
        result = d.prepare_bm25_text(["hello", "world"], "hello world", text_search_extension="vchord")
        assert result == "hello world"

    def test_prepare_bm25_text_pgroonga(self, d):
        result = d.prepare_bm25_text(["hello", "world"], "hello world", text_search_extension="pgroonga")
        assert result == "hello world"

        result = d.prepare_bm25_text(["网关计划任务"], "网关计划任务", text_search_extension="pgroonga")
        assert result == "网关计划任务"

    def test_prepare_bm25_text_pg_search(self, d):
        result = d.prepare_bm25_text(["hello", "world"], "hello world", text_search_extension="pg_search")
        assert result == "hello world"


# ---------------------------------------------------------------------------
# PostgreSQLBackend unit tests (no live DB)
# ---------------------------------------------------------------------------


class TestPostgreSQLBackendUnit:
    def test_uninitialized_acquire_raises(self):
        backend = PostgreSQLBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            backend.get_pool()

    def test_uninitialized_get_pool_raises(self):
        backend = PostgreSQLBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            backend.get_pool()

    def test_is_ready_false_before_initialize(self):
        assert PostgreSQLBackend().is_ready is False

    @pytest.mark.asyncio
    async def test_is_ready_false_for_whole_shutdown(self):
        """is_ready must flip before the (awaited, non-instant) pool close, so
        best-effort writers skip instead of racing a closing pool."""
        backend = PostgreSQLBackend()
        ready_during_close = None

        class _SlowClosingPool:
            async def close(self):
                nonlocal ready_during_close
                ready_during_close = backend.is_ready
                await asyncio.sleep(0)

        backend._pool = _SlowClosingPool()
        assert backend.is_ready is True
        await backend.shutdown()
        assert ready_during_close is False
        assert backend.is_ready is False

    @pytest.mark.asyncio
    async def test_init_callback_also_passed_as_setup(self):
        # asyncpg runs RESET ALL when a connection is released back to the pool,
        # which wipes the session GUCs the init callback SET. The same callback
        # must also be wired as setup= so it re-applies on every acquire.
        backend = PostgreSQLBackend()

        async def cb(conn):
            return None

        with patch(
            "hindsight_api.engine.db.postgresql.asyncpg.create_pool",
            new=AsyncMock(return_value=object()),
        ) as create_pool:
            await backend.initialize("postgresql://localhost/test", init_callback=cb)

        kwargs = create_pool.call_args.kwargs
        assert kwargs["init"] is cb
        assert kwargs["setup"] is cb


class _RecordingConnection:
    """Captures every statement, optionally failing the first (batched) one."""

    def __init__(
        self,
        fail_batched: bool = False,
        reject: str | None = None,
        reject_error: type[Exception] = asyncpg.exceptions.UndefinedObjectError,
    ) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self._fail_batched = fail_batched
        self._reject = reject
        self._reject_error = reject_error

    async def execute(self, query: str, *args) -> None:
        self.calls.append((query, args))
        if self._fail_batched and len(self.calls) == 1:
            raise self._reject_error("unrecognized configuration parameter")
        if self._reject is not None and self._reject in args:
            raise self._reject_error("unrecognized configuration parameter")


class TestApplySessionSettings:
    """The pool's setup callback runs on every acquire — it must be one round trip (#3499)."""

    @pytest.fixture(autouse=True)
    def _forget_rejected_settings(self):
        """The rejected-GUC memo is process-wide, so it must not leak between tests."""
        pg_backend._unsupported_settings.clear()
        yield
        pg_backend._unsupported_settings.clear()

    _SETTINGS = [
        ("hnsw.ef_search", "200"),
        ("statement_timeout", "600s"),
        ("pg_trgm.similarity_threshold", "0.3"),
    ]

    @pytest.mark.asyncio
    async def test_all_settings_applied_in_one_statement(self):
        conn = _RecordingConnection()

        await apply_session_settings(conn, self._SETTINGS)

        assert len(conn.calls) == 1, f"expected one round trip, got {len(conn.calls)}: {conn.calls}"
        query, args = conn.calls[0]
        assert query.startswith("SELECT set_config(")
        # Values are bound, not interpolated, and ordered name/value per setting.
        assert args == ("hnsw.ef_search", "200", "statement_timeout", "600s", "pg_trgm.similarity_threshold", "0.3")
        # `false` = session-scoped, so the GUC survives past the current transaction.
        assert ", false)" in query

    @pytest.mark.asyncio
    async def test_no_statement_when_nothing_to_set(self):
        conn = _RecordingConnection()
        await apply_session_settings(conn, [])
        assert conn.calls == []

    @pytest.mark.asyncio
    async def test_falls_back_to_individual_settings_when_batch_fails(self):
        # An extension GUC the cluster doesn't know fails the whole batched
        # statement; the rest must still be applied.
        conn = _RecordingConnection(fail_batched=True, reject="pg_trgm.similarity_threshold")

        await apply_session_settings(conn, self._SETTINGS)

        applied = [args[0] for query, args in conn.calls[1:]]
        assert applied == ["hnsw.ef_search", "statement_timeout", "pg_trgm.similarity_threshold"]
        # The rejected one raised and was skipped rather than aborting setup.
        assert len(conn.calls) == 1 + len(self._SETTINGS)

    @pytest.mark.asyncio
    async def test_a_setting_the_server_rejects_is_not_sent_again(self):
        """Otherwise every acquire re-pays a failed batch plus one statement per setting.

        Reached by any GUC the cluster does not define — pg_trgm when the extension is
        absent, or hnsw.iterative_scan on a pgvector older than 0.8, which reserves the
        "hnsw." prefix and so rejects it rather than accepting a placeholder.
        """
        conn = _RecordingConnection(fail_batched=True, reject="pg_trgm.similarity_threshold")
        await apply_session_settings(conn, self._SETTINGS)

        # Next acquire: one batched statement again, carrying only what the server took.
        conn = _RecordingConnection()
        await apply_session_settings(conn, self._SETTINGS)

        assert len(conn.calls) == 1
        _, args = conn.calls[0]
        assert "pg_trgm.similarity_threshold" not in args
        assert args == ("hnsw.ef_search", "200", "statement_timeout", "600s")

    @pytest.mark.asyncio
    async def test_invalid_name_rejection_is_remembered_like_undefined_object(self):
        """A reserved-prefix GUC rejection (InvalidNameError) must also be permanent.

        The docstring above assumes an old server answers "unrecognized configuration
        parameter" (UndefinedObjectError, 42704). PG16 + pgvector 0.6.0 answers
        `invalid configuration parameter name "hnsw.iterative_scan"` instead —
        InvalidNameError (42602), because the loaded extension reserves the "hnsw."
        prefix but predates the GUC. If only UndefinedObjectError is remembered,
        the name never reaches the unsupported-set, ``setting_rejected_by_server``
        keeps answering False, and retain's link probing sends the GUC via SET LOCAL
        inside its own transaction — aborting the whole link computation on every
        retain (observed in production on PG16 + pgvector 0.6.0, 2026-08-26).
        """
        conn = _RecordingConnection(
            fail_batched=True,
            reject="hnsw.ef_search",
            reject_error=asyncpg.exceptions.InvalidNameError,
        )
        await apply_session_settings(conn, self._SETTINGS)

        assert pg_backend.setting_rejected_by_server("hnsw.ef_search")

        # Next acquire must not re-send the rejected name.
        conn = _RecordingConnection()
        await apply_session_settings(conn, self._SETTINGS)
        assert len(conn.calls) == 1
        _, args = conn.calls[0]
        assert "hnsw.ef_search" not in args

    @pytest.mark.asyncio
    async def test_a_transient_failure_does_not_disable_a_setting(self):
        """Only "unrecognized configuration parameter" is permanent; anything else retries."""

        class _FlakyConnection(_RecordingConnection):
            async def execute(self, query: str, *args) -> None:
                self.calls.append((query, args))
                if len(self.calls) == 1:
                    raise asyncpg.exceptions.UndefinedObjectError("unrecognized configuration parameter")
                if "hnsw.ef_search" in args:
                    raise asyncpg.exceptions.DeadlockDetectedError("transient")

        await apply_session_settings(_FlakyConnection(), self._SETTINGS)

        conn = _RecordingConnection()
        await apply_session_settings(conn, self._SETTINGS)
        assert "hnsw.ef_search" in conn.calls[0][1]


# ---------------------------------------------------------------------------
# Config integration test
# ---------------------------------------------------------------------------


class TestConfig:
    def test_database_backend_field_exists(self):
        # Verify the field exists on the dataclass
        import dataclasses

        from hindsight_api.config import HindsightConfig

        field_names = {f.name for f in dataclasses.fields(HindsightConfig)}
        assert "database_backend" in field_names

    def test_default_database_backend(self):
        from hindsight_api.config import DEFAULT_DATABASE_BACKEND

        assert DEFAULT_DATABASE_BACKEND == "postgresql"


# ---------------------------------------------------------------------------
# Entity expansion CTE tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ops_module", "ops_class", "limit_clause"),
    [
        ("hindsight_api.engine.db.ops_postgresql", "PostgreSQLOps", "LIMIT 7"),
    ],
)
def test_entity_expansion_filters_fact_type_before_per_entity_cap(
    ops_module: str, ops_class: str, limit_clause: str
) -> None:
    """The cap is per entity *and target fact type*, preventing mixed types from
    exhausting a target type's candidate budget before the outer query sees it.
    """
    from importlib import import_module

    ops = getattr(import_module(ops_module), ops_class)()
    cte = ops.build_entity_expansion_cte("memory_units", "unit_entities", 7, _UNBOUNDED_WINDOW)

    lateral_start = cte.index("CROSS JOIN LATERAL")
    lateral_end = cte.index(") t", lateral_start)
    lateral_query = cte[lateral_start:lateral_end]

    assert "mu_target.fact_type = $2" in lateral_query
    assert lateral_query.index("mu_target.fact_type = $2") < lateral_query.index(limit_clause)


@pytest.mark.parametrize(
    "ops_module,ops_class,limit_clause",
    [
        ("hindsight_api.engine.db.ops_postgresql", "PostgreSQLOps", "LIMIT 7"),
    ],
)
def test_entity_expansion_applies_updated_window_before_per_entity_cap(
    ops_module: str, ops_class: str, limit_clause: str
) -> None:
    """Recall's time window bounds the entity fan-out, not just the graph seeds.

    Placement matters as much as presence: filtering after the cap would let
    out-of-window neighbours eat an entity's bounded budget and starve the
    in-window ones.
    """
    from importlib import import_module

    from hindsight_api.engine.db.ops import UpdatedWindow

    ops = getattr(import_module(ops_module), ops_class)()
    window = UpdatedWindow(after=datetime(2026, 1, 1), before=datetime(2026, 2, 1), first_param_index=4)
    cte = ops.build_entity_expansion_cte("memory_units", "unit_entities", 7, window)

    lateral_start = cte.index("CROSS JOIN LATERAL")
    lateral_query = cte[lateral_start : cte.index(") t", lateral_start)]

    assert "mu_target.updated_at > $4" in lateral_query
    assert "mu_target.updated_at < $5" in lateral_query
    assert lateral_query.index("mu_target.updated_at > $4") < lateral_query.index(limit_clause)


@pytest.mark.parametrize(
    "ops_module,ops_class",
    [
        ("hindsight_api.engine.db.ops_postgresql", "PostgreSQLOps"),
    ],
)
def test_semantic_causal_expansion_applies_updated_window(ops_module: str, ops_class: str) -> None:
    """All three link-expansion arms honour the window — both semantic directions
    (the kNN graph is not symmetric, so each is a separate scan) and causal."""
    from importlib import import_module

    from hindsight_api.engine.db.ops import UpdatedWindow

    ops = getattr(import_module(ops_module), ops_class)()
    window = UpdatedWindow(after=datetime(2026, 1, 1), before=None, first_param_index=4)
    cte = ops.build_semantic_causal_cte("memory_links", "memory_units", window)

    assert cte.count("mu.updated_at > $4") == 3
    assert "updated_at <" not in cte


@pytest.mark.parametrize(
    "ops_module,ops_class",
    [
        ("hindsight_api.engine.db.ops_postgresql", "PostgreSQLOps"),
    ],
)
def test_expansion_ctes_omit_window_when_unbounded(ops_module: str, ops_class: str) -> None:
    """An unbounded recall must emit the pre-existing SQL verbatim — no dangling
    placeholders, since every backend binds params positionally and Oracle rejects
    a query that references a bind it was not given."""
    from importlib import import_module

    ops = getattr(import_module(ops_module), ops_class)()

    entity_cte = ops.build_entity_expansion_cte("memory_units", "unit_entities", 7, _UNBOUNDED_WINDOW)
    sem_causal_cte = ops.build_semantic_causal_cte("memory_links", "memory_units", _UNBOUNDED_WINDOW)

    assert "updated_at" not in entity_cte
    assert "updated_at" not in sem_causal_cte
    assert "$4" not in entity_cte + sem_causal_cte


# ---------------------------------------------------------------------------
# PostgreSQL search_vector handling (insert). Since the curation archive drops
# search_vector (#2503), the insert is the single place it is populated, and
# pg_search_vector_expr is its one source of truth (shared with revert recompute).
# ---------------------------------------------------------------------------


class TestPostgreSQLSearchVector:
    @staticmethod
    def _cfg(ext: str, lang: str = "english"):
        from types import SimpleNamespace

        return SimpleNamespace(text_search_extension=ext, text_search_extension_native_language=lang)

    @pytest.mark.parametrize(
        "ext,needle",
        [
            ("native", "to_tsvector('english'::regconfig,"),
            ("vchord", "::bm25_catalog.bm25vector"),
        ],
    )
    def test_expr_builds_vector_for_vector_backends(self, ext, needle):
        from hindsight_api.engine.db.ops_postgresql import pg_search_vector_expr

        expr = pg_search_vector_expr(self._cfg(ext))
        assert expr is not None and needle in expr
        # Always built from the same three carried columns.
        assert "COALESCE(text, '')" in expr and "COALESCE(text_signals, '')" in expr

    @pytest.mark.parametrize("ext", ["pgroonga", "pg_textsearch", "pg_search"])
    def test_expr_none_for_base_column_backends(self, ext):
        from hindsight_api.engine.db.ops_postgresql import pg_search_vector_expr

        # These index the base text columns directly; search_vector stays empty.
        assert pg_search_vector_expr(self._cfg(ext)) is None

    def test_expr_accepts_custom_column_refs(self):
        from hindsight_api.engine.db.ops_postgresql import pg_search_vector_expr

        expr = pg_search_vector_expr(self._cfg("native"), text_col="mu.text", context_col="mu.context")
        assert "COALESCE(mu.text, '')" in expr and "COALESCE(mu.context, '')" in expr

    async def _insert_query(self, ext: str) -> str:
        from hindsight_api.engine.db.ops_postgresql import PostgreSQLOps

        conn = AsyncMock(spec=DatabaseConnection)
        conn.fetch = AsyncMock(return_value=[{"id": "00000000-0000-0000-0000-000000000001"}])
        batch = dict(
            bank_id="b",
            fact_texts=["t"],
            embeddings=["[0.1]"],
            event_dates=[None],
            occurred_starts=[None],
            occurred_ends=[None],
            mentioned_ats=[None],
            contexts=["c"],
            fact_types=["world"],
            metadata_jsons=["{}"],
            chunk_ids=[None],
            document_ids=[None],
            tags_list=[""],
            observation_scopes_list=[None],
            text_signals_list=[None],
            attachment_ids_list=["[]"],
        )
        with patch("hindsight_api.config.get_config", return_value=self._cfg(ext)):
            await PostgreSQLOps().insert_facts_batch(conn=conn, **batch)
        return conn.fetch.call_args.args[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ext", ["native", "vchord"])
    async def test_insert_includes_search_vector_column(self, ext):
        assert "search_vector" in await self._insert_query(ext)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ext", ["pgroonga", "pg_textsearch", "pg_search"])
    async def test_insert_omits_search_vector_column(self, ext):
        assert "search_vector" not in await self._insert_query(ext)


# ---------------------------------------------------------------------------
# normalize_schema tests
# ---------------------------------------------------------------------------


class TestNormalizeSchema:
    """Verify Backend.normalize_schema() returns correct schema for each backend."""

    def test_postgresql_passes_through(self):
        backend = PostgreSQLBackend()
        assert backend.normalize_schema("public") == "public"
        assert backend.normalize_schema("tenant_abc") == "tenant_abc"
        assert backend.normalize_schema(None) is None
