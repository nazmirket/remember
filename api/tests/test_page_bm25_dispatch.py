"""Native PostgreSQL text search for Pages and memory writes."""

from types import SimpleNamespace
from typing import Any

import pytest

from remember_api.engine.db.ops_postgresql import pg_search_vector_expr
from remember_api.engine.sql.postgresql import PageBm25Arm, page_bm25_arm

# Asserts the SQL text this dispatch emits -- the `$3::text` bind and the absence of
# `search_vector`. A store that owns the knowledge index emits no SQL at all, so there is nothing
# here for the assertions to read.
pytestmark = pytest.mark.memory_backend_incompatible


def _cfg(ext: str) -> SimpleNamespace:
    return SimpleNamespace(text_search_extension=ext, text_search_extension_native_language="english")


def _arm() -> PageBm25Arm:
    return page_bm25_arm(table_alias="mm", text_param="$3")


def test_native_uses_tsvector_operators():
    arm = _arm()
    # The mental_models tsvector is generated with the 'english' config, so the
    # query must use 'english' regardless of the configured native language.
    # Joining tokens with OR aligns candidate recall with memory-recall BM25;
    # precision is restored downstream via ts_rank_cd ranking and RRF fusion.
    assert "ts_rank_cd(mm.search_vector, to_tsquery('english', $3))" in arm.order_by
    assert arm.match_filter == "AND mm.search_vector @@ to_tsquery('english', $3)"


def test_text_param_and_alias_are_threaded_through():
    arm = page_bm25_arm(table_alias="kbm", text_param="$7")
    assert "ts_rank_cd(kbm.search_vector, to_tsquery('english', $7))" in arm.order_by
    assert "kbm.search_vector @@ to_tsquery('english', $7)" in arm.match_filter


# --- write side: search_vector tokenization for mental_models ---------------
# mental_models is a two-column (name + content) table whose native search_vector
# is a GENERATED column, so only vchord needs an inline write (native_inline=False).


def test_page_generated_vector_is_not_written_inline():
    assert (
        pg_search_vector_expr(_cfg("native"), text_col="$3", context_col="$5", signals_col=None, native_inline=False)
        is None
    )


def test_memory_units_default_expr_is_unchanged():
    assert pg_search_vector_expr(_cfg("native")) == (
        "to_tsvector('english'::regconfig, "
        "COALESCE(text, '') || ' ' || COALESCE(context, '') || ' ' || COALESCE(text_signals, ''))"
    )


# --- write side: the name bind must be cast --------------------------------


class _RecordingConn:
    """Captures the SQL of the one fetchrow the pinned-model insert issues."""

    def __init__(self) -> None:
        self.sql = ""

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any]:
        self.sql = sql
        return {}


async def _pinned_insert_sql(ext: str, monkeypatch) -> str:
    """The INSERT that ``_insert_pinned_mental_model`` builds for ``ext``.

    The method reads no attribute off ``self``, so an uninitialised instance is
    enough to reach the SQL without a database, an embedder, or an LLM.
    """
    from remember_api.engine import memory_engine as me

    cfg = _cfg(ext)
    cfg.database_schema = "public"  # fq_table() reads it off the same config
    monkeypatch.setattr(me, "get_config", lambda: cfg)
    conn = _RecordingConn()
    await me.MemoryEngine._insert_pinned_mental_model(
        object.__new__(me.MemoryEngine),
        conn,
        mental_model_id="mm-1",
        bank_id="b",
        name="page name",
        source_query="q",
        content="c",
        embedding=None,
        tags=[],
        max_tokens=None,
        trigger=None,
    )
    return conn.sql


async def test_pinned_insert_keeps_the_cast_without_inline_tokenization(monkeypatch):
    # Backends that leave search_vector alone bind $3 once, so the cast is inert
    # there — it stays for one statement shape across backends.
    sql = await _pinned_insert_sql("native", monkeypatch)
    assert "'pinned', $3::text," in sql
    assert "search_vector" not in sql
