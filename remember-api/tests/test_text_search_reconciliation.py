"""Regression coverage for runtime text-search reconciliation.

``ensure_text_search_extension`` issues DDL against a live database at startup,
so these tests drive it through a fake connection that records every statement.
The assertions are about *which* statements are emitted (and that they are all
re-executable), not about a real schema — the shapes themselves are covered by
the migration suites.
"""

import re
from contextlib import contextmanager
from dataclasses import dataclass

import pytest

from remember_api import migrations
from remember_api._text_search import mental_models_text_document

_COUNT_TABLE = re.compile(r"SELECT COUNT\(\*\) FROM \S+\.(\w+)")
# Statements that change the schema; each one must be safely re-executable
# because replicas boot concurrently and all run this reconciliation.
_DDL_PREFIXES = ("ALTER", "CREATE", "DROP")


class _Result:
    def __init__(self, *, scalar=None, row=None):
        self._scalar = scalar
        self._row = row

    def scalar(self):
        return self._scalar

    def fetchone(self):
        return self._row


@dataclass(frozen=True)
class _TableState:
    column: str | None
    index: str | None
    rows: int
    indexdef: str = ""


class _Connection:
    def __init__(self, tables: dict[str, _TableState]):
        self.tables = tables
        self.statements: list[str] = []
        self.table_checks: list[str] = []
        self.commits = 0

    def execute(self, statement, params=None):
        sql = str(statement)
        self.statements.append(sql)
        if "information_schema.tables" in sql:
            table_name = params["table_name"]
            self.table_checks.append(table_name)
            return _Result(scalar=table_name in self.tables)
        if "information_schema.columns" in sql:
            column_type = self.tables[params["table_name"]].column
            return _Result(row=("USER-DEFINED", column_type) if column_type else None)
        if "pg_get_indexdef" in sql:
            table = self.tables[params["table_name"]]
            return _Result(row=(table.index, table.indexdef) if table.index else None)
        count_target = _COUNT_TABLE.search(" ".join(sql.split()))
        if count_target:
            return _Result(scalar=self.tables[count_target.group(1)].rows)
        return _Result()

    def commit(self):
        self.commits += 1

    @property
    def ddl(self) -> list[str]:
        normalized = (" ".join(s.split()) for s in self.statements)
        return [s for s in normalized if s.startswith(_DDL_PREFIXES)]


class _Engine:
    def __init__(self, conn):
        self.conn = conn

    @contextmanager
    def connect(self):
        yield self.conn


def _connect(monkeypatch, tables: dict[str, _TableState]) -> _Connection:
    conn = _Connection(tables)
    monkeypatch.setattr(migrations, "create_engine", lambda *args, **kwargs: _Engine(conn))
    return conn


def _run(monkeypatch, tables: dict[str, _TableState], extension="native") -> _Connection:
    conn = _connect(monkeypatch, tables)
    migrations.ensure_text_search_extension("postgresql://unused", text_search_extension=extension)
    return conn


def _assert_no_schema_changes(conn: _Connection) -> None:
    assert conn.ddl == []
    assert conn.commits == 0


def test_populated_native_state_is_idempotent(monkeypatch):
    conn = _run(
        monkeypatch,
        {
            "memory_units": _TableState(column="tsvector", index="gin", rows=20),
            "mental_models": _TableState(column="tsvector", index="gin", rows=5),
        },
    )
    assert conn.table_checks == ["memory_units", "mental_models"]
    _assert_no_schema_changes(conn)


@pytest.mark.parametrize("column,index", [("text", "pgroonga"), (None, None), ("tsvector", "bm25")])
def test_populated_incompatible_search_column_is_left_untouched(monkeypatch, column, index):
    conn = _connect(
        monkeypatch,
        {
            "memory_units": _TableState(column=column, index=index, rows=20),
            "mental_models": _TableState(column="tsvector", index="gin", rows=5),
        },
    )
    with pytest.raises(RuntimeError, match=r"memory_units\(20 rows\)"):
        migrations.ensure_text_search_extension("postgresql://unused")
    _assert_no_schema_changes(conn)


def test_unsupported_search_extension_is_rejected_before_connecting(monkeypatch):
    monkeypatch.setattr(migrations, "create_engine", lambda *args, **kwargs: pytest.fail("unexpected connection"))
    with pytest.raises(ValueError, match="Only native"):
        migrations.ensure_text_search_extension("postgresql://unused", text_search_extension="pgroonga")


def test_empty_mental_models_native_reconcile_creates_generated_projection(monkeypatch):
    """No write path fills mental_models.search_vector for native, so the column
    has to generate itself (see pg_search_vector_expr's native_inline=False)."""
    conn = _run(
        monkeypatch,
        {
            "memory_units": _TableState(column="tsvector", index="gin", rows=0),
            "mental_models": _TableState(column="text", index="pgroonga", rows=0),
        },
        extension="native",
    )

    assert (
        "ALTER TABLE public.mental_models ADD COLUMN IF NOT EXISTS search_vector tsvector "
        f"GENERATED ALWAYS AS ( to_tsvector('english', {mental_models_text_document()}) ) STORED" in conn.ddl
    )
    assert conn.commits == 1


def test_empty_memory_units_native_reconcile_creates_plain_column(monkeypatch):
    conn = _run(
        monkeypatch,
        {
            "memory_units": _TableState(column="text", index="pgroonga", rows=0),
            "mental_models": _TableState(column="tsvector", index="gin", rows=0),
        },
        extension="native",
    )

    assert "ALTER TABLE public.memory_units ADD COLUMN IF NOT EXISTS search_vector tsvector" in conn.ddl


def test_reconcile_ddl_is_re_executable(monkeypatch):
    """Replicas boot concurrently during a rolling restart and each runs this
    reconciliation, so the loser of the race must not crash on DDL the winner
    already committed."""
    conn = _run(
        monkeypatch,
        {
            "memory_units": _TableState(column=None, index=None, rows=0),
            "mental_models": _TableState(column=None, index=None, rows=0),
        },
    )

    assert conn.ddl, "expected reconciliation to emit DDL"
    for statement in conn.ddl:
        assert re.match(
            r"(CREATE (INDEX|EXTENSION) IF NOT EXISTS|DROP INDEX IF EXISTS"
            r"|ALTER TABLE \S+ (ADD|DROP) COLUMN IF (NOT )?EXISTS)",
            statement,
        ), f"non re-executable DDL: {statement}"


def test_populated_mental_models_missing_index_is_rebuilt_in_place(monkeypatch):
    """What a deployment leaves behind when it moves from a custom store back to Postgres: the
    column is in shape and populated, only the index is gone. Rebuilding it needs no backfill, so
    it must neither refuse (as a backend switch would) nor drop the column."""
    conn = _run(
        monkeypatch,
        {
            "memory_units": _TableState(column="tsvector", index="gin", rows=20),
            "mental_models": _TableState(column="tsvector", index=None, rows=5),
        },
        extension="native",
    )

    assert conn.ddl == [
        "CREATE INDEX IF NOT EXISTS idx_mental_models_text_search ON public.mental_models USING gin(search_vector)"
    ]
    assert conn.commits == 1
