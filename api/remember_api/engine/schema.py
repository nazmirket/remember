"""
Centralized schema-qualified table name helpers.

Single source of truth for producing ``"schema".table_name`` references
that respect both the active schema context and the database backend.
"""

from ..schema_context import get_current_schema

#: Tables accessed through the PostgreSQL memory store and its query helpers.
STORE_TABLES = frozenset(
    {
        "memory_units",
        "memory_links",
        "unit_entities",
        "documents",
        "chunks",
        "entities",
        "entity_cooccurrences",
        "invalidated_memory_units",
    }
)


class StoreTableAccessError(RuntimeError):
    """Code outside the memories store named a table the store owns."""


def _guard(table_name: str) -> None:
    if table_name in STORE_TABLES:
        raise StoreTableAccessError(
            f"{table_name!r} belongs to the memories store; go through get_memories() instead of SQL (#4969)"
        )


def fq_table(table_name: str) -> str:
    """Get fully-qualified table name using the current schema context.

    Prefix with the schema from :func:`remember_api.schema_context.get_current_schema`.
    """
    _guard(table_name)
    return fq_store_table(table_name)


def fq_store_table(table_name: str) -> str:
    """:func:`fq_table` without the guard. Only the Postgres memories store may call it."""
    return f"{get_current_schema()}.{table_name}"


def fq_table_explicit(table: str, schema: str | None = None) -> str:
    """Get fully-qualified table name with an explicit schema override.

    Used by modules that don't rely on the context-variable schema
    (e.g. task_backend, worker poller) and instead pass the schema
    explicitly.
    """
    _guard(table)
    return fq_store_table_explicit(table, schema)


def fq_store_table_explicit(table: str, schema: str | None = None) -> str:
    """:func:`fq_table_explicit` without the guard. Only the Postgres memories store may call it."""
    if schema:
        return f'"{schema}".{table}'
    return table
