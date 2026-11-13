"""PostgreSQL database types and the shared operations object."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import DatabaseBackend, DatabaseConnection
from .result import ResultRow

__all__ = ["DatabaseBackend", "DatabaseConnection", "ResultRow", "get_postgres_ops"]

if TYPE_CHECKING:
    from .ops_postgresql import PostgreSQLOps

_ops: PostgreSQLOps | None = None


def get_postgres_ops() -> PostgreSQLOps:
    """Return the shared PostgreSQL operations object."""
    global _ops
    if _ops is None:
        from .ops_postgresql import PostgreSQLOps

        _ops = PostgreSQLOps()
    return _ops
