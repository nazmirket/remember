"""SQL dialect abstraction layer.

Isolates database-specific SQL syntax (parameter placeholders, JSON operators,
vector distance functions, etc.) behind a common interface.

Usage:
    from hindsight_api.engine.sql import create_sql_dialect, SQLDialect

    dialect = create_sql_dialect("postgresql")
    placeholder = dialect.param(1)  # "$1"
"""

from .base import SQLDialect

__all__ = [
    "SQLDialect",
    "create_sql_dialect",
]


def create_sql_dialect(backend_type: str) -> SQLDialect:
    """Factory: create a SQLDialect by backend name.

    Args:
        backend_type: Currently "postgresql".

    Returns:
        A SQLDialect instance.

    Raises:
        ValueError: If backend_type is not recognized.
    """
    if backend_type == "postgresql":
        from .postgresql import PostgreSQLDialect

        return PostgreSQLDialect()
    raise ValueError(f"Unknown SQL dialect: {backend_type!r}. Supported dialects: 'postgresql'.")
