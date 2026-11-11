"""PostgreSQL migration dispatcher; omitted callbacks are historical no-ops."""

from collections.abc import Callable

from alembic import op


def run_for_dialect(*, pg: Callable[[], None] | None = None) -> None:
    name = op.get_bind().dialect.name
    if name != "postgresql":
        raise RuntimeError(f"Unsupported migration dialect: {name!r}")
    if pg is not None:
        pg()
