"""Historical revision c8e2f4a6b1d3; no PostgreSQL schema changes."""

from collections.abc import Sequence

from remember_api.alembic._dialect import run_for_dialect

revision: str = "c8e2f4a6b1d3"
down_revision: str | Sequence[str] | None = "e5b1c7d3a902"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_upgrade() -> None:
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect()
