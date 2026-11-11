"""Historical revision d3f8a1c6e9b2; no PostgreSQL schema changes."""

from collections.abc import Sequence

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "d3f8a1c6e9b2"
down_revision: str | Sequence[str] | None = "c7e2f9a4d1b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_upgrade() -> None:
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect()
