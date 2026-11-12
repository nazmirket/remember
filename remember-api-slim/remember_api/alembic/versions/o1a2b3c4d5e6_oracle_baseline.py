"""Historical revision o1a2b3c4d5e6; no PostgreSQL schema changes."""

from collections.abc import Sequence

from remember_api.alembic._dialect import run_for_dialect

revision: str = "o1a2b3c4d5e6"
down_revision: str | Sequence[str] | None = "k6l7m8n9o0p1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_upgrade() -> None:
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect()
