"""Add graph_maintenance_queue table

Queue of memory_units whose outgoing temporal/semantic links lost a
neighbour to a delete. Drained by the async graph_maintenance worker,
which tops the unit's links back up using the same probes retain runs.

The queue only targets the link-recompute pass. The worker also runs
bank-wide sweeps (orphan-entity prune, stale-cooccurrence prune) on each
invocation; those don't need per-target queueing.

Revision ID: b5a4c3e2f1d8
Revises: e9b2c7d1f3a4
Create Date: 2026-05-27
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "b5a4c3e2f1d8"
down_revision: str | Sequence[str] | None = "e9b2c7d1f3a4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    # Composite PK gives us natural ON CONFLICT DO NOTHING dedup when the same
    # unit is enqueued from overlapping deletes. No FK to memory_units: if the
    # unit is deleted between enqueue and drain, the worker observes it's gone
    # and skips — a cascade would erase the work order, but that work has
    # already been satisfied (no surviving row to maintain).
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}graph_maintenance_queue (
            bank_id     TEXT NOT NULL,
            unit_id     UUID NOT NULL,
            enqueued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (bank_id, unit_id)
        )
        """
    )
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_graph_maintenance_queue_bank_enqueued
        ON {schema}graph_maintenance_queue (bank_id, enqueued_at)
        """
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_graph_maintenance_queue_bank_enqueued")
    op.execute(f"DROP TABLE IF EXISTS {schema}graph_maintenance_queue")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
