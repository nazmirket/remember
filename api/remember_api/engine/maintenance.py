"""Run recurring retention, consolidation, Page refresh and attachment maintenance.

Each job operates on the configured local PostgreSQL schema. Jobs run on their own
cadences; startup jitter and bounded deletes limit contention across processes.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from ..config import ConfigLike, get_config
from ..models import RequestContext
from ..schema_context import _current_schema
from .db_utils import acquire_with_retry
from .retain.attachment_store import ATTACHMENT_SWEEP_GRACE
from .schema import fq_table, fq_table_explicit

if TYPE_CHECKING:
    from .memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

# Short tick so jobs with different cadences share one loop without per-job tasks.
_TICK_SECONDS = 60
# How long stop() lets the job in progress reach a stopping point before cancelling it.
_STOP_GRACE_SECONDS = 5.0
# How long stop() then waits for the cancel to land. A job that suppresses or delays
# cancellation (a call that does not yield, a cleanup that catches CancelledError) is
# left behind after this rather than waited on.
_STOP_CANCEL_GRACE_SECONDS = 5.0
# ── retention sweep pacing ────────────────────────────────────────────────────
# Retention used to issue one unbounded `DELETE FROM <table> WHERE started_at <
# cutoff` per schema. On a table with a real backlog that is a single statement
# holding row locks for minutes while it reads the whole expired range — and the
# maintenance loop runs in every API/worker process with no leader election, so
# every pod issued it at the same hourly boundary. Observed as two concurrent
# 330s+ deletes pinned on IO.DataFileRead, blocking each other on row locks,
# saturating RDS I/O and tripling recall latency.
#
# The fix is to design the collision out rather than elect one sweeper: each chunk
# claims its rows with FOR UPDATE SKIP LOCKED, so concurrent sweepers take
# *disjoint* chunks instead of waiting on each other, and the total work stays the
# number of expired rows however many pods join in. Chunks are short, index-driven
# transactions with a pause between them, so no statement holds locks for long and
# the deletes never monopolise disk I/O.
_RETENTION_BATCH_SIZE = 2000
# Ceiling on chunks per table per schema per run: a backstop against looping
# forever on a table that is filling faster than it drains. A full run therefore
# removes at most 2M rows per schema, and the next sweep (whose cadence is
# REMEMBER_API_RETENTION_SWEEP_INTERVAL_SECONDS, default hourly) continues.
_RETENTION_MAX_BATCHES = 1000
# Breather between chunks. Paces one process at ~8k rows/s worst case; several
# pods sweeping at once multiply that, which is still orders of magnitude gentler
# than the unbounded delete this replaces.
_RETENTION_BATCH_PAUSE_SECONDS = 0.25
# ── attachment sweep ──────────────────────────────────────────────────────────
# A retain's attachments are stored before the retain runs; one its document
# still does not reference a day later — the document was never created, or an
# update to it failed — belongs to a retain that failed. Only the past
# week is checked, so a daily run asks about each attachment about seven times rather
# than about every attachment the bank ever held.
# An attachment orphaned before the job existed, or while every process was down
# (or its retain queued) for over a week, falls out of the window: that is what
# `remember-admin repair-bank --attachments` is for, the same rule with no
# lower bound.
_ATTACHMENT_SWEEP_LOOKBACK = timedelta(days=7)


@dataclass(frozen=True)
class _DueScheduledRefresh:
    """A discovery row whose schedule has fired; ``full`` when it is the full refresh's."""

    row: Any
    full: bool


@dataclass(frozen=True)
class _ReconcileTarget:
    """A bank the consolidation reconcile should reschedule, and the schema it lives in."""

    schema_name: str
    bank_id: str


class MaintenanceLoop:
    """Owns the single periodic maintenance task for a :class:`MemoryEngine`."""

    def __init__(self, engine: "MemoryEngine | None") -> None:
        """``engine`` is optional so the loop can be exercised for its scheduling alone — the
        cadence tests construct it that way. Every job that needs the engine checks first."""
        self._engine = engine
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        # Monotonic timestamps of the last run per job, keyed by job name.
        self._last_run: dict[str, float] = {}

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the loop if any maintenance job is enabled. Idempotent."""
        if self._task and not self._task.done():
            return
        # PostgreSQL-only: the retention sweeps target PG-only tables (audit_log,
        # llm_requests) and the reconcile relies on PG-only PL/pgSQL routines
        # installed by the maintenance-routines migration. Oracle support is
        # intentionally absent (mirrors that PG-only migration).
        if not self._any_job_enabled():
            logger.debug("Maintenance loop not started: no jobs enabled")
            return
        self._stop.clear()
        try:
            self._task = asyncio.create_task(self._run())
        except RuntimeError:
            logger.debug("Cannot start maintenance loop: no running event loop")

    async def stop(self) -> None:
        """Stop the loop: let the current job reach its next stopping point, then cancel it.

        A job can run for a long time (a scheduled-refresh pass over many models, a
        reconcile over many banks), and shutdown waits on this. Every job is safe to
        cut short: each unit of work is its own transaction, and whatever was not
        reached is still due on the next tick of the next process.
        """
        self._stop.set()
        task = self._task
        self._task = None
        if task is None or task.done():
            return
        _, pending = await asyncio.wait({task}, timeout=_STOP_GRACE_SECONDS)
        if pending:
            logger.info(f"Maintenance: cancelling the job in progress after {_STOP_GRACE_SECONDS:.0f}s to stop")
            task.cancel()
            _, pending = await asyncio.wait({task}, timeout=_STOP_CANCEL_GRACE_SECONDS)
            if pending:
                logger.warning(
                    f"Maintenance: the job in progress did not stop within {_STOP_CANCEL_GRACE_SECONDS:.0f}s "
                    "of being cancelled; leaving it behind"
                )
                return
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Maintenance tick failed while stopping")

    @staticmethod
    def _any_job_enabled() -> bool:
        cfg = get_config()
        reconcile_on = cfg.consolidation_reconcile_interval_seconds > 0
        # Not gated on audit_log_enabled: that is per-bank overridable, so rows
        # can exist even when the deployment default is off. Retention is driven
        # purely by the (server-level) window.
        sweep_on = cfg.retention_sweep_interval_seconds > 0
        audit_on = sweep_on and cfg.audit_log_retention_days > 0
        llm_on = sweep_on and cfg.llm_trace_enabled and cfg.llm_trace_retention_days > 0
        mm_refresh_on = cfg.mental_model_refresh_tick_seconds > 0
        op_cleanup_on = cfg.operation_cleanup_interval_seconds > 0 and cfg.operation_retention_days > 0
        attachment_sweep_on = cfg.attachment_sweep_interval_seconds > 0
        return reconcile_on or audit_on or llm_on or mm_refresh_on or op_cleanup_on or attachment_sweep_on

    # ── loop ───────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        if not await self._wait_start_jitter():
            return
        while not self._stop.is_set():
            try:
                await self._tick()
            except Exception:
                logger.exception("Maintenance tick failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=_TICK_SECONDS)
            except asyncio.TimeoutError:
                pass

    async def _wait_start_jitter(self) -> bool:
        """Delay the first tick by a random offset. Returns False if stopped while waiting.

        Every job is due the first time ``_is_due`` sees it, so N processes started
        together would run all of them at the same instant — the one moment where
        maintenance queries and overlapping DELETEs would otherwise collide.
        Spreading the *first* tick is enough: from then on each process keeps its
        own phase.
        """
        jitter = get_config().maintenance_start_jitter_seconds
        if jitter <= 0:
            return True
        delay = random.uniform(0, jitter)
        logger.debug(f"Maintenance loop: delaying first tick by {delay:.1f}s")
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=delay)
        except asyncio.TimeoutError:
            return True
        return False

    def _is_due(self, job: str, interval_seconds: int) -> bool:
        """True if ``job`` has never run or its interval has elapsed; marks it run now."""
        now = time.monotonic()
        last = self._last_run.get(job)
        if last is not None and (now - last) < interval_seconds:
            return False
        self._last_run[job] = now
        return True

    async def _tick(self) -> None:
        cfg = get_config()
        retention_interval = cfg.retention_sweep_interval_seconds
        if retention_interval > 0 and self._is_due("retention", retention_interval):
            await self._run_timed("retention", self._run_retention(cfg))
        interval = cfg.consolidation_reconcile_interval_seconds
        if interval > 0 and self._is_due("reconcile", interval):
            await self._run_timed("consolidation reconcile", self._run_reconcile())
        mm_interval = cfg.mental_model_refresh_tick_seconds
        if mm_interval > 0 and self._is_due("mm_refresh", mm_interval):
            await self._run_timed("scheduled mental model refresh", self._run_scheduled_mm_refresh())
        cleanup_interval = cfg.operation_cleanup_interval_seconds
        if (
            cleanup_interval > 0
            and cfg.operation_retention_days > 0
            and self._is_due("operation_cleanup", cleanup_interval)
        ):
            await self._run_timed("operation cleanup", self._run_operation_cleanup(cfg))
        attachment_interval = cfg.attachment_sweep_interval_seconds
        if attachment_interval > 0 and self._is_due("attachment_sweep", attachment_interval):
            await self._run_timed("attachment sweep", self._run_attachment_sweep())

    async def _run_timed(self, name: str, coro: Coroutine[Any, Any, None]) -> None:
        """Run a maintenance job and emit one timing line for it.

        Each job keeps its own summary log (counts of work done); this adds a
        single, uniform line per run so the cost of every sweep is observable.
        """
        start = time.monotonic()
        try:
            await coro
        finally:
            logger.info(f"Maintenance: {name} took {time.monotonic() - start:.3f}s")

    # ── retention ──────────────────────────────────────────────────────────

    async def _run_retention(self, cfg: ConfigLike) -> None:
        # Retention days are static server-level configuration.
        # Not gated on audit_log_enabled: it is per-bank overridable, so a bank
        # may be writing audit rows while the deployment default is off. Gating
        # the purge on the global flag would let those rows accumulate forever.
        if cfg.audit_log_retention_days > 0:
            await self._purge_expired("audit_log", "started_at", cfg.audit_log_retention_days)
        if cfg.llm_trace_enabled and cfg.llm_trace_retention_days > 0:
            await self._purge_expired("llm_requests", "started_at", cfg.llm_trace_retention_days)

    async def _purge_expired(self, table: str, ts_col: str, days: int) -> int:
        """Delete expired rows in bounded batches from the local schema."""
        schema = get_config().database_schema or "public"
        qschema = '"' + schema.replace('"', '""') + '"'
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        try:
            deleted = await self._purge_table_in_batches(f"{qschema}.{table}", ts_col, cutoff)
        except Exception as e:
            logger.warning(f"Retention sweep failed for {schema}.{table}: {e}")
            return 0
        if deleted:
            logger.info(f"Retention sweep {schema}.{table}: DELETE {deleted}")
        return deleted

    async def _purge_table_in_batches(self, table: str, ts_col: str, cutoff: datetime) -> int:
        """Delete expired rows from one qualified table in bounded chunks.

        Rows are claimed oldest-first off the ``(started_at)`` index with FOR UPDATE
        SKIP LOCKED, which is what makes a leaderless fleet safe: a chunk never
        waits on another sweeper (or on a writer still finishing its own row), and
        two processes sweeping the same table take disjoint chunks rather than
        redoing each other's work. Each chunk commits on its own, so no transaction
        holds locks longer than one batch.
        """
        backend = self._engine._backend
        deleted = 0
        for batch in range(_RETENTION_MAX_BATCHES):
            if self._stop.is_set():
                break
            if batch:
                await asyncio.sleep(_RETENTION_BATCH_PAUSE_SECONDS)
            async with acquire_with_retry(backend, max_retries=1) as conn, conn.transaction():
                removed = await conn.fetchval(
                    f"""
                    WITH expired AS (
                        SELECT id FROM {table}
                        WHERE {ts_col} < $1
                        ORDER BY {ts_col}
                        LIMIT $2
                        FOR UPDATE SKIP LOCKED
                    ), removed AS (
                        DELETE FROM {table} t USING expired e WHERE t.id = e.id RETURNING 1
                    )
                    SELECT count(*) FROM removed
                    """,
                    cutoff,
                    _RETENTION_BATCH_SIZE,
                )
            deleted += removed
            # A short chunk means the expired range is drained — or that another
            # sweeper holds the rest, which is equally a reason to stop.
            if removed < _RETENTION_BATCH_SIZE:
                break
        else:
            logger.warning(
                f"Retention sweep hit its per-run batch ceiling on {table} after {deleted} row(s); "
                "the remainder is left for the next run"
            )
        return deleted

    # ── terminal operation cleanup ─────────────────────────────────────────

    async def _run_operation_cleanup(self, cfg: ConfigLike) -> None:
        """Prune one bounded batch of expired local terminal operations."""
        schema = get_config().database_schema or "public"
        table = fq_table_explicit("async_operations", schema)
        cutoff = datetime.now(timezone.utc) - timedelta(days=cfg.operation_retention_days)
        try:
            backend = self._engine._backend
            async with acquire_with_retry(backend, max_retries=1) as conn:
                async with conn.transaction():
                    deleted = await backend.ops.prune_terminal_operations(
                        conn, table, cutoff, batch_size=cfg.operation_cleanup_batch_size
                    )
            if deleted:
                logger.info(f"Operation cleanup pruned {deleted} expired terminal operations from {schema}")
        except Exception as e:
            logger.warning(f"Operation cleanup failed for schema {schema}: {e}")

    # ── attachment sweep ─────────────────────────────────────────────────────

    async def _run_attachment_sweep(self) -> None:
        """Delete the attachments of retains that failed before writing their document.

        Scan banks with attachments in the local schema and remove unreferenced files.
        The delete is idempotent if two maintenance processes overlap.
        """
        engine = self._engine
        now = datetime.now(timezone.utc)
        created_before = now - ATTACHMENT_SWEEP_GRACE
        created_after = created_before - _ATTACHMENT_SWEEP_LOOKBACK
        try:
            async with acquire_with_retry(engine._backend, max_retries=1) as conn:
                schema = get_config().database_schema or "public"
                table = fq_table_explicit("attachments", schema)
                rows = await conn.fetch(
                    f"SELECT DISTINCT $1::text AS schema_name, bank_id FROM {table} "
                    "WHERE created_at >= $2 AND created_at < $3",
                    schema,
                    created_after,
                    created_before,
                )
        except Exception as e:
            logger.warning(f"Attachment sweep discovery failed: {e}")
            return
        if not rows:
            return
        removed = 0
        for row in rows:
            if self._stop.is_set():
                break
            schema, bank_id = row["schema_name"], row["bank_id"]
            token = _current_schema.set(schema)
            try:
                context = RequestContext(internal=True)
                removed += await engine.reclaim_unreferenced_attachments(
                    bank_id, created_after, created_before, context
                )
            except Exception as e:
                logger.warning(f"Attachment sweep failed for bank {bank_id} in {schema}: {e}")
            finally:
                _current_schema.reset(token)
        if removed:
            logger.info(f"Attachment sweep: deleted {removed} attachment(s) of retains that failed")

    # ── consolidation reconcile ──────────────────────────────────────────────

    async def _run_reconcile(self) -> None:
        """Re-schedule consolidation for banks with eligible-but-unscheduled facts."""
        engine = self._engine
        try:
            async with acquire_with_retry(engine._backend, max_retries=1) as conn:
                from .memories import sql_memories

                schema = get_config().database_schema or "public"
                rows = await sql_memories().banks_needing_consolidation(conn=conn, schema=schema)
        except Exception as e:
            logger.warning(f"Consolidation reconcile discovery failed: {e}")
            return

        if not rows:
            return

        targets = [_ReconcileTarget(schema_name=r["schema_name"], bank_id=r["bank_id"]) for r in rows]
        if not targets:
            return

        submitted = 0
        for target in targets:
            if self._stop.is_set():
                break
            schema = target.schema_name
            bank_id = target.bank_id
            token = _current_schema.set(schema)
            try:
                context = RequestContext(internal=True)
                resolved = await engine._config_resolver.resolve_full_config(bank_id, context)
                # Mirror the retain-time auto-consolidation gate (memory_engine): both
                # observations and auto-consolidation must be enabled for this bank.
                if not (resolved.enable_observations and resolved.enable_auto_consolidation):
                    continue
                await engine.submit_async_consolidation(bank_id=bank_id, request_context=context)
                submitted += 1
            except Exception as e:
                logger.warning(f"Consolidation reconcile failed for bank {bank_id} in {schema}: {e}")
            finally:
                _current_schema.reset(token)

        if submitted:
            logger.info(f"Consolidation reconcile: scheduled {submitted} bank(s)")

    # ── scheduled mental model refresh ───────────────────────────────────────

    async def _run_scheduled_mm_refresh(self) -> None:
        """Refresh mental models whose ``trigger.refresh_cron`` or ``full_refresh_cron`` is due.

        Discovery reads cron-scheduled models without an in-flight refresh from
        the local schema. Cron *due-ness* is evaluated here in
        Python — a scheduled fire has elapsed when the most recent cron boundary at
        or before now is later than ``last_refreshed_at`` — because cron arithmetic
        isn't expressible in plain SQL. Each due model is refreshed only when it is
        actually stale, so a schedule that fires while nothing changed costs a
        cheap staleness query, not an LLM call.

        ``full_refresh_cron`` is the same check against its own clock,
        ``last_full_refreshed_at`` (``last_refreshed_at`` moves on every delta, so a
        weekly full refresh keyed off it would never come due on a model that deltas
        daily). Its gate is different on purpose: not "stale" but "a refresh ran
        since the last full one" — a full refresh exists to clear the drift past
        delta edits left, which a quiet bank still carries. When both schedules are
        due the full refresh is the one submitted; it covers the other.
        """
        engine = self._engine
        try:
            async with acquire_with_retry(engine._backend, max_retries=1) as conn:
                schema = get_config().database_schema or "public"
                models = fq_table_explicit("mental_models", schema)
                operations = fq_table_explicit("async_operations", schema)
                rows = await conn.fetch(
                    f"""
                    SELECT $1::text AS schema_name, mm.bank_id::text, mm.id::text AS mental_model_id,
                           mm.trigger->>'refresh_cron' AS refresh_cron, mm.last_refreshed_at,
                           mm.trigger->>'full_refresh_cron' AS full_refresh_cron, mm.last_full_refreshed_at
                    FROM {models} mm
                    WHERE (COALESCE(mm.trigger->>'refresh_cron', '') <> ''
                        OR COALESCE(mm.trigger->>'full_refresh_cron', '') <> '')
                      AND NOT EXISTS (
                          SELECT 1 FROM {operations} o
                          WHERE o.bank_id = mm.bank_id AND o.operation_type = 'refresh_mental_model'
                            AND o.status IN ('pending', 'processing')
                            AND o.task_payload->>'mental_model_id' = mm.id::text
                      )
                    """,
                    schema,
                )
        except Exception as e:
            logger.warning(f"Scheduled mental model refresh discovery failed: {e}")
            return
        if not rows:
            return

        from croniter import croniter

        now = datetime.now(timezone.utc)

        def fired_since(cron: str | None, last: datetime | None, row: Any) -> bool:
            if not cron:
                return False
            try:
                prev_fire = croniter(cron, now).get_prev(datetime)
            except (ValueError, KeyError) as e:
                logger.warning(
                    f"Scheduled mental model refresh: skipping invalid cron {cron!r} for "
                    f"{row['schema_name']}/{row['mental_model_id']}: {e}"
                )
                return False
            return last is None or prev_fire > last

        due: list[_DueScheduledRefresh] = []
        for row in rows:
            last_full = row.get("last_full_refreshed_at")
            full_due = fired_since(row.get("full_refresh_cron"), last_full, row) and (
                last_full is None or row["last_refreshed_at"] is None or row["last_refreshed_at"] > last_full
            )
            if full_due or fired_since(row["refresh_cron"], row["last_refreshed_at"], row):
                due.append(_DueScheduledRefresh(row=row, full=full_due))
        if not due:
            return

        submitted = 0
        skipped_fresh = 0
        skipped_in_flight = 0
        skipped_paused = 0
        for item in due:
            if self._stop.is_set():
                break
            row = item.row
            schema = row["schema_name"]
            bank_id = row["bank_id"]
            mm_id = row["mental_model_id"]
            token = _current_schema.set(schema)
            try:
                context = RequestContext(internal=True)
                # Skip if nothing in the model's scope changed since its last
                # refresh — a scheduled refresh must not regenerate identical
                # content. compute_mental_model_is_stale needs the model's tags +
                # trigger, which the discovery routine doesn't return, so re-read
                # the row under the bank's schema context. A full refresh skips this
                # gate (its own was applied above — see the docstring).
                if not item.full:
                    async with acquire_with_retry(engine._backend, max_retries=1) as conn:
                        mm_row = await conn.fetchrow(
                            f"SELECT id, tags, trigger, last_refreshed_at, last_memory_seen_at "
                            f"FROM {fq_table('mental_models')} "
                            "WHERE bank_id = $1 AND id = $2",
                            bank_id,
                            mm_id,
                        )
                        if mm_row is None:
                            continue
                        is_stale = await engine.compute_mental_model_is_stale(conn, bank_id, mm_row)
                    if not is_stale:
                        skipped_fresh += 1
                        continue
                # skip_if_in_flight makes the enqueue itself idempotent. The discovery
                # routine already excludes models with a pending/processing refresh,
                # but that exclusion is a *read*: this loop runs in every process, so
                # every process saw the same "nothing in flight" snapshot and inserted
                # its own operation — one queued wave per process (#3210). The insert
                # now carries the check, so a second one is never created.
                result = await engine.submit_async_refresh_mental_model(
                    bank_id=bank_id,
                    mental_model_id=mm_id,
                    request_context=context,
                    skip_if_in_flight=True,
                    automatic=True,
                    mode="full" if item.full else None,
                )
                if result.get("paused"):
                    skipped_paused += 1
                elif result.get("deduplicated"):
                    skipped_in_flight += 1
                else:
                    submitted += 1
            except Exception as e:
                logger.warning(f"Scheduled mental model refresh failed for {mm_id} in {schema}: {e}")
            finally:
                _current_schema.reset(token)

        if submitted or skipped_fresh or skipped_in_flight or skipped_paused:
            logger.info(
                f"Scheduled mental model refresh: scheduled {submitted} model(s)"
                + (f", {skipped_fresh} up-to-date" if skipped_fresh else "")
                + (f", {skipped_in_flight} already in flight" if skipped_in_flight else "")
                + (f", {skipped_paused} paused after a failed refresh" if skipped_paused else "")
            )
