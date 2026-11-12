"""
Database migration management using Alembic.

This module provides programmatic access to run database migrations
on application startup. It is designed to be safe for concurrent
execution using PostgreSQL advisory locks to coordinate between
distributed workers.

Migrations target the configured local PostgreSQL schema.

Important: All migrations must be backward-compatible to allow
safe rolling deployments.

No alembic.ini required - all configuration is done programmatically.
"""

import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script.revision import ResolutionError
from alembic.util.exc import CommandError
from sqlalchemy import Connection, create_engine, text
from sqlalchemy.pool import NullPool

from ._pg_extensions import (
    create_extension,
    ensure_extensions_in_public,
    extension_schema,
    relocate_extension_to_public,
)
from ._text_search import mental_models_text_document
from ._vector_index import (
    bootstrap_extension,
    configured_vector_extension,
    detect_vector_extension,
    index_type_keyword,
    index_using_clause,
    minimum_rows_for_index,
    should_defer_index_creation,
    uses_per_bank_vector_indexes,
)
from .config import ENV_MIGRATION_DATABASE_URL, ENV_MIGRATION_ISOLATION, get_config
from .db_url import is_oracle_url, to_libpq_url
from .utils import mask_network_location

logger = logging.getLogger(__name__)

#: Set in the migration child's env so it does not spawn a child of its own.
_CHILD_MARKER = "_REMEMBER_MIGRATION_CHILD"

# Advisory lock ID for migrations (arbitrary unique number)
MIGRATION_LOCK_ID = 123456789

# Alembic's command.upgrade() is NOT thread-safe: it uses module-level global
# proxies (context._proxy, script) that get overwritten when two threads call
# upgrade() concurrently.  This causes migrations to target the wrong schema
# and crash with "relation already exists" or KeyError: 'script'.
# Serialize all Alembic invocations with a process-level lock.
_alembic_lock = threading.Lock()


def _set_alembic_main_option(config: Config, name: str, value: str) -> None:
    """Set an Alembic option without treating URL percent escapes as interpolation."""
    config.set_main_option(name, value.replace("%", "%%"))


def _detect_vector_extension(conn, vector_extension: str = "pgvector") -> str:
    """Validate the configured pgvector extension."""
    return detect_vector_extension(conn, vector_extension)


def _ensure_pgvector_extension_in_public(conn: Connection) -> None:
    """Ensure pgvector is installed in ``public`` before pgvector-backed migrations run."""
    logger.debug("Checking pgvector extension availability...")

    if extension_schema(conn, "vector") is None:
        logger.info("pgvector extension not found, attempting to install...")
        try:
            create_extension(conn, "vector")
            conn.commit()
            logger.info("pgvector extension installed in public schema")
        except Exception as e:
            # Installation failed - this is only fatal if the extension truly
            # doesn't exist; another process may have installed it meanwhile.
            conn.rollback()
            existing = extension_schema(conn, "vector")
            if not existing:
                logger.error(
                    f"pgvector extension is not installed and cannot be installed: {e}. "
                    f"Please ensure pgvector is installed by a database administrator. "
                    f"See: https://github.com/pgvector/pgvector#installation"
                )
                raise RuntimeError(
                    "pgvector extension is required but not installed. Please install it with: CREATE EXTENSION vector;"
                ) from e
            logger.warning(
                f"Could not install pgvector extension (permission denied?), "
                f"but extension exists in '{existing}' schema. Continuing..."
            )

    # Relocate an installation an older version (or an operator) put elsewhere.
    # ALTER EXTENSION ... SET SCHEMA carries its dependent objects along, unlike
    # the DROP ... CASCADE + CREATE this used to do, which took every embedding
    # column with it.
    relocate_extension_to_public(conn, "vector")


def _bootstrap_vector_extension_for_migrations(conn: Connection, vector_extension: str) -> None:
    """Bootstrap the configured vector backend before schema migrations run."""
    if vector_extension == "pgvector":
        _ensure_pgvector_extension_in_public(conn)
    bootstrap_extension(conn, vector_extension)
    # Repair anything an older version installed into a tenant schema, where the
    # runtime (which connects with the default search_path) cannot resolve it — the
    # pg_trgm case that made every retain fail silently in schema mode (#4118).
    ensure_extensions_in_public(conn)
    conn.commit()


def _vector_index_names(
    conn: Connection,
    schema_name: str,
    table_name: str,
    name_like: str | None = None,
    *,
    vector_access_methods_only: bool = True,
) -> list[str]:
    """Names of the vector indexes on ``table_name.embedding``, from the catalog.

    Deliberately NOT ``pg_indexes``: that view renders every row through
    ``pg_get_indexdef()``, which is evaluated for indexes outside the schema we
    asked about. When a concurrent session drops a schema mid-scan — pytest-xdist
    workers do exactly this — the render fails with "cache lookup failed for
    attribute N of relation OID" (an internal_error) and takes the whole
    statement with it. Resolving the relation first and reading ``pg_am`` keeps
    the scan inside one table's own indexes, so an unrelated schema going away
    cannot break it.
    """
    rows = conn.execute(
        text("""
            SELECT i.relname
            FROM pg_class t
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN pg_index x ON x.indrelid = t.oid
            JOIN pg_class i ON i.oid = x.indexrelid
            JOIN pg_am am ON am.oid = i.relam
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(x.indkey)
            WHERE n.nspname = :schema
              AND t.relname = :table
              AND a.attname = 'embedding'
              AND (NOT :vector_ams_only OR am.amname IN ('hnsw', 'vchordrq', 'diskann', 'scann'))
              AND (:name_like IS NULL OR i.relname LIKE :name_like)
        """),
        {
            "schema": schema_name,
            "table": table_name,
            "name_like": name_like,
            "vector_ams_only": vector_access_methods_only,
        },
    ).fetchall()
    return [row[0] for row in rows]


def _drop_index(conn: Connection, schema_name: str, index_name: str) -> None:
    """Drop one index. DDL identifiers cannot be bound parameters, so escape inline."""
    safe_schema = schema_name.replace('"', '""')
    safe_index = index_name.replace('"', '""')
    conn.execute(text(f'DROP INDEX IF EXISTS "{safe_schema}"."{safe_index}"'))


def _get_schema_lock_id(schema: str) -> int:
    """
    Generate a unique advisory lock ID for a schema.

    Uses hash of schema name to create a deterministic lock ID.
    """
    # Use hash to create a unique lock ID per schema
    # Keep within PostgreSQL's bigint range
    hash_bytes = hashlib.sha256(schema.encode()).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big") % (2**31)


#: How often to poll for the migration advisory lock.
_LOCK_POLL_INTERVAL_SECS = 0.5

#: How long a wait has to last before it is worth a log line, and how often to
#: repeat it afterwards.  Queuing briefly behind another worker is routine; only
#: a wait that outlasts this is a symptom.
_LOCK_WAIT_REPORT_AFTER_SECS = 5.0
_LOCK_WAIT_REPORT_INTERVAL_SECS = 30.0


def _advisory_lock_holder(conn: Connection, lock_id: int) -> str | None:
    """Describe who holds the migration advisory lock, or None if not found.

    Best effort: this runs while a migrator is stuck waiting, so any error
    here must not break the wait loop — an empty catalog row just means we
    could not identify the holder.  A failure leaves the transaction aborted,
    so the caller must end it before its next statement — the wait loop's
    ``conn.commit()`` does, which is why this runs before it.  A diagnostic
    must never turn a wait into a failed migration.

    ``pg_advisory_lock(bigint)`` stores the key as classid = high 32 bits,
    objid = low 32 bits, objsubid = 1; the two-argument form uses objsubid = 2.
    Matching objsubid = 1 therefore excludes a two-argument lock with a
    colliding objid, and classid = 0 (``lock_id`` is reduced mod 2**31, so its
    high word is always zero) excludes a 64-bit key that shares the low word.
    """
    try:
        row = conn.execute(
            text(
                "SELECT a.pid, coalesce(a.application_name, ''), coalesce(a.client_addr::text, ''), "
                "coalesce(a.query, '') "
                "FROM pg_locks l "
                "LEFT JOIN pg_stat_activity a ON a.pid = l.pid "
                "WHERE l.locktype = 'advisory' AND l.classid = 0 AND l.objid = :lock_id "
                "AND l.objsubid = 1 AND l.granted "
                "LIMIT 1"
            ),
            {"lock_id": lock_id},
        ).fetchone()
    except Exception as e:
        logger.debug("Could not inspect pg_locks for the migration advisory lock holder: %s", e)
        return None
    if not row:
        return None
    pid, app_name, client_addr, query = row[0], row[1] or "", row[2] or "", (row[3] or "").strip()
    parts = [f"pid={pid}"]
    if app_name:
        parts.append(f"app={app_name}")
    if client_addr:
        parts.append(f"client={client_addr}")
    if query:
        parts.append(f"query={query[:120]}")
    return ", ".join(parts)


def _run_migrations_internal(database_url: str, script_location: str, schema: str | None = None) -> None:
    """
    Internal function to run migrations without locking.

    Args:
        database_url: SQLAlchemy database URL
        script_location: Path to alembic scripts
        schema: Target schema (None for default/public)
    """
    schema_name = schema or "public"
    logger.info(f"Running database migrations to head for schema '{schema_name}'...")
    logger.info(f"Database URL: {mask_network_location(database_url)}")
    logger.info(f"Script location: {script_location}")

    # Create Alembic configuration programmatically (no alembic.ini needed)
    alembic_cfg = Config()

    # Set the script location (where alembic versions are stored)
    _set_alembic_main_option(alembic_cfg, "script_location", script_location)

    # Set the database URL
    _set_alembic_main_option(alembic_cfg, "sqlalchemy.url", database_url)

    # Configure logging (optional, but helps with debugging)
    # Uses Python's logging system instead of alembic.ini
    _set_alembic_main_option(alembic_cfg, "prepend_sys_path", ".")

    # Set path_separator to avoid deprecation warning
    _set_alembic_main_option(alembic_cfg, "path_separator", "os")

    # If targeting a specific schema, pass it to env.py via config
    # env.py will handle setting search_path and version_table_schema
    if schema:
        _set_alembic_main_option(alembic_cfg, "target_schema", schema)

    # Run migrations under a process-level lock.  Alembic uses module-level
    # global proxies that are not thread-safe, so concurrent command.upgrade()
    # calls from different threads corrupt each other's context.
    try:
        with _alembic_lock:
            command.upgrade(alembic_cfg, "heads")
    except (ResolutionError, CommandError) as e:
        # command.upgrade() wraps ResolutionError in CommandError via
        # ScriptDirectory._catch_revision_errors, so the wrapped form is what
        # actually reaches us; re-raise CommandErrors with any other cause.
        if isinstance(e, CommandError) and not isinstance(e.__cause__, ResolutionError):
            raise
        # This happens during rolling deployments when a newer version of the code
        # has already run migrations, and this older replica doesn't have the new
        # migration files. The database is already at a newer revision than we know.
        # This is safe to ignore - the newer code has already applied its migrations.
        logger.warning(
            f"Database is at a newer migration revision than this code version knows about. "
            f"This is expected during rolling deployments. Skipping migrations. Error: {e}"
        )
        return

    logger.info(f"Database migrations completed successfully for schema '{schema_name}'")


def _should_isolate_migrations() -> bool:
    """Whether to run the migration in a subprocess instead of in this process.

    Controlled by ``REMEMBER_API_MIGRATION_ISOLATION``:

        true    isolate — keeps alembic's import graph and its sync engine (psycopg2)
                out of a long-lived server process
        false   (default) never isolate; run in the calling process

    ``_CHILD_MARKER`` stops the child from recursing.
    """
    if os.environ.get(_CHILD_MARKER):
        return False
    return get_config().migration_isolation == "true"


def _run_in_migration_child(target: str, kwargs: dict) -> None:
    """Run the migration in a subprocess so this process never imports psycopg2.

    Alembic drives PostgreSQL through SQLAlchemy's sync engine, i.e. psycopg2, and a
    long-lived server has no other reason to carry that import graph and its thread
    pool for the rest of its life.

    The boundary is the whole migration entrypoint rather than each ``create_engine``
    call: schema migration also reaches ``ensure_embedding_dimension`` and the vector /
    text-search extension helpers, each of which opens its own sync engine. Isolating the
    entrypoint covers all of them in one child instead of one spawn apiece.

    The migration itself is short, rare and not on any hot path, so paying a process
    spawn for it is free.

    The payload travels over stdin so database credentials are not exposed in argv.
    The child inherits stdout and stderr so migration output remains visible.
    """
    payload = json.dumps({"target": target, "kwargs": kwargs})
    env = {
        **os.environ,
        ENV_MIGRATION_ISOLATION: "false",
        _CHILD_MARKER: "1",
    }
    logger.info("Running migrations in a subprocess (see %s)", ENV_MIGRATION_ISOLATION)
    result = subprocess.run(
        [sys.executable, "-m", "remember_api.migrations"],
        input=payload,
        env=env,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Migration subprocess failed (exit {result.returncode}); see the child's output above.")


def run_migrations(
    database_url: str,
    script_location: str | None = None,
    schema: str | None = None,
    migration_database_url: str | None = None,
) -> None:
    """
    Run database migrations to the latest version using programmatic Alembic configuration.

    This function is safe to call from multiple distributed workers simultaneously:
    - Uses PostgreSQL advisory lock to ensure only one worker runs migrations at a time
    - Other workers wait for the lock, then verify migrations are complete
    - If schema is already up-to-date, this is a fast no-op

    A configured schema can be used instead of public.

    Args:
        database_url: SQLAlchemy database URL (e.g., "postgresql://user:pass@host/db")
        script_location: Path to alembic migrations directory (e.g., "/path/to/alembic").
                        If None, defaults to the remember_api/alembic directory.
        schema: Target PostgreSQL schema name. If None, uses default (public).
                When specified, creates the schema if needed and runs migrations there.

    Raises:
        RuntimeError: If migrations fail to complete
        FileNotFoundError: If script_location doesn't exist

    Example:
        # Using default location and public schema
        run_migrations("postgresql://user:pass@host/db")

        # Run migrations for a configured local schema
        run_migrations("postgresql://user:pass@host/db", schema="remember")

        # Using custom location (when importing from another project)
        run_migrations(
            "postgresql://user:pass@host/db",
            script_location="/path/to/copied/_alembic"
        )
    """
    # Prefer a dedicated migration URL that bypasses connection poolers (e.g.
    # PgBouncer in transaction mode).  Session-level advisory locks don't
    # survive a PgBouncer transaction-mode cycle, so the distributed lock is
    # ineffective when the app URL goes through a pooler.  Configure
    # REMEMBER_API_MIGRATION_DATABASE_URL to the direct PostgreSQL endpoint
    # (e.g. remember-pg-rw) to restore correct locking behaviour.
    # When isolation is on, keep psycopg2 out of this process entirely.
    # ``_CHILD_MARKER`` stops the child from recursing.
    if _should_isolate_migrations():
        _run_in_migration_child(
            "run_migrations",
            {
                "database_url": database_url,
                "script_location": script_location,
                "schema": schema,
                "migration_database_url": migration_database_url,
            },
        )
        return

    raw_url = migration_database_url or database_url
    if is_oracle_url(raw_url):
        raise ValueError("Only PostgreSQL migrations are supported")
    migration_url = to_libpq_url(raw_url)

    try:
        # Determine script location
        if script_location is None:
            # Default: use the alembic directory inside the remember_api package
            # This file is in: remember_api/migrations.py
            # Alembic is in: remember_api/alembic/
            package_dir = Path(__file__).parent
            script_location = str(package_dir / "alembic")

        script_path = Path(script_location)
        if not script_path.exists():
            raise FileNotFoundError(
                f"Alembic script location not found at {script_location}. Database migrations cannot be run."
            )

        # Use schema-specific lock ID for multi-tenant isolation
        lock_id = _get_schema_lock_id(schema) if schema else MIGRATION_LOCK_ID
        schema_name = schema or "public"

        # Use PostgreSQL advisory lock to coordinate between distributed workers.
        #
        # IMPORTANT: We must avoid holding an open transaction on the advisory-lock
        # connection while CREATE INDEX CONCURRENTLY runs inside a migration.
        # CONCURRENTLY waits for ALL active transactions to finish before the index
        # becomes valid.  If the advisory-lock connection (or any waiting worker's
        # connection) holds an open transaction, CONCURRENTLY deadlocks:
        #   - migration worker waits for other workers' transactions to close
        #   - other workers wait for the advisory lock to be released
        #
        # Fix:
        #   1. Use pg_try_advisory_lock (non-blocking) in a poll loop instead of
        #      blocking pg_advisory_lock, so we can COMMIT the transaction between
        #      retries.  Between retries the connection holds no open transaction.
        #   2. After acquiring the lock, COMMIT the transaction on the advisory-lock
        #      connection itself before running migrations.  pg_advisory_lock is
        #      session-level, so the lock survives the COMMIT.
        # NullPool: do not retain the connection in a pool after the migration.
        # Each schema migration opens a few short-lived engines (here plus the
        # ensure_* steps); with the default QueuePool those connections linger
        # until GC, and running many schemas in parallel (migration_concurrency)
        # multiplies that footprint and exhausts max_connections — observed as
        # "FATAL: sorry, too many clients already" sweeping 20k schemas at
        # concurrency 12. NullPool closes the connection on return.
        engine = create_engine(migration_url, poolclass=NullPool)
        with engine.connect() as conn:
            logger.debug(f"Acquiring migration advisory lock for schema '{schema_name}' (id={lock_id})...")
            waited_since = time.monotonic()
            next_report_at = _LOCK_WAIT_REPORT_AFTER_SECS
            while True:
                acquired = conn.execute(text(f"SELECT pg_try_advisory_lock({lock_id})")).scalar()
                if acquired:
                    break
                # A worker stuck here used to look like a silent hang: no log
                # line, no timeout, for as long as the lock stays taken.  Say
                # who is holding it, so an operator can find the blocking
                # backend (and see the pooler-recycled-leak shape of #4611).
                #
                # Report once the wait passes _LOCK_WAIT_REPORT_AFTER_SECS and
                # then every _LOCK_WAIT_REPORT_INTERVAL_SECS: queuing briefly
                # behind another worker is routine during a rolling deploy, and
                # one line per poll buries the signal in exactly the incident
                # this is meant to explain.
                waited = time.monotonic() - waited_since
                if waited >= next_report_at:
                    next_report_at = waited + _LOCK_WAIT_REPORT_INTERVAL_SECS
                    # Look the holder up BEFORE the commit below: this statement
                    # autobegins a transaction, and running it after the commit
                    # would hold its snapshot across the sleep — the very thing
                    # the commit exists to prevent.
                    holder = _advisory_lock_holder(conn, lock_id)
                    logger.warning(
                        "Waiting for migration advisory lock (id=%s, schema=%s) held by %s; "
                        "waited %.0fs so far, polling every %ss. If the holder is a stale pooled "
                        "backend, point %s at the direct PostgreSQL endpoint to bypass the pooler.",
                        lock_id,
                        schema_name,
                        holder or "another session (holder not found in pg_locks)",
                        waited,
                        _LOCK_POLL_INTERVAL_SECS,
                        ENV_MIGRATION_DATABASE_URL,
                    )
                # Commit the transaction so this connection holds no open snapshot
                # while waiting.  This prevents blocking CREATE INDEX CONCURRENTLY
                # that may be running in the migration worker.
                conn.commit()
                time.sleep(_LOCK_POLL_INTERVAL_SECS)

            # Everything from here on is inside the try, so the lock is released
            # however we leave — the commit below included.
            try:
                # Commit AFTER acquiring the lock too.  pg_advisory_lock is
                # session-level and survives the COMMIT, but the open transaction
                # on this connection would otherwise block any CREATE INDEX
                # CONCURRENTLY in the migration.
                conn.commit()
                logger.debug("Migration advisory lock acquired")

                vector_extension = configured_vector_extension()
                _bootstrap_vector_extension_for_migrations(conn, vector_extension)

                # Commit any pending transaction on the advisory-lock connection
                # before running migrations.  Some code paths above (e.g., the
                # pgvector extension check) may have started a transaction via
                # SQLAlchemy's autobegin.  If we leave it open, CREATE INDEX
                # CONCURRENTLY inside a migration will deadlock waiting for it.
                conn.commit()

                # Run migrations while holding the lock
                _run_migrations_internal(migration_url, script_location, schema=schema)
            finally:
                # Release the lock even when the migration failed.  If anything
                # above failed, this connection's transaction is aborted and
                # pg_advisory_unlock would raise InFailedSqlTransaction — the
                # lock would stay on the backend (a pooled one never releases
                # it, wedging every later migrator: #4611).  Roll the aborted
                # transaction back first so the unlock can run, and never let a
                # failing unlock mask the original error.
                try:
                    conn.rollback()
                except Exception as rollback_error:
                    logger.warning(
                        "Could not roll back the migration connection before releasing the advisory lock: %s",
                        rollback_error,
                    )
                try:
                    conn.execute(text(f"SELECT pg_advisory_unlock({lock_id})"))
                    logger.debug("Migration advisory lock released")
                except Exception as unlock_error:
                    logger.error(
                        "Failed to release migration advisory lock (id=%s) — the lock may stay on the "
                        "backend until the connection closes; close pooled connections or recycle the "
                        "pooler backend manually: %s",
                        lock_id,
                        unlock_error,
                    )

    except FileNotFoundError:
        logger.error(f"Alembic script location not found at {script_location}")
        raise
    except SystemExit as e:
        # Catch sys.exit() calls from Alembic
        logger.error(f"Alembic called sys.exit() with code: {e.code}", exc_info=True)
        raise RuntimeError(f"Database migration failed with exit code {e.code}") from e
    except Exception as e:
        logger.error(f"Failed to run database migrations: {e}", exc_info=True)
        raise RuntimeError("Database migration failed") from e


def _drop_embedding_vector_indexes(conn: Connection, schema_name: str, table_name: str) -> None:
    """Drop every vector index on ``table_name.embedding`` (HNSW, DiskANN, vchordrq, ScaNN)."""
    for index_name in _vector_index_names(conn, schema_name, table_name):
        _drop_index(conn, schema_name, index_name)


def _migrate_table_embedding_dimension(
    conn: Connection,
    schema_name: str,
    table_name: str,
    required_dimension: int,
    vector_ext: str,
    *,
    indexed: bool = True,
) -> None:
    """
    Migrate the embedding column of a single table to the required dimension.

    - If dimensions match: no action needed
    - If dimensions differ and table is empty: ALTER COLUMN to new dimension
    - If dimensions differ and table has data: raise error with migration guidance

    ``indexed=False`` keeps the column but with no vector index at all: any existing one is
    dropped and none is created, so the pgvector 2000-dimension index limit does not apply.
    """
    current_dim = conn.execute(
        text("""
            SELECT atttypmod
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = :schema
              AND c.relname = :table
              AND a.attname = 'embedding'
        """),
        {"schema": schema_name, "table": table_name},
    ).scalar()

    if current_dim is None:
        logger.debug(f"No embedding column found on {table_name}, skipping")
        return

    if not indexed:
        # Also on the dimension-match path: the base migrations create this index, so a
        # deployment that switches to a custom store still carries one until it is dropped here.
        _drop_embedding_vector_indexes(conn, schema_name, table_name)
        conn.commit()

    if current_dim == required_dimension:
        logger.debug(f"Embedding dimension OK for {table_name}: {current_dim}")
        return

    logger.info(
        f"Embedding dimension mismatch on {table_name}: database has {current_dim}, model requires {required_dimension}"
    )

    row_count = conn.execute(
        text(f"SELECT COUNT(*) FROM {schema_name}.{table_name} WHERE embedding IS NOT NULL")
    ).scalar()

    if row_count > 0:
        raise RuntimeError(
            f"Cannot change embedding dimension from {current_dim} to {required_dimension}: "
            f"{table_name} table contains {row_count} rows with embeddings. "
            f"To change dimensions, you must either:\n"
            f"  1. Re-embed all data: DELETE FROM {schema_name}.{table_name}; then restart\n"
            f"  2. Use a model with {current_dim}-dimensional embeddings"
        )

    logger.info(f"Altering {table_name}.embedding column dimension from {current_dim} to {required_dimension}")

    _drop_embedding_vector_indexes(conn, schema_name, table_name)
    conn.execute(
        text(f"ALTER TABLE {schema_name}.{table_name} ALTER COLUMN embedding TYPE vector({required_dimension})")
    )
    conn.commit()

    if not indexed:
        logger.info(f"Changed {table_name}.embedding dimension to {required_dimension} (no vector index)")
        return

    _create_embedding_vector_index(conn, schema_name, table_name, required_dimension, vector_ext, row_count)
    logger.info(f"Successfully changed {table_name}.embedding dimension to {required_dimension}")


def _has_embedding_vector_index(conn: Connection, schema_name: str, table_name: str) -> bool:
    return bool(_vector_index_names(conn, schema_name, table_name))


def _create_embedding_vector_index(
    conn: Connection,
    schema_name: str,
    table_name: str,
    dimension: int,
    vector_ext: str,
    row_count: int,
) -> None:
    """Build the vector index on ``table_name.embedding`` for the detected extension."""
    if table_name == "memory_units" and uses_per_bank_vector_indexes(vector_ext):
        # Per-bank backends serve every memory_units vector search from the
        # idx_mu_emb_* partial indexes (owned by the maintenance sweep). A global
        # index here is never used and doubles the vector write cost (#5707).
        logger.debug(f"Per-bank vector backend ({vector_ext}); no global vector index on memory_units")
        return
    if vector_ext == "pgvector" and dimension > 2000:
        raise RuntimeError(
            f"Embedding dimension {dimension} exceeds pgvector HNSW index limit of 2000. "
            f"Use an embedding model with <= 2000 dimensions, or switch to a vector extension "
            f"that supports higher dimensions (e.g., pgvectorscale/DiskANN or AlloyDB ScaNN)."
        )

    index_type = index_type_keyword(vector_ext)
    if should_defer_index_creation(vector_ext, row_count):
        minimum_rows = minimum_rows_for_index(vector_ext)
        logger.warning(
            "Skipping %s index recreation on %s: AlloyDB ScaNN AUTO indexes need at least %s populated "
            "embedding rows; table currently has %s",
            vector_ext,
            table_name,
            minimum_rows,
            row_count,
        )
        return

    conn.execute(
        text(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding_{index_type}
            ON {schema_name}.{table_name}
            {index_using_clause(vector_ext)}
        """)
    )
    logger.info(f"Created {index_type} index on {table_name} for {dimension}-dimensional embeddings")
    conn.commit()


def ensure_embedding_dimension(
    database_url: str,
    required_dimension: int,
    schema: str | None = None,
    vector_extension: str = "pgvector",
) -> None:
    """
    Ensure the embedding column dimension matches the model's dimension for all tables.

    Checks and adjusts memory_units.embedding and mental_models.embedding:
    - If dimensions match: no action needed
    - If dimensions differ and table is empty: ALTER COLUMN to new dimension
    - If dimensions differ and table has data: raise error with migration guidance

    Args:
        database_url: SQLAlchemy database URL
        required_dimension: The embedding dimension required by the model
        schema: Target PostgreSQL schema name (None for public)
        vector_extension: Configured vector extension ("pgvector", "vchord", "pgvectorscale", or "scann")

    Raises:
        RuntimeError: If dimension mismatch with existing data
    """
    schema_name = schema or "public"

    engine = create_engine(to_libpq_url(database_url), poolclass=NullPool)
    with engine.connect() as conn:
        # Check if memory_units table exists (proxy for schema being initialized)
        table_exists = conn.execute(
            text("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = :schema AND table_name = 'memory_units'
                )
            """),
            {"schema": schema_name},
        ).scalar()

        if not table_exists:
            logger.debug(f"memory_units table does not exist in schema '{schema_name}', skipping dimension check")
            return

        # Detect which vector extension is available
        vector_ext = _detect_vector_extension(conn, vector_extension)
        logger.info(f"Using vector extension: {vector_ext}")

        _migrate_table_embedding_dimension(conn, schema_name, "memory_units", required_dimension, vector_ext)
        _migrate_table_embedding_dimension(
            conn, schema_name, "mental_models", required_dimension, vector_ext, indexed=True
        )
        if not _has_embedding_vector_index(conn, schema_name, "mental_models"):
            # A deployment that ran with a custom store and moved back to Postgres has the column at
            # the right dimension but no index (the store-owned branch above dropped it). Without
            # this the resize path is the only thing that ever builds it, and a matching dimension
            # never resizes — every page search would seq-scan until the model changed.
            row_count = conn.execute(
                text(f"SELECT COUNT(*) FROM {schema_name}.mental_models WHERE embedding IS NOT NULL")
            ).scalar()
            _create_embedding_vector_index(
                conn, schema_name, "mental_models", required_dimension, vector_ext, row_count
            )
        # NOTE: invalidated_memory_units is deliberately omitted. The curation archive has no
        # embedding column at all (dropped in migration d4f6a8c2e1b3) — invalidate stores no
        # embedding and revert recomputes one — so there is no archive vector to re-dimension
        # and a model switch can't trip a dimension mismatch there (#2209).


def ensure_vector_extension(
    database_url: str,
    vector_extension: str = "pgvector",
    schema: str | None = None,
) -> None:
    """
    Ensure the vector indexes match the configured vector extension.

    This function checks the current vector index type in the database
    and adjusts it if necessary:
    - If index type matches configured extension: no action needed
    - If they differ and tables are empty: drop old indexes, recreate with new type
    - If they differ and tables have data: raise error with migration guidance

    Args:
        database_url: SQLAlchemy database URL
        vector_extension: Configured vector extension (pgvector)
        schema: Target PostgreSQL schema name (None for public)

    Raises:
        RuntimeError: If extension mismatch with existing data
    """
    schema_name = schema or "public"

    engine = create_engine(to_libpq_url(database_url), poolclass=NullPool)
    with engine.connect() as conn:
        # Detect which vector extension should be used
        target_ext = _detect_vector_extension(conn, vector_extension)
        logger.info(f"Target vector extension: {target_ext}")

        # Tables with vector indexes to check
        tables_to_check = [
            ("memory_units", "idx_memory_units_embedding"),
            ("learnings", "idx_learnings_embedding"),
            ("pinned_reflections", "idx_pinned_reflections_embedding"),
        ]

        target_index_type = index_type_keyword(target_ext)

        mismatched_tables = []
        tables_with_data = []

        for table_name, index_name in tables_to_check:
            # Check if table exists
            table_exists = conn.execute(
                text("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = :schema AND table_name = :table_name
                    )
                """),
                {"schema": schema_name, "table_name": table_name},
            ).scalar()

            if not table_exists:
                logger.debug(f"Table {table_name} does not exist in schema '{schema_name}', skipping")
                continue

            row_count = conn.execute(
                text(f"SELECT COUNT(*) FROM {schema_name}.{table_name} WHERE embedding IS NOT NULL")
            ).scalar()

            # Check current index type. Read from the table's own indexes, not
            # pg_indexes: that view renders every index in the database, and a
            # schema dropped concurrently (tenant delete) fails the render — see
            # _vector_index_names.
            current_index_rows = conn.execute(
                text("""
                    SELECT pg_get_indexdef(i.oid), i.relname
                    FROM pg_class t
                    JOIN pg_namespace n ON n.oid = t.relnamespace
                    JOIN pg_index x ON x.indrelid = t.oid
                    JOIN pg_class i ON i.oid = x.indexrelid
                    WHERE n.nspname = :schema
                      AND t.relname = :table_name
                      AND i.relname LIKE :index_pattern
                """),
                {"schema": schema_name, "table_name": table_name, "index_pattern": "%embedding%"},
            ).fetchall()

            if table_name == "memory_units":
                # Per-bank backends never use a GLOBAL memory_units vector index.
                # Every vector search is bank + fact_type scoped, and is served
                # either by the bank's own partial index or — for a bank below
                # the size threshold, which is most of them — by an exact
                # (bank_id, fact_type) B-tree scan plus a top-N sort. The planner
                # never picks a global index when bank_id is in the WHERE clause,
                # which is exactly why migration d5e6f7a8b9c0 drops it for these
                # backends. The partial indexes themselves are owned by the
                # maintenance sweep (engine/vector_index_health.py), not by this
                # reconcile and not by bank creation.
                #
                # The reconcile is strictly hands-off here, in BOTH directions:
                # never create the index (dead weight — an older version of this
                # branch did, which is how legacy schemas ended up carrying it),
                # and never drop or rebuild one that exists. Runtime DROP/CREATE
                # INDEX takes an ACCESS EXCLUSIVE lock on memory_units at
                # unpredictable times (startup, tenant provisioning); index DDL
                # belongs in the versioned migration path. Leftover globals are
                # removed by migrations f2a6d8c4b1e9 and e8c1a4f7b2d9. This continue also keeps
                # memory_units out of the type-mismatch reconcile below, which
                # would otherwise recreate a global index with the new type on a
                # backend switch.
                if current_index_rows:
                    # Migrations are at head here, and they drop every global
                    # index on per-bank backends — so one that survived needs an
                    # operator, not a promise that a migration will handle it.
                    drops = "; ".join(
                        f'DROP INDEX CONCURRENTLY "{schema_name}"."{row[1]}"' for row in current_index_rows
                    )
                    logger.warning(
                        f"Unused global vector index on {schema_name}.memory_units with per-bank backend "
                        f"({target_ext}); it slows every write. Drop it with: {drops}"
                    )
                else:
                    logger.debug(
                        f"Per-bank vector backend ({target_ext}); skipping global {index_name} creation on {table_name}"
                    )
                continue

            current_index_info = current_index_rows[0] if current_index_rows else None
            if not current_index_info:
                logger.warning(f"No embedding index found for {table_name}, will create it if safe")
                mismatched_tables.append((table_name, index_name, None, row_count))
                continue

            indexdef = current_index_info[0].lower()
            if "hnsw" in indexdef:
                current_index_type = "hnsw"
            else:
                logger.warning(f"Unknown index type for {table_name}: {indexdef}")
                continue

            # Check if index type matches target
            if current_index_type != target_index_type:
                logger.info(
                    f"Index type mismatch on {table_name}: current={current_index_type}, target={target_index_type}"
                )
                mismatched_tables.append((table_name, index_name, current_index_type, row_count))

                if row_count > 0:
                    tables_with_data.append((table_name, row_count, current_index_type))
            else:
                logger.debug(f"Index type OK for {table_name}: {current_index_type}")

        # If no mismatches, we're done
        if not mismatched_tables:
            logger.debug(f"All vector indexes match configured extension: {target_ext}")
            return

        # Do not replace an incompatible populated index automatically.
        if tables_with_data:
            table_list = ", ".join([f"{table}({count} rows)" for table, count, _ in tables_with_data])
            current_index_type = tables_with_data[0][2]
            raise RuntimeError(
                f"Cannot change vector extension from {current_index_type} to {target_index_type}: "
                f"the following tables contain data: {table_list}. "
                f"To change vector extension, you must either:\n"
                f"  1. Re-embed all data: DELETE FROM {schema_name}.memory_units; "
                f"DELETE FROM {schema_name}.learnings; DELETE FROM {schema_name}.pinned_reflections; then restart\n"
                f"  2. Inspect the existing index before changing the database."
            )

        logger.info(f"Reconciling vector indexes for {target_ext}")

        for table_name, index_name, current_type, row_count in mismatched_tables:
            # Drop existing index if it exists
            if current_type:
                logger.info(f"Dropping {current_type} index on {table_name}")
                conn.execute(text(f"DROP INDEX IF EXISTS {schema_name}.{index_name}"))

            # Create new index with appropriate type
            if target_ext == "pgvector":
                # Check embedding dimension — pgvector HNSW indexes only support up to 2000 dims
                embed_dim = conn.execute(
                    text("""
                        SELECT atttypmod
                        FROM pg_attribute a
                        JOIN pg_class c ON a.attrelid = c.oid
                        JOIN pg_namespace n ON c.relnamespace = n.oid
                        WHERE n.nspname = :schema AND c.relname = :table_name AND a.attname = 'embedding'
                    """),
                    {"schema": schema_name, "table_name": table_name},
                ).scalar()

                if embed_dim and embed_dim > 2000:
                    raise RuntimeError(
                        f"Embedding dimension {embed_dim} on {table_name} exceeds pgvector HNSW index limit of 2000. "
                        f"Use an embedding model with <= 2000 dimensions."
                    )

            logger.info(f"Creating {target_index_type} index on {table_name}")
            conn.execute(
                text(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {schema_name}.{table_name}
                    {index_using_clause(target_ext)}
                """)
            )

        conn.commit()
        logger.info(f"Successfully reconciled vector indexes for {target_ext}")


def ensure_text_search_extension(
    database_url: str,
    text_search_extension: str = "native",
    schema: str | None = None,
) -> None:
    """Check the local native search columns and repair missing indexes.

    A populated table with an incompatible search column is left untouched:
    recreating it could discard search data that cannot be rebuilt here.
    """
    if text_search_extension != "native":
        raise ValueError("Only native PostgreSQL text search is supported")

    schema_name = schema or "public"
    engine = create_engine(to_libpq_url(database_url), poolclass=NullPool)
    with engine.connect() as conn:
        mismatched = []
        missing_indexes = []
        populated = []
        for table_name in ("memory_units", "mental_models"):
            exists = conn.execute(
                text("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = :schema AND table_name = :table_name
                    )
                """),
                {"schema": schema_name, "table_name": table_name},
            ).scalar()
            if not exists:
                continue
            column = conn.execute(
                text("""
                    SELECT data_type, udt_name FROM information_schema.columns
                    WHERE table_schema = :schema AND table_name = :table_name
                      AND column_name = 'search_vector'
                """),
                {"schema": schema_name, "table_name": table_name},
            ).fetchone()
            index = conn.execute(
                text("""
                    SELECT am.amname, pg_get_indexdef(i.oid)
                    FROM pg_class t
                    JOIN pg_namespace n ON n.oid = t.relnamespace
                    JOIN pg_index x ON x.indrelid = t.oid
                    JOIN pg_class i ON i.oid = x.indexrelid
                    JOIN pg_am am ON am.oid = i.relam
                    WHERE n.nspname = :schema AND t.relname = :table_name
                      AND i.relname = :index_name
                """),
                {"schema": schema_name, "table_name": table_name, "index_name": f"idx_{table_name}_text_search"},
            ).fetchone()
            column_type = column[1] if column else None
            index_type = index[0] if index else None
            if column_type == "tsvector" and index_type == "gin":
                continue
            if column_type == "tsvector" and index_type is None:
                missing_indexes.append(table_name)
                continue
            mismatched.append((table_name, column_type, index_type))
            row_count = conn.execute(text(f"SELECT COUNT(*) FROM {schema_name}.{table_name}")).scalar()
            if row_count:
                populated.append((table_name, row_count))

        if populated:
            table_list = ", ".join(f"{table}({count} rows)" for table, count in populated)
            raise RuntimeError(f"Cannot rebuild incompatible text search columns on populated tables: {table_list}")

        for table_name, column_type, index_type in mismatched:
            if index_type:
                conn.execute(text(f"DROP INDEX IF EXISTS {schema_name}.idx_{table_name}_text_search"))
            if column_type:
                conn.execute(text(f"ALTER TABLE {schema_name}.{table_name} DROP COLUMN IF EXISTS search_vector"))
            if table_name == "mental_models":
                conn.execute(
                    text(f"""
                    ALTER TABLE {schema_name}.{table_name}
                    ADD COLUMN IF NOT EXISTS search_vector tsvector
                    GENERATED ALWAYS AS (
                        to_tsvector('english', {mental_models_text_document()})
                    ) STORED
                """)
                )
            else:
                conn.execute(
                    text(f"ALTER TABLE {schema_name}.{table_name} ADD COLUMN IF NOT EXISTS search_vector tsvector")
                )
            missing_indexes.append(table_name)

        for table_name in missing_indexes:
            conn.execute(
                text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_text_search
                ON {schema_name}.{table_name} USING gin(search_vector)
            """)
            )
        if mismatched or missing_indexes:
            conn.commit()


def _migrate_one_schema_pg(
    database_url: str,
    schema: str,
    *,
    migration_database_url: str | None,
    embedding_dimension: int | None,
    vector_extension: str,
    text_search_extension: str,
    ensure_extensions: bool,
) -> str:
    """Run migrations and reconcile indexes for the configured local schema."""
    run_migrations(database_url, schema=schema, migration_database_url=migration_database_url)
    if embedding_dimension is not None:
        ensure_embedding_dimension(
            database_url,
            embedding_dimension,
            schema=schema,
            vector_extension=vector_extension,
        )
    if ensure_extensions:
        ensure_vector_extension(
            database_url,
            vector_extension=vector_extension,
            schema=schema,
        )
        ensure_text_search_extension(
            database_url,
            text_search_extension=text_search_extension,
            schema=schema,
        )
    return schema


def run_local_migrations(
    database_url: str,
    schema: str,
    *,
    migration_database_url: str | None = None,
    embedding_dimension: int | None = None,
    vector_extension: str = "pgvector",
    text_search_extension: str = "native",
    ensure_extensions: bool = True,
) -> None:
    """Migrate and reconcile the configured schema, optionally in a child process."""
    options = dict(
        database_url=database_url,
        schema=schema,
        migration_database_url=migration_database_url,
        embedding_dimension=embedding_dimension,
        vector_extension=vector_extension,
        text_search_extension=text_search_extension,
        ensure_extensions=ensure_extensions,
    )
    if _should_isolate_migrations():
        _run_in_migration_child("run_local_migrations", options)
        return
    _migrate_one_schema_pg(
        database_url,
        schema,
        migration_database_url=migration_database_url,
        embedding_dimension=embedding_dimension,
        vector_extension=vector_extension,
        text_search_extension=text_search_extension,
        ensure_extensions=ensure_extensions,
    )


def _main() -> None:
    """Entry point for the migration subprocess (see ``_run_in_migration_child``).

    Invoked as ``python -m remember_api.migrations`` with the JSON payload on stdin.
    Kept deliberately thin: it exists only so the psycopg2 import happens in a
    throwaway process, and it re-enters ``run_migrations`` with ``_CHILD_MARKER``
    set so the subprocess branch is skipped.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    payload = json.loads(sys.stdin.read())
    os.environ[_CHILD_MARKER] = "1"
    targets = {
        "run_migrations": run_migrations,
        "run_local_migrations": run_local_migrations,
    }
    target = payload["target"]
    if target not in targets:
        raise SystemExit(f"unknown migration target {target!r}; expected one of {sorted(targets)}")
    targets[target](**payload["kwargs"])


if __name__ == "__main__":
    _main()
