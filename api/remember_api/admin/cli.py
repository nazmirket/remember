"""PostgreSQL-only admin utilities (backup, restore, migration, worker management).

Not supported on Oracle backends. Uses asyncpg.connect() directly, binary COPY,
and TRUNCATE CASCADE — all inherently PG-specific.
"""

import asyncio
import io
import json
import logging
import struct
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg
import typer

from ..config import DEFAULT_DATABASE_SCHEMA, RememberConfig, load_dotenv_for_entrypoint
from ..db_url import is_oracle_url
from ..engine.memories import get_memories, sql_memories
from ..engine.retain.attachment_store import ATTACHMENT_SWEEP_GRACE, reclaim_unreferenced_attachments
from ..engine.retain.bank_utils import _vector_index_clause
from ..engine.schema import fq_table_explicit as _fq_table
from ..engine.storage import PostgreSQLFileStorage, bank_storage_prefix
from ..engine.vector_index_health import (
    BankIndexResult,
    drop_orphaned_bank_indexes,
    list_bank_ids,
    reconcile_bank_vector_indexes,
)
from ..pg0 import parse_pg0_url, resolve_database_url
from ..schema_context import _current_schema
from .reembed import SchemaReembed, finish_schema, reembed_schema

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(name="remember-admin", help="Remember administrative commands")

# Tables to backup/restore in foreign-key dependency order (parents first).
# Restore COPYs in this order and TRUNCATEs in reverse, so every child must
# appear after the tables it references.
#
# This must cover EVERY persistent PostgreSQL table in the schema — a missing
# entry silently drops that table's data on restore (and, worse, restore's
# `TRUNCATE banks CASCADE` wipes any FK-to-banks child like mental_models even
# when it was never backed up). test_admin_backup_restore.py asserts this list
# equals the live schema's tables, so adding a migration that creates a table
# without adding it here fails CI. Oracle-only tables (e.g. observation_sources)
# are intentionally absent — admin backup/restore is PostgreSQL-only.
BACKUP_TABLES = [
    "banks",
    # After "banks" for the same reason as "attachments" below: it FKs to it.
    "bank_aliases",
    # After "banks": attachments references it, so restore's forward COPY needs
    # the parent present, and the reversed TRUNCATE must clear the child first.
    "attachments",
    "documents",
    "entities",
    "chunks",
    "memory_units",
    "invalidated_memory_units",
    "unit_entities",
    "entity_cooccurrences",
    "memory_links",
    "observation_history",
    "mental_models",
    "mental_model_history",
    "knowledge_pages",
    "directives",
    "async_operations",
    "webhooks",
    "file_storage",
    # Knowledge banks: documents FK to banks, passages FK to documents.
    "kb_documents",
    "kb_passages",
    "kb_passage_vectors",
    "audit_log",
    "llm_requests",
    "graph_maintenance_queue",
    "entity_maintenance_queue",
]

MANIFEST_VERSION = "2"


@dataclass(frozen=True)
class BackupColumn:
    """A PostgreSQL column shape required to decode a binary COPY stream."""

    name: str
    type_name: str


@dataclass(frozen=True)
class TableRestorePlan:
    """How one table's backed-up binary COPY stream is replayed onto the target.

    ``columns`` is the target column list handed to ``copy_to_table``, in stream
    order. When the target no longer has a backed-up column, its field is stripped
    from every tuple (``dropped_field_indices``) before the stream is replayed —
    binary COPY is positional, so the column list and the tuple fields must agree.
    """

    columns: list[str]
    dropped_field_indices: tuple[int, ...]
    source_field_count: int


# Header of a PostgreSQL binary COPY stream: an 11-byte signature, an int32 flags
# field, and an int32 header-extension length followed by that many bytes.
_COPY_BINARY_SIGNATURE = b"PGCOPY\n\xff\r\n\x00"
_COPY_BINARY_HEADER_LEN = len(_COPY_BINARY_SIGNATURE) + 8


def _strip_binary_copy_fields(data: bytes, plan: TableRestorePlan) -> bytes:
    """Drop `plan.dropped_field_indices` from every tuple of a binary COPY stream.

    Restore used to reject a backup whose columns the target no longer had — the
    preflight raised "target is missing backup columns …", which made any backup
    taken before a column-dropping migration unrestorable afterwards. Those columns
    are now ignored instead, but they cannot simply be left out of the
    ``copy_to_table`` column list: binary COPY carries no column identities, so each
    tuple's fields are matched to the column list purely by position and an unedited
    stream would desynchronise (or, worse, land values in the wrong columns). So the
    stream itself is rewritten here.

    Tuple format: int16 field count, then per field an int32 length (-1 for NULL)
    followed by that many bytes. An int16 of -1 is the end-of-data trailer.
    """
    if not plan.dropped_field_indices:
        return data
    if not data.startswith(_COPY_BINARY_SIGNATURE):
        raise ValueError("Backup stream is not in PostgreSQL binary COPY format")

    (extension_len,) = struct.unpack_from("!i", data, len(_COPY_BINARY_SIGNATURE) + 4)
    pos = _COPY_BINARY_HEADER_LEN + extension_len
    out = bytearray(data[:pos])

    dropped = set(plan.dropped_field_indices)
    kept_count = plan.source_field_count - len(dropped)
    while True:
        (field_count,) = struct.unpack_from("!h", data, pos)
        pos += 2
        if field_count == -1:  # end-of-data trailer
            out += struct.pack("!h", -1)
            break
        if field_count != plan.source_field_count:
            raise ValueError(
                f"Backup stream tuple has {field_count} fields, manifest declares {plan.source_field_count}"
            )
        out += struct.pack("!h", kept_count)
        for index in range(field_count):
            (length,) = struct.unpack_from("!i", data, pos)
            pos += 4
            payload = b"" if length == -1 else data[pos : pos + length]
            pos += max(length, 0)
            if index in dropped:
                continue
            out += struct.pack("!i", length)
            out += payload
    return bytes(out)


async def _table_columns(conn: asyncpg.Connection, schema: str, table: str) -> list[BackupColumn]:
    rows = await conn.fetch(
        """
        SELECT a.attname AS name, pg_catalog.format_type(a.atttypid, a.atttypmod) AS type_name
        FROM pg_catalog.pg_attribute AS a
        JOIN pg_catalog.pg_class AS c ON c.oid = a.attrelid
        JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
        WHERE n.nspname = $1 AND c.relname = $2 AND a.attnum > 0 AND NOT a.attisdropped
          AND a.attgenerated = ''
        ORDER BY a.attnum
        """,
        schema,
        table,
    )
    return [BackupColumn(name=row["name"], type_name=row["type_name"]) for row in rows]


async def _validate_restore_schema(
    conn: asyncpg.Connection, manifest: dict[str, Any], schema: str
) -> dict[str, TableRestorePlan]:
    """Validate every COPY stream against the target before destructive work starts.

    A column the target no longer has is **not** an error: a migration that drops a
    column would otherwise make every backup taken before it permanently
    unrestorable. Such columns are skipped (their fields are stripped from the
    stream by ``_strip_binary_copy_fields``) and reported, so the operator sees what
    was discarded instead of the restore failing outright.

    Type mismatches remain fatal. Type equality is an exact ``format_type`` string
    match. This is deliberately stricter than binary-COPY wire compatibility (e.g.
    ``varchar`` and ``text`` share a binary format yet compare unequal here): we
    would rather fail a genuinely-restorable backup with a clear, actionable error
    than silently risk a subtle binary mismatch. Restores blocked this way can be
    recovered by aligning the target schema.
    """
    plans: dict[str, TableRestorePlan] = {}
    errors: list[str] = []
    for table, table_manifest in manifest["tables"].items():
        source_columns = [BackupColumn(**column) for column in table_manifest["columns"]]
        target_by_name = {column.name: column for column in await _table_columns(conn, schema, table)}
        unknown = [
            (index, column.name) for index, column in enumerate(source_columns) if column.name not in target_by_name
        ]
        mismatched = [
            f"{column.name} ({column.type_name} in backup, {target_by_name[column.name].type_name} in target)"
            for column in source_columns
            if column.name in target_by_name and target_by_name[column.name].type_name != column.type_name
        ]
        if mismatched:
            errors.append(f"{table}: incompatible column types: {', '.join(mismatched)}")
        if unknown:
            typer.echo(
                f"  {table}: ignoring {len(unknown)} backup column(s) absent from the target schema: "
                f"{', '.join(name for _, name in unknown)}"
            )
        plans[table] = TableRestorePlan(
            columns=[column.name for column in source_columns if column.name in target_by_name],
            dropped_field_indices=tuple(index for index, _ in unknown),
            source_field_count=len(source_columns),
        )

    if errors:
        details = "; ".join(errors)
        raise ValueError(f"Backup schema is incompatible with target schema '{schema}': {details}")
    return plans


def _quote_identifier(identifier: str) -> str:
    """Quote a PostgreSQL identifier for catalog-derived dynamic SQL."""
    return '"' + identifier.replace('"', '""') + '"'


async def _sync_owned_sequences(conn: asyncpg.Connection, schema: str, tables: list[str]) -> None:
    """Advance non-cyclic identity sequences past restored column values."""
    sequences = await conn.fetch(
        """
        SELECT tbl.relname AS table_name,
               attr.attname AS column_name,
               seq_ns.nspname AS sequence_schema,
               seq.relname AS sequence_name,
               pg_seq.seqstart AS sequence_start,
               pg_seq.seqincrement AS sequence_increment,
               pg_seq.seqmin AS sequence_min,
               pg_seq.seqmax AS sequence_max
        FROM pg_catalog.pg_class AS tbl
        JOIN pg_catalog.pg_namespace AS tbl_ns ON tbl_ns.oid = tbl.relnamespace
        JOIN pg_catalog.pg_attribute AS attr
          ON attr.attrelid = tbl.oid AND attr.attnum > 0 AND NOT attr.attisdropped
        JOIN pg_catalog.pg_depend AS dep
          ON dep.refobjid = tbl.oid
         AND dep.refobjsubid = attr.attnum
         AND dep.classid = 'pg_catalog.pg_class'::regclass
         AND dep.refclassid = 'pg_catalog.pg_class'::regclass
         AND dep.deptype = 'i'
        JOIN pg_catalog.pg_class AS seq ON seq.oid = dep.objid AND seq.relkind = 'S'
        JOIN pg_catalog.pg_namespace AS seq_ns ON seq_ns.oid = seq.relnamespace
        JOIN pg_catalog.pg_sequence AS pg_seq ON pg_seq.seqrelid = seq.oid
        WHERE tbl_ns.nspname = $1
          AND tbl.relname = ANY($2::text[])
          AND attr.attidentity <> ''
          AND NOT pg_seq.seqcycle
        ORDER BY tbl.relname, attr.attnum
        """,
        schema,
        tables,
    )

    for sequence in sequences:
        aggregate = "MAX" if sequence["sequence_increment"] > 0 else "MIN"
        qualified_table = f"{_quote_identifier(schema)}.{_quote_identifier(sequence['table_name'])}"
        column = _quote_identifier(sequence["column_name"])
        restored_edge = await conn.fetchval(
            f"SELECT {aggregate}({column}) FROM {qualified_table} "
            f"WHERE {column} BETWEEN {int(sequence['sequence_min'])} AND {int(sequence['sequence_max'])}"
        )
        qualified_sequence = (
            f"{_quote_identifier(sequence['sequence_schema'])}.{_quote_identifier(sequence['sequence_name'])}"
        )
        restart_value = sequence["sequence_start"] if restored_edge is None else restored_edge
        await conn.execute(f"ALTER SEQUENCE {qualified_sequence} RESTART WITH {int(restart_value)}")
        if restored_edge is not None:
            # RESTART is transactional; consuming the restored edge gives the
            # sequence setval(edge, true) semantics without calculating a value
            # beyond a finite sequence's min/max bound.
            await conn.fetchval("SELECT nextval($1::regclass)", qualified_sequence)


def _effective_backup_tables() -> list[str]:
    """Return the tables owned by the local database."""
    return list(BACKUP_TABLES)


async def _admin_connect(db_url: str) -> asyncpg.Connection:
    """Open a raw asyncpg connection to an admin DB URL.

    ``resolve_database_url`` handles both plain ``postgres://`` (passthrough) and
    ``pg0://`` (boots the embedded server and returns its real libpq URL), so this
    is the only step needed to connect. JSON codecs are registered so ``jsonb``
    columns decode to Python objects (used by the export row dumps).
    """
    _pg0 = parse_pg0_url(db_url)
    is_pg0, instance_name = _pg0.is_pg0, _pg0.instance_name
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    conn = await asyncpg.connect(await resolve_database_url(db_url))
    for type_name in ("json", "jsonb"):
        await conn.set_type_codec(type_name, encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
    return conn


async def _backup(
    database_url: str,
    output_path: Path,
    schema: str = "public",
    backup_tables: list[str] | None = None,
) -> dict[str, Any]:
    """Backup all tables to a zip file using binary COPY protocol.

    ``backup_tables`` defaults to the local database's ``BACKUP_TABLES``.
    """
    backup_tables = backup_tables if backup_tables is not None else BACKUP_TABLES
    # sql_memories(), not get_memories(): backup reads this Postgres schema's own tables, whatever the store.
    pg_store = sql_memories()
    conn = await asyncpg.connect(database_url)
    try:
        tables: dict[str, Any] = {}
        manifest: dict[str, Any] = {
            "version": MANIFEST_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "schema": schema,
            "tables": tables,
        }

        # Use a transaction with REPEATABLE READ isolation to get a consistent
        # snapshot across all tables. This prevents race conditions where
        # entity_cooccurrences could reference entities created after the
        # entities table was backed up.
        async with conn.transaction(isolation="repeatable_read"):
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, table in enumerate(backup_tables, 1):
                    typer.echo(f"  [{i}/{len(backup_tables)}] Backing up {table}...", nl=False)

                    buffer = io.BytesIO()

                    columns = await _table_columns(conn, schema, table)

                    # Pin the ordered columns into both the stream and manifest.
                    # PostgreSQL binary COPY does not encode column identities, so
                    # restore must validate this shape before truncating any data.
                    # asyncpg requires schema_name as separate parameter
                    await conn.copy_from_table(
                        table,
                        schema_name=schema,
                        columns=[column.name for column in columns],
                        output=buffer,
                        format="binary",
                    )

                    data = buffer.getvalue()
                    zf.writestr(f"{table}.bin", data)

                    # Get row count for manifest. Through the Postgres store, which alone may
                    # name its own tables (#4969) — this walks every table of the schema.
                    row_count = await pg_store.admin_count_rows(conn=conn, schema=schema, table=table)
                    tables[table] = {
                        "rows": row_count,
                        "size_bytes": len(data),
                        "columns": [{"name": column.name, "type_name": column.type_name} for column in columns],
                    }

                    typer.echo(f" {row_count} rows")

                zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        return manifest
    finally:
        await conn.close()


async def _restore(
    database_url: str,
    input_path: Path,
    schema: str = "public",
    backup_tables: list[str] | None = None,
) -> dict[str, Any]:
    """Restore all tables from a zip file using binary COPY protocol.

    ``backup_tables`` defaults to the local database's ``BACKUP_TABLES``. Tables
    named here but absent from the archive are truncated then skipped.
    """
    backup_tables = backup_tables if backup_tables is not None else BACKUP_TABLES
    conn = await asyncpg.connect(database_url)
    try:
        with zipfile.ZipFile(input_path, "r") as zf:
            # Read and validate manifest
            manifest: dict[str, Any] = json.loads(zf.read("manifest.json"))
            if manifest.get("version") != MANIFEST_VERSION:
                raise ValueError(f"Unsupported backup version: {manifest.get('version')}")

            # Complete the compatibility check before entering the transaction
            # that truncates tables. This turns historical schema drift into an
            # actionable error without risking the target's existing data.
            restore_plans = await _validate_restore_schema(conn, manifest, schema)

            # Use a transaction for atomic restore - either all tables are
            # restored or none are, preventing partial/inconsistent state.
            async with conn.transaction():
                typer.echo("  Clearing existing data...")
                # Truncate tables in reverse order (respects FK constraints). Through the
                # Postgres store, which alone may name its own tables (#4969).
                # sql_memories(), not get_memories(): restore rewrites this Postgres schema's own tables,
                # whatever the store.
                await sql_memories().admin_truncate_tables(
                    conn=conn, schema=schema, tables=list(reversed(backup_tables))
                )

                # Restore tables in forward order
                for i, table in enumerate(backup_tables, 1):
                    filename = f"{table}.bin"
                    if filename not in zf.namelist():
                        typer.echo(f"  [{i}/{len(backup_tables)}] {table}: skipped (not in backup)")
                        continue

                    expected_rows = manifest["tables"].get(table, {}).get("rows", "?")
                    typer.echo(f"  [{i}/{len(backup_tables)}] Restoring {table}... {expected_rows} rows")

                    plan = restore_plans[table]
                    # Strips the fields of any column the target no longer has;
                    # a no-op when the schemas still line up.
                    buffer = io.BytesIO(_strip_binary_copy_fields(zf.read(filename), plan))
                    # asyncpg requires schema_name as separate parameter
                    await conn.copy_to_table(
                        table,
                        schema_name=schema,
                        columns=plan.columns,
                        source=buffer,
                        format="binary",
                    )

                typer.echo("  Synchronizing identity sequences...")
                await _sync_owned_sequences(conn, schema, backup_tables)

        return manifest
    finally:
        await conn.close()


async def _run_backup(db_url: str, output: Path, schema: str = "public") -> dict[str, Any]:
    """Resolve database URL and run backup."""
    _pg0 = parse_pg0_url(db_url)
    is_pg0, instance_name = _pg0.is_pg0, _pg0.instance_name
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    resolved_url = await resolve_database_url(db_url)
    return await _backup(resolved_url, output, schema, backup_tables=_effective_backup_tables())


async def _run_restore(db_url: str, input_file: Path, schema: str = "public") -> dict[str, Any]:
    """Resolve database URL and run restore."""
    _pg0 = parse_pg0_url(db_url)
    is_pg0, instance_name = _pg0.is_pg0, _pg0.instance_name
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    resolved_url = await resolve_database_url(db_url)
    return await _restore(resolved_url, input_file, schema, backup_tables=_effective_backup_tables())


@app.command()
def backup(
    output: Path = typer.Argument(..., help="Output file path (.zip)"),
    schema: str = typer.Option("public", "--schema", "-s", help="Database schema to backup"),
):
    """Backup the Remember database to a zip file."""
    config = RememberConfig.from_env()

    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set REMEMBER_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    if output.suffix != ".zip":
        output = output.with_suffix(".zip")

    typer.echo(f"Backing up database (schema: {schema}) to {output}...")

    manifest = asyncio.run(_run_backup(config.database_url, output, schema))

    total_rows = sum(t["rows"] for t in manifest["tables"].values())
    typer.echo(f"Backed up {total_rows} rows across {len(manifest['tables'])} tables")
    typer.echo(f"Backup saved to {output}")


@app.command()
def restore(
    input_file: Path = typer.Argument(..., help="Input backup file (.zip)"),
    schema: str = typer.Option("public", "--schema", "-s", help="Database schema to restore to"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Restore the database from a backup file. WARNING: This deletes all existing data."""
    config = RememberConfig.from_env()

    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set REMEMBER_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    if not input_file.exists():
        typer.echo(f"Error: File not found: {input_file}", err=True)
        raise typer.Exit(1)

    if not yes:
        typer.confirm(
            "This will DELETE all existing data and replace it with the backup. Continue?",
            abort=True,
        )

    typer.echo(f"Restoring database (schema: {schema}) from {input_file}...")

    manifest = asyncio.run(_run_restore(config.database_url, input_file, schema))

    total_rows = sum(t["rows"] for t in manifest["tables"].values())
    typer.echo(f"Restored {total_rows} rows across {len(manifest['tables'])} tables")
    typer.echo("Restore complete")


async def _run_migration(
    db_url: str,
    schema: str | None = None,
    base_schema: str = DEFAULT_DATABASE_SCHEMA,
    embedding_dimension: int | None = None,
    ensure_extensions: bool = True,
) -> str:
    """Resolve the database URL and migrate the selected local schema."""
    from ..migrations import run_local_migrations

    _pg0 = parse_pg0_url(db_url)
    is_pg0, instance_name = _pg0.is_pg0, _pg0.instance_name
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    resolved_url = await resolve_database_url(db_url)

    config = RememberConfig.from_env()
    target_schema = schema or base_schema or DEFAULT_DATABASE_SCHEMA

    # Run the local migration outside the event loop.
    await asyncio.to_thread(
        run_local_migrations,
        resolved_url,
        target_schema,
        embedding_dimension=embedding_dimension,
        vector_extension=config.vector_extension,
        text_search_extension=config.text_search_extension,
        ensure_extensions=ensure_extensions,
    )

    return target_schema


@app.command(name="run-db-migration")
def run_db_migration(
    schema: str | None = typer.Option(
        None,
        "--schema",
        "-s",
        help="Database schema to run migrations on. If omitted, migrate the configured base schema.",
    ),
    embedding_dimension: int | None = typer.Option(
        None,
        "--embedding-dimension",
        help="Expected embedding dimension to enforce after migrations. Omit to skip dimension sync.",
    ),
    skip_extension_reconcile: bool = typer.Option(
        False,
        "--skip-extension-reconcile",
        help=(
            "Skip the post-migration vector / text-search index reconcile. This step only does "
            "work when the configured backend (REMEMBER_API_VECTOR_EXTENSION / "
            "REMEMBER_API_TEXT_SEARCH_EXTENSION) differs from a schema's existing indexes — a "
            "rare, operator-driven change. Only use when you have NOT changed the backend; a "
            "backend change still needs a normal run to reshape the indexes."
        ),
    ),
):
    """Run database migrations to the latest version."""
    config = RememberConfig.from_env()

    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set REMEMBER_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    if schema:
        typer.echo(f"Running database migrations for schema: {schema}...")
    else:
        typer.echo("Running database migrations for the configured base schema...")
    if skip_extension_reconcile:
        typer.echo("Skipping post-migration extension reconcile (--skip-extension-reconcile).")

    target_schema = asyncio.run(
        _run_migration(
            config.database_url,
            schema=schema,
            base_schema=config.database_schema,
            embedding_dimension=embedding_dimension,
            ensure_extensions=not skip_extension_reconcile,
        )
    )

    typer.echo(f"Database migrations completed successfully for schema '{target_schema}'")


@dataclass
class RepairSweep:
    """Results and any incomplete work from a local schema repair."""

    banks: list[BankIndexResult]
    skipped_schema: str | None
    # Observations removed because a source memory they cite no longer exists (#5937);
    # in a dry run, how many would be.
    orphan_observations: int = 0
    # Observations whose keyword index was re-tokenized from their text (#5299, VectorChord only);
    # in a dry run, how many would be.
    rebuilt_search_vectors: int = 0
    # Counted apart from ``banks``, which stays empty on a backend without per-bank indexes.
    banks_scanned: int = 0
    # Attachments their document does not reference, left by retains that failed
    # (#6350), with --attachments; in a dry run, how many would be removed.
    unreferenced_attachments: int = 0


async def _run_repair_bank(
    db_url: str,
    *,
    base_schema: str,
    schema: str | None,
    bank_id: str | None,
    dry_run: bool,
    attachments: bool = False,
) -> RepairSweep:
    """Reconcile per-(bank, fact_type) vector index coverage, and remove orphan observations.

    A single autocommit connection is used because ``CREATE INDEX CONCURRENTLY``
    cannot run inside a transaction block. The index step is skipped on a backend
    without per-bank indexes; the observation sweep runs everywhere.

    Deliberately unbudgeted, unlike the background operation: this is an operator
    asking for convergence now, across as many banks as they named.
    """
    target_schema = schema or base_schema or DEFAULT_DATABASE_SCHEMA
    index_clause = _vector_index_clause()
    pg_store = sql_memories()

    conn = await _admin_connect(db_url)
    results: list[BankIndexResult] = []
    orphan_observations = 0
    rebuilt_search_vectors = 0
    banks_scanned = 0
    unreferenced_attachments = 0
    # Include all attachments older than the normal sweep's grace period.
    attachments_before = datetime.now(timezone.utc) - ATTACHMENT_SWEEP_GRACE
    attachments_after = datetime(1970, 1, 1, tzinfo=timezone.utc)
    try:
        # Keep completed bank results if a later repair fails.
        bank_ids: list[str] = []
        schema_results: list[BankIndexResult] = []
        schema_orphans = 0
        schema_rebuilt = 0
        schema_attachments = 0
        schema_banks_done = 0
        schema_file_storage = _admin_file_storage(conn, target_schema) if attachments else None
        try:
            bank_ids = [bank_id] if bank_id else await list_bank_ids(conn, target_schema)
            for bid in bank_ids:
                if index_clause is not None:
                    schema_results.append(
                        await reconcile_bank_vector_indexes(conn, target_schema, bid, index_clause, dry_run=dry_run)
                    )
                schema_orphans += await pg_store.admin_sweep_orphan_observations(
                    conn=conn, schema=target_schema, bank_id=bid, dry_run=dry_run
                )
                # After the sweep, so observations it just removed are not re-tokenized.
                schema_rebuilt += await pg_store.admin_rebuild_observation_search_vectors(
                    conn=conn, schema=target_schema, bank_id=bid, dry_run=dry_run
                )
                if schema_file_storage is not None:
                    # The shared rule resolves tables through fq_table.
                    token = _current_schema.set(target_schema)
                    try:
                        schema_attachments += await reclaim_unreferenced_attachments(
                            conn,
                            schema_file_storage,
                            get_memories(),
                            bid,
                            created_after=attachments_after,
                            created_before=attachments_before,
                            dry_run=dry_run,
                        )
                    finally:
                        _current_schema.reset(token)
                schema_banks_done += 1
        except Exception as exc:  # noqa: BLE001 — one bad schema must not abort the sweep
            results.extend(schema_results)
            orphan_observations += schema_orphans
            rebuilt_search_vectors += schema_rebuilt
            unreferenced_attachments += schema_attachments
            banks_scanned += schema_banks_done
            done = f", {schema_banks_done} of {len(bank_ids)} bank(s) done" if schema_banks_done else ""
            typer.echo(f"  schema '{target_schema}': skipped ({exc}){done}", err=True)
            return RepairSweep(
                banks=results,
                skipped_schema=target_schema,
                orphan_observations=orphan_observations,
                rebuilt_search_vectors=rebuilt_search_vectors,
                banks_scanned=banks_scanned,
                unreferenced_attachments=unreferenced_attachments,
            )
        results.extend(schema_results)
        orphan_observations += schema_orphans
        rebuilt_search_vectors += schema_rebuilt
        unreferenced_attachments += schema_attachments
        banks_scanned += schema_banks_done
        # Only in --all mode: an index whose bank row is gone is unreachable
        # from every bank-scoped path, so this is the one place that can collect
        # it. Normally finds nothing — delete_bank drops a bank's indexes while it
        # still knows their names — but a deployment that hit the #3485 wall could
        # not run delete_bank at all.
        #
        # Best-effort, and deliberately NOT a "skipped schema": the banks here were
        # reconciled, and reporting the schema as skipped would tell the operator
        # to re-run a sweep that already did its work. Per-index drop failures are
        # handled inside drop_orphaned_bank_indexes; this catches only the catalog
        # read behind it.
        orphans: list[str] = []
        if not bank_id and index_clause is not None:
            try:
                orphans = await drop_orphaned_bank_indexes(conn, target_schema, dry_run=dry_run)
            except Exception as exc:  # noqa: BLE001 — the banks were repaired; orphan collection is a nicety
                typer.echo(
                    f"  schema '{target_schema}': banks repaired, but orphan collection failed ({exc})", err=True
                )
        if orphans:
            typer.echo(
                f"  schema '{target_schema}': {len(orphans)} orphaned index(es) "
                f"{'to drop (dry-run)' if dry_run else 'dropped'} (no matching bank)"
            )
        typer.echo(
            f"  schema '{target_schema}': {len(bank_ids)} bank(s) scanned, "
            f"{sum(r.already_present for r in schema_results)} present, "
            f"{sum(r.created for r in schema_results)} created, "
            f"{sum(r.dropped for r in schema_results)} dropped, "
            f"{sum(r.skipped for r in schema_results)} to-create (dry-run), "
            f"{sum(r.would_drop for r in schema_results)} to-drop (dry-run), "
            f"{sum(r.failed for r in schema_results)} failed, "
            f"{schema_orphans} orphan observation(s) {'to remove (dry-run)' if dry_run else 'removed'}, "
            f"{schema_rebuilt} observation keyword index(es) {'to rebuild (dry-run)' if dry_run else 'rebuilt'}"
            + (
                f", {schema_attachments} unreferenced attachment(s) {'to remove (dry-run)' if dry_run else 'removed'}"
                if attachments
                else ""
            )
        )
        return RepairSweep(
            banks=results,
            skipped_schema=None,
            orphan_observations=orphan_observations,
            rebuilt_search_vectors=rebuilt_search_vectors,
            banks_scanned=banks_scanned,
            unreferenced_attachments=unreferenced_attachments,
        )
    finally:
        await conn.close()


@app.command(name="repair-bank")
def repair_bank(
    bank_id: str | None = typer.Option(
        None,
        "--bank",
        "-b",
        help="Bank id to repair. Mutually exclusive with --all.",
    ),
    all_banks: bool = typer.Option(
        False,
        "--all",
        help="Repair every bank in the selected local schema.",
    ),
    schema: str | None = typer.Option(
        None,
        "--schema",
        "-s",
        help="Local schema to repair. Defaults to the configured schema.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report what would be repaired without creating, dropping or deleting anything.",
    ),
    attachments: bool = typer.Option(
        False,
        "--attachments",
        help="Also delete attachments their document does not reference (left by retains that failed), "
        "however old. Reads each document that has attachments.",
    ),
):
    """Repair a bank's derived state: vector index coverage, orphan observations, and (under
    VectorChord) the observations' keyword index.

    Orphan observations cite a source memory that no longer exists. Every delete
    path removes them today, but older releases leaked some (#5937); they stay
    recallable and export skips them. Each one is deleted and its surviving
    sources are requeued, the same rule a live delete applies; they are
    re-consolidated on the bank's next consolidation run.

    Under VectorChord, an observation's keyword index is re-tokenized from its
    current text. Before #5299 a consolidation rewrite or merge changed the text
    without it, so keyword recall kept matching the old wording. Every
    observation is rebuilt, since a stale one cannot be told apart from the rest.

    With REMEMBER_API_VECTOR_INDEX_MIN_ROWS at its default of 0 every bank is
    owed all three indexes from the moment it exists, and this rebuilds whatever
    is missing or invalid — a bank whose creation lost its DDL to a deadlock, one
    restored around it, or one whose access method drifted after a backend
    switch. Nothing is dropped in that mode.

    With a threshold set, a (bank, fact_type) earns its index once it holds that
    many rows; below it the planner answers the same query exactly, and faster,
    from the (bank_id, fact_type) B-tree plus a top-N sort. This command then
    also drops what no longer qualifies. Either way it sheds indexes orphaned by
    a deleted bank, and all DDL is CONCURRENTLY, so it never blocks the live
    fleet.

    With a threshold set, writes keep this converged on their own — every write
    that could move a bank across it queues a vector_index_maintenance operation.
    Reach for the command when you want convergence without waiting for a write:
    after a restore or upgrade, after a backend switch, or to shed indexes in
    bulk on a deployment recovering from lock-table exhaustion (#3485).

    With --attachments, also deletes the images and files of retains that failed
    after the API stored them: attachments, a day or more old, that their
    document's text does not reference — the document was never created, or the
    update that brought them failed. The daily maintenance sweep applies the same
    rule to the past week only; this covers everything before it (#6350). It reads
    every document that has an attachment, through the memories store, so it is
    opt-in rather than part of every repair.

    Idempotent and safe to re-run.
    """
    if bool(bank_id) == all_banks:
        typer.echo("Error: pass exactly one of --bank <id> or --all.", err=True)
        raise typer.Exit(2)

    config = RememberConfig.from_env()
    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set REMEMBER_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    if is_oracle_url(config.database_url):
        typer.echo("Error: repair-bank is PostgreSQL-only; Oracle backends are not supported.", err=True)
        raise typer.Exit(1)

    # Backends with a single global vector index (AlloyDB ScaNN) have no per-bank
    # indexes to repair; the observation sweep still runs.
    if _vector_index_clause() is None:
        typer.echo("Configured vector backend does not use per-bank vector indexes — skipping index repair.")

    target = f"bank '{bank_id}'" if bank_id else "all banks"
    scope = f"schema '{schema or config.database_schema or DEFAULT_DATABASE_SCHEMA}'"
    typer.echo(f"Repairing {target} across {scope}...")
    if dry_run:
        typer.echo("Dry run: nothing will be created, dropped or deleted.")

    sweep = asyncio.run(
        _run_repair_bank(
            config.database_url,
            base_schema=config.database_schema,
            schema=schema,
            bank_id=bank_id,
            dry_run=dry_run,
            attachments=attachments,
        )
    )
    results = sweep.banks

    total_banks = sweep.banks_scanned
    total_present = sum(r.already_present for r in results)
    total_created = sum(r.created for r in results)
    total_dropped = sum(r.dropped for r in results)
    total_skipped = sum(r.skipped for r in results)
    total_would_drop = sum(r.would_drop for r in results)
    total_failed = sum(r.failed for r in results)
    typer.echo(
        f"Done: {total_banks} bank(s) scanned, "
        f"{total_present} already present, {total_created} created, {total_dropped} dropped, "
        f"{total_skipped} to-create (dry-run), {total_would_drop} to-drop (dry-run), "
        f"{total_failed} failed, "
        f"{sweep.orphan_observations} orphan observation(s) {'to remove (dry-run)' if dry_run else 'removed'}, "
        f"{sweep.rebuilt_search_vectors} observation keyword index(es) "
        f"{'to rebuild (dry-run)' if dry_run else 'rebuilt'}"
        + (
            f", {sweep.unreferenced_attachments} unreferenced attachment(s) "
            f"{'to remove (dry-run)' if dry_run else 'removed'}"
            if attachments
            else ""
        )
    )
    if total_failed:
        failed_names = [name for r in results for name in r.failed_indexes]
        typer.echo(f"Failed indexes (dropped, retry with a re-run): {', '.join(failed_names)}", err=True)
    if sweep.skipped_schema:
        typer.echo(
            f"Schema '{sweep.skipped_schema}' was not fully repaired. Re-run once the cause is cleared.", err=True
        )
    if total_failed or sweep.skipped_schema:
        raise typer.Exit(1)


class RenameBankError(Exception):
    """A rename-bank precondition failed; nothing was changed."""


# FKs whose own columns include bank_id and that cannot be deferred as declared.
_RIGID_BANK_ID_FKS_SQL = """
    SELECT c.conrelid::regclass::text AS tbl, c.conname
    FROM pg_constraint c
    JOIN pg_namespace n ON n.oid = c.connamespace
    WHERE c.contype = 'f' AND n.nspname = $1 AND NOT c.condeferrable
      AND EXISTS (
          SELECT 1 FROM pg_attribute a
          WHERE a.attrelid = c.conrelid AND a.attnum = ANY (c.conkey) AND a.attname = 'bank_id'
      )
"""


async def _set_fks_deferrable(conn: asyncpg.Connection, fks: list[asyncpg.Record], clause: str) -> None:
    """Flip the given FKs' deferrability in one short transaction.

    ``ALTER CONSTRAINT`` only touches the catalog, but it still takes a table lock
    and queues behind any open transaction on the table — and everything else
    queues behind it. ``lock_timeout`` turns a long wait into a loud failure
    instead of a tenant-wide stall.
    """
    async with conn.transaction():
        await conn.execute("SET LOCAL lock_timeout = '5s'")
        for fk in fks:
            await conn.execute(f"ALTER TABLE {fk['tbl']} ALTER CONSTRAINT {_quote_identifier(fk['conname'])} {clause}")


async def _rename_bank(
    conn: asyncpg.Connection, schema: str, old_bank_id: str, new_bank_id: str, *, dry_run: bool
) -> dict[str, int]:
    """Move every row of ``old_bank_id`` to ``new_bank_id`` in one transaction.

    ``bank_id`` is the key every bank-scoped table carries, several composite FKs
    included (``documents(id, bank_id)``, ``mental_models(id, bank_id)``), so no
    update order satisfies an immediate FK check. The FKs are made DEFERRABLE
    just for the rename and put back afterwards, each flip in its own short
    transaction — doing it inside the rename would hold those table locks, and
    block every bank in the schema, for its whole duration. This is runtime DDL
    rather than a migration on purpose: rename is a rare admin operation, and the
    schema stays exactly as the migrations declare it.

    Only FKs this call flipped are put back. If that restore cannot happen (the
    process dies, or the restore times out on a lock) they stay DEFERRABLE
    INITIALLY IMMEDIATE, which checks every normal write exactly as before; a
    later rename will not restore them, since it only flips what it finds
    NOT DEFERRABLE, so the failure is reported with the constraint names.

    Returns the rows moved per table (tables the bank had no rows in are omitted).
    """
    rigid = await conn.fetch(_RIGID_BANK_ID_FKS_SQL, schema)
    try:
        await _set_fks_deferrable(conn, rigid, "DEFERRABLE INITIALLY IMMEDIATE")
    except asyncpg.exceptions.LockNotAvailableError as exc:
        raise RenameBankError(
            "could not lock the bank tables within 5s to prepare the rename; retry when fewer long transactions run"
        ) from exc
    try:
        return await _move_bank_rows(conn, schema, old_bank_id, new_bank_id, dry_run=dry_run)
    finally:
        # Never raise from here: it would replace the rename's own outcome (success
        # or its real error) with a failure that leaves every write checked as before.
        try:
            await _set_fks_deferrable(conn, rigid, "NOT DEFERRABLE")
        except Exception as exc:  # noqa: BLE001
            names = ", ".join(f"{fk['tbl']}.{fk['conname']}" for fk in rigid)
            typer.echo(
                f"Warning: could not restore NOT DEFERRABLE on {names} ({exc}). They still enforce every write "
                "as before; restore with ALTER TABLE ... ALTER CONSTRAINT ... NOT DEFERRABLE.",
                err=True,
            )


async def _move_bank_rows(
    conn: asyncpg.Connection, schema: str, old_bank_id: str, new_bank_id: str, *, dry_run: bool
) -> dict[str, int]:
    """The rename transaction proper; the bank_id FKs must already be DEFERRABLE.

    Deferring them lets the tables move in any order, and forcing them back to
    IMMEDIATE before the commit — or before the dry run's rollback — makes a
    missed table fail the whole rename instead of stranding rows. Tables are read
    from the catalog, so extension tables and tables added later are covered
    without a list to maintain.

    The ``FOR UPDATE`` on the bank row serialises the rename against writers:
    anything inserting a FK child of the bank takes a key-share lock on that row,
    so it either commits before the rename reads, or waits and then fails its FK
    check rather than writing under an id that no longer exists.
    """
    banks = _fq_table("banks", schema)
    tx = conn.transaction()
    await tx.start()
    try:
        if not await conn.fetchval(f"SELECT 1 FROM {banks} WHERE bank_id = $1 FOR UPDATE", old_bank_id):
            raise RenameBankError(f"bank '{old_bank_id}' does not exist in schema '{schema}'")
        if await conn.fetchval(f"SELECT 1 FROM {banks} WHERE bank_id = $1", new_bank_id):
            raise RenameBankError(f"bank '{new_bank_id}' already exists in schema '{schema}'")
        # The API refuses a bank under another bank's alias (it would take the name over,
        # since a real bank wins resolution). Onto its own alias is the end of the
        # migration aliases exist for: the alias is the bank's id now, so it goes.
        aliases = _fq_table("bank_aliases", schema)
        alias_owner = await conn.fetchval(f"SELECT bank_id FROM {aliases} WHERE alias = $1", new_bank_id)
        if alias_owner is not None and alias_owner != old_bank_id:
            raise RenameBankError(f"'{new_bank_id}' is an alias of bank '{alias_owner}' in schema '{schema}'")
        if alias_owner is not None:  # a dry run rolls this back with the rest
            await conn.execute(f"DELETE FROM {aliases} WHERE alias = $1", new_bank_id)
        # An operation's task_payload names its bank, so one run after the rename
        # would work on (and could recreate) the old id.
        active = await conn.fetchval(
            f"SELECT count(*) FROM {_fq_table('async_operations', schema)} "
            "WHERE bank_id = $1 AND status IN ('pending', 'processing')",
            old_bank_id,
        )
        if active:
            raise RenameBankError(
                f"bank '{old_bank_id}' has {active} pending or processing operation(s); "
                "wait for them to finish or cancel them, then retry"
            )
        tables = await conn.fetch(
            """
            SELECT c.table_name
            FROM information_schema.columns c
            JOIN information_schema.tables t
              ON t.table_schema = c.table_schema AND t.table_name = c.table_name
            WHERE c.table_schema = $1 AND c.column_name = 'bank_id' AND t.table_type = 'BASE TABLE'
            ORDER BY c.table_name
            """,
            schema,
        )
        await conn.execute("SET CONSTRAINTS ALL DEFERRED")
        # Every table, store tables included, so the rewrite goes through the Postgres store,
        # which alone may name its own tables (#4969).
        # sql_memories(), not get_memories(): the rename walks this Postgres schema's own tables, whatever the store.
        moved = await sql_memories().admin_move_bank_id(
            conn=conn,
            schema=schema,
            tables=[row["table_name"] for row in tables],
            old_bank_id=old_bank_id,
            new_bank_id=new_bank_id,
        )
        await conn.execute("SET CONSTRAINTS ALL IMMEDIATE")
    except BaseException:
        await tx.rollback()
        raise
    if dry_run:
        await tx.rollback()
    else:
        await tx.commit()
    return moved


async def _move_bank_files(conn: asyncpg.Connection, db_url: str, schema: str, old_id: str, new_id: str) -> int:
    """Re-key the bank's stored files, now that its rows carry the new id.

    A file's key spells the bank id (``bank_storage_prefix``), and no column move
    reaches inside a key: left alone, the bank's bytes stay under the old prefix,
    where neither the download route nor ``delete_bank``'s sweep — both of which
    look under the bank's *current* prefix — can reach them. The sweep missing
    them is the leak (#4502): on S3 or GCS the bank goes and its bytes stay, still
    billed.

    Copy first, repoint the row, and only then drop the old prefix, so an
    interrupted move leaves every row pointing at bytes that exist.

    ponytail: copies through this process, one object at a time. A server-side
    copy (S3 CopyObject) or, on the native backend, an UPDATE of file_storage
    would move bytes without reading them — worth it if renames get big or common.
    """
    old_prefix = bank_storage_prefix(old_id, schema)
    new_prefix = bank_storage_prefix(new_id, schema)
    # Resolve pg0 URLs the same way as the other admin operations.
    pool = await asyncpg.create_pool(await resolve_database_url(db_url), min_size=1, max_size=2)
    try:
        storage = PostgreSQLFileStorage(
            pool_getter=lambda: pool,
            schema=schema,
        )
        # Keys written before the tenant layout sit outside the prefix and stay where
        # they are: delete_bank sweeps those from their rows, which the rename carries
        # to the new id. `documents` is a store table, so the Postgres store reads and
        # repoints the keys (#4969).
        # sql_memories(), not get_memories(): the rename walks this Postgres schema's own tables, whatever the store.
        pg_store = sql_memories()
        rows = await pg_store.admin_bank_file_keys(conn=conn, schema=schema, bank_id=new_id, prefix=old_prefix)
        columns = {"attachments": "storage_key", "documents": "file_storage_key"}
        moved = 0
        for row in rows:
            new_key = new_prefix + row["key"][len(old_prefix) :]
            try:
                data = await storage.retrieve(row["key"])
            except FileNotFoundError:
                # The row outlived its bytes (an earlier leak swept, a manual
                # cleanup). Leave it pointing where it does rather than at a key
                # with nothing behind it, and carry on: the rename is committed.
                typer.echo(f"Warning: {row['key']} has no stored bytes; its row keeps the old key.")
                continue
            await storage.store(file_data=data, key=new_key)
            await pg_store.admin_repoint_file_key(
                conn=conn,
                schema=schema,
                table=row["table_name"],
                column=columns[row["table_name"]],
                bank_id=new_id,
                old_key=row["key"],
                new_key=new_key,
            )
            moved += 1
        # The originals, plus whatever else the bank left under the old prefix:
        # export and import archives, which no row names and which their
        # operation can produce again.
        try:
            await storage.delete_prefix(old_prefix)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"Warning: could not clear the old prefix {old_prefix} ({exc}); its files are now orphans.")
        return moved
    finally:
        await pool.close()


async def _run_rename_bank(
    db_url: str, schema: str, old_bank_id: str, new_bank_id: str, *, dry_run: bool
) -> dict[str, int]:
    """Rename the bank, then move its stored files and rebuild its vector indexes.

    Those indexes are partial on a ``bank_id`` literal, so after the rename they
    cover nothing; the reconcile sees the stale predicate and rebuilds them
    CONCURRENTLY, which is why it runs after the commit, outside the transaction.
    Until it finishes, recall on the bank runs without its index.
    """
    conn = await _admin_connect(db_url)
    try:
        moved = await _rename_bank(conn, schema, old_bank_id, new_bank_id, dry_run=dry_run)
        if not dry_run:
            files = await _move_bank_files(conn, db_url, schema, old_bank_id, new_bank_id)
            typer.echo(f"Stored files: {files} re-keyed under the new bank id")
        index_clause = _vector_index_clause()
        if not dry_run and index_clause is not None:
            # The rename is already committed here, so a failure in the rebuild must
            # not surface as a bare traceback that buries that fact.
            try:
                result = await reconcile_bank_vector_indexes(conn, schema, new_bank_id, index_clause)
            except Exception as exc:  # noqa: BLE001 — the rename is committed; say so rather than traceback
                typer.echo(
                    f"Vector indexes: could not be rebuilt ({exc}). The rename itself succeeded. "
                    f"Re-run `remember-admin repair-bank --bank {new_bank_id}` once the cause is cleared; "
                    f"until then recall on this bank runs without its index.",
                    err=True,
                )
                return moved
            typer.echo(f"Vector indexes: {result.created} rebuilt, {result.dropped} dropped, {result.failed} failed")
            if result.failed:
                typer.echo(f"Re-run `remember-admin repair-bank --bank {new_bank_id}` to retry.", err=True)
        return moved
    finally:
        await conn.close()


@app.command(name="rename-bank")
def rename_bank(
    old_bank_id: str = typer.Option(..., "--from", help="Current bank id."),
    new_bank_id: str = typer.Option(..., "--to", help="New bank id. Must not exist yet."),
    schema: str | None = typer.Option(
        None,
        "--schema",
        "-s",
        help="Database schema the bank lives in. Defaults to the configured base schema.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run the whole rename, report what would move, then roll back.",
    ),
) -> None:
    """Rename a bank's id in place, keeping every memory, document and mental model.

    One transaction moves every row of the bank to the new id. Stop the bank's
    clients first: anything still calling the old id gets a 404 or, if it creates
    banks on demand, a new empty bank under the old id. Refused while the bank
    has pending or processing operations.
    """
    if not new_bank_id.strip() or new_bank_id == old_bank_id:
        typer.echo("Error: --to must be a non-empty id different from --from.", err=True)
        raise typer.Exit(2)

    config = RememberConfig.from_env()
    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set REMEMBER_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    # The rename flips FK deferrability with PostgreSQL's ALTER CONSTRAINT and runs
    # over asyncpg; Oracle can only change deferrability by recreating the
    # constraint. Refuse before connecting rather than fail halfway.
    if is_oracle_url(config.database_url):
        typer.echo("Error: rename-bank is PostgreSQL-only; Oracle backends are not supported.", err=True)
        raise typer.Exit(1)

    target_schema = schema or config.database_schema or DEFAULT_DATABASE_SCHEMA
    typer.echo(f"Renaming bank '{old_bank_id}' -> '{new_bank_id}' in schema '{target_schema}'...")
    try:
        moved = asyncio.run(
            _run_rename_bank(config.database_url, target_schema, old_bank_id, new_bank_id, dry_run=dry_run)
        )
    except RenameBankError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    for table, count in sorted(moved.items()):
        typer.echo(f"  {table}: {count}")
    if dry_run:
        typer.echo(f"Dry run: {sum(moved.values())} row(s) would move; rolled back.")
    else:
        typer.echo(
            f"Renamed: {sum(moved.values())} row(s) moved. Point clients and API keys at '{new_bank_id}'; "
            "API servers may serve the old id from cache until bank_info_cache_ttl_seconds elapses."
        )


class _SingleConnectionPool:
    """Pool adapter over the admin CLI's one raw connection.

    File storage takes a pool because the API serves many requests from one; the
    CLI has a single connection and a single export running on it, so acquiring
    hands the same connection back. Only the native (PostgreSQL) backend touches
    this at all — an S3/GCS/Azure deployment never calls the pool.
    """

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def acquire(self) -> asyncpg.Connection:
        return self._conn

    async def release(self, conn: asyncpg.Connection) -> None:
        """No-op: the caller owns the connection's lifetime, and closes it."""


def _admin_file_storage(conn: asyncpg.Connection, schema: str) -> Any:
    """File storage for an export run from the CLI, on this instance's config.

    Attachment bytes live here rather than in a column, so a bank with
    attachments cannot be exported without it.
    """
    return PostgreSQLFileStorage(
        pool_getter=lambda: _SingleConnectionPool(conn),
        schema=schema,
    )


async def _decommission_worker(db_url: str, worker_id: str, schema: str = "public") -> int:
    """Release all tasks owned by a worker, setting them back to pending status."""
    _pg0 = parse_pg0_url(db_url)
    is_pg0, instance_name = _pg0.is_pg0, _pg0.instance_name
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    resolved_url = await resolve_database_url(db_url)

    conn = await asyncpg.connect(resolved_url)
    try:
        table = _fq_table("async_operations", schema)
        result = await conn.fetch(
            f"""
            UPDATE {table}
            SET status = 'pending', worker_id = NULL, claimed_at = NULL, updated_at = now()
            WHERE worker_id = $1 AND status = 'processing'
            RETURNING operation_id
            """,
            worker_id,
        )
        return len(result)
    finally:
        await conn.close()


@app.command(name="decommission-worker")
def decommission_worker(
    worker_id: str = typer.Argument(..., help="Worker ID to decommission"),
    schema: str = typer.Option("public", "--schema", "-s", help="Database schema"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Release all tasks owned by a worker (sets status back to pending).

    Use this command when a worker has crashed or been removed without graceful shutdown.
    All tasks that were being processed by the worker will be released back to the queue
    so other workers can pick them up.
    """
    config = RememberConfig.from_env()

    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set REMEMBER_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    if not yes:
        typer.confirm(
            f"This will release all tasks owned by worker '{worker_id}' back to pending. Continue?",
            abort=True,
        )

    typer.echo(f"Decommissioning worker '{worker_id}' (schema: {schema})...")

    count = asyncio.run(_decommission_worker(config.database_url, worker_id, schema))

    if count > 0:
        typer.echo(f"Released {count} task(s) from worker '{worker_id}'")
    else:
        typer.echo(f"No tasks found for worker '{worker_id}'")


async def _decommission_all_workers(db_url: str, schema: str = "public") -> list[dict[str, Any]]:
    """Release all processing tasks from all workers, setting them back to pending status."""
    _pg0 = parse_pg0_url(db_url)
    is_pg0, instance_name = _pg0.is_pg0, _pg0.instance_name
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    resolved_url = await resolve_database_url(db_url)

    conn = await asyncpg.connect(resolved_url)
    try:
        table = _fq_table("async_operations", schema)
        rows = await conn.fetch(
            f"""
            UPDATE {table}
            SET status = 'pending', worker_id = NULL, claimed_at = NULL, updated_at = now()
            WHERE status = 'processing'
            RETURNING operation_id, worker_id, operation_type
            """,
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


@app.command(name="decommission-workers")
def decommission_workers(
    schema: str = typer.Option("public", "--schema", "-s", help="Database schema"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Release all processing tasks from all workers (sets status back to pending).

    Use this command to recover from situations where one or more workers have crashed
    or been removed without graceful shutdown. All tasks currently in 'processing' status
    will be released back to the queue regardless of which worker owns them.
    """
    config = RememberConfig.from_env()

    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set REMEMBER_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    if not yes:
        typer.confirm(
            "This will release ALL processing tasks from ALL workers back to pending. Continue?",
            abort=True,
        )

    typer.echo(f"Decommissioning all workers (schema: {schema})...")

    released = asyncio.run(_decommission_all_workers(config.database_url, schema))

    if released:
        # Group by worker_id for summary
        by_worker: dict[str, int] = {}
        for row in released:
            wid = row["worker_id"] or "unknown"
            by_worker[wid] = by_worker.get(wid, 0) + 1

        typer.echo(f"Released {len(released)} task(s):")
        for wid, count in by_worker.items():
            typer.echo(f"  {wid}: {count} task(s)")
    else:
        typer.echo("No processing tasks found")


async def _worker_status(db_url: str, schema: str = "public") -> list[dict[str, Any]]:
    """Get all processing tasks grouped by worker with their last update time."""
    _pg0 = parse_pg0_url(db_url)
    is_pg0, instance_name = _pg0.is_pg0, _pg0.instance_name
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    resolved_url = await resolve_database_url(db_url)

    conn = await asyncpg.connect(resolved_url)
    try:
        table = _fq_table("async_operations", schema)
        rows = await conn.fetch(
            f"""
            SELECT worker_id, operation_id, operation_type, bank_id,
                   claimed_at, updated_at,
                   now() - claimed_at AS running_for,
                   now() - updated_at AS last_update_ago
            FROM {table}
            WHERE status = 'processing'
            ORDER BY worker_id, claimed_at
            """,
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


@app.command(name="worker-status")
def worker_status(
    schema: str = typer.Option("public", "--schema", "-s", help="Database schema"),
):
    """Show all currently processing tasks grouped by worker.

    Displays each worker's active tasks with operation type, bank, how long
    the task has been running, and when it was last updated. Useful for
    identifying dead workers with orphaned tasks.
    """
    config = RememberConfig.from_env()

    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set REMEMBER_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    rows = asyncio.run(_worker_status(config.database_url, schema))

    if not rows:
        typer.echo("No processing tasks found")
        return

    # Group by worker_id
    by_worker: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        wid = row["worker_id"] or "unknown"
        by_worker.setdefault(wid, []).append(row)

    typer.echo(f"Processing tasks across {len(by_worker)} worker(s):\n")
    for wid, tasks in by_worker.items():
        typer.echo(f"Worker: {wid} ({len(tasks)} task(s))")
        for task in tasks:
            op_id = str(task["operation_id"])[:8]
            running_for = task["running_for"]
            last_update = task["last_update_ago"]
            typer.echo(
                f"  {op_id}  {task['operation_type']:<20s} bank={task['bank_id']}"
                f"  running={running_for}  last_update={last_update} ago"
            )
        typer.echo("")


async def _run_reembed(
    db_url: str, schema: str, *, finish: bool, batch_size: int, concurrency: int, sleep_s: float
) -> SchemaReembed:
    from ..engine.embeddings import create_embeddings_from_env, embedding_model_id

    resolved_url = await resolve_database_url(db_url)
    embeddings = None
    if not finish:
        embeddings = create_embeddings_from_env()
        await embeddings.initialize()
        typer.echo(f"New embedding model: {embedding_model_id(embeddings)}")
    conn = await _admin_connect(db_url)
    try:
        if finish:
            return await finish_schema(conn, schema)
        assert embeddings is not None
        typer.echo(f"Re-embedding schema '{schema}'...")
        return await reembed_schema(
            conn, resolved_url, schema, embeddings, batch_size=batch_size, concurrency=concurrency, sleep_s=sleep_s
        )
    finally:
        await conn.close()


@app.command(name="reembed")
def reembed(
    schema: str | None = typer.Option(None, "--schema", "-s", help="Local schema to re-embed."),
    finish: bool = typer.Option(
        False, "--finish", help="Remove the write guard, once every server runs the new embedding model."
    ),
    batch_size: int = typer.Option(256, "--batch-size", help="Rows embedded and written per transaction."),
    concurrency: int = typer.Option(
        4,
        "--concurrency",
        min=1,
        help="Batches sent to the embedding model at once. Postgres still sees one connection.",
    ),
    sleep_s: float = typer.Option(0.0, "--sleep", help="Seconds to pause between batches, to go easy on Postgres."),
):
    """Re-embed every stored vector with the embedding model configured in the environment.

    Run it with the NEW model's REMEMBER_API_EMBEDDINGS_* settings; servers keep running on the
    old one. Recall keeps working on the old vectors (without their index) while the new ones are
    written to a side column in small batches; then a short write lock swaps the columns. From
    then on Postgres refuses vector writes from servers still on the old model (workers retry
    them later, HTTP answers 503): restart every server with the new model, then run
    `reembed --finish`. Until a server restarts its recall embeds queries with the old model.

    PostgreSQL only. Resumable: re-run after an interruption and it continues.
    """
    config = RememberConfig.from_env()
    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        raise typer.Exit(1)
    if is_oracle_url(config.database_url):
        typer.echo("Error: reembed is PostgreSQL-only.", err=True)
        raise typer.Exit(1)
    target_schema = schema or config.database_schema or DEFAULT_DATABASE_SCHEMA
    result = asyncio.run(
        _run_reembed(
            config.database_url,
            target_schema,
            finish=finish,
            batch_size=batch_size,
            concurrency=concurrency,
            sleep_s=sleep_s,
        )
    )
    rows = f", {result.rows_embedded} rows embedded" if result.rows_embedded else ""
    typer.echo(f"  {result.schema}: {result.status}{rows}")
    if not finish:
        typer.echo("Next: restart every server with the new model, then run `remember-admin reembed --finish`.")


@app.command(
    name="chatgpt",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True, "help_option_names": []},
)
def chatgpt(ctx: typer.Context):
    """Sign in with ChatGPT for the chatgpt LLM provider (login | status | logout)."""
    from ..engine.providers.chatgpt_auth import main as chatgpt_main

    raise typer.Exit(chatgpt_main(ctx.args))


def main():
    load_dotenv_for_entrypoint()
    app()


if __name__ == "__main__":
    main()
