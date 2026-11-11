"""Alembic environment for the local PostgreSQL backend.

Migrations use the sync psycopg2 driver and schema-specific search paths.
The existing revision tree is retained for PostgreSQL databases.
"""

import logging
import os
from pathlib import Path

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import Connection, engine_from_config, pool
from sqlalchemy.engine import Engine

from hindsight_api.db_url import is_oracle_url, to_libpq_url
from hindsight_api.models import Base


def load_env() -> None:
    """Load environment variables from .env (skipped if already configured)."""
    if os.getenv("HINDSIGHT_API_DATABASE_URL"):
        return

    root_dir = Path(__file__).parent.parent.parent
    env_file = root_dir / ".env"

    if env_file.exists():
        load_dotenv(env_file)


load_env()

config = context.config
target_metadata = Base.metadata


def get_database_url() -> str:
    """Resolve the migration URL from Alembic config or env, normalizing per-dialect."""
    database_url = config.get_main_option("sqlalchemy.url")
    if not database_url:
        database_url = os.getenv("HINDSIGHT_API_DATABASE_URL")
        if not database_url:
            raise ValueError(
                "Database URL not found. "
                "Set HINDSIGHT_API_DATABASE_URL environment variable or pass database_url to run_migrations()."
            )

    if is_oracle_url(database_url):
        raise ValueError("Only PostgreSQL migrations are supported")
    database_url = to_libpq_url(database_url)

    # Alembic stores options through ConfigParser, where '%' is interpolation.
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return database_url


def run_migrations_offline() -> None:
    logging.info("running offline")
    database_url = get_database_url()

    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def _configure_pg_session(engine: Engine, connection: Connection, target_schema: str | None) -> None:
    """PG-only: ensure the session is RW (Supabase) and bind ``search_path``."""
    from sqlalchemy import event, text

    @event.listens_for(engine, "connect")
    def set_read_write_mode(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE")
        if target_schema:
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{target_schema}"')
            cursor.execute(f'SET search_path TO "{target_schema}", public')
        cursor.close()

    connection.execute(text("SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE"))
    if target_schema:
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{target_schema}"'))
        connection.execute(text(f'SET search_path TO "{target_schema}", public'))
    connection.commit()


def run_migrations_online() -> None:
    get_database_url()
    target_schema = config.get_main_option("target_schema")

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        _configure_pg_session(connectable, connection, target_schema)
        if target_schema:
            from hindsight_api.alembic._new_schema_indexes import (
                build_new_schema_indexes_plainly,
                is_new_schema,
            )

            # Before alembic creates `alembic_version`, which is what "new" is judged by.
            if is_new_schema(connection, target_schema):
                build_new_schema_indexes_plainly(connectable, target_schema)
            # The check autobegan a transaction; end it so alembic opens its own (its
            # autocommit blocks require one it owns).
            connection.commit()

        context_opts = {
            "connection": connection,
            "target_metadata": target_metadata,
        }
        if target_schema:
            context_opts["version_table_schema"] = target_schema

        context.configure(**context_opts)

        with context.begin_transaction():
            context.run_migrations()

        # Always commit. PG needs it for the explicit RW-mode SET to persist;
        # Oracle needs it because each DDL auto-commits but the trailing
        # ``UPDATE alembic_version`` is plain DML that would otherwise stay in
        # an open transaction and roll back when the connection closes —
        # producing the "schema is created but the version row is one revision
        # behind" failure mode.
        connection.commit()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
