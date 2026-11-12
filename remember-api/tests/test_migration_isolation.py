"""REMEMBER_API_MIGRATION_ISOLATION decides where migrations run.

Alembic drives PostgreSQL through SQLAlchemy's sync engine, i.e. psycopg2. "true"
keeps that import graph and its thread pool out of a long-lived server process by
running the migration in a child; "false", the default, runs it in-process.
"""

import io
import json
import os
from unittest.mock import patch

import pytest

from remember_api import migrations
from remember_api.config import _parse_migration_isolation


def _isolates(mode: str, monkeypatch) -> bool:
    monkeypatch.setenv("REMEMBER_API_MIGRATION_ISOLATION", mode)
    monkeypatch.delenv(migrations._CHILD_MARKER, raising=False)
    # get_config caches, so read the decision through a config carrying this mode.
    with patch.object(migrations, "get_config") as get_config:
        get_config.return_value.migration_isolation = mode
        return migrations._should_isolate_migrations()


def test_true_isolates(monkeypatch):
    assert _isolates("true", monkeypatch) is True


def test_false_does_not_isolate(monkeypatch):
    assert _isolates("false", monkeypatch) is False


def test_child_never_recurses(monkeypatch):
    """The subprocess must run the migration, not spawn another one."""
    monkeypatch.setenv(migrations._CHILD_MARKER, "1")
    with patch.object(migrations, "get_config") as get_config:
        get_config.return_value.migration_isolation = "true"
        assert migrations._should_isolate_migrations() is False


@pytest.mark.parametrize("bad", ["", "yes", "no", "1", "0", "auto", "always", "never", "subprocess"])
def test_rejects_unknown_values(monkeypatch, bad):
    """Silently defaulting would run migrations in the wrong process, invisibly.

    "1"/"0"/"yes"/"no" are rejected on purpose: the flag is not a general bool parser,
    and the two spellings it does take are the ones documented.
    """
    monkeypatch.setenv("REMEMBER_API_MIGRATION_ISOLATION", bad)
    with pytest.raises(ValueError, match="REMEMBER_API_MIGRATION_ISOLATION"):
        _parse_migration_isolation()


def test_defaults_to_in_process(monkeypatch):
    monkeypatch.delenv("REMEMBER_API_MIGRATION_ISOLATION", raising=False)
    assert _parse_migration_isolation() == "false"


def test_case_and_whitespace_are_tolerated(monkeypatch):
    monkeypatch.setenv("REMEMBER_API_MIGRATION_ISOLATION", "  TRUE  ")
    assert _parse_migration_isolation() == "true"


def test_payload_travels_on_stdin_not_argv():
    """The child receives database settings on stdin, while output streams normally."""
    payload = {"database_url": "postgresql://user:password@localhost/db", "schema": "local"}
    with patch.object(migrations.subprocess, "run") as run:
        run.return_value.returncode = 0
        migrations._run_in_migration_child("run_local_migrations", payload)
    (argv,), kwargs = run.call_args
    assert argv[1:] == ["-m", "remember_api.migrations"]
    assert "password" not in " ".join(argv)
    assert json.loads(kwargs["input"])["kwargs"] == payload
    assert "capture_output" not in kwargs and "stdout" not in kwargs
    assert kwargs["env"][migrations._CHILD_MARKER] == "1"


def test_child_failure_is_raised(monkeypatch):
    with patch.object(migrations.subprocess, "run") as run:
        run.return_value.returncode = 3
        with pytest.raises(RuntimeError, match="exit 3"):
            migrations._run_in_migration_child("run_migrations", {"database_url": "postgresql://x"})


def test_main_reads_stdin_and_dispatches(monkeypatch):
    """The child entrypoint reconstructs the exact call the parent asked for."""
    kwargs = {"database_url": "postgresql://x", "schema": "tenant_a"}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"target": "run_migrations", "kwargs": kwargs})))
    with patch.object(migrations, "run_migrations") as run_migrations_:
        migrations._main()
    run_migrations_.assert_called_once_with(**kwargs)
    assert os.environ[migrations._CHILD_MARKER] == "1"


def test_main_rejects_an_unknown_target(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"target": "drop_everything", "kwargs": {}})))
    with pytest.raises(SystemExit, match="drop_everything"):
        migrations._main()
