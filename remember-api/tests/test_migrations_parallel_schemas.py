"""The local migration runs its steps once, in order."""

from remember_api import migrations


def test_local_migration_runs_all_steps_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(migrations, "_should_isolate_migrations", lambda: False)
    for name in (
        "run_migrations",
        "ensure_embedding_dimension",
        "ensure_vector_extension",
        "ensure_text_search_extension",
    ):
        monkeypatch.setattr(
            migrations, name, lambda *args, _name=name, **kwargs: calls.append((_name, kwargs.get("schema")))
        )
    migrations.run_local_migrations("postgresql://local", "local_schema", embedding_dimension=384)
    assert calls == [
        (name, "local_schema")
        for name in (
            "run_migrations",
            "ensure_embedding_dimension",
            "ensure_vector_extension",
            "ensure_text_search_extension",
        )
    ]


def test_local_migration_can_skip_optional_reconciliation(monkeypatch):
    calls = []
    monkeypatch.setattr(migrations, "_should_isolate_migrations", lambda: False)
    monkeypatch.setattr(migrations, "run_migrations", lambda *args, **kwargs: calls.append("migration"))
    monkeypatch.setattr(migrations, "ensure_vector_extension", lambda *args, **kwargs: calls.append("vector"))
    monkeypatch.setattr(migrations, "ensure_text_search_extension", lambda *args, **kwargs: calls.append("text"))
    migrations.run_local_migrations("postgresql://local", "local_schema", ensure_extensions=False)
    assert calls == ["migration"]
