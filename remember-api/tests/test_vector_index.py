from pathlib import Path

from remember_api import _vector_index
from remember_api._vector_index import (
    ann_search_tuning_settings,
    bootstrap_extension,
    configured_vector_extension,
    index_using_clause,
    validate_extension,
)
from tests.pg_extension_fakes import FakePgConnection


def test_index_using_clause_pgvector_matches_existing_clause():
    assert index_using_clause("pgvector") == "USING hnsw (embedding vector_cosine_ops)"


def test_bootstrap_extension_skips_ddl_when_installed():
    """An installed extension must not be re-created.

    CREATE EXTENSION IF NOT EXISTS is a no-op on a writable session but an
    error on a read-only one, and that pointless failure aborts the
    transaction — which used to strand the migration advisory lock (#4611).
    """
    conn = FakePgConnection(extensions={"vector": ("public", True)})

    bootstrap_extension(conn, "pgvector")

    assert conn.created_extensions() == []


def test_bootstrap_extension_still_installs_when_missing():
    conn = FakePgConnection()

    bootstrap_extension(conn, "pgvector")

    assert conn.created_extensions() == ["vector"]


def test_ann_search_tuning_settings_pgvector_dispatches_hnsw_ef_search():
    # Looked up rather than compared as a whole tuple: the profiles also carry the
    # iterative-scan settings, which test_ann_iterative_scan.py covers. Pinning the
    # exact tuple here would make every future addition to a profile fail this test
    # for a reason it is not about.
    assert dict(ann_search_tuning_settings("pgvector", kind="low_latency"))["hnsw.ef_search"] == "60"
    assert dict(ann_search_tuning_settings("pgvector", kind="high_recall"))["hnsw.ef_search"] == "200"


def test_configured_vector_extension_defaults_to_pgvector(monkeypatch):
    monkeypatch.delenv("REMEMBER_API_VECTOR_EXTENSION", raising=False)
    assert configured_vector_extension() == "pgvector"


def test_configured_vector_extension_reads_env_and_lowercases(monkeypatch):
    monkeypatch.setenv("REMEMBER_API_VECTOR_EXTENSION", "PGVECTOR")
    assert configured_vector_extension() == "pgvector"


def test_configured_vector_extension_rejects_unknown_value(monkeypatch):
    import pytest

    monkeypatch.setenv("REMEMBER_API_VECTOR_EXTENSION", "bogus")
    with pytest.raises(ValueError):
        configured_vector_extension()


def test_alembic_vector_migrations_freeze_vector_sql_locally():
    # Pre-existing vchord-related migrations keep their vector SQL frozen
    # (inline helpers, no import from remember_api._vector_index) so their
    # historical behaviour cannot drift when _vector_index.py is updated.
    # New vchord migrations are reviewed for this trade-off individually and
    # may import the shared helpers when they want to share the canonical
    # source of truth (see b8c9d0e1f2a3_vchord_cosine_opclass.py).
    migration_dir = Path(__file__).resolve().parent.parent / "remember_api/alembic/versions"
    frozen_migrations = [
        "5a366d414dce_initial_schema.py",
        "a4b5c6d7e8f9_fix_per_bank_vector_index_type.py",
        "d5e6f7a8b9c0_add_bank_internal_id_and_per_bank_hnsw.py",
        "n9i0j1k2l3m4_learnings_and_pinned_reflections.py",
    ]

    for migration in frozen_migrations:
        text = (migration_dir / migration).read_text()
        assert "remember_api._vector_index" not in text


class _ThresholdConfig:
    def __init__(self, min_rows: int):
        self.vector_index_min_rows = min_rows


def _with_threshold(monkeypatch, min_rows: int) -> None:
    monkeypatch.setattr(_vector_index, "get_config", lambda: _ThresholdConfig(min_rows), raising=False)
    monkeypatch.setattr("remember_api.config.get_config", lambda: _ThresholdConfig(min_rows))


def test_build_bound_is_the_threshold_itself(monkeypatch):
    """The build side is a floor, inclusive: exactly the threshold earns an index."""
    _with_threshold(monkeypatch, 10_000)

    assert _vector_index.per_bank_index_build_bound() == 10_000


def test_zero_threshold_is_eager_not_a_zero_minimum(monkeypatch):
    """0 turns the threshold off rather than setting it to nothing.

    The distinction is the whole point of the default. Read as "no minimum", 0
    still routed every bank through the lazy path: an index earned on first write
    via a visible async operation, and every later write paying a coverage check
    to rediscover there was nothing to do. Read as "off", the bank gets its
    indexes when it is created and nothing looks at row counts ever again — which
    is what the deployment had before the threshold existed.
    """
    _with_threshold(monkeypatch, 0)
    assert _vector_index.per_bank_indexes_are_eager()

    _with_threshold(monkeypatch, 10_000)
    assert not _vector_index.per_bank_indexes_are_eager()


def test_an_emptied_partition_loses_its_index_at_every_threshold(monkeypatch):
    """The keep bound never falls to zero, whatever the ratio works out to.

    Without a floor of 1 a partition emptied of rows still clears the bound, so a
    bank written to once and then cleared would hold three ANN indexes over
    nothing — forever, because nothing writes to an emptied bank. That is the
    accumulation the threshold exists to prevent.
    """
    for threshold in (1, 10_000):
        _with_threshold(monkeypatch, threshold)
        assert _vector_index.per_bank_index_keep_bound() >= 1


def test_the_hysteresis_band_is_non_empty(monkeypatch):
    """Keeping has to start strictly below building, with room in between.

    With a single boundary a bank hovering at the threshold rebuilds and drops
    the same ANN index on alternating writes. The band is what the planner leaves
    alone: a partition inside it neither earns an index nor loses the one it has.
    """
    _with_threshold(monkeypatch, 10_000)

    keep = _vector_index.per_bank_index_keep_bound()
    build = _vector_index.per_bank_index_build_bound()

    assert keep < build
    assert build - keep > 1, "no row count sits strictly inside the band"


def test_other_vector_extensions_are_rejected():
    import pytest

    for extension in ("vchord", "pgvectorscale", "pg_diskann", "scann"):
        with pytest.raises(ValueError):
            validate_extension(extension)
