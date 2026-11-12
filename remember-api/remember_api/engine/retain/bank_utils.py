"""
bank profile utilities for disposition and mission management.
"""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict, cast

from pydantic import BaseModel, Field

from ..._vector_index import index_using_clause, per_bank_indexes_are_eager, uses_per_bank_vector_indexes
from ...config import get_config
from ...schema_context import get_current_schema
from ..db_utils import acquire_with_retry, retry_with_backoff
from ..response_models import DispositionTraits, id_problem
from ..schema import fq_table

if TYPE_CHECKING:
    from ..db.base import DatabaseConnection
    from ..db.ops import DataAccessOps

logger = logging.getLogger(__name__)

# Fact types that get per-bank partial vector indexes, mapped to their 4-char index suffix.
_BANK_INDEX_FACT_TYPES: dict[str, str] = {
    "world": "worl",
    "experience": "expr",
    "observation": "obsv",
}


def _bank_index_name(ft: str, internal_id: str) -> str:
    """Deterministic, schema-safe vector index name for a (bank, fact_type) pair.

    Uses the first 16 hex chars of internal_id (8 bytes of entropy) — unique
    enough in practice, fits comfortably within PostgreSQL's 63-char identifier limit.
    """
    uid = str(internal_id).replace("-", "")[:16]
    return f"idx_mu_emb_{_BANK_INDEX_FACT_TYPES[ft]}_{uid}"


def _vector_index_clause() -> str | None:
    """Return the USING clause for per-bank vector indexes, if this backend uses them."""
    ext = get_config().vector_extension
    if not uses_per_bank_vector_indexes(ext):
        return None
    return index_using_clause(ext)


async def create_bank_vector_indexes(
    conn: "DatabaseConnection", bank_id: str, internal_id: str, *, ops: "DataAccessOps"
) -> None:
    """Create per-(bank, fact_type) partial vector indexes for a newly created bank.

    Only does anything when the size threshold is off — the default — where a
    bank is entitled to its indexes from the moment it exists and building them
    here is instant, because the bank is empty. That is the behaviour that
    predates the threshold, and keeping it means the default deployment never
    submits a ``vector_index_maintenance`` operation and never pays a coverage
    check on a write.

    With ``REMEMBER_API_VECTOR_INDEX_MIN_ROWS`` set, a fresh bank has not earned
    anything yet, so this is a no-op and the maintenance operation builds an
    index CONCURRENTLY if and when the bank grows into one — which also keeps
    CREATE INDEX's ShareLock on the shared memory_units table off the write path.

    Builds pgvector HNSW indexes for the local bank.

    bank_id is escaped for SQL literal safety (apostrophes doubled).

    ``ops`` is required rather than defaulting to None: it is only dereferenced
    in the eager branch, so a caller that forgot it used to work fine until the
    threshold happened to be off, and then failed on an AttributeError from
    inside bank creation.
    """
    if not per_bank_indexes_are_eager():
        return

    index_clause = _vector_index_clause()
    if index_clause is None:
        logger.debug("Skipping per-bank vector indexes for configured backend")
        return

    from ..memories import sql_memories

    await sql_memories().create_bank_vector_indexes(
        conn=conn,
        ops=ops,
        fq_table=fq_table,
        bank_id=bank_id,
        internal_id=internal_id,
        index_clause=index_clause,
        fact_types=_BANK_INDEX_FACT_TYPES,
    )


async def drop_bank_vector_indexes(conn: "DatabaseConnection", internal_id: str, *, ops: "DataAccessOps") -> None:
    """Drop per-(bank, fact_type) partial vector indexes for a bank being deleted.

    Called before the bank row is deleted so internal_id is still known.
    Idempotent via DROP INDEX IF EXISTS.

    On Oracle, this is a no-op (uses single global vector index).
    """
    await ops.drop_bank_vector_indexes(
        conn,
        get_current_schema(),
        internal_id,
        _BANK_INDEX_FACT_TYPES,
    )


DEFAULT_DISPOSITION = {
    "skepticism": 3,
    "literalism": 3,
    "empathy": 3,
}


class BankProfile(TypedDict):
    """Type for bank profile data."""

    name: str
    disposition: DispositionTraits
    mission: str


@dataclass
class BankProfileResult:
    """Result of a get-or-create bank lookup.

    ``created`` is True when the bank row was freshly inserted on this call,
    which callers use to drive the one-time REMEMBER_API_DEFAULT_BANK_TEMPLATE hook.
    """

    profile: BankProfile
    created: bool


class MissionMergeResponse(BaseModel):
    """LLM response for mission merge."""

    mission: str = Field(description="Merged mission in first person perspective")


async def get_bank_profile(pool, bank_id: str) -> BankProfile:
    """
    Get bank profile (name, disposition + mission).
    Auto-creates bank with default values if not exists.

    Args:
        pool: Database connection pool
        bank_id: bank IDentifier

    Returns:
        BankProfile with name, typed DispositionTraits, and mission
    """
    result = await get_or_create_bank_profile(pool, bank_id)
    return result.profile


async def bank_exists_on_conn(conn: "DatabaseConnection", bank_id: str) -> bool:
    """Existence probe for a bank row on a caller-supplied connection.

    Deliberately narrower than ``get_bank_profile_if_exists_on_conn``: callers on
    the lazy-create hot path only need to know whether the row is there, so this
    selects a constant instead of reading and JSON-decoding the disposition.
    """
    return await conn.fetchval(f"SELECT 1 FROM {fq_table('banks')} WHERE bank_id = $1", bank_id) is not None


async def bank_kind_on_conn(conn: "DatabaseConnection", bank_id: str) -> str | None:
    """The bank's kind ('memory' or 'knowledge'), or None when there is no such bank."""
    return await conn.fetchval(f"SELECT kind FROM {fq_table('banks')} WHERE bank_id = $1", bank_id)


async def bank_kind(pool, bank_id: str) -> str | None:
    async with acquire_with_retry(pool) as conn:
        return await bank_kind_on_conn(conn, bank_id)


async def bank_exists(pool, bank_id: str) -> bool:
    """Dedicated-connection variant of ``bank_exists_on_conn``.

    A bare SELECT with no surrounding transaction — this is the read-only fast
    path taken by every write to an already-existing bank, so it must not pay
    for a BEGIN/COMMIT round trip.
    """
    async with acquire_with_retry(pool) as conn:
        return await bank_exists_on_conn(conn, bank_id)


async def get_bank_profile_if_exists_on_conn(conn: "DatabaseConnection", bank_id: str) -> BankProfile | None:
    """Get bank profile (name, disposition + mission) on conn without auto-creating.

    Returns None if the bank does not exist.
    """
    row = await conn.fetchrow(
        f"""
        SELECT name, disposition, mission
        FROM {fq_table("banks")} WHERE bank_id = $1
        """,
        bank_id,
    )
    if not row:
        return None
    disposition_data = row["disposition"]
    if isinstance(disposition_data, str):
        disposition_data = json.loads(disposition_data)
    return BankProfile(
        name=row["name"],
        disposition=DispositionTraits(**disposition_data),
        mission=row["mission"] or "",
    )


async def get_bank_profile_if_exists(pool, bank_id: str) -> BankProfile | None:
    """
    Get bank profile (name, disposition + mission) without auto-creating.

    Returns None if the bank does not exist. This is the read-only variant
    of get_bank_profile, intended for read endpoints where a bank that
    doesn't exist should surface as 404 rather than be silently created.

    Args:
        pool: Database connection pool
        bank_id: bank IDentifier

    Returns:
        BankProfile if the bank exists, otherwise None.
    """
    # Cached per process (see engine/bank_info_cache): on a hit this takes no pooled connection at
    # all, which is what lets a store-owned retain -- one that writes nothing to Postgres -- run
    # without touching the pool. A miss reads exactly as before.
    from .. import bank_info_cache

    async def _load() -> dict:
        async with acquire_with_retry(pool) as conn:
            profile = await get_bank_profile_if_exists_on_conn(conn, bank_id)
        # "The bank does not exist" has to be representable as a cached value, or every miss for a
        # missing bank re-reads it. `BankProfile` is a TypedDict, so the only thing that has to be
        # unpacked is `disposition`, which is a pydantic model the cache would otherwise hand back
        # by reference and let a caller mutate for every other holder of the entry.
        if profile is None:
            return {}
        return {**profile, "disposition": profile["disposition"].model_dump()}

    row = await bank_info_cache.get_or_load(bank_id, "profile", _load)
    if not row:
        return None
    return BankProfile(
        name=row["name"],
        disposition=DispositionTraits(**row["disposition"]),
        mission=row["mission"] or "",
    )


def validate_new_bank_id(bank_id: str) -> None:
    """Reject a bank id that must not be created. Existing banks are never re-checked."""
    from remember_api.extensions import OperationValidationError

    problem = id_problem(bank_id, "Bank id")
    if problem:
        raise OperationValidationError(problem, status_code=400)
    # Reserved so a store-owned backend can keep its own bookkeeping objects (`_index`, `_meta`, ...)
    # next to a bank's objects without escaping bank names. Existing banks are not re-checked.
    if bank_id.startswith("_"):
        raise OperationValidationError("Bank id must not start with '_': that prefix is reserved", status_code=400)


async def create_bank_row_on_conn(conn: "DatabaseConnection", bank_id: str, *, ops: "DataAccessOps") -> bool:
    """Idempotently insert a default bank row and its vector indexes on conn.

    Uses ``INSERT ... ON CONFLICT (bank_id) DO NOTHING RETURNING bank_id``.
    Returns True if the bank was freshly inserted on this call, False if it already existed.

    The ``created`` flag drives one-time side effects (per-bank vector indexes
    here, the default-bank-template hook in the caller), so it has to be exact on
    both dialects. It is: the Oracle layer strips RETURNING from ON CONFLICT DO
    NOTHING, but ``fetchval`` compensates — it returns the first bind argument on
    a successful insert and None when it suppresses ORA-00001 (see
    ``db/oracle.py``). Use ``fetchval`` here, not ``fetchrow``, which has no such
    compensation and would report every fresh Oracle insert as an existing row.
    """
    # internal_id is minted here rather than defaulted server-side so its value is
    # known without a RETURNING round-trip: the index names derive from it, both
    # for the eager create below and for the maintenance operation when a
    # threshold is set.
    # A bank must not be born under a name that already routes somewhere else: the
    # alias would be dead the moment this commits, since resolution prefers a real
    # bank. The mirror of the NOT EXISTS in `bank_aliases.create_alias` -- the two
    # tables cannot be covered by one constraint, so each write path checks the
    # other. Only on the create path, which already ran an existence probe, so the
    # common "bank exists" case pays nothing.
    from .. import bank_aliases

    await bank_aliases.refuse_alias_as_bank_id_on_conn(conn, bank_id)

    internal_id = uuid.uuid4()
    inserted = await conn.fetchval(
        f"""
        INSERT INTO {fq_table("banks")} (bank_id, name, disposition, mission, internal_id)
        VALUES ($1, $2, $3::jsonb, $4, $5)
        ON CONFLICT (bank_id) DO NOTHING
        RETURNING bank_id
        """,
        bank_id,
        bank_id,  # Default name is the bank_id
        json.dumps(DEFAULT_DISPOSITION),
        "",
        internal_id,
    )

    created = inserted is not None
    if created:
        # Fresh insert — create per-bank vector indexes (instant on an empty
        # bank). A no-op unless the size threshold is off; see
        # create_bank_vector_indexes.
        await create_bank_vector_indexes(conn, bank_id, str(internal_id), ops=ops)

    return created


async def create_bank_if_missing(pool, bank_id: str) -> bool:
    """Dedicated-connection variant of ``create_bank_row_on_conn``.

    Returns True if freshly created on this call, False if it already existed.
    """

    # Retried as a whole transaction. With the size threshold off (the default)
    # a fresh SQL-owned bank builds its per-(bank, fact_type) partial vector
    # indexes with a plain CREATE INDEX — it must, since this runs inside the
    # bank-create tx and CONCURRENTLY cannot — and that CREATE takes a ShareLock
    # on the shared memory_units table, which can deadlock with concurrent
    # writers. Even with no DDL to issue, the lazy create can lose a deadlock
    # (40P01 / ORA-00060) to a concurrent writer touching the same bank row.
    # The body is idempotent
    # (INSERT ... ON CONFLICT DO NOTHING + CREATE INDEX IF NOT EXISTS), so
    # retrying the whole tx stays correct and cheap.
    async def _create() -> bool:
        async with acquire_with_retry(pool) as conn:
            async with conn.transaction():
                return await create_bank_row_on_conn(conn, bank_id, ops=pool.ops)

    return await retry_with_backoff(_create)


async def get_or_create_bank_profile(pool, bank_id: str) -> BankProfileResult:
    """
    Get bank profile, auto-creating with defaults if it doesn't exist.

    Same as get_bank_profile, but also reports whether the bank was freshly
    created on this call (``BankProfileResult.created``). Used by the memory
    engine to apply the REMEMBER_API_DEFAULT_BANK_TEMPLATE hook on first bank
    creation.

    Acquires its own connection. When the caller already holds a connection and
    wants the bank row to share its transaction (so the lazy bank-create commits
    or rolls back atomically with the caller's write), use
    ``get_or_create_bank_profile_on_conn`` instead.
    """

    # Retried as a whole transaction — see the deadlock note on
    # ``create_bank_if_missing``; the body is idempotent, so retrying is safe.
    async def _create() -> BankProfileResult:
        async with acquire_with_retry(pool) as conn:
            async with conn.transaction():
                return await get_or_create_bank_profile_on_conn(conn, bank_id, ops=pool.ops)

    return await retry_with_backoff(_create)


async def get_or_create_bank_profile_on_conn(
    conn: "DatabaseConnection", bank_id: str, *, ops: "DataAccessOps"
) -> BankProfileResult:
    """
    Connection-bound variant of ``get_or_create_bank_profile``.

    Runs the SELECT, the ``INSERT ... ON CONFLICT DO NOTHING`` and the per-bank
    vector index creation on the caller-supplied ``conn``. When ``conn`` is
    inside an open transaction, the lazy bank-create therefore commits (or rolls
    back) atomically with whatever bank-scoped write the caller performs on the
    same connection — closing the window where a freshly-created bank could
    outlive a write that ultimately failed.

    ``ops`` is the backend's dialect ops object (``backend.ops``), needed for
    per-bank vector index DDL.
    """
    profile = await get_bank_profile_if_exists_on_conn(conn, bank_id)
    if profile is not None:
        return BankProfileResult(
            profile=profile,
            created=False,
        )

    # Bank doesn't exist, create with defaults.
    created = await create_bank_row_on_conn(conn, bank_id, ops=ops)

    return BankProfileResult(
        profile=BankProfile(name=bank_id, disposition=DispositionTraits(**DEFAULT_DISPOSITION), mission=""),
        created=created,
    )


async def update_bank_disposition(pool, bank_id: str, disposition: dict[str, int]) -> None:
    """
    Update bank disposition traits.

    Args:
        pool: Database connection pool
        bank_id: bank IDentifier
        disposition: Dict with skepticism, literalism, empathy (all 1-5)
    """
    # Ensure bank exists first
    await get_bank_profile(pool, bank_id)

    async with acquire_with_retry(pool) as conn:
        await conn.execute(
            f"""
            UPDATE {fq_table("banks")}
            SET disposition = $2::jsonb,
                updated_at = NOW()
            WHERE bank_id = $1
            """,
            bank_id,
            json.dumps(disposition),
        )

    # The profile row just changed; the next read must not serve the entry loaded before it.
    from .. import bank_info_cache

    await bank_info_cache.invalidate(bank_id, "profile")


async def set_bank_mission(pool, bank_id: str, mission: str) -> None:
    """
    Set bank mission (replacing any existing mission).

    Args:
        pool: Database connection pool
        bank_id: bank IDentifier
        mission: The mission text
    """
    # Ensure bank exists first
    await get_bank_profile(pool, bank_id)

    async with acquire_with_retry(pool) as conn:
        await conn.execute(
            f"""
            UPDATE {fq_table("banks")}
            SET mission = $2,
                updated_at = NOW()
            WHERE bank_id = $1
            """,
            bank_id,
            mission,
        )

    # The profile row just changed; the next read must not serve the entry loaded before it.
    from .. import bank_info_cache

    await bank_info_cache.invalidate(bank_id, "profile")


# Sort floor for banks that have never been written to and carry no created_at.
_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _as_utc(ts: datetime | None) -> datetime | None:
    """Normalize a DB timestamp to an aware UTC datetime so values stay comparable."""
    if ts is None:
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


def _aliases_explaining(
    bank_id: str, name: str | None, search_query: str | None, matched: dict[str, list[str]]
) -> list[str]:
    """The aliases worth reporting for one search hit — often none.

    This answers "why is this row in my results", so it is empty whenever the row
    already explains itself: a bank whose own id or name contains the search text
    needs no footnote, and searching "acme" would otherwise tag the acme row with
    every acme-* alias it happens to own. Only a bank reached *purely* through an
    alias gets one.
    """
    if not search_query:
        return []
    needle = search_query.upper()
    if needle in bank_id.upper() or needle in (name or "").upper():
        return []
    return matched.get(bank_id, [])


async def list_banks(pool, *, search_query: str | None = None) -> list:
    """
    List banks with summary stats, optionally narrowed by a search string.

    ``last_document_at`` is document *ingestion* time (when a document first
    landed), while ``last_write_at`` is the last time anything was written to
    the bank — a document re-retained/appended to, or a fact stored. Appending
    to a long-lived document does not move ``last_document_at``, which is why
    the two differ and why UIs showing "last write" must use ``last_write_at``.

    ``fact_count`` comes back 0 here and is filled per page by
    :func:`apply_sql_fact_counts`. Counting every bank's facts in this query meant
    aggregating all of ``memory_units`` to render one page of banks, which is what made a
    365-bank list time out (#4468). The write watermark still has to be read for every bank,
    since it orders the list — but as one index entry per bank off an index that already
    exists, not a GROUP BY over the whole table. Measured on 365 banks / 1.8M facts: 131 ms
    to 6 ms, and no longer growing with the number of facts in the system.

    Args:
        pool: Database connection pool
        search_query: Case-insensitive substring matched against bank ID, name and aliases

    Returns:
        List of dicts with bank info and stats (fact_count, last_document_at, last_write_at),
        most recently written bank first.
    """
    # Spelled out as UPPER(...) LIKE UPPER(...) rather than ILIKE: the Oracle
    # rewriter only recognizes ILIKE on an unqualified column, and these are
    # alias-qualified.
    # Knowledge banks share the `banks` table but have their own list (see remember_api/knowledge).
    where_clause = "WHERE b.kind = 'memory'"
    params: list[str] = []
    if search_query:
        # Aliases are matched too, so searching the id a caller actually uses finds
        # the bank it reaches. Mid-migration that is the only id someone may know:
        # without this, the new id is live in the API but invisible in the picker.
        where_clause = (
            "WHERE b.kind = 'memory' AND (UPPER(b.bank_id) LIKE UPPER($1) OR UPPER(COALESCE(b.name, '')) LIKE UPPER($2) "
            f"OR EXISTS (SELECT 1 FROM {fq_table('bank_aliases')} a "
            "WHERE a.bank_id = b.bank_id AND UPPER(a.alias) LIKE UPPER($3)))"
        )
        params = [f"%{search_query}%", f"%{search_query}%", f"%{search_query}%"]

    from ..memories import sql_memories

    async with acquire_with_retry(pool) as conn:
        # The bank rows with their document and fact write watermarks, read from Postgres for
        # every bank: the ids are not known before this query, so it cannot be split by owner.
        rows = await sql_memories().list_bank_rows(
            conn=conn, fq_table=fq_table, where_clause=where_clause, params=params
        )

        # Which alias made a bank match, so a result found by an id the bank does not
        # itself carry can say so — otherwise searching "acme-prod" silently returns
        # "acme" and the hit looks wrong. A separate small query rather than an
        # aggregate in the one above: array aggregation is spelled differently on
        # PostgreSQL and Oracle, and this runs only when a search was typed.
        matched_aliases: dict[str, list[str]] = {}
        if search_query:
            for alias_row in await conn.fetch(
                f"SELECT bank_id, alias FROM {fq_table('bank_aliases')} "
                "WHERE UPPER(alias) LIKE UPPER($1) ORDER BY alias",
                f"%{search_query}%",
            ):
                matched_aliases.setdefault(alias_row["bank_id"], []).append(alias_row["alias"])

        result = []
        # Banks are ordered by last write in Python rather than SQL: GREATEST() has
        # different NULL semantics on PostgreSQL vs Oracle, and the bank list is small.
        sort_keys: dict[str, datetime] = {}

        for row in rows:
            disposition_data = row["disposition"]
            if isinstance(disposition_data, str):
                disposition_data = json.loads(disposition_data)

            last_doc = _as_utc(row["last_document_at"])
            created_at = _as_utc(row["created_at"])
            updated_at = _as_utc(row["updated_at"])
            # Last write = newest of "a document was (re-)retained" and "a fact was stored".
            # Appending to an existing document only bumps documents.updated_at, and facts
            # written outside a retain (consolidation, curation, import) only bump memory_units.
            write_times = [t for t in (_as_utc(row["last_document_write_at"]), _as_utc(row["last_fact_at"])) if t]
            last_write = max(write_times) if write_times else None

            sort_keys[row["bank_id"]] = last_write or created_at or _UNIX_EPOCH
            result.append(
                {
                    "bank_id": row["bank_id"],
                    "name": row["name"],
                    "disposition": disposition_data,
                    "mission": row["mission"] or "",
                    "created_at": created_at.isoformat() if created_at else None,
                    "updated_at": updated_at.isoformat() if updated_at else None,
                    # Filled per page by apply_sql_fact_counts: counting
                    # every bank's facts here is what made this query scan all of memory_units.
                    "fact_count": 0,
                    "last_document_at": last_doc.isoformat() if last_doc else None,
                    "last_write_at": last_write.isoformat() if last_write else None,
                    # Empty unless this search matched one of the bank's aliases. Not
                    # the bank's full alias list: it answers "why is this row here",
                    # which is only a question when something was searched for.
                    "matched_aliases": _aliases_explaining(row["bank_id"], row["name"], search_query, matched_aliases),
                }
            )

    # The row dicts are heterogeneous, so their value type is a union; the id is a str.
    result.sort(key=lambda bank: sort_keys[cast(str, bank["bank_id"])], reverse=True)
    return result


@dataclass
class BankPage:
    """One page of bank rows, plus how many banks matched before paging."""

    banks: list[dict]
    total: int


async def list_banks_page(pool, *, limit: int, offset: int, search_query: str | None = None) -> BankPage:
    """Return a page of local banks, ordered by their latest write."""
    banks = await list_banks(pool, search_query=search_query)
    return BankPage(banks=banks[offset : offset + limit], total=len(banks))


async def apply_sql_fact_counts(pool, banks: list[dict]) -> None:
    """Fill memory counts for the returned page of local banks."""
    if not banks:
        return
    from ..memories import sql_memories

    async with acquire_with_retry(pool) as conn:
        counts = await sql_memories().bank_fact_counts(
            conn=conn, fq_table=fq_table, bank_ids=[bank["bank_id"] for bank in banks]
        )
    for bank in banks:
        bank["fact_count"] = counts.get(bank["bank_id"], 0)
