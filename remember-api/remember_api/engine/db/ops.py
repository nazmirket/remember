"""Data types and SQL helpers shared by PostgreSQL operations."""

from dataclasses import dataclass
from datetime import datetime

from .result import ResultRow


class ChunkIdOwnedByAnotherBank(Exception):
    """A chunk upsert hit a ``chunks`` row that belongs to a different bank.

    ``chunks`` is keyed on ``chunk_id`` alone, so the row can only be one bank's. Ids
    built by ``engine/chunk_ids.py`` cannot collide across banks; ones written before
    that fix can, and overwriting is how #4244 leaked one bank's chunk into another.
    """

    def __init__(self, chunk_ids: list[str]) -> None:
        self.chunk_ids = chunk_ids
        super().__init__(f"Chunk id(s) already owned by another bank, refusing to overwrite: {chunk_ids}")


#: The ``memory_units`` columns every link-expansion arm projects, in the order the
#: arms are ``UNION ALL``-ed together.  Order is part of the contract, not a style
#: choice: the arms are combined positionally, so two arms listing the same columns
#: in different orders would union cleanly and silently mis-assign every value.
MEMORY_UNIT_COLUMNS: tuple[str, ...] = (
    "id",
    "text",
    "context",
    "event_date",
    "occurred_start",
    "occurred_end",
    "mentioned_at",
    "fact_type",
    "document_id",
    "chunk_id",
    "tags",
    "proof_count",
)


def memory_unit_columns(alias: str = "", *, indent: int = 0) -> str:
    """The shared ``memory_units`` projection for link expansion, optionally alias-qualified.

    Single source of truth for a list that link expansion repeats ~20 times across
    both backends — once per ``UNION ALL`` arm, again in the PostgreSQL semantic
    arm's ``GROUP BY``, and again in each dialect's outer re-projection.  Spelling
    it out per site made adding a column a 20-edit change where *every* miss is a
    failure: a dropped column breaks the union arity, a reordered one corrupts the
    results silently (see ``MEMORY_UNIT_COLUMNS``), and a ``GROUP BY`` left behind
    is a runtime SQL error on a path only Oracle or a live recall exercises.

    The score/weight/source expressions that follow the projection stay at the call
    sites — they are what actually differs between the arms, and hiding them here
    would trade a real distinction for a false one.

    Args:
        alias: Correlation name to qualify each column with (e.g. ``"mu"``).  Pass
            ``""`` for bare columns, as subquery re-projections and ``GROUP BY``
            lists need.
        indent: Spaces to indent continuation lines by, so the generated SQL stays
            readable in logs and ``EXPLAIN`` output.
    """
    prefix = f"{alias}." if alias else ""
    columns = [f"{prefix}{column}" for column in MEMORY_UNIT_COLUMNS]
    # Four per line keeps the longest alias-qualified row well inside the 120-column
    # limit the rest of the file is formatted to.
    lines = [", ".join(columns[i : i + 4]) for i in range(0, len(columns), 4)]
    return (",\n" + " " * indent).join(lines)


def key_serialization_sql(table: str, alias: str) -> str:
    """SQL predicate keeping one ``serialization_key`` to a single in-flight run.

    Two kinds of work carry a key. A retain that targets exactly one document
    carries that document: appending to a document is a read-modify-write over
    its whole text, so two concurrent retains for one document can only produce
    a lost update or a wasted extraction — never more throughput. A mental-model
    refresh carries ``mental_model:<id>``: two refreshes of one model both write
    the model, so the one that finished second used to win regardless of which
    read more, and in delta mode they moved each other's watermark.

    This predicate makes the queue reflect that: a candidate is claimable only
    when no peer for the same key is already ``processing``, and only when it is
    the oldest claimable pending peer for that key. Peers must also share the
    candidate's ``operation_type``: document ids are caller-supplied, so a key
    alone cannot keep a document from colliding with a model's refreshes.

    Ordering, not just exclusion, is the point. Appends are cumulative, so the
    order they commit in is the order the document ends up in; claiming them by
    ``(created_at, operation_id)`` makes that the submission order. It also
    stops a single claim batch from taking several peers at once, which
    excluding busy keys alone would not prevent.

    Rows with a NULL ``serialization_key`` — multi-document batches, and every
    operation whose runs are independent — are unaffected, and keys are
    independent of one another, so this costs no parallelism across a busy bank:
    only the operations that were racing each other for one document, or one
    model, are put in a line.

    A peer wedged in 'processing' holds its key until claim recovery releases
    it, the same caveat ``bank_serialization_sql`` carries and the same general
    gap.

    The candidate row is always 'pending' and the 'pending' branch is
    strictly-older, so the subquery can never match the candidate itself. The
    fragment carries no SQL comments on purpose — it is rewritten for Oracle by
    regex (``db/oracle.py``), and the peer alias is left at the ``doc_peer`` it
    was born with for the same reason: nothing about this text is worth a
    rewriter surprise.

    Args:
        table: Fully-qualified async_operations table.
        alias: Alias of the outer candidate row in the calling query.
    """
    return f"""
        ({alias}.serialization_key IS NULL OR NOT EXISTS (
            SELECT 1 FROM {table} doc_peer
            WHERE doc_peer.bank_id = {alias}.bank_id
              AND doc_peer.serialization_key = {alias}.serialization_key
              AND doc_peer.operation_type = {alias}.operation_type
              AND (
                  doc_peer.status = 'processing'
                  OR (doc_peer.status = 'pending'
                      AND doc_peer.task_payload IS NOT NULL
                      AND (doc_peer.next_retry_at IS NULL OR doc_peer.next_retry_at <= NOW())
                      AND (doc_peer.created_at < {alias}.created_at
                           OR (doc_peer.created_at = {alias}.created_at
                               AND doc_peer.operation_id < {alias}.operation_id)))
              )
        ))
    """


_BANK_SERIALIZED_OPERATION_TYPES = ("graph_maintenance", "consolidation")


def bank_serialization_sql(table: str, alias: str, operation_type: str | None = None) -> str:
    """SQL predicate keeping one bank to a single in-flight run, per operation type.

    Applies to the operation types whose runs for a bank are *interchangeable*
    — the payload carries only ``bank_id`` and the job drains that bank's whole
    backlog — so a second concurrent run for the same bank recomputes the first
    one's work and adds nothing:

    * ``graph_maintenance`` (#3230). ``claim_graph_maintenance_batch`` locks
      queue rows ``FOR UPDATE`` *without* ``SKIP LOCKED`` (it is written
      assuming a single runner per bank), so concurrent runs also convoy on each
      other's row locks while each holds a worker slot.
    * ``consolidation`` (#3700). Two runs claimed together read the same
      ``total_unconsolidated`` backlog and hand the same memories to the LLM
      twice — pure duplicated inference on the slowest path in the system.

    Ordering, not just exclusion, is the point, and both halves of the predicate
    are needed:

    * The ``processing`` branch is the guarantee across batches and across
      workers: a bank with a run in flight is not claimed again.
    * The strictly-older ``pending`` branch is the guarantee *within* one batch.
      Excluding busy banks alone does not give it: with several pending rows and
      nothing yet processing, one batch claims them all. Several pending rows per
      bank are reachable through the recovery paths
      (``_reclaim_own_processing_tasks`` resets *all* of a worker's processing
      rows in one statement, from ``recover_own_tasks`` at startup and
      ``release_own_tasks`` at shutdown, plus ``_schedule_retry`` /
      ``_defer_operation`` / ``remember-admin recover``). It also covers the
      window in which a peer is claimed but not yet committed as ``processing``:
      a concurrent worker reads it as the older ``pending`` row and stands down.

    This is a **predicate**, not a separate claim phase, because
    ``graph_maintenance`` cannot afford to be one: it has no reserved-slot floor
    (``WORKER_SLOT_TYPE_DEFAULTS`` gives consolidation 2 and graph_maintenance
    0), and the poller's fairness pass calls ``claim_tasks`` with
    ``shared_limit=1``, so a phase after the generic shared-pool query would let
    a single pending retain starve it indefinitely. As a predicate it keeps
    competing by ``created_at``. Consolidation has its own claim path (bank
    priority tiers); the same predicate goes on every query there.

    Suppression here only *defers* — unlike the submit-time dedup in
    ``_submit_async_operation``, nothing is dropped. That is why scoped
    consolidations (``observation_scopes``, which submit-time dedup deliberately
    exempts because a scoped run covers only its tag subset) need no carve-out:
    they queue behind the bank's in-flight run instead of racing it, and are
    claimed on a later poll with a fresh watermark.

    The caveat is unchanged: a row wedged in 'processing' holds its bank until
    something releases it (``remember-admin recover``, or a restart with a
    stable ``REMEMBER_API_WORKER_ID`` so ``recover_own_tasks`` matches it). That
    is a general gap in claim recovery, not specific to these operation types.

    Rows of any other operation type are unaffected, and banks are independent
    of one another, so this costs no parallelism anywhere else.

    The candidate row is always 'pending' and the 'pending' branch is
    strictly-older, so the subquery can never match the candidate itself. The
    fragment carries no SQL comments on purpose — it is rewritten for Oracle by
    regex (``db/oracle.py``).

    Args:
        table: Fully-qualified async_operations table.
        alias: Alias of the outer candidate row in the calling query.
        operation_type: The one type the calling query is already restricted to,
            if it is restricted to one — consolidation has its own claim path.
            Matches peers against that constant and drops the guard on the
            candidate's type, which can only be true there. Leave unset for the
            mixed-type claim queries, where the candidate's own type has to pick
            its peers and rows of every other type must fall straight through.
            Inlined into the SQL text, so it is checked against the known set.
    """
    if operation_type is None:
        types = ", ".join(f"'{t}'" for t in _BANK_SERIALIZED_OPERATION_TYPES)
        candidate_guard = f"{alias}.operation_type NOT IN ({types}) OR "
        peer_type = f"{alias}.operation_type"
    elif operation_type in _BANK_SERIALIZED_OPERATION_TYPES:
        candidate_guard = ""
        peer_type = f"'{operation_type}'"
    else:
        raise ValueError(f"{operation_type} is not serialised per bank: {_BANK_SERIALIZED_OPERATION_TYPES}")

    return f"""
        ({candidate_guard}NOT EXISTS (
            SELECT 1 FROM {table} bank_peer
            WHERE bank_peer.bank_id = {alias}.bank_id
              AND bank_peer.operation_type = {peer_type}
              AND (
                  bank_peer.status = 'processing'
                  OR (bank_peer.status = 'pending'
                      AND bank_peer.task_payload IS NOT NULL
                      AND (bank_peer.next_retry_at IS NULL OR bank_peer.next_retry_at <= NOW())
                      AND (bank_peer.created_at < {alias}.created_at
                           OR (bank_peer.created_at = {alias}.created_at
                               AND bank_peer.operation_id < {alias}.operation_id)))
              )
        ))
    """


@dataclass
class ClaimedOperations:
    """One claim cycle's rows, plus where the bank rotation got to.

    ``claim_tasks`` picks the bank whose turn it is *inside* the claim query, so
    the caller cannot know which bank that was from the rows alone — the
    rotation row is not distinguishable from the ones claimed by age. Handing
    the cursor back keeps the rotation's state in the poller (where the tenant
    rotation already lives) without costing a second statement to ask.
    """

    rows: list[ResultRow]
    """Claimed rows, rotation row first, then oldest-first."""

    next_bank_cursor: str
    """Bank served by the rotation, to claim past next time. Empty string starts
    a new round — which is what an empty rotation tier means: no bank sorts after
    the cursor any more."""


@dataclass
class TagListingParts:
    """Backend-specific SQL fragments for the tag listing query."""

    tag_source: str
    non_empty_check: str
    tag_col: str
    bank_prefix: str


@dataclass(frozen=True)
class UpdatedWindow:
    """Recall's ``created_after``/``created_before`` bounds, as SQL for graph expansion.

    Recall applies the window to ``updated_at`` — a consolidation touch makes a
    fact current again — so link expansion has to bound the same column its seed
    query does.  Filtering only the seeds is not enough: a single in-window seed
    would otherwise drag its whole neighbourhood (shared entities, semantic kNN
    links, causal links) into the results no matter how old those neighbours are.

    ``first_param_index`` is where the bounds land in the owning query's param
    list, so each call site keeps the placeholder numbering next to the params it
    binds.  Rendering is per-alias because the same window is applied to several
    correlation names within one query.
    """

    after: datetime | None
    before: datetime | None
    first_param_index: int

    def clause(self, alias: str) -> str:
        """``AND <alias>.updated_at > $n ...`` — empty when the window is unbounded."""
        parts: list[str] = []
        index = self.first_param_index
        if self.after is not None:
            parts.append(f" AND {alias}.updated_at > ${index}")
            index += 1
        if self.before is not None:
            parts.append(f" AND {alias}.updated_at < ${index}")
        return "".join(parts)

    @property
    def params(self) -> list[datetime]:
        """The bound values, in placeholder order. Append to the owning param list."""
        return [bound for bound in (self.after, self.before) if bound is not None]


@dataclass(frozen=True)
class LinkExpansionRows:
    """The three link-expansion signals, kept apart until they are scored.

    They cannot be concatenated at the SQL layer: each carries a different score
    scale (shared-entity count, kNN weight, causal weight) and the caller applies
    a different transformation to each before summing them.
    """

    entity: list[ResultRow]
    semantic: list[ResultRow]
    causal: list[ResultRow]
