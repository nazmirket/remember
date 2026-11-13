"""Data types and helpers for the local PostgreSQL memory store."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from ..search.tags import TagsMatch
from ..search.types import RetrievalResult

logger = logging.getLogger(__name__)

# Keys used in an implementation's opaque metadata bag for the `memory_units`
# columns it has no first-class model of. These round-trip verbatim: they are
# stored without interpretation and returned on every hit, which is what lets
# recall rebuild a full result row without touching Postgres.
#
# Nothing here is queryable — an implementation cannot filter or sort on these. A
# column that retrieval must *filter* on has to be modelled properly instead.
META_CONTEXT = "context"
META_DOCUMENT_ID = "document_id"
META_CHUNK_ID = "chunk_id"
META_METADATA_JSON = "metadata_json"
META_OBSERVATION_SCOPES = "observation_scopes"
META_TEXT_SIGNALS = "text_signals"
META_CREATED_AT = "created_at"
#: When the memory last changed, and the contract every write path owes it (#3490):
#: a write that changes what the memory *is* — text, context, dates, fact_type, tags,
#: metadata, embedding, an observation's sources — stamps ``updated_at``, so a consumer
#: chasing ``WHERE updated_at > watermark`` sees the change. Those consumers are
#: incremental export, cache invalidation, the mental-model staleness check
#: (:meth:`any_memory_updated_since`) and its delta refresh — and recall's own
#: ``created_after`` / ``created_before`` window, which despite the name filters on this
#: column, so what stamps it also decides what a date-bounded recall returns.
#:
#: The consolidation *scheduler* is the one deliberate exception: when a pass records
#: that it folded a fact (or requeues one whose observation went away) it writes only
#: ``consolidated_at`` / ``consolidation_failed_at``, which are scheduler state rather
#: than the memory. Stamping there would make every pass look like an edit to every fact
#: it folded — re-flagging mental models stale and re-feeding unchanged facts to a delta
#: refresh. :meth:`PostgresMemories.mark_consolidated` and the requeue sites that clear
#: the markers inline therefore leave the column alone.
#:
#: The exemption is that *situation*, not the two columns: a write that clears the markers
#: as part of a real change to the memory still stamps — :meth:`restore_memory` brings an
#: archived memory back and resets it for re-consolidation in one statement, and that is an
#: edit. A store that owns memories itself is expected to keep the same contract.
#:
#: No timestamp can report a hard delete; a consumer that must catch those needs a
#: content fingerprint, not a watermark.
META_UPDATED_AT = "updated_at"
# Observation bookkeeping. `source_memory_ids` is a JSON list: an implementation
# with no edge relation carries an observation's sources denormalised.
META_SOURCE_MEMORY_IDS = "source_memory_ids"
META_CONSOLIDATED_AT = "consolidated_at"
# A *positive* flag mirroring META_CONSOLIDATED_AT, because a metadata predicate
# can only match equality — there is no "key is absent". Consolidation's candidate
# query is "not yet consolidated", so it needs a value to match on: every memory is
# written with "0" and flipped to "1" once folded into an observation.
META_CONSOLIDATED_FLAG = "consolidated"
#: The attachments a fact was drawn from, as a JSON list of short ids — the per-fact
#: provenance the extractor records. Carried on the memory so read surfaces can resolve
#: it from the rows the store already returned, without a second lookup.
META_ATTACHMENT_IDS = "attachment_ids"

# Keys in a store-owned DOCUMENT record's metadata map (string -> string, so structured values
# travel as one JSON string each). Unlike the memory bag above these describe the document, and
# the same record read serves every one of them.
#: The document's replayable retain parameters, as one JSON object.
DOC_META_RETAIN_PARAMS = "retain_params"
#: The names the caller gave this document's attachments, as a JSON object of short id ->
#: filename. On the document rather than the fact because a filename describes the reference,
#: not the bytes: the same image can be "diagram.png" in one document and "fig-2.png" in another.
#: The authority is `attachments.filename`, written at the ingress on every backend; this is the
#: store's own copy, which the paths that replay stored text restate their names from.
DOC_META_ATTACHMENT_FILENAMES = "attachment_filenames"
#: Where the document's original upload lives in Remember's ``file_storage`` — the key
#: ``documents.file_storage_key`` holds for a bank whose documents live in SQL. The upload's name
#: and content type ride on the record's own ``file_original_name`` / ``file_content_type``.
DOC_META_FILE_STORAGE_KEY = "file_storage_key"
CONSOLIDATED_NO = "0"
CONSOLIDATED_YES = "1"


def document_record_metadata(
    retain_params: "dict | None", attachment_filenames: "Mapping[str, str] | None" = None
) -> dict[str, str]:
    """The metadata map a store-owned document record is written with.

    One builder for every write path, because a path that forgets a key does not fail -- it
    writes a record without it, and the key reads back as absent until the next full re-ingest.
    The map REPLACES the record's previous one: a write carries every name the document should
    keep, which is why the paths that re-send stored text (append, reprocess, an edit re-sending
    placeholders) carry the names they read back with it.
    """
    out: dict[str, str] = {}
    if retain_params:
        out[DOC_META_RETAIN_PARAMS] = json.dumps(retain_params)
    names = {str(k): str(v) for k, v in (attachment_filenames or {}).items() if k and v}
    if names:
        out[DOC_META_ATTACHMENT_FILENAMES] = json.dumps(names, sort_keys=True)
    return out


def document_attachment_filenames(record: "Mapping | None") -> dict[str, str]:
    """A store-owned document record's attachment names (short id -> filename); ``{}`` if none.

    Tolerant by design: a record written before the key existed, or one whose value does not
    parse, has no names -- which reads back as a null ``filename``, exactly what it read before.
    """
    raw = ((record or {}).get("metadata") or {}).get(DOC_META_ATTACHMENT_FILENAMES)
    if not raw:
        return {}
    try:
        decoded = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return {}
    if not isinstance(decoded, dict):
        return {}
    return {str(k): str(v) for k, v in decoded.items() if k and v}


# ------------------------------------------------------------------ document and chunk types


@dataclass(frozen=True)
class AttachmentRef:
    """Attachment short ids named by one document: the filename lives on the document edge."""

    document_id: str | None
    attachment_ids: list[str]


@dataclass
class ObservationChunkIds:
    """The chunk ids of each observation's sources, for recall ``include_chunks``.

    ``chunk_ids_by_observation`` is in observation-rank order, each list in source order
    and not yet deduplicated across observations. ``sources_by_observation`` is set only
    when answering had to read the observations for their sources, so the caller can put
    them back on its results rather than read them a second time.
    """

    chunk_ids_by_observation: dict[str, list[str]]
    sources_by_observation: dict[str, list[str]] | None = None


@dataclass
class DocumentSourceUnits:
    """A document's experience/world memory ids and its total memory count, read before a delete."""

    unit_ids: list[str]
    units_count: int


@dataclass
class DeletedDocument:
    """What deleting a document found: whether it existed, and its uploaded file's storage key."""

    deleted: bool
    file_storage_key: str | None


@dataclass
class DocumentTags:
    """A document's current tags. ``tags`` is ``None`` when they could not be read — the
    document is absent, or its record does not carry them — which is never "no tags"."""

    found: bool
    tags: list[str] | None


def _epoch_ms_to_datetime(value: Any) -> datetime | None:
    """Epoch milliseconds -> aware datetime, for records read from a store rather than from SQL.

    A store returns timestamps as integers; the SQL rows these records stand in for come back as
    datetimes, and the response builders call ``.isoformat()`` on them. Normalising here keeps
    those builders unaware of where the record came from. ``0`` means unset, not the epoch.
    """
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


#: Prefix for the per-source metadata key an observation carries, one per source.
#: The forward list (:data:`META_SOURCE_MEMORY_IDS`) reads an observation's
#: sources; these read the other direction — "observations built on this fact" —
#: as an equality predicate rather than a corpus walk.
META_SOURCE_KEY_PREFIX = "src:"


def source_key(unit_id: str) -> str:
    """The metadata key marking an observation as built on ``unit_id``."""
    return f"{META_SOURCE_KEY_PREFIX}{unit_id}"


@dataclass
class CausalEdgeRecord:
    """A causal edge, resolved to the target's unit id."""

    target_unit_id: str
    relation_type: str  # "caused_by" for retain; legacy types on transfer import
    weight: float = 1.0


@dataclass
class StoredMemory:
    """A memory read by address rather than by ranking.

    What comes back from a get-by-id or a scan: no arm scores, because nothing
    ranked it. Shaped like a `memory_units` row so the callers that render one
    (the curation UI, export) need no second shape.
    """

    unit_id: str
    text: str
    fact_type: str
    context: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict | None = None
    proof_count: int = 1
    event_date: datetime | None = None
    occurred_start: datetime | None = None
    occurred_end: datetime | None = None
    mentioned_at: datetime | None = None
    created_at: datetime | None = None
    # Write time, as opposed to the four content times above: when the memory was last
    # written, which is the watermark a caller compares against to detect a change. Distinct
    # from `created_at`, which never moves after the first write.
    updated_at: datetime | None = None
    # When a curation edit last changed the memory; ``None`` for one still exactly as extracted.
    # The curation views report it, so a store has to hand it back for them to say so.
    edited_at: datetime | None = None
    # Which observation scopes a memory is routed to. Consolidation reads it off
    # its candidates to decide which observation each one belongs in, so it has
    # to survive the round trip through the store.
    observation_scopes: list | None = None
    entity_ids: list[str] = field(default_factory=list)
    source_memory_ids: list[str] = field(default_factory=list)
    consolidated_at: datetime | None = None
    # Derived kNN edges `(target_unit_id, weight)`, populated only when the read
    # asked for them — the ranking path never does.
    semantic_edges: list[tuple[str, float]] = field(default_factory=list)
    # Intrinsic causal edges the memory was written with, same shape the write model
    # carries. Populated only when the read asked for edges. A store that keeps memories
    # outside SQL has no `memory_links` table to reconstruct these from, so without them
    # on the read model an export of such a bank silently loses every causal relation.
    causal_edges: list[CausalEdgeRecord] = field(default_factory=list)
    # Short ids of the attachments this fact was drawn from (see META_ATTACHMENT_IDS).
    # Carried so the list and detail views resolve them from this read alone.
    attachment_ids: list[str] = field(default_factory=list)


@dataclass
class ScanPage:
    """One page of a scan, plus the cursor for the next.

    ``next_page_token`` is empty when the walk is exhausted. It is a *position*,
    not a snapshot: concurrent writes can shift later pages, so a scan is
    eventually-complete browsing rather than a consistent iterator.
    """

    memories: list[StoredMemory] = field(default_factory=list)
    next_page_token: str = ""


@dataclass
class FactRecord:
    """One memory unit, as an implementation that owns the store needs to see it.

    There is no row behind this — it is the *whole* record — so it carries every
    column recall returns, plus the edges that would otherwise have become
    `memory_links` and `unit_entities` rows.
    """

    unit_id: str  # UUID string
    text: str
    # A float list, or the pgvector literal '[0.1,...]' — Remember produces both.
    embedding: list[float] | str
    fact_type: str
    tags: list[str] = field(default_factory=list)
    proof_count: int = 1
    context: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    metadata: dict | None = None
    observation_scopes: list | str | None = None
    # Entity names + spelled-out date tokens Remember folds into its BM25 document.
    text_signals: str | None = None
    event_date: datetime | None = None
    occurred_start: datetime | None = None
    occurred_end: datetime | None = None
    mentioned_at: datetime | None = None
    created_at: datetime | None = None
    # What would have become `unit_entities` rows: the unit→entity posting travels with the
    # memory, and the ids point into the store's own entity registry.
    entity_ids: list[str] = field(default_factory=list)
    # What would have become causal `memory_links` rows.
    causal_edges: list[CausalEdgeRecord] = field(default_factory=list)
    # Observations only: the facts this observation was consolidated from.
    source_memory_ids: list[str] = field(default_factory=list)
    # When this memory was folded into an observation (sources only).
    consolidated_at: datetime | None = None
    # Short ids of the attachments this fact was drawn from — what Postgres keeps in
    # `memory_units.attachment_ids`. Empty for a fact stated in plain text.
    attachment_ids: list[str] = field(default_factory=list)

    def metadata_bag(self) -> dict[str, str]:
        """Render the non-modelled columns as an opaque str→str bag."""
        bag: dict[str, str] = {}
        if self.context:
            bag[META_CONTEXT] = self.context
        if self.document_id:
            bag[META_DOCUMENT_ID] = self.document_id
        if self.chunk_id:
            bag[META_CHUNK_ID] = self.chunk_id
        if self.metadata:
            bag[META_METADATA_JSON] = json.dumps(self.metadata)
        if self.observation_scopes is not None:
            bag[META_OBSERVATION_SCOPES] = json.dumps(self.observation_scopes)
        if self.text_signals:
            bag[META_TEXT_SIGNALS] = self.text_signals
        if self.created_at is not None:
            bag[META_CREATED_AT] = self.created_at.isoformat()
        # Remember filters recall's created_after/created_before window on
        # updated_at. A freshly written fact has updated_at == created_at.
        stamp = self.created_at
        if stamp is not None:
            bag[META_UPDATED_AT] = stamp.isoformat()
        if self.source_memory_ids:
            # Forward direction: the list, for reading an observation's sources back.
            bag[META_SOURCE_MEMORY_IDS] = json.dumps(self.source_memory_ids)
            # Backward direction: one key per source, so "observations built on
            # this fact" is an equality predicate rather than a corpus walk.
            for source_id in self.source_memory_ids:
                bag[source_key(source_id)] = "1"
        if self.consolidated_at is not None:
            bag[META_CONSOLIDATED_AT] = self.consolidated_at.isoformat()
        # Observations are not themselves consolidated, so only sources carry the flag.
        if self.fact_type != "observation":
            bag[META_CONSOLIDATED_FLAG] = CONSOLIDATED_YES if self.consolidated_at else CONSOLIDATED_NO
        if self.attachment_ids:
            # Deduplicated in first-seen order, the same normalisation the SQL write applies.
            bag[META_ATTACHMENT_IDS] = json.dumps(list(dict.fromkeys(self.attachment_ids)))
        return bag


def build_text_signals(fact) -> str | None:
    """Entity names + spelled-out dates — the enrichment Remember folds into BM25.

    Mirrors the signal construction the `memory_units` INSERT performs, so an
    implementation that owns the store produces the same searchable document the
    SQL path does.
    """
    parts: list[str] = []
    if fact.entities:
        parts.extend(e.name for e in fact.entities)
    stamps = [fact.occurred_start]
    if fact.occurred_end and fact.occurred_end != fact.occurred_start:
        stamps.append(fact.occurred_end)
    for stamp in stamps:
        if stamp is None:
            continue
        try:
            parts.append(stamp.strftime("%B %d %Y").lstrip("0").replace(" 0", " "))
        except (ValueError, AttributeError):
            pass
    return " ".join(parts) if parts else None


def build_fact_records(
    unit_ids: list[str],
    facts: list,
    document_id: str | None = None,
    unit_entity_ids: dict[str, list[str]] | None = None,
) -> list[FactRecord]:
    """Turn the retain pipeline's facts into records, edges resolved.

    ``unit_entity_ids`` is the unit→entity posting that would otherwise become
    `unit_entities` rows; causal relations become the memory's causal edges. Both
    travel with the memory, which is why a store that owns them writes once rather
    than inserting and then linking.

    Only called by implementations that own the store — the Postgres one already
    wrote all of this and never builds a record.
    """
    now = datetime.now(timezone.utc)
    records: list[FactRecord] = []
    for index, (unit_id, fact) in enumerate(zip(unit_ids, facts)):
        entity_ids = (unit_entity_ids or {}).get(str(unit_id))
        if entity_ids is None:
            entity_ids = [str(e.entity_id) for e in (fact.entities or []) if e.entity_id is not None]

        causal_edges = []
        for relation in fact.causal_relations or []:
            target = relation.target_fact_index
            # Targets are indices into this batch; a stale index would otherwise
            # produce an edge pointing at the wrong memory.
            if not isinstance(target, int) or not 0 <= target < len(unit_ids) or target == index:
                continue
            causal_edges.append(
                CausalEdgeRecord(target_unit_id=str(unit_ids[target]), relation_type=relation.relation_type)
            )

        records.append(
            FactRecord(
                unit_id=str(unit_id),
                text=fact.fact_text,
                # Unpacked here on purpose: `FactRecord` is the contract every store
                # implementation reads, and retain's packed `array("f")` is an internal
                # carrying format (#3756). The list exists only for the record.
                embedding=list(fact.embedding),
                fact_type=fact.fact_type,
                tags=fact.tags or [],
                context=fact.context,
                document_id=fact.document_id or document_id,
                chunk_id=fact.chunk_id,
                metadata=fact.metadata,
                observation_scopes=fact.observation_scopes,
                text_signals=build_text_signals(fact),
                event_date=fact.occurred_start if fact.occurred_start is not None else fact.mentioned_at,
                occurred_start=fact.occurred_start,
                occurred_end=fact.occurred_end,
                mentioned_at=fact.mentioned_at,
                created_at=now,
                entity_ids=entity_ids,
                causal_edges=causal_edges,
                attachment_ids=list(getattr(fact, "attachment_ids", None) or []),
            )
        )
    return records


def watermark_may_postdate(last_refreshed_at: datetime | None, watermark: datetime | None) -> bool:
    """Cheap half of staleness: rule a model current from the bank watermark alone.

    False is exact — nothing in the bank has been written since the model last
    read the memories, so nothing in its scope has either, and no scoped query is
    needed. True means only that *something* was written, possibly outside the
    model's tags, so it is a signal to go and ask
    (:func:`_mental_model_stale_scope` plus the store's scoped check), never an
    answer to report. Surfaces used to report it as "may need refresh" because
    asking per model was expensive; they now ask the scoped question instead
    (#3291), and this stays what it always was — a free way to skip it. Every
    staleness caller runs it first (#4169): the scoped check is bounded by the
    writes since the model's own watermark, so a model this settles is exactly the
    one that would have paid the most to be asked.
    """
    if last_refreshed_at is None:
        return True  # Never refreshed — nothing to be current with.
    if watermark is None:
        return False  # Empty bank.
    # Oracle's TIMESTAMP columns come back naive while the bank watermark is aware;
    # compare both in UTC, as engine/reflect/tools.py already does for last_refreshed_at.
    if last_refreshed_at.tzinfo is None:
        last_refreshed_at = last_refreshed_at.replace(tzinfo=timezone.utc)
    if watermark.tzinfo is None:
        watermark = watermark.replace(tzinfo=timezone.utc)
    return watermark > last_refreshed_at


@dataclass(frozen=True)
class MemoryScopeWatermark:
    """One "has this scope changed?" question, for the batched staleness check.

    ``key`` is opaque to the store — it is whatever the caller wants the answer
    reported under (a mental-model id, a page id) — and the scope
    fields are the same ones :meth:`PostgresMemories.any_memory_updated_since`
    takes for a single model, so the two surfaces cannot drift apart.
    """

    key: str
    since: datetime
    fact_types: list[str] | None = None
    tags: list[str] | None = None
    tags_match: TagsMatch = "any"
    tag_groups: list | None = None


@dataclass
class RelinkPassResult:
    """What one relink drain got through.

    ``queue_exhausted`` is False when the pass stopped on its deadline (or the
    runaway-iteration cap) with rows still queued — not a failure, since every
    batch commits before the next is claimed, but the caller needs to know the
    queue is not empty so it can arrange for the rest to be picked up.
    """

    units_processed: int = 0
    links_added: int = 0
    queue_exhausted: bool = True


@dataclass
class EntityPrunePassResult:
    """What one entity-prune drain got through.

    ``entities_examined`` counts candidates claimed, not rows deleted: most
    candidates turn out to be alive and are kept, which is the pass working as
    intended rather than wasted effort.
    """

    entities_examined: int = 0
    orphan_entities_pruned: int = 0
    stale_cooccurrences_pruned: int = 0
    queue_exhausted: bool = True


@dataclass
class RecallArms:
    """One fact_type's per-arm candidate lists from :meth:`PostgresMemories.recall_unified`.

    Each list holds ``RetrievalResult`` items, unfused — RRF/rerank happen downstream.
    ``temporal`` is empty unless a window was given; ``graph`` is empty when that arm is off.
    """

    semantic: list = field(default_factory=list)
    bm25: list = field(default_factory=list)
    graph: list = field(default_factory=list)
    temporal: list = field(default_factory=list)


#: How many semantic hits seed the graph arm's link expansion — its entry points into the graph.
GRAPH_SEED_LIMIT = 20

#: Most edges of each kind one graph-view answer carries: the links a store reads (with the ones
#: observations inherit from them), and the edges the engine derives (shared source, shared
#: entity) -- so at most twice this in all. The UI cannot usefully draw more, and uncapped a bank
#: of a few thousand observations derives hundreds of thousands -- seconds of Python on the event
#: loop and a multi-MB answer (#5703).
GRAPH_MAX_EDGES = 10_000


@dataclass
class SemanticBm25Result:
    """One fact_type's dense + keyword candidates, plus the graph arm's seeds.

    What a store's combined semantic/BM25 read hands back (the Postgres store's ``search``). Declared
    here rather than in the Postgres store's modules so a store implementing the same read does not
    have to import them.
    """

    semantic: list[RetrievalResult]
    bm25: list[RetrievalResult]
    graph_seeds: list[RetrievalResult] | None


@dataclass
class MemoryLocation:
    """Where one memory lives and what it is: the answer to the lookups that precede a delete
    or a history read. ``source_memory_ids`` is filled only by :meth:`PostgresMemories.observation_head`."""

    unit_id: str
    bank_id: str
    fact_type: str
    source_memory_ids: list[str] = field(default_factory=list)


@dataclass
class TypedMemoryScope:
    """A bank's memories of one fact_type, as a typed bank clear needs them: the ids (source
    types only, for the stale-observation sweep) and the count it reports."""

    unit_ids: list[str] = field(default_factory=list)
    count: int = 0


@dataclass
class BankContentCounts:
    """What a whole-bank delete reports having removed."""

    memory_units: int = 0
    entities: int = 0
    documents: int = 0


# What the retain path's document reads return.


@dataclass
class DocumentBase:
    """The stored document an append builds on: its body and the version it was read at.

    ``content_hash`` is the version: a store-owned write compare-and-sets against it (see
    :meth:`PostgresMemories.put_document`).
    """

    original_text: str | None
    content_hash: str | None
    attachment_filenames: dict[str, str] = field(default_factory=dict)


@dataclass
class ExistingChunk:
    """A chunk already stored for a document: its id, position and content hash."""

    chunk_id: str
    chunk_index: int
    content_hash: str | None


@dataclass
class DocumentChunkState:
    """A stored document's version, body (when asked for) and chunks, as a delta retain diffs them.

    ``content_hash`` is the version the delta's write compare-and-sets against (see
    :meth:`PostgresMemories.put_document`)."""

    content_hash: str | None
    original_text: str | None
    chunks: list[ExistingChunk] = field(default_factory=list)
    # The document's tags at load, before the retain rewrites them; None when not readable.
    tags: list[str] | None = None


def document_chunk_state(
    record: "Mapping | None", *, bank_id: str, document_id: str, include_text: bool
) -> DocumentChunkState:
    """A store-owned document record read as a :class:`DocumentChunkState`; empty if ``None``.

    The record's own chunk hashes, not a download of every chunk's text: they were computed with
    the same ``sha256(chunk.encode()).hexdigest()`` retain compares with. A chunk_id is
    ``build_chunk_id(bank_id, document_id, index)`` by construction, so the record carries none.
    All of it comes from the ONE record, so the chunks diffed and the hash the write is
    conditional on describe the same version.
    """
    from ..chunk_ids import build_chunk_id

    rec = record or {}
    return DocumentChunkState(
        content_hash=rec.get("content_hash"),
        original_text=rec.get("original_text") if include_text else None,
        tags=list(rec["tags"] or []) if "tags" in rec else None,
        chunks=[
            ExistingChunk(
                chunk_id=build_chunk_id(bank_id, document_id, index),
                chunk_index=index,
                content_hash=chunk_hash,
            )
            for index, chunk_hash in enumerate(rec.get("chunk_hashes") or [])
        ],
    )


@dataclass
class RelabelResult:
    """What relabelling a document's memories did: how many it updated, and which of them
    (``experience``/``world`` only) changed tags or observation scoping — the ones whose
    observations are no longer valid."""

    updated: int
    rescoped_unit_ids: list[str] = field(default_factory=list)


@dataclass
class RetagResult:
    """What retagging or rescoping a document's memories did: how many observations it deleted,
    and how many memories it queued again for consolidation because their observation scopes
    moved. Both zero when nothing moved."""

    invalidated_observations: int = 0
    requeued_units: int = 0


class EntityResolverHandle(Protocol):
    """What the engine holds as its entity resolver: the Postgres store's SQL resolver, built once at
    startup through ``sql_memories()`` and handed to :meth:`PostgresMemories.resolve_entities`.

    A Protocol rather than the Postgres ``EntityResolver`` class: this module may not import the
    Postgres store's modules (``tests/test_store_table_boundary.py``). The two per-task stats
    hooks are what the retain and import paths call on every batch whatever store owns the bank;
    the memory edit calls the reassert / link pair after :meth:`PostgresMemories.resolve_entities`
    resolved something.
    """

    def discard_pending_stats(self) -> None: ...

    def forget_pending_mentions(self, entity_ids: list[str]) -> None: ...

    async def flush_pending_stats(self, conn: Any | None = None) -> None: ...

    async def reassert_entities_batch(self, bank_id: str, resolved_entities: list[Any], conn) -> None: ...

    async def link_units_to_entities_batch(
        self,
        unit_entity_pairs: list[tuple[str, str]] | list[tuple[str, str, datetime | None]],
        conn=None,
        bank_id: str | None = None,
    ) -> None: ...


# Deprecated names from before pages were renamed. Remove in 2.x.


__all__ = [
    "CONSOLIDATED_NO",
    "CONSOLIDATED_YES",
    "META_ATTACHMENT_IDS",
    "META_CHUNK_ID",
    "META_CONSOLIDATED_AT",
    "META_CONSOLIDATED_FLAG",
    "META_CONTEXT",
    "META_CREATED_AT",
    "META_DOCUMENT_ID",
    "META_METADATA_JSON",
    "META_OBSERVATION_SCOPES",
    "META_SOURCE_KEY_PREFIX",
    "META_SOURCE_MEMORY_IDS",
    "META_TEXT_SIGNALS",
    "META_UPDATED_AT",
    "BankContentCounts",
    "CausalEdgeRecord",
    "EntityPrunePassResult",
    "FactRecord",
    "GRAPH_SEED_LIMIT",
    "MemoryLocation",
    "RelinkPassResult",
    "ScanPage",
    "SemanticBm25Result",
    "StoredMemory",
    "TypedMemoryScope",
    "build_fact_records",
    "build_text_signals",
    "source_key",
]
