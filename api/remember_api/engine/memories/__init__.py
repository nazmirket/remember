"""Local PostgreSQL memory store and process-wide access."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    DOC_META_ATTACHMENT_FILENAMES,
    META_ATTACHMENT_IDS,
    META_CHUNK_ID,
    CausalEdgeRecord,
    FactRecord,
    MemoryScopeWatermark,
    RecallArms,
    ScanPage,
    StoredMemory,
    build_fact_records,
    build_text_signals,
    document_attachment_filenames,
    document_record_metadata,
    source_key,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .postgres import PostgresMemories

_memories: PostgresMemories | None = None


def get_memories() -> PostgresMemories:
    """Return the process-wide local PostgreSQL memory store."""
    global _memories
    if _memories is None:
        from .postgres import PostgresMemories

        _memories = PostgresMemories()
    return _memories


def sql_memories() -> PostgresMemories:
    """Return the local memory store for PostgreSQL-specific operations."""
    return get_memories()


__all__ = [
    "DOC_META_ATTACHMENT_FILENAMES",
    "META_ATTACHMENT_IDS",
    "META_CHUNK_ID",
    "CausalEdgeRecord",
    "FactRecord",
    "MemoryScopeWatermark",
    "RecallArms",
    "ScanPage",
    "StoredMemory",
    "build_fact_records",
    "build_text_signals",
    "document_attachment_filenames",
    "document_record_metadata",
    "get_memories",
    "source_key",
    "sql_memories",
]
