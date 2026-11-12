"""Streaming exports: the ZIP writer and the file-storage backends it streams through."""

import io
import json
import uuid
import zipfile
import zlib
from collections.abc import AsyncIterator

import pytest

from remember_api.engine.storage.postgresql import _CHUNKED_MANIFEST_SIGNATURE, PG_STREAM_CHUNK_SIZE


# --- PostgreSQL file storage, against the real database -------------------------


def _pg_key(name: str) -> str:
    return f"tests/streaming/{uuid.uuid4().hex}/{name}"


async def _pieces(data: bytes, size: int = 1024 * 1024) -> AsyncIterator[bytes]:
    for i in range(0, len(data), size):
        yield data[i : i + size]


async def test_postgres_short_stream_is_one_plain_row(memory):
    """A stream under one chunk is stored exactly as store() would store it."""
    fs = memory._file_storage
    key = _pg_key("small.txt")
    try:
        await fs.store_stream(key, _pieces(b"small data"))
        assert await fs.retrieve(key) == b"small data"
        assert not await fs.exists(f"{key}#chunk=000000")
    finally:
        await fs.delete(key)


async def test_postgres_empty_file(memory):
    fs = memory._file_storage
    key = _pg_key("empty.bin")
    try:
        await fs.store_stream(key, _pieces(b""))
        assert await fs.exists(key)
        assert await fs.get_size(key) == 0
        assert await fs.retrieve(key) == b""
        assert b"".join([c async for c in fs.retrieve_stream(key)]) == b""
    finally:
        await fs.delete(key)
    assert not await fs.exists(key)


async def test_postgres_plain_row_is_streamed_in_bounded_ranges(memory):
    """A large file written by store() — every attachment — never comes back in one piece."""
    fs = memory._file_storage
    key = _pg_key("attachment.bin")
    data = bytes(range(256)) * (10 * 1024 * 1024 // 256)  # 10 MiB, not a chunked file
    try:
        await fs.store(data, key)
        pieces = [c async for c in fs.retrieve_stream(key)]
        assert [len(p) for p in pieces] == [PG_STREAM_CHUNK_SIZE, PG_STREAM_CHUNK_SIZE, 2 * 1024 * 1024]
        assert b"".join(pieces) == data
        assert await fs.get_size(key) == len(data)
    finally:
        await fs.delete(key)


async def test_postgres_aborted_stream_leaves_nothing(memory):
    fs = memory._file_storage
    key = _pg_key("aborted.bin")

    async def failing() -> AsyncIterator[bytes]:
        yield b"A" * PG_STREAM_CHUNK_SIZE
        yield b"B" * 1024
        raise RuntimeError("disconnect")

    with pytest.raises(RuntimeError, match="disconnect"):
        await fs.store_stream(key, failing())
    assert not await fs.exists(key)
    assert not await fs.exists(f"{key}#chunk=000000")


@pytest.mark.parametrize("rewrite", ["store", "store_stream"])
async def test_postgres_overwriting_a_chunked_file_drops_its_old_chunks(memory, rewrite):
    """Replacing a chunked file must not strand the previous version's chunk rows."""
    fs = memory._file_storage
    key = _pg_key("rewritten.bin")
    try:
        await fs.store_stream(key, _pieces(b"X" * (9 * 1024 * 1024)))
        assert await fs.exists(f"{key}#chunk=000002")
        if rewrite == "store":
            await fs.store(b"short", key)
        else:
            await fs.store_stream(key, _pieces(b"Y" * (5 * 1024 * 1024)))
        assert not await fs.exists(f"{key}#chunk=000002")
        if rewrite == "store":
            assert not await fs.exists(f"{key}#chunk=000000")
            assert await fs.retrieve(key) == b"short"
        else:
            assert await fs.retrieve(key) == b"Y" * (5 * 1024 * 1024)
    finally:
        await fs.delete(key)


async def test_postgres_row_that_only_looks_like_a_manifest_is_a_plain_file(memory):
    fs = memory._file_storage
    key = _pg_key("lookalike.bin")
    data = _CHUNKED_MANIFEST_SIGNATURE + b"not-a-manifest"
    await fs.store(data, key)
    assert await fs.retrieve(key) == data
    assert await fs.get_size(key) == len(data)
    await fs.delete(key)
    assert not await fs.exists(key)
