"""Bank-scoped file storage: the tenant-scoped key prefix, and what deletes the files under it."""

import uuid

import pytest

from remember_api.engine.response_models import ID_MAX_BYTES
from remember_api.engine.storage import bank_storage_prefix, key_segment


@pytest.mark.parametrize("bank_id", [".", "..", "a/b", "a%2Fb", "user.name", "ü b"])
def test_every_bank_id_is_one_key_segment_an_object_store_accepts(bank_id):
    prefix = bank_storage_prefix(bank_id)
    segments = prefix.rstrip("/").split("/")
    assert len(segments) == 4 and segments[2] == "banks"
    assert segments[3] not in (".", "..")


def test_distinct_bank_ids_never_share_a_prefix():
    ids = [".", "..", "%2E", "a.b", "a%2Eb", "a/b", "a%2Fb"]
    assert len({bank_storage_prefix(b) for b in ids}) == len(ids)


def test_an_empty_bank_id_is_refused():
    with pytest.raises(ValueError):
        bank_storage_prefix("")


def test_maximum_bank_id_stays_within_object_store_key_budget():
    """The longest bank id creation accepts still leaves room for the rest of an object-store key (#4391)."""
    # Percent-encoding turns every UTF-8 byte of a CJK character into 3 key bytes: the worst case.
    bank_id = "界" * (ID_MAX_BYTES // len("界".encode()))
    assert len(key_segment(bank_id)) == 3 * ID_MAX_BYTES

    # S3-compatible stores cap object keys at 1,024 bytes. The bank-id limit counts bytes before
    # encoding, so this fails if the limit or the encoding grows past what a file key can hold.
    object_key = f"{bank_storage_prefix(bank_id, schema='tenant')}files/{uuid.uuid4()}/original-name.txt"
    assert len(object_key.encode()) < 1024


@pytest.mark.asyncio
async def test_postgres_delete_prefix_treats_like_wildcards_literally(memory):
    storage = memory._file_storage
    await storage.store(file_data=b"x", key="t/banks/a%2F_/1")
    await storage.store(file_data=b"x", key="t/banks/aXX2FZ/1")

    assert await storage.delete_prefix("t/banks/a%2F_/") == 1
    assert await storage.exists("t/banks/aXX2FZ/1")
    await storage.delete("t/banks/aXX2FZ/1")


@pytest.mark.asyncio
async def test_deleting_a_bank_deletes_everything_under_its_prefix(api_client, memory):
    bank_id = f"files-{uuid.uuid4().hex[:8]}"
    other_bank = f"{bank_id}-other"
    for bank in (bank_id, other_bank):
        response = await api_client.put(f"/v1/default/banks/{bank}", json={})
        assert response.status_code in (200, 201), response.text
    export_key = f"{bank_storage_prefix(bank_id)}exports/{uuid.uuid4()}/transfer.zip"
    other_key = f"{bank_storage_prefix(other_bank)}exports/{uuid.uuid4()}/transfer.zip"
    for key in (export_key, other_key):
        await memory._file_storage.store(file_data=b"zip", key=key)

    response = await api_client.delete(f"/v1/default/banks/{bank_id}")
    assert response.status_code == 200, response.text

    assert not await memory._file_storage.exists(export_key)
    # Same name plus a suffix: a prefix without its trailing "/" would sweep it too.
    assert await memory._file_storage.exists(other_key)
    await memory._file_storage.delete(other_key)


@pytest.mark.asyncio
async def test_a_file_under_another_tenant_is_not_downloadable_through_a_same_named_bank(api_client, memory):
    """Object stores share one bucket across tenants: authorizing the bank id alone is not enough."""
    bank_id = f"files-{uuid.uuid4().hex[:8]}"
    response = await api_client.put(f"/v1/default/banks/{bank_id}", json={})
    assert response.status_code in (200, 201), response.text
    foreign_key = f"tenants/someone-else/banks/{bank_id}/exports/{uuid.uuid4()}/transfer.zip"
    await memory._file_storage.store(file_data=b"zip", key=foreign_key)

    try:
        response = await api_client.get(f"/v1/default/files/download/{foreign_key}")
        assert response.status_code == 404
    finally:
        await memory._file_storage.delete(foreign_key)


@pytest.mark.asyncio
async def test_chunked_streaming_file_storage_lifecycle_real_db(memory):
    """Verify chunked streaming (>4MB) lifecycle on real database backend."""
    storage = memory._file_storage
    key = f"tests/streaming/{uuid.uuid4().hex}/large_file.bin"
    # 4.5 MB of data to trigger chunking (> 4MB PG_STREAM_CHUNK_SIZE)
    total_size = int(4.5 * 1024 * 1024)
    chunk_piece = b"0123456789ABCDEF" * 4096  # 64 KB pieces

    async def sample_stream():
        emitted = 0
        while emitted < total_size:
            to_send = min(len(chunk_piece), total_size - emitted)
            yield chunk_piece[:to_send]
            emitted += to_send

    try:
        # 1. Store via store_stream (>4MB triggers chunked storage)
        stored_key = await storage.store_stream(key, sample_stream())
        assert stored_key == key
        assert await storage.exists(key)
        assert await storage.exists(f"{key}#chunk=000000")
        assert await storage.exists(f"{key}#chunk=000001")

        # 2. get_size returns the manifest total_bytes
        size = await storage.get_size(key)
        assert size == total_size

        # 3. retrieve_stream reads back exact stream chunks
        read_bytes = bytearray()
        async for chunk in storage.retrieve_stream(key):
            read_bytes.extend(chunk)
        assert len(read_bytes) == total_size

        # 4. retrieve reads back full bytes
        full_data = await storage.retrieve(key)
        assert len(full_data) == total_size
        assert bytes(read_bytes) == full_data

        # 5. delete cleans up manifest and virtual chunks
        await storage.delete(key)
        assert not await storage.exists(key)
        assert not await storage.exists(f"{key}#chunk=000000")
        assert not await storage.exists(f"{key}#chunk=000001")
        assert await storage.get_size(key) is None
    finally:
        await storage.delete(key)
