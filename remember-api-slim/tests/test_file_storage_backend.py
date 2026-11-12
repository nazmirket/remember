"""Storage used by retained document attachments."""

import pytest


@pytest.fixture
def sample_txt_content():
    """Create simple text content."""
    return b"This is a test document.\nIt contains some important information.\nAlice works at Google."


@pytest.mark.asyncio
async def test_file_storage_postgresql(memory_no_llm_verify, sample_txt_content):
    """Test file storage in PostgreSQL."""
    # Test that files are stored and retrieved correctly
    storage = memory_no_llm_verify._file_storage

    # Store a file
    key = "test/file1.txt"
    stored_key = await storage.store(
        file_data=sample_txt_content,
        key=key,
        metadata={"content_type": "text/plain"},
    )

    assert stored_key == key

    # Retrieve the file
    retrieved = await storage.retrieve(key)
    assert retrieved == sample_txt_content

    # Check if file exists
    exists = await storage.exists(key)
    assert exists is True

    # Delete the file
    await storage.delete(key)

    # Check file no longer exists
    exists_after = await storage.exists(key)
    assert exists_after is False
