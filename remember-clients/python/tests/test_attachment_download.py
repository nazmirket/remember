"""
The generated client returns an inline attachment's bytes unchanged (#4292).

The spec used to list ``application/json`` next to ``application/octet-stream``
for this endpoint, so the generated client decoded the body as UTF-8 text and a
PNG failed on its 0x89 magic byte. This goes through the real server and the
generated method so a spec regression shows up as a client failure.

These tests require a running Remember API server.
"""

import base64
import os
import uuid

import pytest

from remember_client import Remember
from remember_client.remember_client import _run_async

REMEMBER_API_URL = os.getenv("REMEMBER_API_URL", "http://localhost:8888")

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def client():
    with Remember(base_url=REMEMBER_API_URL) as client:
        yield client


@pytest.fixture
def bank_id(client):
    bid = f"test_bank_{uuid.uuid4().hex[:12]}"
    yield bid
    try:
        client.delete_bank(bank_id=bid)
    except Exception:
        # Best-effort cleanup: must not mask the test result.
        pass


def test_inline_image_round_trips_byte_identical(client, bank_id):
    client.retain(
        bank_id=bank_id,
        document_id="d1",
        content=[
            {"type": "text", "text": "Alice stood in front of the Brandenburg Gate."},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(PNG_BYTES).decode()},
            },
        ],
    )

    document = _run_async(client._documents_api.get_document(bank_id, "d1"))
    assert document.attachments, "the retained document lists no attachments"
    attachment = document.attachments[0]
    assert attachment.media_type == "image/png"

    data = _run_async(client._memory_api.get_bank_attachment(bank_id, attachment.id))

    assert bytes(data) == PNG_BYTES
