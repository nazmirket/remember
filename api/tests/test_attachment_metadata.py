"""Attachment metadata encoding retains its input meaning."""

import json
from types import SimpleNamespace

import pytest

from remember_api.engine.memories.base import (
    DOC_META_ATTACHMENT_FILENAMES,
    META_ATTACHMENT_IDS,
    FactRecord,
    build_fact_records,
    document_attachment_filenames,
    document_record_metadata,
)
from remember_api.engine.retain.attachment_content import compute_attachment_hash, short_attachment_id

UNIT_A = "00000000-0000-0000-0000-00000000000a"
UNIT_PLAIN = "00000000-0000-0000-0000-0000000000cc"
SHOT = short_attachment_id(compute_attachment_hash(b"the vpn reset screenshot"))
DIAGRAM = "0f1e2d3c4b5a"
SHOT_NAME = "vpn-reset.png"


def test_metadata_bag_carries_attachment_ids_deduplicated_in_order():
    record = FactRecord(
        unit_id=UNIT_A, text="t", embedding=[0.0], fact_type="world", attachment_ids=[SHOT, DIAGRAM, SHOT]
    )

    bag = record.metadata_bag()

    assert json.loads(bag[META_ATTACHMENT_IDS]) == [SHOT, DIAGRAM]


def test_metadata_bag_omits_the_key_for_a_fact_with_no_attachments():
    """A plain-text fact is the overwhelmingly common case: no key, not an empty list."""
    record = FactRecord(unit_id=UNIT_A, text="t", embedding=[0.0], fact_type="world")

    assert META_ATTACHMENT_IDS not in record.metadata_bag()


def test_build_fact_records_carries_each_facts_attachment_ids():
    """The builder the store-owned retain paths use must not drop the extractor's attribution."""

    def _fact(ids):
        return SimpleNamespace(
            fact_text="t",
            embedding=[0.0],
            fact_type="world",
            tags=[],
            context=None,
            document_id="doc",
            chunk_id=None,
            metadata=None,
            observation_scopes=None,
            entities=[],
            causal_relations=[],
            occurred_start=None,
            occurred_end=None,
            mentioned_at=None,
            attachment_ids=ids,
        )

    records = build_fact_records([UNIT_A, UNIT_PLAIN], [_fact([SHOT]), _fact([])])

    assert [r.attachment_ids for r in records] == [[SHOT], []]
    assert json.loads(records[0].metadata_bag()[META_ATTACHMENT_IDS]) == [SHOT]


def test_document_record_metadata_round_trips_the_names():
    """The names are a JSON string in a string -> string map, next to the retain params."""
    metadata = document_record_metadata({"context": "kb"}, {SHOT: SHOT_NAME, DIAGRAM: ""})

    assert all(isinstance(v, str) for v in metadata.values())
    assert json.loads(metadata["retain_params"]) == {"context": "kb"}
    # An empty name is no name: it must not come back as a filename of "".
    assert document_attachment_filenames({"metadata": metadata}) == {SHOT: SHOT_NAME}


def test_document_record_metadata_omits_the_key_when_nothing_is_named():
    assert document_record_metadata(None, {}) == {}
    assert DOC_META_ATTACHMENT_FILENAMES not in document_record_metadata({"context": "kb"}, None)


@pytest.mark.parametrize(
    "record",
    [None, {}, {"metadata": {}}, {"metadata": {DOC_META_ATTACHMENT_FILENAMES: "{not json"}}],
    ids=["absent", "no-metadata", "no-key", "unparseable"],
)
def test_a_record_without_readable_names_has_none(record):
    """A record written before the key existed reads back exactly as it did: no names."""
    assert document_attachment_filenames(record) == {}
