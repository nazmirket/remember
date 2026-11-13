"""File storage backends for uploaded files."""

from urllib.parse import quote

from ...schema_context import get_current_schema
from .postgresql import PostgreSQLFileStorage

__all__ = ["PostgreSQLFileStorage", "bank_storage_prefix", "key_segment"]


def bank_storage_prefix(bank_id: str, schema: str | None = None) -> str:
    """The key prefix every file a bank stores lives under, in the current tenant.

    The tenant comes first because object-store backends share one bucket
    across every tenant schema: without it two tenants' banks with the same id
    wrote the same keys, and a content-addressed attachment one of them deleted
    was the other's too.

    ``schema`` names the tenant explicitly, for callers outside a request that
    have no current schema to read — the admin CLI's rename, which re-keys a
    bank's files and so must spell both the old prefix and the new one.
    """
    return f"tenants/{key_segment(schema or get_current_schema())}/banks/{key_segment(bank_id)}/"


def key_segment(value: str) -> str:
    """Encode a name as exactly one key segment, injectively.

    Escaping rather than validating: bank ids are caller-chosen and already in use
    with dots, spaces and non-ASCII. Percent-encoding with nothing marked safe
    encodes ``/`` (so one bank cannot nest under another's prefix and be swept with
    it) and ``%`` itself (so two names never share an encoding). ``quote`` leaves
    ``.`` alone, and object stores refuse ``.`` and ``..`` as path segments, so dots
    are encoded as well.
    """
    if not value:
        raise ValueError("A storage key segment cannot be empty")
    return quote(value, safe="").replace(".", "%2E")
