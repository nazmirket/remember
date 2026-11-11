"""File storage backends for uploaded files."""

from collections.abc import Callable
from urllib.parse import quote

from ...schema_context import get_current_schema
from .base import FileStorage
from .postgresql import PostgreSQLFileStorage

__all__ = ["FileStorage", "PostgreSQLFileStorage", "bank_storage_prefix", "create_file_storage", "key_segment"]


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


def create_file_storage(
    storage_type: str,
    pool_getter: Callable | None = None,
    schema: str | None = None,
    schema_getter: Callable | None = None,
    **kwargs,
) -> FileStorage:
    """
    Create file storage backend based on configuration.

    Args:
        storage_type: "native" (PostgreSQL BYTEA) or a registered storage extension
        pool_getter: Database pool getter (required for native)
        schema: Static database schema (for native single-tenant)
        schema_getter: Callable returning current schema at query time (for native multi-tenant)
        **kwargs: Additional args passed to storage backend

    Returns:
        FileStorage instance

    Raises:
        ValueError: If storage_type is unknown or required args are missing
    """
    # A deployment can register its own file backend through the extension
    # mechanism, exactly as it registers a memories store. Checked first so an
    # external store can supply one without this factory naming any provider.
    extension_storage = _load_file_storage_extension()
    if extension_storage is not None:
        return extension_storage

    if storage_type == "native":
        if not pool_getter:
            raise ValueError("pool_getter required for native (PostgreSQL) storage")
        return PostgreSQLFileStorage(pool_getter=pool_getter, schema=schema, schema_getter=schema_getter)
    raise ValueError(f"Unknown storage type: {storage_type}. Built-in storage supports 'native'.")


def _load_file_storage_extension() -> FileStorage | None:
    """Load a file-storage backend registered through the extension mechanism.

    ``HINDSIGHT_API_FILE_STORAGE_EXTENSION=module.path:ClassName`` names a
    :class:`FileStorage` implementation; every other ``HINDSIGHT_API_FILE_STORAGE_*``
    variable is handed to it as a lowercased config dict. This keeps
    provider-specific backends out of this factory — the same pattern the memories
    store uses via ``HINDSIGHT_API_MEMORIES_EXTENSION``. Returns ``None`` when no
    extension is configured, so the built-in backends below stay the default.
    """
    import importlib
    import os

    from ...config import get_config

    ext_path = get_config().file_storage_extension
    if not ext_path:
        return None
    if ":" not in ext_path:
        raise ValueError(
            f"Invalid HINDSIGHT_API_FILE_STORAGE_EXTENSION '{ext_path}'. Expected format 'module.path:ClassName'."
        )
    module_path, class_name = ext_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if not (isinstance(cls, type) and issubclass(cls, FileStorage)):
        raise ValueError(f"File storage extension '{class_name}' must inherit from FileStorage.")

    prefix = "HINDSIGHT_API_FILE_STORAGE_"
    config = {
        key[len(prefix) :].lower(): value
        for key, value in os.environ.items()
        if key.startswith(prefix) and key != "HINDSIGHT_API_FILE_STORAGE_EXTENSION"
    }
    return cls(config)
