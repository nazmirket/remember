"""Coverage for the lazy bank-create fast path in ``MemoryEngine._ensure_bank_exists``.

The contract under test:

* An already-existing bank costs one bare SELECT — no write statement, no
  ``validate_create_bank`` call, no surrounding transaction.
* A missing bank probes once, validates once, then issues one idempotent insert.
* A rejected ``validate_create_bank`` leaves no bank row behind.
* Concurrent callers may all validate, but exactly one wins the insert and
  exactly one applies the default bank template.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable

import pytest
import pytest_asyncio

from remember_api import RequestContext
from remember_api.engine.db_utils import acquire_with_retry
from remember_api.engine.memory_engine import MemoryEngine
from remember_api.engine.response_models import ID_MAX_BYTES
from remember_api.engine.retain import bank_utils
from remember_api.extensions import OperationValidationError


@pytest_asyncio.fixture
async def bank_name(memory: MemoryEngine, request_context: RequestContext) -> AsyncIterator[Callable[[str], str]]:
    """Mint unique bank ids and drop whatever they named at teardown."""
    created: list[str] = []

    def _mint(prefix: str) -> str:
        bank_id = f"test-{prefix}-{uuid.uuid4().hex[:8]}"
        created.append(bank_id)
        return bank_id

    yield _mint

    for bank_id in created:
        await memory.delete_bank(bank_id, request_context=request_context)


def spy_on(module_or_obj, name: str, bank_id: str, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch ``name`` with a pass-through that appends to the returned list for ``bank_id``.

    The wrapped callable takes the bank id as its second positional argument,
    which is true of every ``bank_utils`` helper spied on here.
    """
    calls: list[str] = []
    real = getattr(module_or_obj, name)

    async def _spy(first, b_id, *args, **kwargs):
        if b_id == bank_id:
            calls.append(b_id)
        return await real(first, b_id, *args, **kwargs)

    monkeypatch.setattr(module_or_obj, name, _spy)
    return calls


@pytest.mark.asyncio
async def test_existing_bank_is_read_only(
    memory: MemoryEngine,
    request_context: RequestContext,
    monkeypatch: pytest.MonkeyPatch,
    bank_name: Callable[[str], str],
) -> None:
    """An existing bank issues no insert."""
    bank_id = bank_name("exists")
    backend = await memory._get_backend()
    await bank_utils.create_bank_if_missing(backend, bank_id)

    inserts = spy_on(bank_utils, "create_bank_row_on_conn", bank_id, monkeypatch)

    assert await memory._ensure_bank_exists(bank_id, request_context) is False
    async with acquire_with_retry(backend) as conn:
        async with conn.transaction():
            assert await memory._ensure_bank_exists(bank_id, request_context, conn=conn) is False

    assert inserts == [], "An existing bank must not issue an insert on either path"


@pytest.mark.asyncio
@pytest.mark.parametrize("on_conn", [True, False], ids=["on-conn", "own-conn"])
async def test_missing_bank_probes_once_then_inserts_once(
    memory: MemoryEngine,
    request_context: RequestContext,
    monkeypatch: pytest.MonkeyPatch,
    bank_name: Callable[[str], str],
    on_conn: bool,
) -> None:
    """A missing bank costs one existence probe and one insert."""
    bank_id = bank_name("missing")
    backend = await memory._get_backend()

    probes = spy_on(bank_utils, "bank_kind_on_conn" if on_conn else "bank_kind", bank_id, monkeypatch)
    inserts = spy_on(bank_utils, "create_bank_row_on_conn", bank_id, monkeypatch)

    async def ensure() -> bool:
        if not on_conn:
            return await memory._ensure_bank_exists(bank_id, request_context)
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                return await memory._ensure_bank_exists(bank_id, request_context, conn=conn)

    assert await ensure() is True
    assert len(probes) == 1, "Expected exactly one existence probe"
    assert len(inserts) == 1, "Expected exactly one insert"

    # The bank now exists: the second call returns on the probe.
    assert await ensure() is False
    assert len(probes) == 2
    assert len(inserts) == 1, "Existing bank must not re-insert"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_id",
    # "é" is 2 bytes: 194 > 192. No NUL case: PG cannot encode 0x00 in text, so such an id fails
    # at the existence probe on every endpoint, before creation is ever reached.
    ["", "é" * 97, "tab\tin-id", "line\nbreak", "_reserved", "team/standup"],
    ids=["empty", "too-many-bytes", "tab", "newline", "underscore-prefix", "slash"],
)
async def test_invalid_new_bank_id_is_rejected_before_insert(
    memory: MemoryEngine, request_context: RequestContext, bad_id: str
) -> None:
    """A new bank id that is empty, over the byte limit, has control characters or "/", or starts with `_` is a 400."""
    backend = await memory._get_backend()
    with pytest.raises(OperationValidationError) as exc_info:
        await memory._ensure_bank_exists(bad_id, request_context)
    assert exc_info.value.status_code == 400
    assert await bank_utils.bank_exists(backend, bad_id) is False


@pytest.mark.asyncio
async def test_bank_id_limit_counts_bytes_and_spares_existing_banks(
    memory: MemoryEngine, request_context: RequestContext, bank_name: Callable[[str], str]
) -> None:
    """Exactly the byte limit is accepted; a longer id that already exists stays usable."""
    at_limit = bank_name("x")
    at_limit += "é" * ((ID_MAX_BYTES - len(at_limit.encode())) // 2)
    at_limit += "a" * (ID_MAX_BYTES - len(at_limit.encode()))
    assert len(at_limit.encode()) == ID_MAX_BYTES
    assert await memory._ensure_bank_exists(at_limit, request_context) is True

    # A bank created before the limit existed: inserted directly, bypassing the check.
    legacy = bank_name("legacy") + "a" * 200
    backend = await memory._get_backend()
    await bank_utils.create_bank_if_missing(backend, legacy)
    assert await memory._ensure_bank_exists(legacy, request_context) is False


@pytest.mark.asyncio
async def test_concurrent_creation_applies_the_template_once(
    memory: MemoryEngine,
    request_context: RequestContext,
    monkeypatch: pytest.MonkeyPatch,
    bank_name: Callable[[str], str],
) -> None:
    """Two racing callers produce one created=True and one template application."""
    bank_id = bank_name("concurrent")
    backend = await memory._get_backend()

    templates: list[str] = []
    real_apply = memory._apply_default_bank_template

    async def spy_apply(b_id, ctx):
        if b_id == bank_id:
            templates.append(b_id)
        return await real_apply(b_id, ctx)

    monkeypatch.setattr(memory, "_apply_default_bank_template", spy_apply)

    results = await asyncio.gather(
        memory._ensure_bank_exists(bank_id, request_context),
        memory._ensure_bank_exists(bank_id, request_context),
    )

    assert sorted(results) == [False, True], "Exactly one caller must win the insert"
    assert templates == [bank_id], "Default bank template must be applied exactly once"
    assert await bank_utils.bank_exists(backend, bank_id) is True


@pytest.mark.asyncio
async def test_bank_utils_primitives(memory: MemoryEngine, bank_name: Callable[[str], str]) -> None:
    """The extracted primitives behave standalone, including the ON CONFLICT arm."""
    bank_id = bank_name("primitives")
    backend = await memory._get_backend()

    async with acquire_with_retry(backend) as conn:
        async with conn.transaction():
            assert await bank_utils.bank_exists_on_conn(conn, bank_id) is False
            assert await bank_utils.get_bank_profile_if_exists_on_conn(conn, bank_id) is None

            assert await bank_utils.create_bank_row_on_conn(conn, bank_id, ops=backend.ops) is True
            # ON CONFLICT DO NOTHING: the second insert reports no creation.
            assert await bank_utils.create_bank_row_on_conn(conn, bank_id, ops=backend.ops) is False

            assert await bank_utils.bank_exists_on_conn(conn, bank_id) is True
            profile = await bank_utils.get_bank_profile_if_exists_on_conn(conn, bank_id)
            assert profile is not None
            assert profile["name"] == bank_id
