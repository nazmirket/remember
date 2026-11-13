"""Model changes defer writes without masking ordinary failures."""

import pytest

from remember_api.worker.backpressure import is_reembed_guard


class ReembedGuard(Exception):
    sqlstate = "HS503"


def test_guard_and_vector_dimension_mismatches_are_recognized():
    assert is_reembed_guard(ReembedGuard("writes are shed"))
    assert is_reembed_guard(Exception("expected 768 dimensions, not 384"))


def test_wrapped_guard_is_recognized():
    try:
        try:
            raise ReembedGuard("restart required")
        except ReembedGuard as inner:
            raise RuntimeError("retain failed") from inner
    except RuntimeError as outer:
        assert is_reembed_guard(outer)


@pytest.mark.parametrize("message", ["bad document", "duplicate key", "StatusCode.UNAVAILABLE", "writes are shed"])
def test_ordinary_failures_are_not_deferred(message):
    assert not is_reembed_guard(Exception(message))


def test_cycle_terminates():
    a, b = Exception("a"), Exception("b")
    a.__cause__, b.__cause__ = b, a
    assert not is_reembed_guard(a)
