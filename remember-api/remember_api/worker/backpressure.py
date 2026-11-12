"""Classify retryable local write failures for the worker."""

from ..engine.embeddings import is_embedding_model_mismatch


def _any_cause(exc: BaseException, predicate) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if predicate(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def is_reembed_guard(exc: BaseException) -> bool:
    """Defer writes until the daemon restarts after a model change."""
    return _any_cause(exc, is_embedding_model_mismatch)


def is_append_conflict(exc: BaseException) -> bool:
    """Retry a concurrent document append from a fresh read."""
    from ..engine.retain.types import ConcurrentAppendConflict

    return _any_cause(exc, lambda current: isinstance(current, ConcurrentAppendConflict))
