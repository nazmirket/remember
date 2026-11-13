"""Errors used by local memory operations."""


class OperationValidationError(Exception):
    """The operation was refused before it did any work.

    The HTTP layer answers ``status_code`` with ``reason``, and the operation is counted
    with ``success="rejected"`` rather than as a failure (see metrics.OperationOutcome).
    """

    def __init__(self, reason: str, status_code: int = 403):
        self.reason = reason
        self.status_code = status_code
        super().__init__(f"Operation validation failed: {reason}")
