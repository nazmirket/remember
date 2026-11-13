"""Task execution hooks for the local queue and synchronous tests."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


def fq_table(table: str, schema: str | None = None) -> str:
    """Get fully-qualified table name with optional schema prefix."""
    from .schema import fq_table_explicit

    return fq_table_explicit(table, schema)


class TaskBackend(ABC):
    """
    Abstract base class for task execution backends.

    Implementations must:
    1. Store/publish task events (as serializable dicts)
    2. Execute tasks through a provided executor callback (optional)

    The backend treats tasks as pure dictionaries that can be serialized
    and stored in the database. The executor (typically MemoryEngine.execute_task)
    receives the dict and routes it to the appropriate handler.
    """

    def __init__(self):
        """Initialize the task backend."""
        self._executor: Callable[[dict[str, Any]], Awaitable[None]] | None = None
        self._initialized = False

    def set_executor(self, executor: Callable[[dict[str, Any]], Awaitable[None]]):
        """
        Set the executor callback for processing tasks.

        Args:
            executor: Async function that takes a task dict and executes it
        """
        self._executor = executor

    @abstractmethod
    async def initialize(self):
        """
        Initialize the backend (e.g., connect to database).
        """
        pass

    @abstractmethod
    async def submit_task(self, task_dict: dict[str, Any]):
        """
        Submit a task for execution.

        Args:
            task_dict: Task as a dictionary (must be serializable)
        """
        pass

    @abstractmethod
    async def shutdown(self):
        """
        Shutdown the backend gracefully.
        """
        pass

    async def _execute_task(self, task_dict: dict[str, Any]):
        """
        Execute a task through the registered executor.

        Args:
            task_dict: Task dictionary to execute

        Raises:
            Exception: Re-raised from executor on failure.
        """
        if self._executor is None:
            task_type = task_dict.get("type", "unknown")
            logger.warning(f"No executor registered, skipping task {task_type}")
            return

        await self._executor(task_dict)


class SyncTaskBackend(TaskBackend):
    """
    Synchronous task backend that executes tasks immediately.

    This is useful for tests and embedded/CLI usage where we don't want
    background workers. Tasks are executed inline rather than being queued.
    """

    async def initialize(self):
        """No-op for sync backend."""
        self._initialized = True
        logger.debug("SyncTaskBackend initialized")

    async def submit_task(self, task_dict: dict[str, Any]):
        """
        Execute the task immediately (synchronously).

        Args:
            task_dict: Task dictionary to execute
        """
        if not self._initialized:
            await self.initialize()

        await self._execute_task(task_dict)

    async def shutdown(self):
        """No-op for sync backend."""
        self._initialized = False
        logger.debug("SyncTaskBackend shutdown")


class QueuedTaskBackend(TaskBackend):
    """Operations are persisted by the engine and claimed by the local worker."""

    def __init__(self, pool_getter=None, schema_getter=None):
        super().__init__()
        self._pool_getter = pool_getter
        self._schema_getter = schema_getter

    async def initialize(self):
        self._initialized = True
        logger.debug("QueuedTaskBackend initialized")

    async def submit_task(self, task_dict: dict[str, Any]):
        """No-op: the row already exists in async_operations; a worker will claim it."""
        task_type = task_dict.get("type", "unknown")
        logger.debug(f"QueuedTaskBackend: submit_task no-op for {task_type} (will be picked up by poller)")

    async def shutdown(self):
        self._initialized = False
        logger.debug("QueuedTaskBackend shutdown")

    async def wait_for_pending_tasks(self, timeout: float = 120.0):
        """
        Wait for pending tasks to be processed.

        In the broker model, this polls the database to check if tasks
        for this process have been completed. This is useful in tests
        when worker_enabled=True (API processes its own tasks).

        Args:
            timeout: Maximum time to wait in seconds
        """
        import asyncio

        from .db_utils import acquire_with_retry

        if self._pool_getter is None:
            return
        pool = self._pool_getter()
        schema = self._schema_getter() if self._schema_getter else None
        table = fq_table("async_operations", schema)

        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            # Check if there are any pending tasks with payloads
            async with acquire_with_retry(pool) as conn:
                count = await conn.fetchval(
                    f"""
                    SELECT COUNT(*) FROM {table}
                    WHERE status = 'pending' AND task_payload IS NOT NULL
                    """
                )

            if count == 0:
                return

            await asyncio.sleep(0.5)

        logger.warning(f"Timeout waiting for pending tasks after {timeout}s")
