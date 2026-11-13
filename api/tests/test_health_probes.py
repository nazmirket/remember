"""Liveness must never depend on the database (#3329).

A liveness probe wired to a DB check restarts every pod at once when the
database is merely slow: in-flight requests die, claimed async operations are
requeued with ``retry_count`` incremented, and each restarted process reconnects
to re-warm its pool against a database that is already saturated. These tests
pin the split — ``/health/live`` answers without touching the database, while
``/health`` and ``/health/ready`` keep failing closed so readiness gates traffic.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

_DB_DOWN = {"status": "unhealthy", "database": "error", "error": "connection refused"}
_DB_UP = {"status": "healthy", "database": "connected", "db_acquire_ms": 0.4}


def _api_client(health: dict) -> tuple[TestClient, MagicMock]:
    """API app whose engine reports ``health``, with no database behind it."""
    from remember_api.api.http import create_app

    memory = MagicMock()
    # Copy: the worker handler enriches the payload it gets back in place.
    memory.health_check = AsyncMock(return_value=dict(health))
    return TestClient(create_app(memory, initialize_memory=False)), memory


# ---------------------------------------------------------------------------
# Liveness: DB-free
# ---------------------------------------------------------------------------


def test_api_liveness_stays_200_when_database_is_down():
    client, memory = _api_client(_DB_DOWN)

    response = client.get("/health/live")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "alive"
    assert body["version"]
    assert body["uptime_seconds"] >= 0
    # The point of the endpoint: it never asks the engine about the database.
    assert memory.health_check.await_count == 0


# ---------------------------------------------------------------------------
# Readiness: still gated on the database
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/health", "/health/ready"])
def test_api_readiness_fails_when_database_is_down(path):
    client, _ = _api_client(_DB_DOWN)

    response = client.get(path)

    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"


def test_api_health_and_ready_return_the_same_payload():
    """/health is documented as the alias of /health/ready — keep it true."""
    client, _ = _api_client(_DB_UP)

    health = client.get("/health")
    ready = client.get("/health/ready")

    assert health.status_code == ready.status_code == 200
    assert health.json() == ready.json()
    from remember_api.api import http

    assert health.json()["checkout_source"] == str(Path(http.__file__).resolve())


# ---------------------------------------------------------------------------
# Poller progress stamp
# ---------------------------------------------------------------------------


def test_poll_age_is_none_until_the_first_cycle_completes():
    from remember_api.worker import WorkerPoller

    poller = WorkerPoller(backend=MagicMock(), worker_id="w-test", executor=AsyncMock())

    assert poller.seconds_since_last_poll is None


@pytest.mark.asyncio
async def test_poll_age_is_stamped_by_the_claim_loop():
    """One pass of the loop must refresh the stamp, whether or not it claimed work."""
    from remember_api.worker import WorkerPoller

    poller = WorkerPoller(backend=MagicMock(), worker_id="w-test", executor=AsyncMock(), poll_interval_ms=1)
    poller.recover_own_tasks = AsyncMock()
    poller.claim_batch = AsyncMock(return_value=[])
    poller._log_progress_if_due = AsyncMock()

    # Let the loop run a couple of cycles, then signal shutdown.
    run = asyncio.create_task(poller.run())
    await asyncio.sleep(0.05)
    poller._shutdown.set()
    await run

    assert poller.seconds_since_last_poll is not None
    assert poller.seconds_since_last_poll >= 0
