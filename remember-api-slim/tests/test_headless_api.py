"""The local API keeps its machine-readable schema without browser dashboards."""

from types import SimpleNamespace

import httpx
import pytest

from remember_api.api import create_app as create_unified_app
from remember_api.api.http import create_app


@pytest.mark.asyncio
async def test_browser_docs_are_disabled_but_schema_remains_available():
    app = create_app(SimpleNamespace(audit_logger=None), initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/docs")).status_code == 404
        assert (await client.get("/redoc")).status_code == 404
        schema = await client.get("/openapi.json")
        assert schema.status_code == 200
        assert "/v1/default/banks/{bank_id}/memories" in schema.json()["paths"]


@pytest.mark.asyncio
async def test_minimal_app_disables_browser_docs():
    app = create_unified_app(SimpleNamespace(), http_api_enabled=False, initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/docs")).status_code == 404
        assert (await client.get("/redoc")).status_code == 404
        assert (await client.get("/openapi.json")).status_code == 200
