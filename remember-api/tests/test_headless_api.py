"""The local API keeps its machine-readable schema without browser dashboards."""

from types import SimpleNamespace

import httpx
import pytest

from remember_api.api.http import create_app


@pytest.mark.asyncio
async def test_browser_docs_are_disabled_but_schema_remains_available():
    app = create_app(SimpleNamespace(audit_logger=None), initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/docs")).status_code == 404
        assert (await client.get("/redoc")).status_code == 404
        assert (await client.get("/mcp/")).status_code == 404
        assert (await client.get("/v1/default/banks/test/webhooks")).status_code == 404
        assert (await client.post("/v1/default/banks/test/transfer/export")).status_code == 404
        schema = await client.get("/openapi.json")
        assert schema.status_code == 200
        assert "/v1/default/banks/{bank_id}/memories" in schema.json()["paths"]


def test_local_api_does_not_load_http_extensions(monkeypatch):
    monkeypatch.setenv("REMEMBER_API_HTTP_EXTENSION", "nonexistent:ExternalExtension")
    app = create_app(SimpleNamespace(audit_logger=None), initialize_memory=False)
    assert app is not None
