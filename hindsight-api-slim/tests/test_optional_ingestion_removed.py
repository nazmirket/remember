"""The local fork does not expose optional document-ingestion endpoints."""

from unittest.mock import MagicMock

from hindsight_api.api.http import create_app


def test_optional_ingestion_routes_are_absent():
    app = create_app(MagicMock(), initialize_memory=False)
    routes = {route.path for route in app.routes}
    assert "/v1/default/banks/{bank_id}/files/retain" not in routes
    assert not any(path.startswith("/v1/default/knowledge-banks") for path in routes)
    assert "/v1/default/banks/{bank_id}/memories" in routes
    assert "/v1/default/banks/{bank_id}/pages/tree" in routes


def test_version_reports_removed_features_disabled(monkeypatch):
    from fastapi.testclient import TestClient

    from hindsight_api.config import clear_config_cache

    monkeypatch.setenv("HINDSIGHT_API_ENABLE_KNOWLEDGE_BANKS", "true")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_FILE_UPLOAD_API", "true")
    clear_config_cache()
    app = create_app(MagicMock(), initialize_memory=False)
    client = TestClient(app)
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json()["features"]["file_upload_api"] is False
    assert response.json()["features"]["knowledge_banks"] is False
    assert client.post("/v1/default/banks/test/files/retain").status_code == 404
    assert client.get("/v1/default/knowledge-banks").status_code == 404
    clear_config_cache()
