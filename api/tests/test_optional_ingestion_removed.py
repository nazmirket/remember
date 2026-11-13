"""The local fork does not expose optional document-ingestion endpoints."""

from unittest.mock import MagicMock

from remember_api.api.http import create_app
from remember_api.config import RememberConfig
from remember_api.engine.memory_engine import MemoryEngine


def test_optional_ingestion_routes_are_absent():
    app = create_app(MagicMock(), initialize_memory=False)
    routes = {route.path for route in app.routes}
    assert "/v1/default/banks/{bank_id}/files/retain" not in routes
    assert not any(path.startswith("/v1/default/knowledge-banks") for path in routes)
    assert "/v1/default/banks/{bank_id}/memories" in routes
    assert "/v1/default/banks/{bank_id}/pages/tree" in routes


def test_removed_ingestion_has_no_parser_handler_or_settings():
    assert not hasattr(MemoryEngine, "_handle_file_convert_retain")
    assert not any(
        name.startswith(("file_parser", "knowledge_embeddings")) for name in RememberConfig.__dataclass_fields__
    )


def test_version_reports_removed_features_disabled(monkeypatch):
    from fastapi.testclient import TestClient

    from remember_api.config import clear_config_cache

    clear_config_cache()
    app = create_app(MagicMock(), initialize_memory=False)
    client = TestClient(app)
    response = client.get("/version")
    assert response.status_code == 200
    assert "file_upload_api" not in response.json()["features"]
    assert "knowledge_banks" not in response.json()["features"]
    assert client.post("/v1/default/banks/test/files/retain").status_code == 404
    assert client.get("/v1/default/knowledge-banks").status_code == 404
    clear_config_cache()
