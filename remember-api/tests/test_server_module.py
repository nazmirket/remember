"""
Tests for remember_api.server module (multi-worker code path).

The server.py module is used when running with multiple workers:
    uvicorn remember_api.server:app --workers 2

This module executes code at import time, creating the app at module level.
These tests ensure that extensions are properly loaded in this code path,
which was previously a regression that caused authentication bypass in production.
"""

import importlib
import sys
from unittest.mock import MagicMock, patch


def _clean_server_module():
    """Remove remember_api.server from sys.modules for fresh import."""
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith("remember_api.server")]
    for mod in modules_to_remove:
        del sys.modules[mod]


class TestServerModuleExtensionLoading:
    """Tests that server.py correctly loads extensions when configured via environment."""

    def test_server_loads_tenant_extension_when_configured(self, monkeypatch):
        """
        Verify that server.py loads tenant extension from REMEMBER_API_TENANT_EXTENSION.

        This test catches the regression where server.py didn't call load_extension(),
        causing authentication to be bypassed in multi-worker deployments.
        """
        # Set up environment to configure a tenant extension
        monkeypatch.setenv(
            "REMEMBER_API_TENANT_EXTENSION",
            "tests.test_server_module:MockTenantExtension",
        )

        _clean_server_module()

        # Track what extensions were loaded via load_extension
        loaded_extensions = {}

        # Get the real load_extension function
        from remember_api.extensions.loader import load_extension as real_load_extension

        def tracking_load_extension(name, base_class):
            """Track calls to load_extension and delegate to original."""
            result = real_load_extension(name, base_class)
            loaded_extensions[name] = result
            return result

        # Patch at source level BEFORE importing server
        # Note: We patch the entire remember_api module namespace
        with (
            patch("remember_api.MemoryEngine") as mock_engine,
            patch("remember_api.api.create_app") as mock_create_app,
            patch("remember_api.config.get_config") as mock_get_config,
            patch("remember_api.extensions.load_extension", side_effect=tracking_load_extension),
        ):
            mock_config = MagicMock()
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_engine.return_value = MagicMock()
            mock_create_app.return_value = MagicMock()

            # Now import server - this triggers module-level code
            import remember_api.server

        # Verify TENANT extension was loaded
        assert "TENANT" in loaded_extensions, (
            "server.py did not call load_extension('TENANT', ...) - extensions not loaded!"
        )
        assert loaded_extensions["TENANT"] is not None, (
            "load_extension('TENANT', ...) returned None despite env var being set"
        )
        assert isinstance(loaded_extensions["TENANT"], MockTenantExtension), (
            f"Expected MockTenantExtension, got {type(loaded_extensions['TENANT'])}"
        )

    def test_server_loads_operation_validator_when_configured(self, monkeypatch):
        """
        Verify that server.py loads operation validator from REMEMBER_API_OPERATION_VALIDATOR_EXTENSION.
        """
        monkeypatch.setenv(
            "REMEMBER_API_OPERATION_VALIDATOR_EXTENSION",
            "tests.test_server_module:MockOperationValidator",
        )

        _clean_server_module()

        loaded_extensions = {}

        from remember_api.extensions.loader import load_extension as real_load_extension

        def tracking_load_extension(name, base_class):
            result = real_load_extension(name, base_class)
            loaded_extensions[name] = result
            return result

        with (
            patch("remember_api.MemoryEngine") as mock_engine,
            patch("remember_api.api.create_app") as mock_create_app,
            patch("remember_api.config.get_config") as mock_get_config,
            patch("remember_api.extensions.load_extension", side_effect=tracking_load_extension),
        ):
            mock_config = MagicMock()
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_engine.return_value = MagicMock()
            mock_create_app.return_value = MagicMock()

            import remember_api.server

        assert "OPERATION_VALIDATOR" in loaded_extensions, (
            "server.py did not call load_extension('OPERATION_VALIDATOR', ...)"
        )
        assert loaded_extensions["OPERATION_VALIDATOR"] is not None
        assert isinstance(loaded_extensions["OPERATION_VALIDATOR"], MockOperationValidator)

    def test_server_passes_extensions_to_memory_engine(self, monkeypatch):
        """
        Verify that server.py passes loaded extensions to MemoryEngine constructor.

        This is the critical test - even if extensions are loaded, they must be
        passed to MemoryEngine for authentication to work.
        """
        monkeypatch.setenv(
            "REMEMBER_API_TENANT_EXTENSION",
            "tests.test_server_module:MockTenantExtension",
        )

        _clean_server_module()

        memory_engine_calls = []

        def capture_memory_engine(*args, **kwargs):
            memory_engine_calls.append({"args": args, "kwargs": kwargs})
            return MagicMock()

        with (
            patch("remember_api.MemoryEngine", side_effect=capture_memory_engine),
            patch("remember_api.api.create_app") as mock_create_app,
            patch("remember_api.config.get_config") as mock_get_config,
        ):
            mock_config = MagicMock()
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_create_app.return_value = MagicMock()

            import remember_api.server

        # Verify MemoryEngine was called
        assert len(memory_engine_calls) == 1, "MemoryEngine should be called exactly once"

        call_kwargs = memory_engine_calls[0]["kwargs"]

        # THE CRITICAL ASSERTION: tenant_extension must be passed and not None
        assert "tenant_extension" in call_kwargs, "MemoryEngine was not called with tenant_extension parameter!"
        assert call_kwargs["tenant_extension"] is not None, (
            "tenant_extension was None - server.py did not pass loaded extension to MemoryEngine!"
        )

    def test_server_works_without_extensions(self, monkeypatch):
        """
        Verify that server.py works correctly when no extensions are configured.
        """
        # Ensure no extension env vars are set
        monkeypatch.delenv("REMEMBER_API_TENANT_EXTENSION", raising=False)
        monkeypatch.delenv("REMEMBER_API_OPERATION_VALIDATOR_EXTENSION", raising=False)

        _clean_server_module()

        memory_engine_calls = []

        def capture_memory_engine(*args, **kwargs):
            memory_engine_calls.append({"args": args, "kwargs": kwargs})
            return MagicMock()

        with (
            patch("remember_api.MemoryEngine", side_effect=capture_memory_engine),
            patch("remember_api.api.create_app") as mock_create_app,
            patch("remember_api.config.get_config") as mock_get_config,
        ):
            mock_config = MagicMock()
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_create_app.return_value = MagicMock()

            import remember_api.server

        # Should work without extensions
        assert len(memory_engine_calls) == 1
        call_kwargs = memory_engine_calls[0]["kwargs"]

        # Extensions should be None when not configured
        assert call_kwargs.get("tenant_extension") is None
        assert call_kwargs.get("operation_validator") is None


# Mock extensions for testing
from remember_api.extensions import (
    OperationValidatorExtension,
    RecallContext,
    ReflectContext,
    RequestContext,
    RetainContext,
    TenantContext,
    TenantExtension,
    ValidationResult,
)


class MockTenantExtension(TenantExtension):
    """Mock tenant extension for testing server.py extension loading."""

    async def authenticate(self, request_context: RequestContext) -> TenantContext:
        return TenantContext(schema_name="public")

    async def list_tenants(self) -> list:
        from remember_api.extensions.tenant import Tenant

        return [Tenant(schema="public")]


class MockOperationValidator(OperationValidatorExtension):
    """Mock operation validator for testing server.py extension loading."""

    def __init__(self, config: dict):
        super().__init__(config)

    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        return ValidationResult.accept()
