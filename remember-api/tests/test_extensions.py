"""Local token authentication for memory operations."""

import pytest

from remember_api.extensions.builtin.tenant import DefaultTenantExtension
from remember_api.extensions.tenant import AuthenticationError
from remember_api.models import RequestContext


class TestMemoryEngineTenantAuth:
    """Tests for tenant authentication in MemoryEngine."""

    @pytest.mark.asyncio
    async def test_retain_requires_tenant_request_when_extension_configured(self, memory_with_tenant):
        """Retain fails without RequestContext when tenant extension is configured."""
        memory = memory_with_tenant

        with pytest.raises(AuthenticationError) as exc_info:
            await memory.retain_batch_async(
                bank_id="test-bank",
                contents=[{"content": "test"}],
                request_context=None,  # Missing!
            )

        assert "RequestContext is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retain_succeeds_with_valid_tenant_request(self, memory_with_tenant):
        """Retain succeeds with valid RequestContext."""
        memory = memory_with_tenant

        # Should not raise
        await memory.retain_batch_async(
            bank_id="test-bank-tenant",
            contents=[{"content": "test content"}],
            request_context=RequestContext(api_key="test-api-key"),
        )

    @pytest.mark.asyncio
    async def test_retain_fails_with_invalid_api_key(self, memory_with_tenant):
        """Retain fails with invalid API key."""
        memory = memory_with_tenant

        with pytest.raises(AuthenticationError) as exc_info:
            await memory.retain_batch_async(
                bank_id="test-bank",
                contents=[{"content": "test"}],
                request_context=RequestContext(api_key="wrong-key"),
            )

        assert "Invalid API key" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_recall_requires_tenant_request_when_extension_configured(self, memory_with_tenant):
        """Recall fails without RequestContext when tenant extension is configured."""
        memory = memory_with_tenant

        with pytest.raises(AuthenticationError):
            await memory.recall_async(
                bank_id="test-bank",
                query="test query",
                fact_type=["world"],
                request_context=None,
            )

    @pytest.mark.asyncio
    async def test_list_audit_logs_fails_with_invalid_api_key(self, memory_with_tenant):
        """Audit log listing goes through tenant auth like other ops."""
        memory = memory_with_tenant

        with pytest.raises(AuthenticationError) as exc_info:
            await memory.list_audit_logs(
                "test-bank",
                request_context=RequestContext(api_key="wrong-key"),
            )

        assert "Invalid API key" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_audit_log_stats_fails_with_invalid_api_key(self, memory_with_tenant):
        """Audit log stats goes through tenant auth like other ops."""
        memory = memory_with_tenant

        with pytest.raises(AuthenticationError):
            await memory.audit_log_stats(
                "test-bank",
                request_context=RequestContext(api_key="wrong-key"),
            )

    @pytest.mark.asyncio
    async def test_no_tenant_request_needed_without_extension(self, memory):
        """Operations work with empty RequestContext when no tenant extension configured."""
        # Should not raise - no tenant extension configured, just pass empty RequestContext
        await memory.retain_batch_async(
            bank_id="test-bank-no-tenant",
            contents=[{"content": "test content"}],
            request_context=RequestContext(),
        )


@pytest.fixture
def memory_with_tenant(memory, monkeypatch):
    """Memory engine with a tenant extension (API key auth)."""
    monkeypatch.setenv("REMEMBER_API_TOKEN", "test-api-key")
    tenant_ext = DefaultTenantExtension({})
    memory._tenant_extension = tenant_ext
    return memory
