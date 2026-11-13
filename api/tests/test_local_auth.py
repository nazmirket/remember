"""Optional authentication at the local daemon boundary."""

import pytest

from remember_api.auth import LocalAuthenticator
from remember_api.auth import AuthenticationError
from remember_api.models import RequestContext


@pytest.mark.asyncio
async def test_local_token_accepts_only_the_configured_value(monkeypatch):
    monkeypatch.setenv("REMEMBER_API_TOKEN", "local-test-token")
    tenant = LocalAuthenticator("public")
    for key in (None, "wrong-token", "lócál-token"):
        with pytest.raises(AuthenticationError):
            await tenant.authenticate(RequestContext(api_key=key))
    assert await tenant.authenticate(RequestContext(api_key="local-test-token")) == "public"


@pytest.mark.asyncio
async def test_without_local_token_no_authentication_is_required(monkeypatch):
    monkeypatch.delenv("REMEMBER_API_TOKEN", raising=False)
    tenant = LocalAuthenticator("public")
    assert await tenant.authenticate(RequestContext()) == "public"
