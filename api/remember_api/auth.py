"""Authentication for the local memory daemon."""

import hmac
import os

from .config import get_config
from .models import RequestContext


class AuthenticationError(Exception):
    """Raised when authentication fails."""

    def __init__(self, reason: str, headers: dict[str, str] | None = None):
        self.reason = reason
        self.headers = headers or {}
        super().__init__(f"Authentication failed: {reason}")


class LocalAuthenticator:
    """Authenticate the optional bearer token and return the configured schema."""

    def __init__(self, schema: str | None = None):
        self._schema = schema if schema is not None else get_config().database_schema
        self._api_token = os.getenv("REMEMBER_API_TOKEN") or None

    async def authenticate(self, context: RequestContext) -> str:
        if self._api_token and not hmac.compare_digest(
            (context.api_key or "").encode("utf-8"), self._api_token.encode("utf-8")
        ):
            raise AuthenticationError("Invalid API key")
        return self._schema
