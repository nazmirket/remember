"""Built-in tenant extension implementations."""

import hmac
import os

from remember_api.config import get_config
from remember_api.extensions.tenant import AuthenticationError, TenantContext, TenantExtension
from remember_api.models import RequestContext


class DefaultTenantExtension(TenantExtension):
    """
    Default single-tenant authentication for the local daemon.

    Uses the configured schema from
    REMEMBER_API_DATABASE_SCHEMA (defaults to 'public').

    Features:
    - Optional bearer-token authentication via REMEMBER_API_TOKEN
    - Uses configured schema from environment
    - No authentication when no token is configured

    Configuration:
        REMEMBER_API_DATABASE_SCHEMA=your-schema (optional, defaults to 'public')

    This implementation is automatically enabled by the local daemon.
    """

    def __init__(self, config: dict[str, str]):
        super().__init__(config)
        # Cache the schema at initialization for consistency
        # Support explicit schema override via config, otherwise use environment
        self._schema = config.get("schema", get_config().database_schema)
        self._api_token = os.getenv("REMEMBER_API_TOKEN") or None

    async def authenticate(self, context: RequestContext) -> TenantContext:
        """Authenticate an optional local bearer token and return the configured schema."""
        if self._api_token and not hmac.compare_digest(
            (context.api_key or "").encode("utf-8"), self._api_token.encode("utf-8")
        ):
            raise AuthenticationError("Invalid API key")
        return TenantContext(schema_name=self._schema)
