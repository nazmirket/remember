"""Authentication result for the local database schema."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from remember_api.extensions.base import Extension
from remember_api.models import RequestContext


class AuthenticationError(Exception):
    """Raised when authentication fails."""

    def __init__(self, reason: str, headers: dict[str, str] | None = None):
        self.reason = reason
        self.headers = headers or {}
        super().__init__(f"Authentication failed: {reason}")


@dataclass
class TenantContext:
    """
    Tenant context returned by authentication.

    Contains the PostgreSQL schema name for tenant isolation.
    All database queries will use fully-qualified table names
    with this schema (e.g., schema_name.memory_units).
    """

    schema_name: str


class TenantExtension(Extension, ABC):
    """Authenticate a request for the configured local schema."""

    @abstractmethod
    async def authenticate(self, context: RequestContext) -> TenantContext:
        """
        Authenticate the action context and return tenant context.

        Args:
            context: The action context containing API key and other auth data.

        Returns:
            TenantContext with the schema_name for database operations.

        Raises:
            AuthenticationError: If authentication fails.
        """
        ...
