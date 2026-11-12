"""Internal interfaces for local memory processing."""

from remember_api.errors import OperationValidationError
from remember_api.extensions.base import Extension
from remember_api.extensions.memory_defense import (
    DefenseAction,
    DefenseDecision,
    DefensePolicy,
    MemoryDefenseExtension,
    PolicyRule,
    apply_redaction,
    parse_policy,
)
from remember_api.extensions.tenant import (
    AuthenticationError,
    TenantContext,
    TenantExtension,
)
from remember_api.models import RequestContext
from remember_api.worker.exceptions import DeferOperation

__all__ = [
    # Base
    "Extension",
    "DeferOperation",
    "OperationValidationError",
    # Tenant/Auth
    "AuthenticationError",
    "RequestContext",
    "TenantContext",
    "TenantExtension",
    # Memory Defense
    "DefenseAction",
    "DefenseDecision",
    "DefensePolicy",
    "MemoryDefenseExtension",
    "PolicyRule",
    "apply_redaction",
    "parse_policy",
]
