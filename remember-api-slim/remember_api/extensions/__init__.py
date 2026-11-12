"""
Remember Extensions System.

Extensions allow customizing and extending Remember behavior without modifying core code.
Extensions are loaded via environment variables pointing to implementation classes.

Example:
    REMEMBER_API_OPERATION_VALIDATOR_EXTENSION=mypackage.validators:MyValidator
    REMEMBER_API_OPERATION_VALIDATOR_MAX_RETRIES=3

    REMEMBER_API_HTTP_EXTENSION=mypackage.http:MyHttpExtension
    REMEMBER_API_HTTP_SOME_CONFIG=value

Extensions receive an ExtensionContext that provides a controlled API for interacting
with the system (e.g., running migrations for tenant schemas).

This package exports the extension *interfaces* only. Concrete implementations —
including the ones bundled under ``remember_api.extensions.builtin`` — are
imported by path at load time, so importing this package never pulls in an
implementation's dependencies. Extensions distributed outside the server live in
a separately installed package.
"""

from remember_api.extensions.bank_tables import BankScopedTable
from remember_api.extensions.base import Extension
from remember_api.extensions.context import DefaultExtensionContext, ExtensionContext
from remember_api.extensions.http import HttpExtension
from remember_api.extensions.llm_request import (
    LLMRequestExtension,
    LLMRequestOverrides,
    LLMRequestRefused,
    LLMRequestTarget,
)
from remember_api.extensions.loader import load_extension
from remember_api.extensions.mcp import MCPExtension
from remember_api.extensions.memory_defense import (
    DefenseAction,
    DefenseDecision,
    DefensePolicy,
    MemoryDefenseExtension,
    PolicyRule,
    apply_redaction,
    parse_policy,
)
from remember_api.extensions.model_call import ModelCall, ModelCallExtension, OperationRun
from remember_api.extensions.operation_validator import (
    # Bank Management operations
    BankListContext,
    BankListResult,
    BankListScope,
    BankReadContext,
    BankReadOperation,
    BankWriteContext,
    BankWriteOperation,
    # Consolidation operation
    ConsolidateContext,
    ConsolidateResult,
    ConsolidationActionsContext,
    CreateBankContext,
    # File Conversion
    # Transfer imports
    ImportArchiveDocument,
    ImportContext,
    # Memory curation
    MemoryCurationAction,
    # Memory review
    MemoryReview,
    MemoryUpdateContext,
    MemoryUpdateResult,
    MentalModelCandidateContext,
    # Mental Model operations
    MentalModelGetContext,
    MentalModelGetResult,
    MentalModelRefreshContext,
    MentalModelRefreshResult,
    # Core operations
    OperationValidationError,
    OperationValidatorExtension,
    PrecheckContext,
    PrecheckOperation,
    RecallContext,
    RecallResult,
    ReflectContext,
    ReflectResultContext,
    RetainAttachmentInfo,
    RetainContext,
    RetainFactsContext,
    RetainResult,
    RetainSource,
    ReviewedConsolidationAction,
    ReviewedFact,
    TagScopeContext,
    ValidationResult,
)
from remember_api.extensions.tenant import (
    AuthenticationError,
    Tenant,
    TenantContext,
    TenantExtension,
)
from remember_api.models import RequestContext
from remember_api.worker.exceptions import DeferOperation

__all__ = [
    # Base
    "Extension",
    "BankScopedTable",
    "load_extension",
    # Context
    "ExtensionContext",
    "DefaultExtensionContext",
    # HTTP Extension
    "HttpExtension",
    # MCP Extension
    "MCPExtension",
    # LLM Request Extension
    "LLMRequestExtension",
    "LLMRequestOverrides",
    "LLMRequestRefused",
    "LLMRequestTarget",
    # Model Call Extension
    "ModelCall",
    "ModelCallExtension",
    "OperationRun",
    # Operation Validator - Core
    "DeferOperation",
    "OperationValidationError",
    "OperationValidatorExtension",
    "PrecheckContext",
    "PrecheckOperation",
    "RecallContext",
    "RecallResult",
    "ReflectContext",
    "ReflectResultContext",
    "RetainAttachmentInfo",
    "RetainContext",
    "RetainResult",
    "RetainSource",
    "ImportArchiveDocument",
    "ImportContext",
    "ValidationResult",
    # Operation Validator - Bank Management
    "BankListContext",
    "BankListResult",
    "BankListScope",
    "BankReadContext",
    "BankReadOperation",
    "BankWriteContext",
    "BankWriteOperation",
    "TagScopeContext",
    "CreateBankContext",
    # Operation Validator - Consolidation
    "ConsolidateContext",
    "ConsolidateResult",
    "ConsolidationActionsContext",
    "MemoryReview",
    "RetainFactsContext",
    "ReviewedConsolidationAction",
    "ReviewedFact",
    # Operation Validator - File Conversion
    # Operation Validator - Memory curation
    "MemoryCurationAction",
    "MemoryUpdateContext",
    "MemoryUpdateResult",
    # Operation Validator - Mental Model
    "MentalModelGetContext",
    "MentalModelGetResult",
    "MentalModelCandidateContext",
    "MentalModelRefreshContext",
    "MentalModelRefreshResult",
    # Tenant/Auth
    "AuthenticationError",
    "RequestContext",
    "Tenant",
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
