"""Local HTTP API for Remember."""

from fastapi import FastAPI

from remember_api import MemoryEngine


def create_app(memory: MemoryEngine, initialize_memory: bool = True) -> FastAPI:
    from .http import create_app as create_http_app

    return create_http_app(memory=memory, initialize_memory=initialize_memory)


# Re-export commonly used items for backwards compatibility
from .http import (
    CreateBankRequest,
    DispositionTraits,
    MemoryItem,
    RecallRequest,
    RecallResponse,
    RecallResult,
    ReflectRequest,
    ReflectResponse,
    RetainRequest,
)

__all__ = [
    "create_app",
    "RecallRequest",
    "RecallResult",
    "RecallResponse",
    "MemoryItem",
    "RetainRequest",
    "ReflectRequest",
    "ReflectResponse",
    "CreateBankRequest",
    "DispositionTraits",
]
