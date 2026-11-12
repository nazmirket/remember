"""Remember API exports.

Heavy engine modules load lazily. Native thread limits are installed before any
ML libraries are imported.
"""

from typing import TYPE_CHECKING

# Cap native ML thread pools (OpenBLAS/OpenMP/MKL) before any import pulls in
# numpy/torch/onnxruntime — they read these env vars only at load time. See
# remember_api/_thread_limits.py for the rationale.
from ._thread_limits import apply_default_thread_limits

apply_default_thread_limits()

__version__ = "1.2.1"

# name -> (module, attribute). Kept as data so `__all__`, `__dir__` and the resolver cannot drift.
_LAZY_EXPORTS: "dict[str, tuple[str, str]]" = {
    "RememberConfig": (".config", "RememberConfig"),
    "get_config": (".config", "get_config"),
    "CrossEncoderModel": (".engine.cross_encoder", "CrossEncoderModel"),
    "LocalSTCrossEncoder": (".engine.cross_encoder", "LocalSTCrossEncoder"),
    "RemoteTEICrossEncoder": (".engine.cross_encoder", "RemoteTEICrossEncoder"),
    "Embeddings": (".engine.embeddings", "Embeddings"),
    "LocalSTEmbeddings": (".engine.embeddings", "LocalSTEmbeddings"),
    "RemoteTEIEmbeddings": (".engine.embeddings", "RemoteTEIEmbeddings"),
    "LLMConfig": (".engine.llm_wrapper", "LLMConfig"),
    "MemoryEngine": (".engine.memory_engine", "MemoryEngine"),
    "EntryPoint": (".engine.search.trace", "EntryPoint"),
    "NodeVisit": (".engine.search.trace", "NodeVisit"),
    "QueryInfo": (".engine.search.trace", "QueryInfo"),
    "SearchPhaseMetrics": (".engine.search.trace", "SearchPhaseMetrics"),
    "SearchSummary": (".engine.search.trace", "SearchSummary"),
    "SearchTrace": (".engine.search.trace", "SearchTrace"),
    "WeightComponents": (".engine.search.trace", "WeightComponents"),
    "SearchTracer": (".engine.search.tracer", "SearchTracer"),
    "RequestContext": (".models", "RequestContext"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    """Resolve a public name on first access, then cache it in the module globals."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attribute)
    # Bind it so every later access is a plain global lookup rather than another __getattr__.
    globals()[name] = value
    return value


def __dir__() -> "list[str]":
    return sorted([*__all__, "__version__"])


if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs, never at runtime
    # `noqa: F401` throughout: these are re-exports, and `__all__` is COMPUTED from
    # `_LAZY_EXPORTS` above so the resolver and the export list cannot drift. A linter cannot
    # see a computed `__all__`, so it reads every one of these as unused.
    from .config import RememberConfig, get_config  # noqa: F401
    from .engine.cross_encoder import CrossEncoderModel, LocalSTCrossEncoder, RemoteTEICrossEncoder  # noqa: F401
    from .engine.embeddings import Embeddings, LocalSTEmbeddings, RemoteTEIEmbeddings  # noqa: F401
    from .engine.llm_wrapper import LLMConfig  # noqa: F401
    from .engine.memory_engine import MemoryEngine  # noqa: F401
    from .engine.search.trace import (  # noqa: F401
        EntryPoint,
        NodeVisit,
        QueryInfo,
        SearchPhaseMetrics,
        SearchSummary,
        SearchTrace,
        WeightComponents,
    )
    from .engine.search.tracer import SearchTracer  # noqa: F401
    from .models import RequestContext  # noqa: F401
