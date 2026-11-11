"""
LLM provider implementations.

This package contains concrete implementations of the LLMInterface for various providers.

The classes load on first access (PEP 562 ``__getattr__``), not at import time. Each
provider pulls in its own SDK, and anything
importing a submodule here (``http.py`` imports ``none_llm``) runs this file first, so
eager imports made every process pay for every SDK, used or not.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .anthropic_llm import AnthropicLLM
    from .claude_code_llm import ClaudeCodeLLM
    from .codex_llm import CodexLLM
    from .llamacpp_llm import LlamaCppLLM
    from .mock_llm import MockLLM
    from .none_llm import NoneLLM
    from .openai_compatible_llm import OpenAICompatibleLLM
    from .openai_responses_llm import OpenAIResponsesLLM

_MODULES = {
    "AnthropicLLM": "anthropic_llm",
    "ClaudeCodeLLM": "claude_code_llm",
    "CodexLLM": "codex_llm",
    "LlamaCppLLM": "llamacpp_llm",
    "MockLLM": "mock_llm",
    "NoneLLM": "none_llm",
    "OpenAICompatibleLLM": "openai_compatible_llm",
    "OpenAIResponsesLLM": "openai_responses_llm",
}

__all__ = [
    "AnthropicLLM",
    "ClaudeCodeLLM",
    "CodexLLM",
    "LlamaCppLLM",
    "MockLLM",
    "NoneLLM",
    "OpenAICompatibleLLM",
    "OpenAIResponsesLLM",
]


def __getattr__(name: str) -> Any:
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value
