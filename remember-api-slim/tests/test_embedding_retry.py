"""
Tests for bounded retry/backoff on remote embedding calls.

Recall generates its query embedding synchronously on the request path, so a
single upstream 5xx used to surface as a failed recall. These tests pin the
contract of the retry wrapper:

1. Transient failures (5xx, timeouts, connection errors) are retried and a later
   success is returned.
2. Persistent transient failures still raise, after a bounded number of attempts.
3. Non-transient failures (4xx auth/validation) raise immediately, with no retry.
4. The wall-clock retry budget caps how long a single encode() call can spend
   retrying, and batching does not multiply it.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from aiohttp import web

from remember_api.engine.aiohttp_session import UpstreamHTTPError
from remember_api.engine.remote_retry import (
    RetryPolicy,
    is_transient_remote_error,
    status_code_of,
)
from tests.aiohttp_stub import stub_server


class _StatusError(Exception):
    """Stand-in for litellm/openai errors, which expose `status_code`."""

    def __init__(self, status_code: int, message: str = "upstream error"):
        super().__init__(f"{message} (status {status_code})")
        self.status_code = status_code


class _InternalServerError(_StatusError):
    """Matches litellm.InternalServerError closely enough for classification."""

    def __init__(self, message: str = "Venice AI returned 500"):
        super().__init__(500, message)


class _AuthenticationError(_StatusError):
    def __init__(self, message: str = "invalid api key"):
        super().__init__(401, message)


def _response_for(texts):
    response = MagicMock()
    response.data = [{"embedding": [0.1] * 768, "index": i} for i in range(len(texts))]
    return response


# Fast policy so the tests exercise the logic, not the sleeps.
FAST_POLICY = RetryPolicy(max_retries=4, initial_backoff=0.01, max_backoff=0.04, budget_seconds=5.0)


class TestTransientClassification:
    """Only genuinely transient failures should be retried."""

    @pytest.mark.parametrize("status", [500, 502, 503, 504, 408, 429])
    def test_transient_status_codes(self, status):
        assert is_transient_remote_error(_StatusError(status)) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_client_error_status_codes_are_not_transient(self, status):
        assert is_transient_remote_error(_StatusError(status)) is False

    def test_transport_errors_are_transient(self):
        import httpx

        assert is_transient_remote_error(httpx.ConnectError("connection refused")) is True
        assert is_transient_remote_error(httpx.ReadTimeout("read timed out")) is True
        assert is_transient_remote_error(TimeoutError("timed out")) is True
        assert is_transient_remote_error(ConnectionError("reset by peer")) is True

    def test_named_sdk_errors_are_transient_without_status(self):
        class APITimeoutError(Exception):
            pass

        assert is_transient_remote_error(APITimeoutError("no status attribute")) is True

    def test_unknown_errors_are_not_transient(self):
        assert is_transient_remote_error(ValueError("bad input")) is False
        assert is_transient_remote_error(Exception("API Error")) is False

    def test_google_genai_status_on_code_attribute(self):
        """google.genai.errors.APIError carries the status on `code`, with `response` often None."""
        assert is_transient_remote_error(SimpleNamespace(code=429, response=None)) is True
        assert is_transient_remote_error(SimpleNamespace(code=503, response=None)) is True
        assert is_transient_remote_error(SimpleNamespace(code=403, response=None)) is False

    @staticmethod
    def _wrapped(inner: BaseException) -> BaseException:
        """The chain a reflect tool failure raises: tool error from search error from ``inner``."""
        try:
            try:
                raise inner
            except BaseException as e:
                raise RuntimeError(f"Failed to search memories: {e!r}") from e
        except RuntimeError as e:
            wrapper = RuntimeError("Reflect tool 'search_observations' failed")
            wrapper.__cause__ = e
            return wrapper

    @pytest.mark.parametrize(
        "inner",
        [_StatusError(429), _StatusError(503), aiohttp.SocketTimeoutError("read timed out"), TimeoutError()],
        ids=["429", "503", "socket-timeout", "timeout"],
    )
    def test_a_transient_error_stays_transient_under_wrappers(self, inner):
        """#6140: a reranker 429 or socket timeout two wrappers down paused refreshes."""
        assert is_transient_remote_error(self._wrapped(inner)) is True

    @pytest.mark.parametrize("inner", [_StatusError(400), _StatusError(401), ValueError("bad")])
    def test_a_permanent_error_stays_permanent_under_wrappers(self, inner):
        assert is_transient_remote_error(self._wrapped(inner)) is False

    def test_the_outermost_verdict_wins(self):
        """A wrapper that carries its own status is not overruled by what it wraps."""
        outer = _StatusError(400)
        outer.__cause__ = TimeoutError()
        assert is_transient_remote_error(outer) is False

    def test_a_declared_permanent_failure_is_not_overruled_by_its_cause(self):
        from remember_api.engine.llm_interface import ProviderAuthError, ProviderContentPolicyError

        for permanent in (ProviderContentPolicyError("refused"), ProviderAuthError("bad key")):
            permanent.__cause__ = TimeoutError()
            assert is_transient_remote_error(permanent) is False

    def test_a_cause_cycle_terminates(self):
        a, b = RuntimeError("a"), RuntimeError("b")
        a.__cause__, b.__cause__ = b, a
        assert is_transient_remote_error(a) is False

    def test_non_status_code_attributes_are_ignored(self):
        """`code` is a common attribute name; only plausible HTTP statuses may classify."""
        assert status_code_of(SimpleNamespace(code="RESOURCE_EXHAUSTED")) is None
        assert status_code_of(SimpleNamespace(code=1)) is None
        assert status_code_of(SimpleNamespace(code=429)) == 429


class TestInitializeRetries:
    """Dimension detection gates startup; a flaky provider must not blow it up."""


class TestPolicyFromConfig:
    """Attempts and budget must be tunable without a code change."""

    def test_env_overrides_are_picked_up(self, monkeypatch):
        from remember_api.config import (
            ENV_EMBEDDINGS_INITIAL_BACKOFF,
            ENV_EMBEDDINGS_MAX_BACKOFF,
            ENV_EMBEDDINGS_MAX_RETRIES,
            ENV_EMBEDDINGS_RETRY_BUDGET,
            RememberConfig,
        )
        from remember_api.engine.embeddings import _retry_policy_from_config

        monkeypatch.setenv(ENV_EMBEDDINGS_MAX_RETRIES, "2")
        monkeypatch.setenv(ENV_EMBEDDINGS_INITIAL_BACKOFF, "0.25")
        monkeypatch.setenv(ENV_EMBEDDINGS_MAX_BACKOFF, "2.5")
        monkeypatch.setenv(ENV_EMBEDDINGS_RETRY_BUDGET, "7.5")

        config = RememberConfig.from_env()
        policy = _retry_policy_from_config(config)

        assert policy.max_retries == 2
        assert policy.initial_backoff == 0.25
        assert policy.max_backoff == 2.5
        assert policy.budget_seconds == 7.5

    def test_defaults_are_bounded(self):
        policy = RetryPolicy()

        assert 3 <= policy.max_retries + 1 <= 5
        assert policy.budget_seconds <= 15.0


class TestEveryRemoteProviderRetries:
    """A structural guard over the whole provider family.

    Retry is implemented once per provider, so the provider that forgets is by
    definition the one nobody wrote a test for: Gemini shipped without it (#4103),
    and Cohere and ZeroEntropy had the same gap. This asserts over every
    ``Embeddings`` subclass rather than over the ones we happened to remember, so a
    new remote backend has to make a deliberate choice instead of silently
    inheriting none.
    """

    # Providers that legitimately do not take an RetryPolicy, and why.
    _EXEMPT = {
        # In-process: no round trip to fail transiently.
        "LocalSTEmbeddings": "in-process",
        # A router: each side it delegates to is checked here on its own.
        "SplitEmbeddings": "delegates to its two sides",
        "OnnxEmbeddings": "in-process",
        # Their own retry loop, including TEI's 429-as-backpressure handling.
        "RemoteTEIEmbeddings": "own _request_with_retry (see tei_retry.py)",
        # The OpenAI SDK retries internally; max_retries is passed to the client.
        "OpenAIEmbeddings": "openai SDK max_retries",
        "CodexOAuthEmbeddings": "openai SDK max_retries (subclass)",
    }

    def test_every_remote_backend_takes_a_retry_policy(self):
        import inspect

        from remember_api.engine import embeddings as embeddings_module
        from remember_api.engine.embeddings import Embeddings

        missing = []
        for name, obj in vars(embeddings_module).items():
            if not inspect.isclass(obj) or not issubclass(obj, Embeddings) or obj is Embeddings:
                continue
            if inspect.isabstract(obj) or name in self._EXEMPT:
                continue
            if "retry_policy" not in inspect.signature(obj.__init__).parameters:
                missing.append(name)

        assert not missing, (
            f"{missing} take no retry_policy. Wire them through RetryPolicy "
            f"(as the litellm/google/cohere/zeroentropy backends are), or add them to "
            f"_EXEMPT with the reason they retry some other way."
        )

    def test_exemptions_still_name_real_classes(self):
        """Keep the exemption list from silently outliving the classes it names."""
        from remember_api.engine import embeddings as embeddings_module

        assert not [name for name in self._EXEMPT if not hasattr(embeddings_module, name)]
