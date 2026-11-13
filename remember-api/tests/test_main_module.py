"""
Tests for remember_api.main module (single-worker code path).

The main.py module is used when running with a single worker:
    remember-api  (or remember-api --workers 1)

When workers=1, main.py creates the app directly and passes it to uvicorn.
These tests ensure that extensions are properly loaded in this code path.

Compare with test_server_module.py which tests the multi-worker path (workers > 1).
"""

import errno
import socket
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestMainModuleExtensionLoading:
    """Tests that main.py correctly loads extensions when configured via environment."""

    @pytest.fixture(autouse=True)
    def _port_is_free(self):
        # These tests run main() against a fixed port; don't let a local listener on it fail them.
        with patch("remember_api.main._port_bind_error", return_value=None):
            yield

    def test_main_works_without_extensions(self, monkeypatch):
        """
        Verify that main.py works correctly when no extensions are configured.
        """
        # Ensure no extension env vars are set
        monkeypatch.setenv("REMEMBER_API_TENANT_EXTENSION", "nonexistent:Tenant")
        monkeypatch.setenv("REMEMBER_API_OPERATION_VALIDATOR_EXTENSION", "nonexistent:Validator")
        monkeypatch.setenv("REMEMBER_API_WORKERS", "1")

        memory_engine_calls = []

        def capture_memory_engine(*args, **kwargs):
            memory_engine_calls.append({"args": args, "kwargs": kwargs})
            return MagicMock()

        with (
            patch("remember_api.main.MemoryEngine", side_effect=capture_memory_engine),
            patch("remember_api.main.create_app") as mock_create_app,
            patch("remember_api.main._get_raw_config") as mock_get_config,
            patch("remember_api.main.print_banner"),
            patch("uvicorn.run"),
        ):
            mock_config = MagicMock()
            mock_config.host = "127.0.0.1"
            mock_config.port = 8888
            mock_config.log_level = "info"
            # argparse defaults: a MagicMock would compare against ints later on.
            mock_config.workers = 1
            mock_config.access_log = False
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "pg0"
            mock_config.read_database_url = None
            mock_config.migration_database_url = None
            mock_get_config.return_value = mock_config
            mock_create_app.return_value = MagicMock()

            with patch.object(sys, "argv", ["remember-api"]):
                from remember_api.main import main

                main()

        # Should work without extensions
        assert len(memory_engine_calls) == 1
        call_kwargs = memory_engine_calls[0]["kwargs"]

        # Extensions should be None when not configured
        assert "tenant_extension" not in call_kwargs
        assert "operation_validator" not in call_kwargs

    def test_main_uses_app_object_for_single_worker(self, monkeypatch):
        """
        Verify that main.py passes the app object (not import string) when workers=1.

        This is important because it means single-worker mode uses the app created
        in main.py (with extensions loaded), not server.py.
        """
        monkeypatch.setenv("REMEMBER_API_WORKERS", "1")
        monkeypatch.delenv("REMEMBER_API_TENANT_EXTENSION", raising=False)

        uvicorn_calls = []

        def capture_uvicorn_run(**kwargs):
            uvicorn_calls.append(kwargs)

        mock_app = MagicMock()

        with (
            patch("remember_api.main.MemoryEngine") as mock_engine,
            patch("remember_api.main.create_app", return_value=mock_app),
            patch("remember_api.main._get_raw_config") as mock_get_config,
            patch("remember_api.main.print_banner"),
            patch("uvicorn.run", side_effect=capture_uvicorn_run),
        ):
            mock_config = MagicMock()
            mock_config.host = "127.0.0.1"
            mock_config.port = 8888
            mock_config.log_level = "info"
            # argparse defaults: a MagicMock would compare against ints later on.
            mock_config.workers = 1
            mock_config.access_log = False
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "pg0"
            mock_config.read_database_url = None
            mock_config.migration_database_url = None
            mock_get_config.return_value = mock_config
            mock_engine.return_value = MagicMock()

            with patch.object(sys, "argv", ["remember-api"]):
                from remember_api.main import main

                main()

        assert len(uvicorn_calls) == 1
        # With workers=1, should pass app object, not import string
        assert uvicorn_calls[0]["app"] is mock_app, "main.py should pass app object (not import string) when workers=1"

    def test_main_sets_keepalive_timeout(self, monkeypatch):
        """
        Verify that uvicorn is configured with timeout_keep_alive > aiohttp's
        default client keepalive timeout (15s), so the server never closes
        connections before the client does.
        """
        monkeypatch.setenv("REMEMBER_API_WORKERS", "1")
        monkeypatch.delenv("REMEMBER_API_TENANT_EXTENSION", raising=False)

        uvicorn_calls = []

        def capture_uvicorn_run(**kwargs):
            uvicorn_calls.append(kwargs)

        with (
            patch("remember_api.main.MemoryEngine") as mock_engine,
            patch("remember_api.main.create_app") as mock_create_app,
            patch("remember_api.main._get_raw_config") as mock_get_config,
            patch("remember_api.main.print_banner"),
            patch("uvicorn.run", side_effect=capture_uvicorn_run),
        ):
            mock_config = MagicMock()
            mock_config.host = "127.0.0.1"
            mock_config.port = 8888
            mock_config.log_level = "info"
            # argparse defaults: a MagicMock would compare against ints later on.
            mock_config.workers = 1
            mock_config.access_log = False
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "pg0"
            mock_config.read_database_url = None
            mock_config.migration_database_url = None
            mock_get_config.return_value = mock_config
            mock_engine.return_value = MagicMock()
            mock_create_app.return_value = MagicMock()

            with patch.object(sys, "argv", ["remember-api"]):
                from remember_api.main import main

                main()

        assert len(uvicorn_calls) == 1
        assert "timeout_keep_alive" in uvicorn_calls[0], "uvicorn config must set timeout_keep_alive"
        assert uvicorn_calls[0]["timeout_keep_alive"] > 15, (
            "timeout_keep_alive must exceed aiohttp's 15s client default"
        )


class TestPortPreflight:
    """#4281: an occupied port must fail before MemoryEngine, embedded PG, models or migrations."""

    @pytest.mark.parametrize("flags", [[], ["--daemon"]])
    def test_occupied_port_exits_before_initialization(self, monkeypatch, capsys, flags):
        with socket.create_server(("127.0.0.1", 0)) as listener:
            port = listener.getsockname()[1]
            monkeypatch.setattr(sys, "argv", ["remember-api", "--host", "127.0.0.1", "--port", str(port), *flags])
            with (
                patch("remember_api.main._PORT_IN_USE_GRACE_SECONDS", 0),
                patch("remember_api.main.load_dotenv_for_entrypoint"),
                patch("remember_api.main.daemonize") as mock_daemonize,
                patch("remember_api.main.MemoryEngine") as mock_engine,
                patch("uvicorn.run") as mock_uvicorn_run,
            ):
                from remember_api.main import main

                with pytest.raises(SystemExit) as exc:
                    main()

            assert exc.value.code == 1
            assert f"cannot bind 127.0.0.1:{port}" in capsys.readouterr().err
            mock_daemonize.assert_not_called()
            mock_engine.assert_not_called()
            mock_uvicorn_run.assert_not_called()
            # The existing listener is left alone and still serves.
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                conn, _ = listener.accept()
                conn.close()

    def test_port_released_within_grace_is_not_an_error(self):
        """A previous instance still releasing the port must not make startup fail."""
        from remember_api.main import _wait_for_port

        # side_effect would RAISE exception instances, so hand them back through a function.
        results = iter([OSError(errno.EADDRINUSE, "Address already in use")] * 2 + [None])
        with (
            patch("remember_api.main._port_bind_error", side_effect=lambda host, port: next(results)) as probe,
            patch("remember_api.main.time.sleep") as sleep,
        ):
            assert _wait_for_port("127.0.0.1", 9177) is None
        assert probe.call_count == 3
        assert sleep.call_count == 2

    def test_port_still_in_use_after_grace_fails(self):
        from remember_api.main import _wait_for_port

        with (
            socket.create_server(("127.0.0.1", 0)) as listener,
            patch("remember_api.main._PORT_IN_USE_GRACE_SECONDS", 0.3),
            patch("remember_api.main._PORT_IN_USE_RETRY_INTERVAL", 0.05),
        ):
            error = _wait_for_port("127.0.0.1", listener.getsockname()[1])
        assert isinstance(error, OSError)
        assert error.errno == errno.EADDRINUSE

    def test_other_bind_errors_are_not_retried(self):
        from remember_api.main import _wait_for_port

        denied = OSError(errno.EACCES, "Permission denied")
        with (
            patch("remember_api.main._port_bind_error", return_value=denied) as probe,
            patch("remember_api.main.time.sleep") as sleep,
        ):
            assert _wait_for_port("127.0.0.1", 80) is denied
        assert probe.call_count == 1
        sleep.assert_not_called()

    def test_bind_error_reports_the_occupied_port(self):
        from remember_api.main import _port_bind_error

        with socket.create_server(("127.0.0.1", 0)) as listener:
            error = _port_bind_error("127.0.0.1", listener.getsockname()[1])
        assert isinstance(error, OSError)
        assert error.errno == errno.EADDRINUSE

    def test_free_port_is_released_after_the_probe(self):
        """The probe must not hold or listen on the port: uvicorn binds it right after."""
        from remember_api.main import _port_bind_error

        with socket.create_server(("127.0.0.1", 0)) as reserved:
            port = reserved.getsockname()[1]
        assert _port_bind_error("127.0.0.1", port) is None
        with socket.socket() as probe_target:
            assert probe_target.connect_ex(("127.0.0.1", port)) != 0
        with socket.create_server(("127.0.0.1", port)):
            pass

    def test_ipv6_host_probes_an_ipv6_socket(self):
        """Mirror uvicorn: a host containing ':' is bound as AF_INET6 (dual-stack untouched)."""
        from remember_api.main import _port_bind_error

        if not socket.has_ipv6:
            pytest.skip("IPv6 unavailable")
        try:
            listener = socket.create_server(("::1", 0), family=socket.AF_INET6)
        except OSError:
            pytest.skip("IPv6 loopback unavailable")
        with listener:
            error = _port_bind_error("::1", listener.getsockname()[1])
        assert isinstance(error, OSError)
        assert error.errno == errno.EADDRINUSE
