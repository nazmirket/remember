"""The API serves only on loopback and keeps the configured local port."""

from types import SimpleNamespace

import pytest

from remember_api.main import _parse_cli_args


def _config(host=None):
    return SimpleNamespace(host=host, port=9001, log_level="info", access_log=False)


def test_defaults_to_loopback_and_configured_port():
    args = _parse_cli_args([], _config()).args
    assert args.host == "127.0.0.1"
    assert args.port == 9001


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_accepts_loopback_hosts(host):
    assert _parse_cli_args(["--host", host], _config()).args.host == host


@pytest.mark.parametrize("host", ["0.0.0.0", "192.0.2.1"])
def test_rejects_non_loopback_cli_hosts(host):
    with pytest.raises(SystemExit):
        _parse_cli_args(["--host", host], _config())


def test_rejects_non_loopback_configured_host():
    with pytest.raises(SystemExit):
        _parse_cli_args([], _config("0.0.0.0"))


@pytest.mark.parametrize("option", ["--reload", "--workers", "--proxy-headers", "--ssl-keyfile"])
def test_rejects_alternate_deployment_options(option):
    with pytest.raises(SystemExit):
        _parse_cli_args([option], _config())


@pytest.mark.parametrize(
    ("database_url", "read_database_url", "migration_database_url"),
    [
        ("postgresql://remote/remember", None, None),
        ("pg0", "postgresql://remote/replica", None),
        ("pg0", None, "postgresql://remote/migrations"),
    ],
)
def test_main_rejects_remote_database_routes(
    monkeypatch, capsys, database_url, read_database_url, migration_database_url
):
    from remember_api import main as main_module

    config = _config()
    config.database_url = database_url
    config.read_database_url = read_database_url
    config.migration_database_url = migration_database_url
    monkeypatch.setattr(main_module, "_get_raw_config", lambda: config)
    monkeypatch.setattr(main_module, "load_dotenv_for_entrypoint", lambda: None)
    monkeypatch.setattr(main_module, "_wait_for_port", lambda *_: None)
    monkeypatch.setattr("sys.argv", ["remember-api"])
    with pytest.raises(SystemExit) as result:
        main_module.main()
    assert result.value.code == 1
    assert "requires its embedded pg0 database" in capsys.readouterr().err
