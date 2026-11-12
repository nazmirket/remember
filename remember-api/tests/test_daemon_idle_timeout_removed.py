import remember_api.daemon as daemon_module
import remember_api.main as main_module


def test_no_idle_timeout_machinery_remains():
    """The daemon must not interrupt long operations with an inactivity timer."""
    assert not hasattr(daemon_module, "IdleTimeoutMiddleware")
    assert not hasattr(daemon_module, "DEFAULT_IDLE_TIMEOUT")
    assert "idle_middleware" not in main_module.main.__code__.co_varnames
