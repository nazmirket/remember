"""Local-store metrics retain their established labels."""

from unittest.mock import MagicMock, patch

import pytest

from remember_api.metrics import MetricsCollector


@pytest.fixture
def collector():
    config = MagicMock(
        metrics_include_bank_id=False,
        metrics_include_tenant=False,
        recall_diagnostic_phases=True,
        recall_phase_sample_every=1,
    )
    meter = MagicMock()
    meter.create_histogram.side_effect = lambda *a, **k: MagicMock()
    meter.create_counter.side_effect = lambda *a, **k: MagicMock()
    with (
        patch("remember_api.metrics.get_meter", return_value=meter),
        patch("remember_api.config.get_config", return_value=config),
    ):
        return MetricsCollector()


def test_local_operation_and_recall_phase_have_no_backend_label(collector):
    with collector.record_operation("recall", bank_id="bank"):
        collector.record_recall_phase("engine_call", 0.01)
    assert "memories_backend" not in collector.operation_duration.record.call_args.args[1]
    assert "memories_backend" not in collector.operation_total.add.call_args.args[1]
    assert "memories_backend" not in collector.recall_phase_duration.record.call_args.args[1]


async def test_retain_preserves_local_store_label():
    from remember_api.engine.retain import timing

    seen = {}

    class Capture:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        async def __aenter__(self):
            return None

        async def __aexit__(self, *args):
            return False

    @timing.timed_retain
    async def retain(self, bank_id, contents):
        return "ok"

    with patch.object(timing, "retain_timing", Capture):
        assert await retain(object(), "bank", [{"content": "x"}]) == "ok"
    assert seen["store"] == "PostgresMemories"
