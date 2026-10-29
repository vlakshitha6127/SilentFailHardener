from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.silentfail.dashboard import (
    DEMO_OUTPUT_DIR,
    DEMO_TESTS_DIR,
    orchestrator as dashboard_orchestrator,
    provider_status_label,
)
from src.silentfail.nebius import NebiusClient, safe_endpoint_label
from src.silentfail.orchestrator import Orchestrator


def test_dashboard_uses_isolated_demo_outputs():
    assert dashboard_orchestrator.data_dir == DEMO_OUTPUT_DIR
    assert dashboard_orchestrator.tests_dir == DEMO_TESTS_DIR
    assert dashboard_orchestrator.data_dir != Path("data")
    assert dashboard_orchestrator.data_dir / "latest_pipeline_run.json" != Path(
        "data/latest_pipeline_run.json"
    )


def test_demo_pipeline_preserves_historical_output(tmp_path):
    protected = Path("data/latest_pipeline_run.json")
    before = protected.read_bytes()
    demo = Orchestrator(
        data_dir=str(tmp_path / "demo-data"),
        tests_dir=str(tmp_path / "demo-tests"),
    )

    result = demo.run_full_pipeline(train_count=1, held_out_count=1)

    assert result["scale_and_cost"]["mode"] == "local_simulation"
    assert (tmp_path / "demo-data" / "latest_pipeline_run.json").exists()
    assert (tmp_path / "demo-tests" / "test_generated_regression.py").exists()
    assert protected.read_bytes() == before


def test_provider_status_requires_verified_probe():
    assert provider_status_label({}) == "Provider status: not probed"
    assert provider_status_label({}, configured_live=True) == "Live provider configured; not verified"
    assert provider_status_label({"status": "local_simulation", "connected": False}) == (
        "Deterministic local simulation"
    )
    assert provider_status_label({"status": "live_provider_error", "connected": False}) == (
        "Configured live provider failed; deterministic local simulation"
    )
    assert provider_status_label({"status": "connected", "connected": True}) == (
        "Live provider verified (model response received)"
    )


def test_provider_probe_is_local_or_truthfully_verified_without_network():
    offline = NebiusClient(api_key="")
    offline_result = offline.probe_gate_g0()
    assert offline_result["status"] == "local_simulation"
    assert offline_result["connected"] is False

    live = NebiusClient(api_key="mock-key")
    live.client = MagicMock()
    live.is_live = True
    live.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="pong"))]
    )
    live_result = live.probe_gate_g0()
    assert live_result["status"] == "connected"
    assert live_result["connected"] is True
    assert live_result["response"] == "pong"

    failed = NebiusClient(api_key="mock-key", base_url="https://user:secret@example.test/v1")
    failed.client = MagicMock()
    failed.is_live = True
    failed.client.chat.completions.create.side_effect = RuntimeError("token=secret")
    failed_result = failed.probe_gate_g0()
    assert failed_result["status"] == "live_provider_error"
    assert failed_result["connected"] is False
    assert "secret" not in failed_result["reason"]
    assert failed_result["endpoint"] == "https://example.test/v1"


def test_safe_endpoint_label_removes_credentials_and_query():
    assert safe_endpoint_label("https://user:secret@example.test:443/v1?token=secret#frag") == (
        "https://example.test:443/v1"
    )
