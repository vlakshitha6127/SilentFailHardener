"""Verify Nebius Gate G0 offline fallback."""
from src.silentfail.nebius import NebiusClient


def test_probe_gate_g0_fallback():
    client = NebiusClient(api_key="")
    result = client.probe_gate_g0()

    assert result["status"] == "local_simulation"
    assert result["connected"] is False
    assert result["latency_ms"] is None
    assert result["measured_rps"] is None
