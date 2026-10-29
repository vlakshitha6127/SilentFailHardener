"""
Regression: NaiveRefundAgent._run_deterministic_naive generates a stable logical_op_id
per refund attempt and passes it to ToolRegistry.issue_refund on every retry.
A timeout_after_success fault on call 0 must not produce a duplicate refund when the
agent retries on call 1 (same logical_op_id hits the cache).

Also verifies that two separate logical operations (different orders) each produce
their own distinct refund.
"""
import pytest

from src.silentfail.models import FaultConfig, Order
from src.silentfail.ledger import WorldState, CausalLedger
from src.silentfail.tools import ToolRegistry, FaultInjector
from src.silentfail.agent import NaiveRefundAgent


def _find_success_observation(transcript):
    for entry in transcript:
        obs = entry.get("observation")
        if isinstance(obs, dict) and obs.get("status") == "ok":
            return obs
    return None


def test_deterministic_retry_timeout_preserves_operation_id():
    """
    Agent retries internally after timeout_after_success.
    The same logical_op_id is used for every attempt so exactly one refund is created.
    """
    ws = WorldState()
    # Use a purely numeric order ID so the agent's regex can parse it.
    ws.orders["1"] = Order(id="1", total=100.0, refunded_amount=0.0)
    ledger = CausalLedger(scenario_id="deterministic_retry_test")

    captured_ops = []
    fault_injector = FaultInjector([
        FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success")
    ])
    tools = ToolRegistry(world_state=ws, ledger=ledger, fault_injector=fault_injector)

    # Spy on issue_refund to capture the operation_id on each call.
    original_issue_refund = ToolRegistry.issue_refund
    def spy(self, *args, **kwargs):
        captured_ops.append(kwargs.get("operation_id"))
        return original_issue_refund(self, *args, **kwargs)
    ToolRegistry.issue_refund = spy

    try:
        agent = NaiveRefundAgent()
        result = agent._run_deterministic_naive(
            user_message="refund 30 order 1",
            tools=tools,
            max_retries=2,
        )
    finally:
        ToolRegistry.issue_refund = original_issue_refund

    # Exactly one refund must exist.
    assert len(ws.refunds) == 1, f"Expected 1 refund, got {len(ws.refunds)}"
    # The success observation must reference the sole refund.
    success_obs = _find_success_observation(result.transcript)
    assert success_obs is not None, "No successful refund observation in transcript"
    assert success_obs["refund_id"] in ws.refunds

    # All retries must have used the same operation_id (one unique value).
    unique_ops = set(captured_ops)
    assert len(unique_ops) == 1, (
        f"Expected a single operation_id across all retries, got: {captured_ops}"
    )


def test_two_distinct_deterministic_refunds():
    """
    Two calls to issue_refund with different operation_ids on the same order must
    each produce a separate refund even when the amounts are identical.
    """
    ws = WorldState()
    ws.orders["ord_a"] = Order(id="ord_a", total=100.0, refunded_amount=0.0)
    ledger = CausalLedger(scenario_id="distinct_ops_test")
    tools = ToolRegistry(world_state=ws, ledger=ledger)

    r1 = tools.issue_refund("ord_a", 30.0, operation_id="op-alpha-001")
    assert r1.get("status") == "ok", f"First refund failed: {r1}"

    r2 = tools.issue_refund("ord_a", 30.0, operation_id="op-beta-002")
    assert r2.get("status") == "ok", f"Second refund failed: {r2}"

    assert r1["refund_id"] != r2["refund_id"], "Different operation_ids must yield different refund_ids"
    assert len(ws.refunds) == 2, f"Expected 2 refunds in world state, got {len(ws.refunds)}"
