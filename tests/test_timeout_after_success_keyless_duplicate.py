"""
Regression: timeout_after_success followed by a retry with the same operation_id
must produce exactly one refund and return the original refund_id.

Two distinct operation_ids for the same order and amount must produce two separate refunds.
"""
import pytest
import uuid

from src.silentfail.models import FaultConfig, Order
from src.silentfail.ledger import WorldState, CausalLedger
from src.silentfail.tools import ToolRegistry, FaultInjector


@pytest.fixture
def world():
    ws = WorldState()
    ws.orders["ord_1"] = Order(id="ord_1", total=100.0, refunded_amount=0.0)
    return ws


def test_keyless_timeout_after_success_duplicate_refund(world):
    """A retry with the same operation_id after a timeout must not create a second refund."""
    op_id = f"op_{uuid.uuid4().hex[:8]}"
    ledger = CausalLedger(scenario_id="timeout_after_success_test")

    # First call: fault fires, mutation is applied, but agent sees timeout.
    fault_injector = FaultInjector([
        FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success")
    ])
    tools1 = ToolRegistry(world_state=world, ledger=ledger, fault_injector=fault_injector)
    resp1 = tools1.issue_refund("ord_1", 30.0, operation_id=op_id)
    assert resp1.get("error") == "timeout", f"Expected timeout, got: {resp1}"
    assert len(world.refunds) == 1, "Refund must have been created despite timeout"
    original_refund_id = next(iter(world.refunds))

    # Retry: same operation_id, no fault — must hit cache, return original result, no new refund.
    tools2 = ToolRegistry(world_state=world, ledger=ledger)
    # Share the same _refund_key_cache so the retry sees the cached result.
    tools2._refund_key_cache = tools1._refund_key_cache
    resp2 = tools2.issue_refund("ord_1", 30.0, operation_id=op_id)
    assert resp2.get("status") == "ok", f"Expected ok on retry, got: {resp2}"
    assert resp2.get("refund_id") == original_refund_id, "Retry must return the original refund_id"
    assert len(world.refunds) == 1, "No additional refund must be created on retry"


def test_two_distinct_keyless_refunds_allowed(world):
    """Two calls with different operation_ids must each create a separate refund."""
    ledger = CausalLedger(scenario_id="distinct_refunds_test")
    tools = ToolRegistry(world_state=world, ledger=ledger)

    op1 = f"op_{uuid.uuid4().hex[:8]}"
    r1 = tools.issue_refund("ord_1", 30.0, operation_id=op1)
    assert r1.get("status") == "ok", f"First refund failed: {r1}"
    first_id = r1["refund_id"]

    op2 = f"op_{uuid.uuid4().hex[:8]}"
    r2 = tools.issue_refund("ord_1", 30.0, operation_id=op2)
    assert r2.get("status") == "ok", f"Second refund failed: {r2}"
    second_id = r2["refund_id"]

    assert first_id != second_id, "Distinct operation_ids must produce distinct refund_ids"
    assert len(world.refunds) == 2, "Both refunds must exist in world state"
