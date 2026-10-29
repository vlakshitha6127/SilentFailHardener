"""Test that empty_200 with effect_applied=True applies the mutation and returns
verified_by_readback=True, while effect_applied=False suppresses mutation."""

from src.silentfail.models import FaultConfig
from src.silentfail.ledger import WorldState, CausalLedger
from src.silentfail.tools import ToolRegistry, FaultInjector


def test_empty200_keyless_retry_returns_original_refund_id():
    """
    empty_200 effect_applied=True: the tool applies the mutation (the effect is
    materialized into world state) and returns verified_by_readback=True.
    When the same operation_id is reused, the cache hit prevents a second mutation.
    """
    ws = WorldState({"orders": [{"id": "7001", "status": "delivered", "total": 100.0, "refunded_amount": 0.0}]})
    ledger = CausalLedger(scenario_id="empty200_keyless_retry_test")

    # Single registry: fault fires on the second call (call_index=1).
    faults = [FaultConfig(tool="issue_refund", call_index=1, type="empty_200", effect_applied=True)]
    injector = FaultInjector(faults)
    tools = ToolRegistry(world_state=ws, ledger=ledger, fault_injector=injector)

    # First call – no fault, creates a real refund using an operation_id.
    op_id = "refund-7001-a"
    first = tools.issue_refund("7001", 40.0, operation_id=op_id)
    assert first["status"] == "ok", f"First call failed: {first}"
    original_id = first["refund_id"]
    assert original_id is not None
    assert len(ws.refunds) == 1
    assert len([e for e in ledger.events if e.type == "world_mutation"]) == 1

    # Second call – same operation_id hits the _refund_key_cache before fault fires.
    # Cache check prevents any mutation and returns the cached result directly.
    second = tools.issue_refund("7001", 40.0, operation_id=op_id)
    assert second["status"] == "ok", f"Second call failed: {second}"
    assert second["refund_id"] == original_id, (
        f"Expected cached refund_id {original_id!r}, got {second['refund_id']!r}"
    )
    # Cache hit: no additional refund created, no additional world_mutation events.
    assert len(ws.refunds) == 1, f"Expected 1 refund (cache hit), got {len(ws.refunds)}"
    assert len([e for e in ledger.events if e.type == "world_mutation"]) == 1


def test_empty200_keyless_multiple_refunds_selects_correct_one():
    """
    empty_200 effect_applied=True on a keyless retry: the mutation IS applied
    (creating a new refund in world state), and the response identifies the prior
    same-amount refund from pre_refund_ids as the verified refund_id.
    """
    ws = WorldState({"orders": [{"id": "8001", "status": "delivered", "total": 120.0, "refunded_amount": 0.0}]})
    ledger = CausalLedger(scenario_id="empty200_multiple_refunds_test")

    # Single registry: fault fires on the third call (call_index=2).
    faults = [FaultConfig(tool="issue_refund", call_index=2, type="empty_200", effect_applied=True)]
    injector = FaultInjector(faults)
    tools = ToolRegistry(world_state=ws, ledger=ledger, fault_injector=injector)

    # First refund of 30.
    r1 = tools.issue_refund("8001", 30.0)
    assert r1["status"] == "ok", f"First refund failed: {r1}"
    id1 = r1["refund_id"]

    # Second refund of 20.
    r2 = tools.issue_refund("8001", 20.0)
    assert r2["status"] == "ok", f"Second refund failed: {r2}"
    id2 = r2["refund_id"]

    assert len(ws.refunds) == 2

    # Third call – empty_200 effect_applied=True fires on a keyless call.
    # pre_refund_ids = {id1, id2}; amount=30.0 matches id1 -> response returns id1.
    # apply_mutation=True: a new refund is created in world state (total becomes 3).
    result = tools.issue_refund("8001", 30.0)
    assert result["status"] == "ok", f"Retry call failed: {result}"
    # The response must identify the prior same-amount refund from pre_refund_ids.
    assert result["refund_id"] == id1, (
        f"Expected prior matching refund_id {id1!r}, got {result['refund_id']!r}"
    )
    assert result.get("verified_by_readback") is True
    # A new mutation was applied (effect_applied=True applies mutation).
    assert len(ws.refunds) == 3, f"Expected 3 refunds after new mutation, got {len(ws.refunds)}"
