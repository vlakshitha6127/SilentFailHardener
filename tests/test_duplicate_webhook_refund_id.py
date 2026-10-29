'''Test that duplicate_webhook returns the original refund_id without extra mutation.'''

from src.silentfail.models import FaultConfig
from src.silentfail.ledger import WorldState, CausalLedger
from src.silentfail.tools import ToolRegistry, FaultInjector


def test_duplicate_webhook_returns_original_refund_id():
    # Initial state with one order.
    ws = WorldState({"orders": [{"id": "6001", "status": "delivered", "total": 120.0, "refunded_amount": 0.0}]})
    ledger = CausalLedger(scenario_id="dup_webhook_id_test")

    # Inject duplicate_webhook fault on the second call (call_index=1).
    faults = [FaultConfig(tool="issue_refund", call_index=1, type="duplicate_webhook")]
    injector = FaultInjector(faults)
    tools = ToolRegistry(world_state=ws, ledger=ledger, fault_injector=injector)

    # First refund creates a real mutation.
    first = tools.issue_refund("6001", 40.0)
    assert first["status"] == "ok"
    original_id = first["refund_id"]
    assert original_id is not None
    assert len(ws.refunds) == 1
    assert len([e for e in ledger.events if e.type == "world_mutation"]) == 1

    # Second call triggers duplicate_webhook.
    second = tools.issue_refund("6001", 40.0)
    # Should return the same refund_id and webhook event.
    assert second["status"] == "ok"
    assert second.get("event") == "webhook.refund.completed"
    assert second.get("refund_id") == original_id
    # No extra refund or mutation.
    assert len(ws.refunds) == 1
    assert len([e for e in ledger.events if e.type == "world_mutation"]) == 1
