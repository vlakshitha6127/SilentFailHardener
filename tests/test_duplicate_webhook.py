'''Test for duplicate_webhook fault not creating extra refund mutation.'''

from src.silentfail.models import FaultConfig
from src.silentfail.ledger import WorldState, CausalLedger
from src.silentfail.tools import ToolRegistry, FaultInjector


def test_duplicate_webhook_no_extra_mutation():
    # Initial world state with a single order.
    initial = {"orders": [{"id": "3001", "status": "delivered", "total": 100.0, "refunded_amount": 0.0}]}
    ws = WorldState(initial)
    ledger = CausalLedger(scenario_id="test_duplicate_webhook")

    # Inject duplicate_webhook on the second call (call_index=1).
    faults = [FaultConfig(tool="issue_refund", call_index=1, type="duplicate_webhook")]
    injector = FaultInjector(faults)
    tools = ToolRegistry(world_state=ws, ledger=ledger, fault_injector=injector)

    # First refund – creates a mutation.
    res1 = tools.issue_refund("3001", 30.0)
    assert res1["status"] == "ok"
    assert len(ws.refunds) == 1
    # Verify a world mutation event recorded.
    mutation_events = [e for e in ledger.events if e.type == "world_mutation"]
    assert len(mutation_events) == 1

    # Second call triggers duplicate_webhook fault.
    res2 = tools.issue_refund("3001", 30.0)
    # The observed response should be the simulated webhook event.
    assert res2["status"] == "ok"
    assert res2.get("event") == "webhook.refund.completed"
    # No additional refund record or world mutation.
    assert len(ws.refunds) == 1
    mutation_events = [e for e in ledger.events if e.type == "world_mutation"]
    assert len(mutation_events) == 1
