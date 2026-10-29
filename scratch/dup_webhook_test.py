from src.silentfail.models import Refund, Order
from src.silentfail.ledger import WorldState, CausalLedger
from src.silentfail.tools import ToolRegistry, FaultInjector
from src.silentfail.models import FaultConfig

initial = {"orders": [{"id": "101", "status": "delivered", "total": 50.0, "refunded_amount": 0.0}]}
ws = WorldState(initial)
ledger = CausalLedger(scenario_id='test')
# Inject duplicate_webhook fault on second call (call_index=1)
faults = [FaultConfig(tool="issue_refund", call_index=1, type="duplicate_webhook")]
injector = FaultInjector(faults)
tools = ToolRegistry(world_state=ws, ledger=ledger, fault_injector=injector)
# first refund
res1 = tools.issue_refund('101', 20.0)
# second refund (should trigger duplicate_webhook)
res2 = tools.issue_refund('101', 20.0)
print('res1', res1)
print('res2', res2)
print('refund count', len(ws.refunds))
print('ledger events', [e.type for e in ledger.events])
