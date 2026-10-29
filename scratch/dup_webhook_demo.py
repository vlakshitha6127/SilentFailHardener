import json
from src.silentfail.models import FaultConfig
from src.silentfail.ledger import WorldState, CausalLedger
from src.silentfail.tools import ToolRegistry, FaultInjector

initial = {"orders": [{"id": "5000", "status": "delivered", "total": 100.0, "refunded_amount": 0.0}]}
ws = WorldState(initial)
ledger = CausalLedger(scenario_id='dup_test')
faults = [FaultConfig(tool='issue_refund', call_index=1, type='duplicate_webhook')]
injector = FaultInjector(faults)
tools = ToolRegistry(world_state=ws, ledger=ledger, fault_injector=injector)
res1 = tools.issue_refund('5000', 25.0)
res2 = tools.issue_refund('5000', 25.0)
print('res1', json.dumps(res1))
print('res2', json.dumps(res2))
print('refunds', len(ws.refunds))
print('events', [e.type for e in ledger.events])
