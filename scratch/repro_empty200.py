import sys
sys.path.append('src')
from silentfail.tools import ToolRegistry, FaultInjector
from silentfail.models import FaultConfig, WorldState, CausalLedger

# Initial world with an order
ws = WorldState({
    "orders": [{"id": "o1", "status": "delivered", "total": 100.0, "refunded_amount": 0.0}],
    "refunds": {}
})
ledger = CausalLedger(scenario_id='test')
# First call without fault
tools = ToolRegistry(ws, ledger)
res1 = tools.issue_refund('o1', 30.0, idempotency_key='key1')
print('first', res1)
print('refund count after first', len(ws.refunds))
# Inject empty_200 with effect_applied=True for second call (same idempotency_key)
faults = [FaultConfig(tool='issue_refund', call_index=0, type='empty_200', effect_applied=True)]
inj = FaultInjector(faults)
tools2 = ToolRegistry(ws, ledger, inj)
res2 = tools2.issue_refund('o1', 30.0, idempotency_key='key1')
print('second', res2)
print('refund count after second', len(ws.refunds))
