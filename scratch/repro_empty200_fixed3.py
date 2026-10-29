import os, sys
sys.path.append(os.path.abspath(''))  # add repo root so 'src' package is importable
from silentfail.tools import ToolRegistry, FaultInjector
from silentfail.models import FaultConfig, WorldState, CausalLedger

# Setup world with one order
ws = WorldState({
    "orders": [{"id": "o1", "status": "delivered", "total": 100.0, "refunded_amount": 0.0}],
    "refunds": {}
})
ledger = CausalLedger(scenario_id='repro')
# First call without fault, with idempotency key
tools = ToolRegistry(ws, ledger)
res1 = tools.issue_refund('o1', 30.0, idempotency_key='key1')
print('first result:', res1)
print('refund count after first:', len(ws.refunds))
# Second call with empty_200 fault effect_applied=True, same idempotency key
faults = [FaultConfig(tool='issue_refund', call_index=0, type='empty_200', effect_applied=True)]
inj = FaultInjector(faults)
tools2 = ToolRegistry(ws, ledger, inj)
res2 = tools2.issue_refund('o1', 30.0, idempotency_key='key1')
print('second result:', res2)
print('refund count after second:', len(ws.refunds))
