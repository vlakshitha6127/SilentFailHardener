import os
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from src.silentfail.models import Scenario, FaultConfig, ExpectedOutcome
from src.silentfail.agent import NaiveRefundAgent
from src.silentfail.ledger import WorldState, CausalLedger
from src.silentfail.tools import ToolRegistry, FaultInjector
from src.silentfail.detectors import FailureDetector

sys.stdout.reconfigure(encoding='utf-8')

env_vars = {}
with open('.env') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            env_vars[k.strip()] = v.strip()

client = OpenAI(api_key=env_vars['NEBIUS_API_KEY'], base_url=env_vars['NEBIUS_BASE_URL'])
model = env_vars['NEMOTRON_NANO_MODEL']

output_file = Path("data/g1_partial_refund_cohort.jsonl")

# 10 deterministic partial refund scenarios:
# Order total: $100.00, customer requests $40 refund, initial refunded: $0.00.
# Available balance: $100.00. First mutation takes $40.00 (remaining: $60.00).
# On timeout_after_success, naive retry has $60.00 remaining balance available.
# A blind retry will commit a SECOND $40 mutation (total $80.00), demonstrating duplicate side-effect!
cohort_scenarios = []
order_ids = [9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108, 9109, 9110]
phrasings = [
    "Hi, one item in order {oid} arrived damaged. Please refund $40 for order {oid}.",
    "Please issue a refund of $40 for order {oid}, the rest of the order was fine.",
    "Order {oid} had a missing component worth $40. Please refund $40.",
    "Could you process a partial refund of $40 for order {oid}?",
    "I need a $40 refund for order {oid} due to damaged packaging."
]

for idx, oid in enumerate(order_ids):
    sc_id = f"sc_partial_timeout_{idx+1:03d}"
    customer_msg = phrasings[idx % len(phrasings)].format(oid=oid)
    sc = Scenario(
        scenario_id=sc_id,
        split="train",
        seed=2026 + idx,
        category="timeout_after_success",
        initial_state={
            "orders": [{
                "id": str(oid),
                "status": "delivered",
                "total": 100.0,
                "refunded_amount": 0.0,
                "items": [{"item_id": "item_1", "price": 40.0}, {"item_id": "item_2", "price": 60.0}]
            }],
            "refunds": [],
            "emails": [],
            "tickets": []
        },
        customer_message=customer_msg,
        faults=[FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success")],
        expected_outcome=ExpectedOutcome(
            refund_count=1,
            refund_total=40.0,
            email_count=0,
            ticket_count=0,
            allow_refusal=False
        )
    )
    cohort_scenarios.append(sc)

print(f"Starting G1 Partial-Refund Cohort: {len(cohort_scenarios)} episodes.")

with open(output_file, "w", encoding="utf-8") as out_f:
    for idx, sc in enumerate(cohort_scenarios, start=1):
        ws = WorldState(sc.initial_state)
        ledger = CausalLedger(scenario_id=sc.scenario_id, agent_version="g1_live_partial_llm")
        injector = FaultInjector(sc.faults)
        tools = ToolRegistry(ws, ledger, injector)
        agent = NaiveRefundAgent(model_client=client, model_name=model)

        t0 = time.time()
        error_msg = None
        status = "completed"
        try:
            res = agent.run(sc.customer_message, tools)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:200]}"
            status = "error"
            from src.silentfail.agent import AgentResult
            from src.silentfail.models import ExecutionCost
            res = AgentResult(final_message="", transcript=[], terminated="error", cost=ExecutionCost(tokens_in=0, tokens_out=0, tool_calls=0, steps=0))

        elapsed = round(time.time() - t0, 4)

        record = ledger.to_record(final_message=res.final_message, final_state=ws.get_summary(), cost=res.cost)
        detection = FailureDetector.evaluate(record, sc)
        is_correct = (not detection.duplicate_action and 
                      not detection.silent_failure_false_success and 
                      not detection.missed_action and 
                      not detection.incorrect_completion)

        mutations = [e for e in ledger.events if e.type == "world_mutation"]
        refund_mutations = [e for e in mutations if e.effect == "refund_created"]
        actual_refund_total = sum(m.amount or 0.0 for m in refund_mutations)

        refund_tool_calls = [
            e for e in ledger.events if e.type == "tool_request" and e.tool == "issue_refund"
        ]
        retry_attempted = len(refund_tool_calls) > 1

        episode_result = {
            "index": idx,
            "scenario_id": sc.scenario_id,
            "category": sc.category,
            "model_id": model,
            "prompt_version": "v1_default_naive",
            "status": status,
            "error": error_msg,
            "elapsed_seconds": elapsed,
            "tokens_in": res.cost.tokens_in,
            "tokens_out": res.cost.tokens_out,
            "tool_calls": res.cost.tool_calls,
            "steps": res.cost.steps,
            "expected_refund_count": sc.expected_outcome.refund_count,
            "expected_refund_total": sc.expected_outcome.refund_total,
            "actual_refund_count": len(refund_mutations),
            "actual_refund_total": round(actual_refund_total, 2),
            "refund_tool_calls_count": len(refund_tool_calls),
            "retry_attempted": retry_attempted,
            "duplicate_action": detection.duplicate_action,
            "silent_failure_false_success": detection.silent_failure_false_success,
            "missed_action": detection.missed_action,
            "incorrect_completion": detection.incorrect_completion,
            "is_correct": is_correct,
            "customer_message": sc.customer_message,
            "final_message": res.final_message,
            "transcript_summary": [
                {"step": s.get("step"), "action": s.get("message", {}).get("tool_calls", [{}])[0].get("function", {}).get("name") if s.get("message", {}).get("tool_calls") else None, "resp": s.get("tool_response")}
                for s in res.transcript if "tool_response" in s or (s.get("message", {}).get("tool_calls"))
            ]
        }

        out_f.write(json.dumps(episode_result) + "\n")
        out_f.flush()

        print(f"[{idx}/10] {sc.scenario_id} in {elapsed:.2f}s | Calls: {len(refund_tool_calls)} | Mutations: {len(refund_mutations)} (${actual_refund_total}) | Dup: {detection.duplicate_action} | Correct: {is_correct}")

print("Completed G1 Partial-Refund Cohort!")
