import os
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from src.silentfail.scenarios import ScenarioGenerator
from src.silentfail.agent import NaiveRefundAgent
from src.silentfail.ledger import WorldState, CausalLedger
from src.silentfail.tools import ToolRegistry, FaultInjector
from src.silentfail.detectors import FailureDetector

# Force utf-8 stdout
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

output_file = Path("data/g1_live_evaluation_results.jsonl")
output_file.parent.mkdir(parents=True, exist_ok=True)

# Generate exactly 50 scenarios
gen = ScenarioGenerator(base_seed=1234)
scenarios = gen.generate_suite(50, split="train")

# Check existing completed scenario IDs if resuming
completed_ids = set()
if output_file.exists():
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    completed_ids.add(data["scenario_id"])
                except Exception:
                    pass

print(f"Starting G1 run: {len(completed_ids)} already completed, {len(scenarios) - len(completed_ids)} remaining.")

with open(output_file, "a", encoding="utf-8") as out_f:
    for idx, sc in enumerate(scenarios, start=1):
        if sc.scenario_id in completed_ids:
            continue

        ws = WorldState(sc.initial_state)
        ledger = CausalLedger(scenario_id=sc.scenario_id, agent_version="g1_live_llm")
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
            # Create fallback empty result
            from src.silentfail.agent import AgentResult
            from src.silentfail.models import ExecutionCost
            res = AgentResult(final_message="", transcript=[], terminated="error", cost=ExecutionCost(tokens_in=0, tokens_out=0, tool_calls=0, steps=0))

        elapsed = round(time.time() - t0, 4)

        # Run deterministic detectors
        record = ledger.to_record(final_message=res.final_message, final_state=ws.get_summary(), cost=res.cost)
        detection = FailureDetector.evaluate(record, sc)
        is_correct = (not detection.duplicate_action and 
                      not detection.silent_failure_false_success and 
                      not detection.missed_action and 
                      not detection.incorrect_completion)

        mutations = [e for e in ledger.events if e.type == "world_mutation"]
        refund_mutations = [e for e in mutations if e.effect == "refund_created"]
        actual_refund_total = sum(m.amount or 0.0 for m in refund_mutations)

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
            "duplicate_action": detection.duplicate_action,
            "silent_failure_false_success": detection.silent_failure_false_success,
            "missed_action": detection.missed_action,
            "incorrect_completion": detection.incorrect_completion,
            "is_correct": is_correct,
            "customer_message": sc.customer_message,
            "final_message": res.final_message,
            "faults": [{"tool": f.tool, "call_index": f.call_index, "type": f.type, "effect_applied": f.effect_applied} for f in sc.faults],
            "transcript_summary": [
                {"step": s.get("step"), "action": s.get("message", {}).get("tool_calls", [{}])[0].get("function", {}).get("name") if s.get("message", {}).get("tool_calls") else None, "resp": s.get("tool_response")}
                for s in res.transcript if "tool_response" in s or (s.get("message", {}).get("tool_calls"))
            ]
        }

        out_f.write(json.dumps(episode_result) + "\n")
        out_f.flush()

        print(f"[{idx}/50] {sc.scenario_id} ({sc.category}) in {elapsed:.2f}s | Correct: {is_correct} | Dup: {detection.duplicate_action} | SilentFail: {detection.silent_failure_false_success}")

print("Completed G1 run!")
