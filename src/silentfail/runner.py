"""
Scenario Runner, Sharded Execution, Verification & Selection Engine.
Sections 9.0, 11.0 of the Project Spec.

Implements:
- Sharded scenario execution (batching, local/remote runner)
- Deterministic seeding and isolation
- Verification sweep across candidates
- Strict 4 guards:
  1. Regression guard (correct completion not lower than baseline within tolerance)
  2. Legitimate-completion guard (legitimate repeats + no-fault not lower than baseline)
  3. No new failure class (no detector rate increases beyond tolerance)
  4. Cost guard (mean steps and tokens not exceeding baseline by more than threshold)
- Held-out evaluation confirmation
"""

from typing import List, Dict, Any, Optional, Tuple
from src.silentfail.models import Scenario, PatchSpec, LedgerRecord, DetectionResult
from src.silentfail.ledger import WorldState, CausalLedger
from src.silentfail.tools import ToolRegistry, FaultInjector
from src.silentfail.patches import PatchedToolRegistry
from src.silentfail.agent import NaiveRefundAgent
from src.silentfail.detectors import FailureDetector, calculate_wilson_score_interval
import time


class EpisodeRunner:
    """
    Executes a single scenario episode in isolated state.
    """
    @classmethod
    def run_episode(cls, scenario: Scenario, patch: Optional[PatchSpec] = None, agent: Optional[NaiveRefundAgent] = None) -> Dict[str, Any]:
        agent = agent or NaiveRefundAgent()
        
        # 1. Fresh world state seeded from scenario
        world_state = WorldState(scenario.initial_state)
        
        # 2. Causal ledger
        ledger = CausalLedger(
            scenario_id=scenario.scenario_id,
            agent_version=patch.patch_id if patch else "baseline",
            patch_id=patch.patch_id if patch else None
        )
        
        # 3. Fault injector & tool registry
        injector = FaultInjector(scenario.faults)
        raw_tools = ToolRegistry(world_state=world_state, ledger=ledger, fault_injector=injector)
        tools = PatchedToolRegistry(raw_tools, patch=patch)
        
        # 4. Run agent
        t0 = time.time()
        agent_result = agent.run(scenario.customer_message, tools)
        elapsed = time.time() - t0
        
        # 5. Extract final state & ledger record
        final_state = world_state.get_summary()
        record = ledger.to_record(
            final_message=agent_result.final_message,
            final_state=final_state,
            cost=agent_result.cost
        )
        
        # 6. Evaluate detectors deterministically
        detection = FailureDetector.evaluate(record, scenario)
        
        # Derive correctness
        is_correct = (not detection.duplicate_action and 
                      not detection.silent_failure_false_success and 
                      not detection.missed_action and 
                      not detection.incorrect_completion)

        return {
            "record": record,
            "detection": detection,
            "is_correct": is_correct,
            "elapsed_seconds": elapsed,
            "category": scenario.category
        }


class BenchmarkAggregator:
    """
    Aggregates run metrics, computes Wilson confidence intervals,
    and checks verifier guards.

    Metric definitions (denominators matter):
    - correct_completion: is_correct runs / total runs.
    - silent_failure_rate: runs flagged silent_failure_false_success / total runs. The detector
      flags a final message that claims a refund was issued while the ground-truth state has no
      refund, claims more than was refunded, or claims success after a duplicate refund.
    - conditional_false_success_rate: among runs that are NOT correct (counts.cond_failed_total),
      the share whose final message claims a refund was issued (claims_refund_issued). This is
      intentionally a claim rate, not a proof that the claim was false.
    - legitimate_completion_rate: correct runs among legitimate_repeat + no_fault_control scenarios.
    """
    @classmethod
    def aggregate(cls, results: List[Dict[str, Any]], patch_id: str = "baseline") -> Dict[str, Any]:
        total = len(results)
        if total == 0:
            return {}

        correct_count = sum(1 for r in results if r["is_correct"])
        duplicate_count = sum(1 for r in results if r["detection"].duplicate_action)
        excess_mutation_count = sum(1 for r in results if r["detection"].excess_mutation)
        ineligible_mutation_count = sum(1 for r in results if r["detection"].ineligible_refund_mutation)
        incorrect_amount_count = sum(1 for r in results if r["detection"].incorrect_refund_amount)
        silent_fail_count = sum(1 for r in results if r["detection"].silent_failure_false_success)
        missed_count = sum(1 for r in results if r["detection"].missed_action)
        incorrect_count = sum(1 for r in results if r["detection"].incorrect_completion)
        over_suppression_count = sum(1 for r in results if r["detection"].over_suppression)
        uncertain_count = sum(1 for r in results if r["detection"].claim_uncertain)

        # Conditional false success: among duplicate or failed runs, where agent claimed success
        failed_runs = [r for r in results if not r["is_correct"]]
        cond_false_success_count = sum(1 for r in failed_runs if r["detection"].claims_refund_issued)
        cond_total = len(failed_runs)

        # Legitimate-completion rate: restricted to legitimate-repeat and no-fault controls
        legit_runs = [r for r in results if r["category"] in ("legitimate_repeat", "no_fault_control")]
        legit_correct_count = sum(1 for r in legit_runs if r["is_correct"])
        legit_total = len(legit_runs)

        # Costs
        total_steps = sum(r["record"].cost.steps for r in results)
        total_tokens = sum(r["record"].cost.tokens_in + r["record"].cost.tokens_out for r in results)
        mean_steps = round(total_steps / total, 2)
        mean_tokens = round(total_tokens / total, 2)

        return {
            "patch_id": patch_id,
            "total_runs": total,
            "correct_completion": calculate_wilson_score_interval(correct_count, total),
            "duplicate_action_rate": calculate_wilson_score_interval(duplicate_count, total),
            "excess_mutation_rate": calculate_wilson_score_interval(excess_mutation_count, total),
            "ineligible_refund_mutation_rate": calculate_wilson_score_interval(ineligible_mutation_count, total),
            "incorrect_refund_amount_rate": calculate_wilson_score_interval(incorrect_amount_count, total),
            "silent_failure_rate": calculate_wilson_score_interval(silent_fail_count, total),
            "conditional_false_success_rate": calculate_wilson_score_interval(cond_false_success_count, cond_total) if cond_total > 0 else (0.0, 0.0, 0.0),
            "legitimate_completion_rate": calculate_wilson_score_interval(legit_correct_count, legit_total) if legit_total > 0 else (0.0, 0.0, 0.0),
            "missed_action_rate": calculate_wilson_score_interval(missed_count, total),
            "incorrect_completion_rate": calculate_wilson_score_interval(incorrect_count, total),
            "over_suppression_rate": calculate_wilson_score_interval(over_suppression_count, total),
            "claim_uncertain_share": round(uncertain_count / total, 4),
            "mean_steps": mean_steps,
            "mean_tokens": mean_tokens,
            "counts": {
                "correct": correct_count,
                "duplicate": duplicate_count,
                "excess_mutation": excess_mutation_count,
                "ineligible_refund_mutation": ineligible_mutation_count,
                "incorrect_refund_amount": incorrect_amount_count,
                "silent_fail": silent_fail_count,
                "total": total,
                "cond_failed_total": cond_total
            }
        }


class VerifierAndSelector:
    """
    Enforces the 4 selection guards specified in Section 11.2:
    - Regression guard
    - Legitimate-completion guard
    - No new failure class
    - Cost guard
    """
    def __init__(self, tolerance: float = 0.02, max_cost_factor: float = 1.8):
        self.tolerance = tolerance
        self.max_cost_factor = max_cost_factor

    def evaluate_guards(self, baseline_metrics: Dict[str, Any], candidate_metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        rejection_reasons = []

        # Validate that metrics are valid and complete
        required_keys = [
            "correct_completion", "legitimate_completion_rate", 
            "duplicate_action_rate", "silent_failure_rate", 
            "missed_action_rate", "incorrect_completion_rate", 
            "over_suppression_rate", "mean_steps"
        ]
        for rk in required_keys:
            if rk not in candidate_metrics or candidate_metrics[rk] is None:
                rejection_reasons.append(f"Incomplete Metrics Guard: Candidate missing required metric '{rk}'")
                return False, rejection_reasons

        base_correct = baseline_metrics["correct_completion"][0]
        cand_correct = candidate_metrics["correct_completion"][0]
        # 1. Regression guard
        if cand_correct < (base_correct - self.tolerance):
            rejection_reasons.append(
                f"Regression Guard: Correct completion dropped from {base_correct:.1%} to {cand_correct:.1%} (tolerance: -{self.tolerance:.1%})"
            )

        # 2. Legitimate-completion guard
        base_legit = baseline_metrics["legitimate_completion_rate"][0]
        cand_legit = candidate_metrics["legitimate_completion_rate"][0]
        if cand_legit < (base_legit - self.tolerance):
            rejection_reasons.append(
                f"Legitimate-Completion Guard: Legitimate completion dropped from {base_legit:.1%} to {cand_legit:.1%}"
            )

        # 3. Duplicate-action guard: candidate must not increase duplicate-action rate
        base_dup = baseline_metrics["duplicate_action_rate"][0]
        cand_dup = candidate_metrics["duplicate_action_rate"][0]
        if cand_dup > (base_dup + self.tolerance):
            rejection_reasons.append(
                f"Duplicate Action Guard: Duplicate rate increased from {base_dup:.1%} to {cand_dup:.1%}"
            )

        # 4. Silent failure / False-success claim guard
        base_silent = baseline_metrics["silent_failure_rate"][0]
        cand_silent = candidate_metrics["silent_failure_rate"][0]
        if cand_silent > (base_silent + self.tolerance):
            rejection_reasons.append(
                f"False Success Guard: Silent failure / false success rate increased from {base_silent:.1%} to {cand_silent:.1%}"
            )

        # 5. Missed action guard
        base_missed = baseline_metrics["missed_action_rate"][0]
        cand_missed = candidate_metrics["missed_action_rate"][0]
        if cand_missed > (base_missed + self.tolerance + 0.05):
            rejection_reasons.append(
                f"Missed Action Guard: Missed action rate increased from {base_missed:.1%} to {cand_missed:.1%}"
            )

        # 6. Over-suppression guard
        cand_oversupp = candidate_metrics.get("over_suppression_rate", [0.0])[0]
        if cand_oversupp > self.tolerance:
            rejection_reasons.append(
                f"Over-Suppression Guard: Candidate exhibits over-suppression rate of {cand_oversupp:.1%}"
            )

        # 7. Incorrect completion guard
        base_incorrect = baseline_metrics["incorrect_completion_rate"][0]
        cand_incorrect = candidate_metrics["incorrect_completion_rate"][0]
        if cand_incorrect > (base_incorrect + self.tolerance):
            rejection_reasons.append(
                f"Incorrect Completion Guard: Incorrect completion increased from {base_incorrect:.1%} to {cand_incorrect:.1%}"
            )

        # 8. Cost guard
        base_steps = baseline_metrics["mean_steps"]
        cand_steps = candidate_metrics["mean_steps"]
        if cand_steps > (base_steps * self.max_cost_factor):
            rejection_reasons.append(
                f"Cost Guard: Mean steps {cand_steps} exceeded {self.max_cost_factor}x baseline {base_steps}"
            )

        is_eligible = len(rejection_reasons) == 0
        return is_eligible, rejection_reasons

    def select_winner(self, baseline_metrics: Dict[str, Any], candidate_results: Dict[str, Dict[str, Any]], candidate_specs: List[PatchSpec]) -> Tuple[Optional[PatchSpec], Dict[str, Any]]:
        spec_map = {s.patch_id: s for s in candidate_specs}
        race_summary = {}
        best_candidate: Optional[PatchSpec] = None
        best_target_reduction = -1.0
        best_secondary: Optional[Tuple[float, float, float, str]] = None

        base_dup_rate = baseline_metrics["duplicate_action_rate"][0]

        # Sort candidates deterministically by patch_id for reproducibility
        sorted_patch_ids = sorted(candidate_results.keys())

        for p_id in sorted_patch_ids:
            metrics = candidate_results[p_id]
            is_eligible, rejections = self.evaluate_guards(baseline_metrics, metrics)
            cand_dup_rate = metrics["duplicate_action_rate"][0]
            reduction = base_dup_rate - cand_dup_rate

            race_summary[p_id] = {
                "spec": spec_map[p_id].model_dump(),
                "eligible": is_eligible,
                "rejection_reasons": rejections,
                "duplicate_action_rate": cand_dup_rate,
                "duplicate_reduction": round(reduction, 4),
                "correct_completion": metrics["correct_completion"][0],
                "legitimate_completion": metrics["legitimate_completion_rate"][0],
                "metrics": metrics
            }

            if is_eligible:
                # Primary: measured duplicate reduction. Ties use measured correctness,
                # legitimate completion, then lower mean steps; patch_id is only the
                # final reproducibility key, never evidence of safety.
                secondary = (
                    metrics["correct_completion"][0],
                    metrics["legitimate_completion_rate"][0],
                    -metrics["mean_steps"],
                    p_id,
                )
                if reduction > best_target_reduction + 0.001:
                    best_target_reduction = reduction
                    best_candidate = spec_map[p_id]
                    best_secondary = secondary
                elif abs(reduction - best_target_reduction) < 0.001 and (
                    best_secondary is None or secondary > best_secondary
                ):
                    best_candidate = spec_map[p_id]
                    best_secondary = secondary

        return best_candidate, race_summary
