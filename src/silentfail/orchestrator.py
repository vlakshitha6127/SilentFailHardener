"""
SilentFail Hardener Orchestrator and Pipeline Coordinator.
Executes the full closed-loop life cycle:
1. Scenario generation (train + held-out)
2. Baseline replay
3. Deterministic failure detection and metrics aggregation
4. Ultra root-cause diagnosis & candidate patch proposal (including tagged controls)
5. Verification sweep across candidates with 4 guards
6. Held-out validation of winning patch
7. Emission of runnable regression test
"""

from typing import Dict, Any, List, Optional
import time
import json
import pprint
from pathlib import Path
from src.silentfail.models import Scenario, PatchSpec
from src.silentfail.scenarios import ScenarioGenerator
from src.silentfail.patches import get_default_candidate_patches
from src.silentfail.runner import EpisodeRunner, BenchmarkAggregator, VerifierAndSelector
from src.silentfail.nebius import NebiusClient

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED_TEST_FILENAME = "test_generated_regression.py"


class Orchestrator:
    def __init__(self, data_dir: str = "data", tests_dir: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Generated regression tests are written next to the repo's tests, anchored to the
        # repository root (not the current working directory).
        self.tests_dir = Path(tests_dir) if tests_dir else REPO_ROOT / "tests"
        self.nebius_client = NebiusClient()
        self.verifier = VerifierAndSelector(tolerance=0.02)
        
        # State cache for dashboard
        self.pipeline_state: Dict[str, Any] = {
            "status": "idle",
            "progress": 0,
            "gate_g0": {},
            "baseline_metrics": {},
            "race_summary": {},
            "winning_patch": None,
            "heldout_metrics_before": {},
            "heldout_metrics_after": {},
            "sample_failing_record": None,
            "scale_and_cost": {},
            "generated_test_code": ""
        }

    def run_full_pipeline(self, train_count: int = 50, held_out_count: int = 30) -> Dict[str, Any]:
        self.pipeline_state["status"] = "running"
        t0_total = time.time()

        # Step 0: Gate G0 probe
        g0_res = self.nebius_client.probe_gate_g0()
        self.pipeline_state["gate_g0"] = g0_res

        # Step 1: Generate Scenarios
        gen = ScenarioGenerator(base_seed=42)
        train_scenarios = gen.generate_suite(train_count, split="train")
        held_out_scenarios = gen.generate_suite(held_out_count, split="held_out", seed_offset=5000)

        # Step 2: Baseline Sweep on Train Scenarios
        baseline_results = []
        episode_durations: List[float] = []
        for sc in train_scenarios:
            res = EpisodeRunner.run_episode(sc)
            baseline_results.append(res)
            episode_durations.append(res["elapsed_seconds"])
            if not res["is_correct"] and not self.pipeline_state["sample_failing_record"]:
                self.pipeline_state["sample_failing_record"] = res["record"].model_dump()

        base_metrics = BenchmarkAggregator.aggregate(baseline_results, "baseline")
        self.pipeline_state["baseline_metrics"] = base_metrics

        # Step 3: Propose candidate patches (Ultra + tagged controls)
        failing_ledgers = [r["record"].model_dump() for r in baseline_results if not r["is_correct"]]
        diag_res = self.nebius_client.diagnose_and_propose_patches(failing_ledgers)
        if isinstance(diag_res, tuple) and len(diag_res) == 2:
            candidates, proposal_metadata = diag_res
            self.pipeline_state["proposal_metadata"] = proposal_metadata
        else:
            candidates = diag_res

        # Step 4: Verification sweep (Candidates x Train Scenarios)
        candidate_metrics: Dict[str, Dict[str, Any]] = {}
        for cand in candidates:
            cand_results = [EpisodeRunner.run_episode(sc, patch=cand) for sc in train_scenarios]
            for cr in cand_results:
                episode_durations.append(cr["elapsed_seconds"])
            candidate_metrics[cand.patch_id] = BenchmarkAggregator.aggregate(cand_results, cand.patch_id)

        # Step 5: Selection rule with 4 guards
        winner, race_summary = self.verifier.select_winner(base_metrics, candidate_metrics, candidates)
        self.pipeline_state["race_summary"] = race_summary
        self.pipeline_state["winning_patch"] = winner.model_dump() if winner else None

        # Step 6: Held-out validation
        heldout_base_res = [EpisodeRunner.run_episode(sc) for sc in held_out_scenarios]
        for hbr in heldout_base_res:
            episode_durations.append(hbr["elapsed_seconds"])
        heldout_base_metrics = BenchmarkAggregator.aggregate(heldout_base_res, "baseline_heldout")
        
        heldout_patch_res = []
        if winner:
            heldout_patch_res = [EpisodeRunner.run_episode(sc, patch=winner) for sc in held_out_scenarios]
            for hpr in heldout_patch_res:
                episode_durations.append(hpr["elapsed_seconds"])
            heldout_patch_metrics = BenchmarkAggregator.aggregate(heldout_patch_res, winner.patch_id)
        else:
            heldout_patch_metrics = {
                "status": "no_candidate_passed_guards",
                "note": "No candidate patch passed verifier guards. Patched held-out evaluation was not run."
            }

        self.pipeline_state["heldout_metrics_before"] = heldout_base_metrics
        self.pipeline_state["heldout_metrics_after"] = heldout_patch_metrics

        # Step 7: Scale and Cost metrics table (Section 9.1)
        # Measured timing over all actually executed episodes
        total_episodes_executed = len(episode_durations)
        actual_total_elapsed = max(0.0001, time.time() - t0_total)
        median_sec = sorted(episode_durations)[len(episode_durations) // 2] if episode_durations else 0.0

        is_cloud_exec = self.nebius_client.is_live and g0_res.get("connected", False)

        self.pipeline_state["scale_and_cost"] = {
            "mode": "nebius_cloud" if is_cloud_exec else "local_simulation",
            "scenarios_train": train_count,
            "scenarios_heldout": held_out_count,
            "candidates_verified": len(candidates),
            "verification_runs": len(candidates) * train_count,
            "total_episodes_evaluated": total_episodes_executed,
            "median_seconds_per_episode": round(median_sec, 6),
            "measured_wall_clock_sec": round(actual_total_elapsed, 4),
            "speedup_factor": None if not is_cloud_exec else round(sum(episode_durations) / actual_total_elapsed, 2),
            "note": "Measured in local deterministic simulation mode." if not is_cloud_exec else "Measured on remote Nebius endpoint."
        }

        # Step 8: Generate Runnable Pytest Regression Test
        test_code = self.generate_regression_test_file(winner, write=True)
        self.pipeline_state["generated_test_code"] = test_code
        self.pipeline_state["status"] = "completed"
        self.pipeline_state["progress"] = 100

        # Save artifacts
        output_path = self.data_dir / "latest_pipeline_run.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.pipeline_state, f, indent=2, default=str)

        return self.pipeline_state

    def generate_regression_test_file(self, winner: Optional[PatchSpec], write: bool = False) -> str:
        """
        Renders the regression test for the actual winning PatchSpec.
        Pure by default (returns source, writes nothing). The pipeline passes write=True.
        With no winner, a notice is rendered (and written) instead; no patch is fabricated.
        """
        test_code = render_regression_test(winner)
        if write:
            self.tests_dir.mkdir(parents=True, exist_ok=True)
            with open(self.tests_dir / GENERATED_TEST_FILENAME, "w", encoding="utf-8", newline="\n") as f:
                f.write(test_code)
        return test_code


def render_regression_test(winner: Optional[PatchSpec]) -> str:
    """Deterministically renders regression-test source from the winning PatchSpec (all fields)."""
    if not winner:
        return '''"""
SilentFail Verification Notice: No candidate patch passed all verifier guards.
No regression test emitted because no safe patch was selected.
"""
'''
    patch_literal = pprint.pformat(winner.model_dump(), sort_dicts=False)
    return TEST_TEMPLATE.replace("__PATCH_LITERAL__", patch_literal)


TEST_TEMPLATE = '''"""
Auto-generated SilentFail Regression Test. Do not edit by hand.
Regenerate with the pipeline (see README); it is rendered from the winning PatchSpec in
data/latest_pipeline_run.json, and tests/test_pipeline_artifact_sync.py checks they match.
Guarantees prevention of duplicate refunds after timeouts
while protecting legitimate repeated partial refunds from over-suppression.
"""

from src.silentfail.models import Scenario, ExpectedOutcome, FaultConfig, PatchSpec
from src.silentfail.runner import EpisodeRunner

WINNING_PATCH = PatchSpec(**__PATCH_LITERAL__)


def test_regression_timeout_duplicate_failure_cured():
    """Hardened patch scenario: winning patch correctly resolves timeout-after-success."""
    scenario = Scenario(
        scenario_id="sc_regression_timeout",
        split="held_out",
        category="timeout_after_success",
        initial_state={"orders": [{"id": "8812", "status": "delivered", "total": 85.0, "refunded_amount": 0.0}]},
        customer_message="Please refund $42.50 for order 8812, package arrived broken.",
        faults=[FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success")],
        expected_outcome=ExpectedOutcome(refund_count=1, refund_total=42.5, email_count=1)
    )

    # 1. Baseline run: operation_id prevents silent duplicate; exactly one refund is recorded.
    base_res = EpisodeRunner.run_episode(scenario, patch=None)
    assert base_res["record"].final_state["refund_count"] == 1

    # 2. Hardened agent must prevent duplicate and succeed cleanly
    patched_res = EpisodeRunner.run_episode(scenario, patch=WINNING_PATCH)
    assert patched_res["detection"].duplicate_action is False
    assert patched_res["record"].final_state["refund_count"] == 1
    assert patched_res["is_correct"] is True


def test_legitimate_repeat_is_not_oversuppressed():
    """Legitimate partial refunds scenario: verifies patch does not over-suppress valid repeats."""
    scenario = Scenario(
        scenario_id="sc_legit_repeat_guard",
        split="held_out",
        category="legitimate_repeat",
        initial_state={"orders": [{"id": "9933", "status": "delivered", "total": 90.0, "refunded_amount": 0.0}]},
        customer_message="Please refund items 1 and 2 for order 9933 as two valid partial refunds.",
        faults=[],
        expected_outcome=ExpectedOutcome(refund_count=2, refund_total=90.0, email_count=1)
    )

    patched_res = EpisodeRunner.run_episode(scenario, patch=WINNING_PATCH)
    assert patched_res["record"].final_state["refund_count"] == 2
    assert patched_res["detection"].over_suppression is False
    assert patched_res["is_correct"] is True
'''
