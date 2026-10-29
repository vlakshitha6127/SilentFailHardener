"""
Auto-generated SilentFail Regression Test. Do not edit by hand.
Regenerate with the pipeline (see README); it is rendered from the winning PatchSpec in
data/latest_pipeline_run.json, and tests/test_pipeline_artifact_sync.py checks they match.
Guarantees prevention of duplicate refunds after timeouts
while protecting legitimate repeated partial refunds from over-suppression.
"""

from src.silentfail.models import Scenario, ExpectedOutcome, FaultConfig, PatchSpec
from src.silentfail.runner import EpisodeRunner

WINNING_PATCH = PatchSpec(**{'patch_id': 'patch_retry_policy',
 'name': 'State-Checked Retry Policy Wrapper',
 'type': 'wrapper.retry_policy',
 'description': 'On timeout/empty-200, looks up a refund recorded under the '
                "operation's stable idempotency key before reporting the "
                'outcome; retries reuse the same key.',
 'is_control': False,
 'config': {}})


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
