import pytest

from src.silentfail.contract_validation import (
    ContractValidationError,
    validate_paired_results,
    validate_split_reconciliation,
    wilson_interval,
)
from src.silentfail.models import LedgerRecord, Scenario


def _pair_scenario(scenario_id, split):
    return Scenario(
        scenario_id=scenario_id,
        split=split,
        category="control",
        initial_state={
            "orders": [{
                "id": "1",
                "status": "delivered",
                "total": 1.0,
                "refunded_amount": 0.0,
                "items": [],
            }],
        },
        customer_message="Refund order 1.",
        faults=[],
        expected_outcome={
            "refund_count": 1,
            "refund_total": 1.0,
            "email_count": 0,
            "ticket_count": 0,
            "allow_refusal": False,
        },
    )


def _pair_result(scenario_id, correct, valid=True, uncertain=False):
    return {
        "record": LedgerRecord(run_id="r", scenario_id=scenario_id),
        "is_correct": correct,
        "contract_observation": {"valid": valid, "claim_uncertain": uncertain},
    }


@pytest.mark.parametrize(
    ("successes", "denominator", "estimate"),
    [(0, 0, None), (0, 4, 0.0), (4, 4, 1.0), (2, 4, 0.5)],
)
def test_wilson_intervals_cover_zero_all_success_all_failure_and_mixed(
    successes, denominator, estimate
):
    interval = wilson_interval(successes, denominator)
    assert interval["numerator"] == successes
    assert interval["denominator"] == denominator
    assert interval["point_estimate"] == estimate
    if denominator:
        assert 0.0 <= interval["lower_95"] <= interval["upper_95"] <= 1.0
    else:
        assert interval["lower_95"] is None
        assert interval["upper_95"] is None


def test_wilson_interval_matches_95_percent_reference_case():
    interval = wilson_interval(5, 10)
    assert interval["point_estimate"] == 0.5
    assert interval["lower_95"] == pytest.approx(0.236593090512564, rel=1e-12)
    assert interval["upper_95"] == pytest.approx(0.763406909487436, rel=1e-12)


@pytest.mark.parametrize(
    "successes, denominator",
    [(-1, 2), (3, 2), (1, -1), (True, 1), (1, True)],
)
def test_wilson_intervals_reject_invalid_counts(successes, denominator):
    with pytest.raises(ContractValidationError):
        wilson_interval(successes, denominator)
    with pytest.raises(ContractValidationError):
        wilson_interval(1, 2, confidence=0.90)


def test_split_count_reconciliation_is_explicit_and_deterministic():
    overall = {"counts": {"total": 5, "correct": 3, "invalid_episode_count": 1}}
    splits = {
        "train": {"counts": {"total": 3, "correct": 2, "invalid_episode_count": 1}},
        "held_out": {"counts": {"total": 2, "correct": 1, "invalid_episode_count": 0}},
    }
    first = validate_split_reconciliation(overall, splits)
    second = validate_split_reconciliation(overall, splits)
    assert first == second
    assert first["valid"] is True


def test_split_count_reconciliation_rejects_mismatch():
    with pytest.raises(ContractValidationError, match="do not reconcile"):
        validate_split_reconciliation(
            {"counts": {"total": 5, "correct": 4}},
            {
                "train": {"counts": {"total": 3, "correct": 2}},
                "held_out": {"counts": {"total": 2, "correct": 1}},
            },
        )


def test_paired_transitions_report_small_sample_mcnemar_and_denominators():
    scenarios = {
        "train": _pair_scenario("train", "train"),
        "held": _pair_scenario("held", "held_out"),
    }
    report = validate_paired_results(
        {
            "baseline": [
                _pair_result("train", False),
                _pair_result("held", True),
            ],
            "hardened": [
                _pair_result("train", True),
                _pair_result("held", False),
            ],
        },
        scenarios,
    )
    transitions = report["transitions"]
    assert transitions["failed_to_corrected"] == 1
    assert transitions["correct_to_regressed"] == 1
    assert transitions["transition_denominator"] == 2
    assert transitions["total_paired_scenarios"] == 2
    assert transitions["correction_rate_among_baseline_failures"] == 1.0
    assert transitions["regression_rate_among_baseline_correct"] == 1.0
    assert transitions["paired_correctness_difference"] == 0.0
    assert transitions["mcnemar_exact_p_value"] == 1.0


def test_exact_mcnemar_reference_case_three_to_zero():
    scenarios = {
        f"s{index}": _pair_scenario(f"s{index}", "train")
        for index in range(3)
    }
    report = validate_paired_results(
        {
            "baseline": [_pair_result(sid, False) for sid in scenarios],
            "hardened": [_pair_result(sid, True) for sid in scenarios],
        },
        scenarios,
    )
    assert report["transitions"]["failed_to_corrected"] == 3
    assert report["transitions"]["correct_to_regressed"] == 0
    assert report["transitions"]["mcnemar_exact_p_value"] == pytest.approx(0.25)


def test_invalid_and_inconclusive_pairs_are_visible_and_excluded_from_transitions():
    scenarios = {
        "train": _pair_scenario("train", "train"),
        "held": _pair_scenario("held", "held_out"),
    }
    report = validate_paired_results(
        {
            "baseline": [
                _pair_result("train", False, valid=False),
                _pair_result("held", False, uncertain=True),
            ],
            "hardened": [
                _pair_result("train", True),
                _pair_result("held", True),
            ],
        },
        scenarios,
    )
    transitions = report["transitions"]
    assert transitions["invalid_pairs"] == 1
    assert transitions["inconclusive_pairs"] == 1
    assert transitions["transition_denominator"] == 0
    assert transitions["failed_to_corrected"] == 0
    assert transitions["paired_correctness_difference"] is None


def test_paired_results_reject_missing_observations_and_invalid_splits():
    scenarios = {"train": _pair_scenario("train", "train")}
    malformed = _pair_result("train", True)
    del malformed["contract_observation"]["claim_uncertain"]
    with pytest.raises(ContractValidationError, match="claim_uncertain"):
        validate_paired_results(
            {"baseline": [malformed], "hardened": [_pair_result("train", True)]},
            scenarios,
        )
    bad_scenario = Scenario.model_construct(
        **{**_pair_scenario("bad", "train").model_dump(), "split": "other"}
    )
    with pytest.raises(ContractValidationError, match="unsupported scenario split"):
        validate_paired_results(
            {"baseline": [_pair_result("bad", True)], "hardened": [_pair_result("bad", True)]},
            {"bad": bad_scenario},
        )
