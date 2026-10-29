import pytest

from src.silentfail.agent import NaiveRefundAgent
from src.silentfail.contract_validation import (
    CONTRACT_SCHEMA_VERSION,
    validate_contract_manifest,
    validate_episode_contract,
)
from src.silentfail.models import Scenario
from src.silentfail.offline_harness import build_manifest
from src.silentfail.runner import EpisodeRunner
from src.silentfail.scenarios import (
    PROPOSED_SCENARIO_CLASSES,
    build_refund_success_email_empty_200_not_applied_scenario,
    build_proposed_scenario,
    proposed_cohort_contracts,
    proposed_scenario_contract,
    replacement_email_contract,
)
from src.silentfail.structural_validation import (
    StructuralValidationError,
    scenario_structural_signature,
    validate_structural_leakage,
)


def _scenario(scenario_id, split="train", **updates):
    value = {
        "scenario_id": scenario_id,
        "split": split,
        "seed": 17,
        "category": "no_fault_control",
        "initial_state": {
            "orders": [{
                "id": "1234",
                "status": "delivered",
                "total": 50.0,
                "refunded_amount": 0.0,
                "items": [{"item_id": "item_1", "price": 50.0}],
            }],
            "refunds": [],
            "emails": [],
            "tickets": [],
        },
        "customer_message": "Please refund $50 for order 1234.",
        "faults": [],
        "expected_outcome": {
            "refund_count": 1,
            "refund_total": 50.0,
            "email_count": 1,
            "ticket_count": 0,
            "allow_refusal": False,
        },
    }
    value.update(updates)
    return Scenario.model_validate(value)


def test_canonically_equivalent_scenarios_have_identical_signatures():
    first = _scenario("sc_train_0001")
    second = _scenario(
        "sc_held_out_9999",
        split="held_out",
        seed=9999,
        initial_state={
            **first.initial_state,
            "orders": [{**first.initial_state["orders"][0], "id": "9876"}],
        },
    )
    assert scenario_structural_signature(first)["sha256"] == scenario_structural_signature(second)["sha256"]


def test_cosmetic_changes_do_not_bypass_duplicate_detection():
    report = validate_structural_leakage([
        _scenario("train-1"),
        _scenario(
            "held-1",
            split="held_out",
            seed=999,
            initial_state={
                **_scenario("temp").initial_state,
                "orders": [{**_scenario("temp").initial_state["orders"][0], "id": "9999"}],
            },
        ),
    ])
    assert report["valid"] is False
    assert report["exact_structural_duplicates"] == [("held-1", "train-1")]


def test_fault_order_and_initial_state_changes_are_independent_dimensions():
    base = _scenario("train-1", faults=[
        {"tool": "issue_refund", "call_index": 0, "type": "timeout_after_success"},
        {"tool": "issue_refund", "call_index": 1, "type": "duplicate_webhook"},
    ])
    changed = _scenario(
        "held-1",
        split="held_out",
        faults=[
            {"tool": "issue_refund", "call_index": 0, "type": "duplicate_webhook"},
            {"tool": "issue_refund", "call_index": 1, "type": "timeout_after_success"},
        ],
        initial_state={
            **base.initial_state,
            "orders": [{**base.initial_state["orders"][0], "status": "pending"}],
        },
    )
    signature = scenario_structural_signature(changed)["components"]
    assert signature["fault_sequence"][0]["type"] == "duplicate_webhook"
    report = validate_structural_leakage(
        [base, changed],
        overlap_exemptions={"no_fault_control": {"message_template"}},
    )
    assert report["valid"] is True
    nearest = report["nearest_training"][0]
    assert "fault_sequence" in nearest["differing_dimensions"]
    assert "initial_state_shape" in nearest["differing_dimensions"]


def test_held_out_requires_two_independent_differences():
    train = _scenario("train-1")
    held = _scenario("held-1", split="held_out", category="timeout_after_success")
    report = validate_structural_leakage([train, held])
    assert report["valid"] is False
    assert any("fewer than two" in error for error in report["errors"])


def test_overlap_exemption_is_explicit_but_never_hides_exact_duplicates():
    train = _scenario("train-1")
    held = _scenario("held-1", split="held_out", category="timeout_after_success")
    report = validate_structural_leakage(
        [train, held],
        overlap_exemptions={"timeout_after_success": {
            "message_template",
            "fault_sequence",
        }},
    )
    assert report["valid"] is False
    assert any("fewer than two" in error for error in report["errors"])


def test_nearest_neighbor_report_is_deterministic():
    scenarios = [
        _scenario("train-b", category="timeout_after_success"),
        _scenario("train-a", category="empty_200_applied"),
        _scenario("held-z", split="held_out", category="compound_faults"),
    ]
    first = validate_structural_leakage(scenarios)
    second = validate_structural_leakage(list(reversed(scenarios)))
    assert first["nearest_training"] == second["nearest_training"]


def test_malformed_structural_input_fails_closed():
    malformed = _scenario("bad").model_dump()
    malformed["customer_message"] = None
    with pytest.raises(StructuralValidationError):
        scenario_structural_signature(malformed)
    with pytest.raises(StructuralValidationError):
        validate_structural_leakage([_scenario("a"), {"scenario_id": "bad"}])


def test_replacement_email_fixture_is_supported_and_distinguishes_refund_from_email():
    scenario = build_refund_success_email_empty_200_not_applied_scenario()
    contract = replacement_email_contract(scenario.scenario_id)
    contracts = validate_contract_manifest(
        build_manifest([scenario]),
        {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [contract]},
        expected_train_count=1,
        expected_held_out_count=0,
    )
    result = EpisodeRunner.run_episode(
        scenario,
        agent=NaiveRefundAgent(model_client=None),
    )
    observation = validate_episode_contract(
        contracts[scenario.scenario_id],
        scenario,
        result,
        configuration="baseline",
    )
    assert observation["valid"] is True
    assert observation["refund_count"] == 1
    assert observation["refund_total"] == 50.0
    assert observation["email_successful_mutation_count"] == 0
    assert observation["email_tool_failure"] is True
    assert observation["fault_observation"]["observed"] == [
        ["send_email", "empty_200", 1]
    ]


def test_all_proposed_classes_have_explicit_fixture_and_contract():
    for index, scenario_class in enumerate(PROPOSED_SCENARIO_CLASSES):
        scenario = build_proposed_scenario(
            scenario_class,
            scenario_id=f"fixture-{index}",
            split="train",
            seed=100 + index,
        )
        valid, message = __import__(
            "src.silentfail.scenarios", fromlist=["ScenarioValidator"]
        ).ScenarioValidator.validate_and_compute_outcome(scenario)
        assert valid, message
        contract = proposed_scenario_contract(scenario)
        assert contract["scenario_id"] == scenario.scenario_id
        assert contract["requested_operation"]["operation_count"] >= 1
        assert "detectors" in contract


def test_proposed_fixture_contracts_pass_schema_validation_without_execution():
    for index, scenario_class in enumerate(PROPOSED_SCENARIO_CLASSES):
        scenario = build_proposed_scenario(
            scenario_class,
            scenario_id=f"schema-{index}",
            split="train",
            seed=300 + index,
        )
        contract = proposed_scenario_contract(scenario)
        validated = validate_contract_manifest(
            build_manifest([scenario]),
            {
                "contract_schema_version": CONTRACT_SCHEMA_VERSION,
                "contracts": [contract],
            },
            expected_train_count=1,
            expected_held_out_count=0,
        )
        assert validated[scenario.scenario_id] == contract


def test_proposed_contract_factory_rejects_incomplete_cohort():
    scenario = build_proposed_scenario(
        "clean_eligible_full_refund",
        scenario_id="only-one",
        split="train",
        seed=1,
    )
    with pytest.raises(ValueError, match="exactly 65"):
        proposed_cohort_contracts([scenario])


def test_proposed_fixture_contracts_capture_retry_and_excess_semantics():
    retry = build_proposed_scenario(
        "same_operation_retry", scenario_id="retry", split="train", seed=11
    )
    retry_contract = proposed_scenario_contract(retry)
    assert retry_contract["requested_operation"]["retry_groups"][0][
        "require_same_operation_id"
    ] is True
    assert retry_contract["requested_operation"]["operations"][0][
        "expected_mutation_count"
    ] == 1
    assert retry_contract["requested_operation"]["retry_groups"][0]["min_attempts"] == 1

    excess = build_proposed_scenario(
        "excess_mutation", scenario_id="excess", split="train", seed=12
    )
    assert [fault.type for fault in excess.faults] == [
        "timeout_after_success",
        "duplicate_webhook",
    ]
    excess_contract = proposed_scenario_contract(excess)
    assert excess_contract["detectors_by_configuration"]["baseline"]["excess_mutation"] is True
    assert excess_contract["detectors_by_configuration"]["hardened"]["excess_mutation"] is False
    assert excess_contract["fault_observations"]["hardened"]["allowed_unobserved"] == [{
        "tool": "issue_refund",
        "type": "duplicate_webhook",
        "occurrence": 1,
    }]


def test_excessive_request_contract_matches_full_balance_fallback():
    scenario = build_proposed_scenario(
        "incorrect_refund_amount", scenario_id="amount-runtime", split="train", seed=13
    )
    contract = proposed_scenario_contract(scenario)
    result = EpisodeRunner.run_episode(
        scenario,
        agent=NaiveRefundAgent(model_client=None),
    )
    observation = validate_episode_contract(
        contract,
        scenario,
        result,
        configuration="baseline",
    )
    assert result["record"].final_state["refund_total"] == 100.0
    assert result["detection"].incorrect_refund_amount is True
    assert observation["valid"] is True
    assert observation["refund_total"] == 100.0


def test_same_operation_readback_success_requires_no_duplicate_retry():
    scenario = build_proposed_scenario(
        "same_operation_retry", scenario_id="retry-runtime", split="train", seed=14
    )
    contract = proposed_scenario_contract(scenario)
    result = EpisodeRunner.run_episode(
        scenario,
        patch=__import__("src.silentfail.models", fromlist=["PatchSpec"]).PatchSpec(
            patch_id="retry",
            name="retry",
            type="wrapper.retry_policy",
            description="test",
        ),
        agent=NaiveRefundAgent(model_client=None),
    )
    observation = validate_episode_contract(
        contract,
        scenario,
        result,
        configuration="hardened",
    )
    operation_id = f"op_{scenario.initial_state['orders'][0]['id']}_req0"
    assert observation["valid"] is True
    assert observation["operation_observation"]["per_operation"][
        operation_id
    ]["request_count"] == 1
    assert observation["operation_observation"]["per_operation"][
        operation_id
    ]["mutation_count"] == 1


def test_excess_mutation_is_allowed_to_be_short_circuited_by_hardening():
    scenario = build_proposed_scenario(
        "excess_mutation", scenario_id="excess-runtime", split="train", seed=15
    )
    contract = proposed_scenario_contract(scenario)
    result = EpisodeRunner.run_episode(
        scenario,
        patch=__import__("src.silentfail.models", fromlist=["PatchSpec"]).PatchSpec(
            patch_id="retry",
            name="retry",
            type="wrapper.retry_policy",
            description="test",
        ),
        agent=NaiveRefundAgent(model_client=None),
    )
    observation = validate_episode_contract(
        contract,
        scenario,
        result,
        configuration="hardened",
    )
    assert observation["valid"] is True
    assert result["detection"].excess_mutation is False
    assert observation["fault_observation"]["observed"] == [
        ["issue_refund", "timeout_after_success", 1]
    ]


def test_incorrect_amount_fixture_records_supported_full_balance_fallback():
    scenario = build_proposed_scenario(
        "incorrect_refund_amount", scenario_id="amount", split="train", seed=13
    )
    contract = proposed_scenario_contract(scenario)
    assert scenario.expected_outcome.refund_count == 1
    assert scenario.expected_outcome.refund_total == 100.0
    assert scenario.expected_outcome.allow_refusal is False
    assert contract["requested_operation"]["amount"] == 150.0
    assert contract["refusal"]["mode"] == "forbidden"
    assert contract["completion_claim"]["kind"] == "success"
    assert contract["detectors"]["incorrect_refund_amount"] is True


def test_contract_assertion_signature_covers_refund_and_fault_observations():
    scenario = build_proposed_scenario(
        "timeout_after_success", scenario_id="shape", split="train", seed=14
    )
    contract = proposed_scenario_contract(scenario)
    changed = {
        **contract,
        "refund": {**contract["refund"], "expected_total": 49.0},
    }
    first = scenario_structural_signature(scenario, contract)
    second = scenario_structural_signature(scenario, changed)
    assert first["sha256"] != second["sha256"]
