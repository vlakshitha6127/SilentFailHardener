import json
import hashlib
from pathlib import Path

import pytest

from src.silentfail.models import PatchSpec, Scenario
from src.silentfail.agent import NaiveRefundAgent
from src.silentfail.ledger import CausalLedger, WorldState
from src.silentfail.tools import ToolRegistry
from src.silentfail.contract_validation import (
    CONTRACT_SCHEMA_VERSION,
    ContractValidationError,
    classify_email,
    episode_accounting,
    validate_paired_results,
    validate_contract_manifest,
    validate_episode_contract,
    validate_aggregate_reconciliation,
)
from src.silentfail.offline_harness import (
    MANIFEST_SCHEMA_VERSION,
    OfflineAgentProvider,
    build_manifest,
    generate_manifest,
    load_manifest,
    run_offline_evaluation,
    patch_provenance,
    verify_expected_patch_fingerprint,
    save_manifest,
)


PATCH = PatchSpec(
    patch_id="patch_idempotency_key",
    name="Idempotency",
    type="wrapper.idempotency_key",
    description="test patch",
)


def _scenario(message: str = "Please refund $50 for order ORD-1234; note \"box\\\\label\".") -> Scenario:
    return Scenario(
        scenario_id="sc_literal",
        split="train",
        seed=17,
        category="no_fault_control",
        initial_state={
            "orders": [{
                "id": "ORD-1234",
                "status": "delivered",
                "total": 50.0,
                "refunded_amount": 0.0,
                "items": [{"item_id": "item_1", "price": 50.0}],
            }],
            "refunds": [],
            "emails": [],
            "tickets": [],
        },
        customer_message=message,
        faults=[],
        expected_outcome={
            "refund_count": 1,
            "refund_total": 50.0,
            "email_count": 1,
            "ticket_count": 0,
            "allow_refusal": False,
        },
    )


def test_manifest_generation_is_deterministic():
    assert generate_manifest(4, base_seed=42) == generate_manifest(4, base_seed=42)


@pytest.mark.parametrize("message", [
    "Please refund $50 for order 1234.",
    "Please refund $42.50 for order 1234.",
    'Please refund order 1234; customer said "box\\\\label".',
    'Please refund $42.50 for order ORD-1234; note "box\\\\label".',
])
def test_manifest_round_trip_preserves_literal_message(tmp_path, message):
    manifest = build_manifest([_scenario(message)])
    path = tmp_path / "manifest.json"
    save_manifest(path, manifest)
    loaded = load_manifest(path)
    assert loaded["scenarios"][0]["customer_message"] == message
    assert loaded["schema_version"] == MANIFEST_SCHEMA_VERSION


def test_manifest_rejects_duplicates_and_unknown_fields(tmp_path):
    manifest = build_manifest([_scenario()])
    manifest["scenarios"].append(dict(manifest["scenarios"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest_from_object(manifest, tmp_path)

    malformed = build_manifest([_scenario()])
    malformed["scenarios"][0]["unsupported"] = True
    with pytest.raises(ValueError, match="unsupported"):
        load_manifest_from_object(malformed, tmp_path)


def load_manifest_from_object(manifest, tmp_path):
    path = tmp_path / "invalid_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return load_manifest(path)


def test_offline_evaluation_isolated_and_aggregates_supported_metrics(tmp_path):
    manifest = build_manifest([_scenario()])
    output = tmp_path / "run-1"
    result = run_offline_evaluation(manifest, output, PATCH, offline_only=True)

    assert result["summary"]["mode"] == "local_simulation"
    assert set(result["episodes"]) == {"baseline", "hardened"}
    assert result["aggregate_metrics"]["baseline"]["total_runs"] == 1
    assert result["aggregate_metrics"]["hardened"]["total_runs"] == 1
    assert result["hashes"]["scenario_manifest_sha256"]
    assert (output / "episodes" / "baseline__sc_literal.json").exists()
    assert (output / "hashes.json").exists()
    assert result["hashes"]["scenario_manifest_sha256"] == hashlib.sha256(
        (output / "scenario_manifest.json").read_bytes()
    ).hexdigest()
    assert result["hashes"]["output_sha256"]["summary.json"] == hashlib.sha256(
        (output / "summary.json").read_bytes()
    ).hexdigest()
    raw = json.loads((output / "episodes" / "hardened__sc_literal.json").read_text(encoding="utf-8"))
    assert raw["customer_message"] == manifest["scenarios"][0]["customer_message"]
    assert "duplicate_action" in raw["detection"]
    assert "excess_mutation" in raw["detection"]
    assert result["aggregate_metrics"]["baseline"]["patch_provenance"]["sha256"]
    assert result["aggregate_metrics"]["hardened"]["patch_provenance"]["sha256"]
    configuration = json.loads((output / "run_configuration.json").read_text(encoding="utf-8"))
    assert configuration["patch_provenance"]["baseline"]["serialization"]["patch"] is None
    assert configuration["patch_provenance"]["hardened"]["serialization"]["patch"]["patch_id"] == PATCH.patch_id


def test_offline_mode_and_output_isolation_fail_closed(tmp_path):
    manifest = build_manifest([_scenario()])
    with pytest.raises(RuntimeError, match="offline_only"):
        run_offline_evaluation(manifest, tmp_path / "run-1", PATCH)

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        run_offline_evaluation(manifest, existing, PATCH, offline_only=True)


def test_protected_path_and_live_client_are_rejected(tmp_path):
    manifest = build_manifest([_scenario()])
    with pytest.raises(ValueError, match="protected"):
        run_offline_evaluation(manifest, Path(__file__).resolve().parents[1] / "README.md", PATCH, offline_only=True)

    class LiveAgent:
        model_client = object()

        def run(self, user_message, tools, config=None):
            raise AssertionError("must not run")

    with pytest.raises(RuntimeError, match="live model client"):
        run_offline_evaluation(
            manifest,
            tmp_path / "live",
            PATCH,
            offline_only=True,
            agent_providers={
                "baseline": OfflineAgentProvider(LiveAgent()),
                "hardened": OfflineAgentProvider(LiveAgent()),
            },
        )
    assert not (tmp_path / "live").exists()


def test_unsupported_provider_is_rejected_without_invoking_it(tmp_path):
    called = False

    class UnsupportedProvider:
        def get_agent(self):
            nonlocal called
            called = True
            raise AssertionError("unsupported provider must not be invoked")

    with pytest.raises(TypeError, match="OfflineAgentProvider"):
        run_offline_evaluation(
            build_manifest([_scenario()]),
            tmp_path / "unsupported",
            PATCH,
            offline_only=True,
            agent_providers={"baseline": UnsupportedProvider(), "hardened": UnsupportedProvider()},
        )
    assert called is False


def test_episode_failure_does_not_publish_partial_run(tmp_path, monkeypatch):
    from src.silentfail import offline_harness

    def fail(*args, **kwargs):
        raise RuntimeError("episode failed")

    monkeypatch.setattr(offline_harness.EpisodeRunner, "run_episode", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="episode failed"):
        run_offline_evaluation(build_manifest([_scenario()]), output, PATCH, offline_only=True)
    assert not output.exists()
    assert not list(tmp_path.glob(".failed.tmp-*"))


def test_serialization_failure_does_not_publish_or_delete_existing_sibling(tmp_path, monkeypatch):
    from src.silentfail import offline_harness

    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    original_write = offline_harness._write_json
    calls = {"count": 0}

    def fail_after_first_write(path, value):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("serialization failed")
        original_write(path, value)

    monkeypatch.setattr(offline_harness, "_write_json", fail_after_first_write)
    output = tmp_path / "failed-serialization"
    with pytest.raises(OSError, match="serialization failed"):
        run_offline_evaluation(build_manifest([_scenario()]), output, PATCH, offline_only=True)
    assert not output.exists()
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not list(tmp_path.glob(".failed-serialization.tmp-*"))


def test_hash_validation_failure_does_not_publish(tmp_path, monkeypatch):
    from src.silentfail import offline_harness

    monkeypatch.setattr(
        offline_harness,
        "_validate_output_hashes",
        lambda destination, hashes: (_ for _ in ()).throw(ValueError("hash validation failed")),
    )
    output = tmp_path / "failed-hash"
    with pytest.raises(ValueError, match="hash validation failed"):
        run_offline_evaluation(build_manifest([_scenario()]), output, PATCH, offline_only=True)
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-hash.tmp-*"))


def test_same_manifest_inputs_and_deterministic_agent_behavior(tmp_path):
    manifest = generate_manifest(3, base_seed=77)
    result_one = run_offline_evaluation(manifest, tmp_path / "one", PATCH, offline_only=True)
    result_two = run_offline_evaluation(manifest, tmp_path / "two", PATCH, offline_only=True)

    for configuration in ("baseline", "hardened"):
        first = result_one["episodes"][configuration]
        second = result_two["episodes"][configuration]
        assert [item["scenario_id"] for item in first] == [item["scenario_id"] for item in second]
        assert [item["customer_message"] for item in first] == [item["customer_message"] for item in second]
        assert result_one["aggregate_metrics"][configuration] == result_two["aggregate_metrics"][configuration]


def _contract(scenario, *, refusal="forbidden", claim_kind="success", claim_amount=50.0, required_email=1, allowed_email=1, detectors=None):
    return {
        "scenario_id": scenario.scenario_id,
        "requested_operation": {"kind": "partial_refund", "amount": 50.0, "operation_count": 1},
        "refund": {"expected_count": scenario.expected_outcome.refund_count, "expected_total": scenario.expected_outcome.refund_total, "amount_tolerance": 0.01},
        "refusal": {"mode": refusal},
        "completion_claim": {"kind": claim_kind, "amount": claim_amount, "amount_tolerance": 0.01},
        "email": {"required_count": required_email, "allowed_count": allowed_email},
        "detectors": detectors or {},
    }


def _numeric_order_scenario(scenario_id, faults, category="timeout_after_success"):
    base = _scenario().model_dump()
    return Scenario.model_validate({
        **base,
        "scenario_id": scenario_id,
        "category": category,
        "customer_message": "Please refund $50 for order 1234.",
        "initial_state": {
            **base["initial_state"],
            "orders": [{
                **base["initial_state"]["orders"][0],
                "id": "1234",
            }],
        },
        "faults": faults,
    })


def test_contract_manifest_validates_and_rejects_contradictions():
    scenario = _scenario()
    manifest = build_manifest([scenario])
    contracts = {
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "contracts": [_contract(scenario)],
    }
    validated = validate_contract_manifest(manifest, contracts, expected_train_count=1, expected_held_out_count=0)
    assert validated[scenario.scenario_id]["scenario_id"] == scenario.scenario_id

    bad = dict(contracts)
    bad["contracts"] = [dict(contracts["contracts"][0], unsupported=True)]
    with pytest.raises(ContractValidationError, match="unsupported"):
        validate_contract_manifest(manifest, bad, expected_train_count=1, expected_held_out_count=0)

    contradictory = dict(contracts)
    contradictory["contracts"] = [dict(contracts["contracts"][0], refund={
        "expected_count": 1, "expected_total": 49.0, "amount_tolerance": 0.01
    })]
    with pytest.raises(ContractValidationError, match="conflicts"):
        validate_contract_manifest(manifest, contradictory, expected_train_count=1, expected_held_out_count=0)


def test_contract_refusal_modes_and_claim_amounts():
    scenario = Scenario.model_validate({
        **_scenario().model_dump(),
        "scenario_id": "sc_contract",
        "initial_state": {
            **_scenario().initial_state,
            "orders": [{**_scenario().initial_state["orders"][0], "id": "1234"}],
        },
        "customer_message": 'Please refund $50 for order 1234; note "box\\\\label".',
    })
    manifest = build_manifest([scenario])
    valid = {
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "contracts": [_contract(scenario)],
    }
    contract = validate_contract_manifest(manifest, valid, expected_train_count=1, expected_held_out_count=0)[scenario.scenario_id]
    from src.silentfail.runner import EpisodeRunner
    actual = EpisodeRunner.run_episode(scenario, patch=None, agent=NaiveRefundAgent(model_client=None))
    observation = validate_episode_contract(contract, scenario, actual)
    assert observation["email_successful_mutation_count"] == 1

    wrong_claim = dict(contract, completion_claim={"kind": "success", "amount": 49.0, "amount_tolerance": 0.01})
    wrong_claim_result = validate_episode_contract(wrong_claim, scenario, actual)
    assert wrong_claim_result["valid"] is False
    assert "claimed amount mismatch" in wrong_claim_result["errors"]

    refusal_scenario = Scenario.model_validate({
        **_scenario().model_dump(),
        "scenario_id": "sc_refusal",
        "category": "ineligible_already_refunded",
        "initial_state": {
            "orders": [{"id": "ORD-1234", "status": "refunded", "total": 50.0, "refunded_amount": 50.0}],
            "refunds": [], "emails": [], "tickets": [],
        },
        "customer_message": "Please refund order ORD-1234.",
        "expected_outcome": {"refund_count": 0, "refund_total": 0.0, "email_count": 0, "ticket_count": 0, "allow_refusal": True},
    })
    refusal_manifest = build_manifest([refusal_scenario])
    refusal_contract = _contract(refusal_scenario, refusal="required", claim_kind="refusal", claim_amount=None, required_email=0, allowed_email=0)
    refusal_contracts = validate_contract_manifest(
        refusal_manifest,
        {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [refusal_contract]},
        expected_train_count=1,
        expected_held_out_count=0,
    )
    from src.silentfail.runner import EpisodeRunner
    refusal_result = EpisodeRunner.run_episode(refusal_scenario, agent=NaiveRefundAgent(model_client=None))
    assert validate_episode_contract(refusal_contracts["sc_refusal"], refusal_scenario, refusal_result)["refund_count"] == 0

    permitted = dict(refusal_contract, refusal={"mode": "permitted"})
    assert validate_contract_manifest(
        refusal_manifest,
        {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [permitted]},
        expected_train_count=1,
        expected_held_out_count=0,
    )
    forbidden = dict(refusal_contract, refusal={"mode": "forbidden"})
    with pytest.raises(ContractValidationError, match="forbidden refusal"):
        validate_contract_manifest(
            refusal_manifest,
            {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [forbidden]},
            expected_train_count=1,
            expected_held_out_count=0,
        )


def test_email_failures_are_distinguished_from_interleaved_other_tools():
    ledger = CausalLedger()
    ledger.record_tool_response({"error": "timeout"}, fault="timeout_after_success", tool="issue_refund")
    ledger.record_tool_response({"error": "timeout"}, fault="timeout_after_success", tool="send_email")
    ledger.record_tool_response({"status": "ok"}, tool="issue_refund")
    record = ledger.to_record("Unable to process refund.", WorldState().get_summary())
    observation = classify_email(record)
    assert observation["tool_failure"] is True
    assert observation["tool_failure_count"] == 1

    old_ledger = CausalLedger()
    old_ledger.record_tool_response({"error": "timeout"}, fault="timeout_after_success")
    old_record = old_ledger.to_record("", WorldState().get_summary())
    assert classify_email(old_record)["tool_failure"] is False


def test_contract_reports_missing_and_excess_email_separately():
    from src.silentfail.runner import EpisodeRunner

    scenario = Scenario.model_validate({
        **_scenario().model_dump(),
        "scenario_id": "sc_email_contract",
        "initial_state": {
            **_scenario().initial_state,
            "orders": [{**_scenario().initial_state["orders"][0], "id": "1234"}],
        },
        "customer_message": "Please refund $50 for order 1234.",
    })
    actual = EpisodeRunner.run_episode(scenario, patch=None, agent=NaiveRefundAgent(model_client=None))
    manifest = build_manifest([scenario])
    base = _contract(scenario)
    validated = validate_contract_manifest(
        manifest,
        {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [base]},
        expected_train_count=1,
        expected_held_out_count=0,
    )[scenario.scenario_id]
    observation = validate_episode_contract(validated, scenario, actual)
    assert observation["email_missing_required"] is False
    assert observation["email_excess"] is False

    result = validate_episode_contract(
        dict(validated, email={"required_count": 2, "allowed_count": 2}),
        scenario,
        actual,
    )
    assert result["valid"] is False
    assert result["email_missing_required"] is True
    assert "missing required email" in result["errors"]
    result = validate_episode_contract(
        dict(validated, email={"required_count": 0, "allowed_count": 0}),
        scenario,
        actual,
    )
    assert result["valid"] is False
    assert result["email_excess"] is True
    assert "excess email" in result["errors"]


def test_claim_validation_marks_missing_and_ambiguous_messages_inconclusive():
    from src.silentfail.runner import EpisodeRunner

    scenario = Scenario.model_validate({
        **_scenario().model_dump(),
        "scenario_id": "sc_claims",
        "initial_state": {
            **_scenario().initial_state,
            "orders": [{**_scenario().initial_state["orders"][0], "id": "1234"}],
        },
        "customer_message": "Please refund $50 for order 1234.",
    })
    actual = EpisodeRunner.run_episode(scenario, agent=NaiveRefundAgent(model_client=None))
    manifest = build_manifest([scenario])
    contract = validate_contract_manifest(
        manifest,
        {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [_contract(scenario)]},
        expected_train_count=1,
        expected_held_out_count=0,
    )[scenario.scenario_id]
    missing = dict(actual)
    missing["record"] = actual["record"].model_copy(update={"final_message": ""})
    missing_result = validate_episode_contract(contract, scenario, missing)
    assert missing_result["claim_uncertain"] is True
    assert missing_result["valid"] is False

    ambiguous = dict(actual)
    ambiguous["record"] = actual["record"].model_copy(update={"final_message": "Your request is being handled."})
    ambiguous_result = validate_episode_contract(contract, scenario, ambiguous)
    assert ambiguous_result["claim_uncertain"] is True
    assert ambiguous_result["valid"] is False


def test_permitted_contract_accepts_refusal_or_success():
    from src.silentfail.runner import EpisodeRunner

    scenario = Scenario.model_validate({
        **_scenario().model_dump(),
        "scenario_id": "sc_permitted",
        "initial_state": {
            **_scenario().initial_state,
            "orders": [{**_scenario().initial_state["orders"][0], "id": "1234"}],
        },
        "customer_message": "Please refund $50 for order 1234.",
        "expected_outcome": {
            "refund_count": 1, "refund_total": 50.0, "email_count": 1,
            "ticket_count": 0, "allow_refusal": True,
        },
    })
    contract_data = _contract(
        scenario,
        refusal="permitted",
        claim_kind="success_or_refusal",
        claim_amount=50.0,
    )
    contract = validate_contract_manifest(
        build_manifest([scenario]),
        {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [contract_data]},
        expected_train_count=1,
        expected_held_out_count=0,
    )[scenario.scenario_id]
    success = EpisodeRunner.run_episode(scenario, agent=NaiveRefundAgent(model_client=None))
    assert validate_episode_contract(contract, scenario, success)["valid"] is True
    refusal = dict(success)
    refusal["record"] = success["record"].model_copy(update={
        "final_message": "Unable to process refund for order 1234.",
        "final_state": {**success["record"].final_state, "refund_count": 0, "refund_total": 0.0, "email_count": 0},
        "events": [],
    })
    refusal_result = validate_episode_contract(contract, scenario, refusal)
    assert refusal_result["valid"] is True
    wrong_amount_contract = dict(contract, completion_claim={
        "kind": "success_or_refusal", "amount": 49.0, "amount_tolerance": 0.01,
    })
    wrong_amount_result = validate_episode_contract(wrong_amount_contract, scenario, success)
    assert wrong_amount_result["valid"] is False
    assert "claimed amount mismatch" in wrong_amount_result["errors"]


def test_contract_requires_exact_train_and_held_out_counts():
    scenario = _scenario()
    contract = {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [_contract(scenario)]}
    with pytest.raises(ContractValidationError, match="expected 39 train and 26 held_out"):
        validate_contract_manifest(build_manifest([scenario]), contract)


def test_pairing_rejects_duplicate_ids_and_reports_split_transitions():
    from src.silentfail.models import DetectionResult, LedgerRecord

    train = _scenario()
    held_out = Scenario.model_validate({**train.model_dump(), "scenario_id": "sc_held", "split": "held_out"})
    def raw(scenario_id, is_correct):
        return {
            "record": LedgerRecord(run_id="r", scenario_id=scenario_id),
            "is_correct": is_correct,
            "contract_observation": {"valid": is_correct, "claim_uncertain": False},
        }
    with pytest.raises(ContractValidationError, match="pairing mismatch"):
        validate_paired_results(
            {"baseline": [raw(train.scenario_id, True), raw(train.scenario_id, True)],
             "hardened": [raw(train.scenario_id, True), raw(held_out.scenario_id, True)]},
            {train.scenario_id: train, held_out.scenario_id: held_out},
        )
    report = validate_paired_results(
        {"baseline": [raw(train.scenario_id, False), raw(held_out.scenario_id, True)],
         "hardened": [raw(train.scenario_id, True), raw(held_out.scenario_id, True)]},
        {train.scenario_id: train, held_out.scenario_id: held_out},
    )
    assert report["splits"]["train"]["baseline_count"] == 1
    assert report["splits"]["held_out"]["hardened_count"] == 1
    assert report["transitions"]["failed_to_corrected"] == 0
    assert report["transitions"]["invalid_pairs"] == 1
    assert report["splits"]["train"]["transitions"]["invalid_pairs"] == 1
    assert report["splits"]["held_out"]["transitions"]["correct_to_unchanged"] == 1


def test_aggregate_reconciliation_rejects_missing_detector_fields():
    from src.silentfail.models import DetectionResult, LedgerRecord

    result = {
        "record": LedgerRecord(run_id="r", scenario_id="s"),
        "detection": DetectionResult(),
        "is_correct": True,
    }
    observation = {
        "valid": True,
        "claim_uncertain": False,
        "email_successful_mutation_count": 0,
        "email_tool_failure": False,
        "email_missing_required": False,
        "email_excess": False,
    }
    with pytest.raises(ContractValidationError, match="missing detector fields"):
        validate_aggregate_reconciliation(
            [result],
            {"counts": {"total": 1, "correct": 1}},
            [observation],
        )


def test_contract_episode_accounting_handles_empty_and_mixed_cohorts():
    from src.silentfail.models import DetectionResult, LedgerRecord

    empty = validate_aggregate_reconciliation([], {
        "counts": {
            "total": 0,
            "correct": 0,
            "claim_uncertain": 0,
            **{name: 0 for name in (
                "duplicate_action", "excess_mutation", "ineligible_refund_mutation",
                "incorrect_refund_amount", "missed_action", "incorrect_completion",
                "over_suppression", "silent_failure_false_success", "legitimate_completion",
                "duplicate",
                "email_successful_mutation", "email_tool_failure", "email_missing_required", "email_excess",
            )},
            **{name: 0 for name in (
                "raw_episode_count", "valid_episode_count", "invalid_episode_count",
                "inconclusive_episode_count", "invalid_or_inconclusive_episode_count",
                "execution_count", "scenario_count",
            )},
        },
    }, [])
    assert empty["episode_counts"] == {
        "raw_episode_count": 0,
        "valid_episode_count": 0,
        "invalid_episode_count": 0,
        "inconclusive_episode_count": 0,
        "invalid_or_inconclusive_episode_count": 0,
        "execution_count": 0,
        "scenario_count": 0,
    }
    results = []
    observations = []
    for index, (correct, uncertain) in enumerate(((True, False), (False, False), (False, True))):
        results.append({
            "record": LedgerRecord(run_id="r", scenario_id=f"s{index}"),
            "detection": DetectionResult(),
            "is_correct": correct,
        })
        observations.append({
            "valid": correct and not uncertain,
            "claim_uncertain": uncertain,
            "email_successful_mutation_count": 0,
            "email_tool_failure": False,
            "email_missing_required": False,
            "email_excess": False,
        })
    counts = {
        "total": 3,
        "correct": 1,
        "claim_uncertain": 1,
        **{name: 0 for name in (
            "duplicate_action", "excess_mutation", "ineligible_refund_mutation",
            "incorrect_refund_amount", "missed_action", "incorrect_completion",
            "over_suppression", "silent_failure_false_success", "legitimate_completion",
            "duplicate",
            "email_successful_mutation", "email_tool_failure", "email_missing_required", "email_excess",
        )},
        **episode_accounting(results, observations),
    }
    report = validate_aggregate_reconciliation(
        results,
        {"counts": counts},
        observations,
    )
    assert report["episode_counts"]["valid_episode_count"] == 1
    assert report["episode_counts"]["invalid_episode_count"] == 1
    assert report["episode_counts"]["inconclusive_episode_count"] == 1
    assert report["rate_denominators"]["correct"] == 3
    assert report["rates"]["correct"] == pytest.approx(1 / 3)
    assert set(report["rates"]) == {
        "correct",
        "duplicate_action",
        "excess_mutation",
        "ineligible_refund_mutation",
        "incorrect_refund_amount",
        "missed_action",
        "incorrect_completion",
        "over_suppression",
        "silent_failure_false_success",
        "legitimate_completion",
        "email_successful_mutation",
        "email_tool_failure",
        "email_missing_required",
        "email_excess",
    }
    assert set(report["rate_denominators"]) == set(report["rates"])
    assert all(value == 3 for value in report["rate_denominators"].values())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("valid", None, "missing required fields"),
        ("claim_uncertain", "false", "must be a boolean"),
        ("email_successful_mutation_count", -1, "non-negative integer"),
        ("email_tool_failure", 0, "must be a boolean"),
    ],
)
def test_contract_episode_accounting_rejects_malformed_observations(field, value, message):
    from src.silentfail.models import LedgerRecord

    observation = {
        "valid": True,
        "claim_uncertain": False,
        "email_successful_mutation_count": 0,
        "email_tool_failure": False,
        "email_missing_required": False,
        "email_excess": False,
    }
    if value is None:
        del observation[field]
    else:
        observation[field] = value
    with pytest.raises(ContractValidationError, match=message):
        episode_accounting(
            [{"record": LedgerRecord(run_id="r", scenario_id="s")}],
            [observation],
        )


def test_patch_provenance_is_deterministic_and_fails_closed_on_changes(tmp_path):
    config = {"scenario_manifest_sha256": "manifest", "offline_only": True}
    first = patch_provenance(PATCH, configuration="hardened", evaluation_config=config)
    second = patch_provenance(PATCH.model_copy(deep=True), configuration="hardened", evaluation_config=dict(config))
    assert first["sha256"] == second["sha256"]
    verify_expected_patch_fingerprint(first["sha256"], second)
    changed = PATCH.model_copy(update={"config": {"retry_limit": 2}})
    changed_provenance = patch_provenance(changed, configuration="hardened", evaluation_config=config)
    assert changed_provenance["sha256"] != first["sha256"]
    with pytest.raises(ValueError, match="does not match"):
        verify_expected_patch_fingerprint(first["sha256"], changed_provenance)
    with pytest.raises(ValueError, match="does not match"):
        run_offline_evaluation(
            build_manifest([_scenario()]),
            tmp_path / "mismatch",
            PATCH,
            offline_only=True,
            expected_patch_fingerprint="0" * 64,
        )
    assert not (tmp_path / "mismatch").exists()


def test_email_classification_reports_unknown_attribution_without_guessing():
    ledger = CausalLedger()
    ledger.record_tool_response({"error": "provider unavailable"})
    record = ledger.to_record("", WorldState().get_summary())
    observation = classify_email(record)
    assert observation["tool_failure_count"] == 0
    assert observation["unknown_attribution"] is True


def test_legacy_contracts_remain_valid_without_extension_fields():
    scenario = _scenario()
    contract = _contract(scenario)
    validated = validate_contract_manifest(
        build_manifest([scenario]),
        {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [contract]},
        expected_train_count=1,
        expected_held_out_count=0,
    )
    assert "contract_features" not in validated[scenario.scenario_id]


def test_contract_extensions_reject_unknown_or_malformed_features():
    scenario = _scenario()
    base = _contract(scenario)
    unknown = dict(base, contract_features=["not_supported"])
    with pytest.raises(ContractValidationError, match="unsupported features"):
        validate_contract_manifest(
            build_manifest([scenario]),
            {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [unknown]},
            expected_train_count=1,
            expected_held_out_count=0,
        )
    malformed = dict(base, contract_features="operation_bindings")
    with pytest.raises(ContractValidationError, match="list of strings"):
        validate_contract_manifest(
            build_manifest([scenario]),
            {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [malformed]},
            expected_train_count=1,
            expected_held_out_count=0,
        )
    missing_feature = dict(base, requested_operation={
        **base["requested_operation"],
        "operations": [{
            "operation_id": "op_1234_req0",
            "amount": 50.0,
            "kind": "partial_refund",
            "expected_mutation_count": 1,
        }],
    })
    with pytest.raises(ContractValidationError, match="require features"):
        validate_contract_manifest(
            build_manifest([scenario]),
            {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [missing_feature]},
            expected_train_count=1,
            expected_held_out_count=0,
        )


def test_configuration_specific_detector_expectations_override_legacy_values():
    from src.silentfail.runner import EpisodeRunner

    scenario = _numeric_order_scenario(
        "sc_config_detectors",
        [{"tool": "issue_refund", "call_index": 0, "type": "timeout_after_success"}],
    )
    contract_data = dict(_contract(scenario, detectors={"duplicate_action": False}))
    contract_data["contract_features"] = ["configuration_expectations"]
    contract_data["detectors_by_configuration"] = {
        "baseline": {"duplicate_action": True},
        "hardened": {"duplicate_action": False},
    }
    contract = validate_contract_manifest(
        build_manifest([scenario]),
        {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [contract_data]},
        expected_train_count=1,
        expected_held_out_count=0,
    )[scenario.scenario_id]
    baseline = EpisodeRunner.run_episode(scenario, agent=NaiveRefundAgent(model_client=None))
    hardened = EpisodeRunner.run_episode(scenario, patch=PATCH, agent=NaiveRefundAgent(model_client=None))
    baseline_observation = validate_episode_contract(contract, scenario, baseline, "baseline")
    assert baseline_observation["valid"] is False
    assert any("detector assertion duplicate_action mismatch" in error for error in baseline_observation["errors"])
    assert validate_episode_contract(contract, scenario, hardened, "hardened")["valid"] is True


def test_exact_operation_bindings_and_retry_groups_validate_ledger_events():
    from src.silentfail.runner import EpisodeRunner

    scenario = _numeric_order_scenario(
        "sc_operation_binding",
        [{"tool": "issue_refund", "call_index": 0, "type": "timeout_after_success"}],
    )
    contract_data = _contract(scenario)
    contract_data["contract_features"] = ["operation_bindings", "fault_observations"]
    contract_data["requested_operation"] = {
        "kind": "partial_refund",
        "amount": 50.0,
        "operation_count": 1,
        "operations": [{
            "operation_id": "op_1234_req0",
            "amount": 50.0,
            "kind": "partial_refund",
            "expected_mutation_count": 1,
        }],
        "retry_groups": [{
            "group_id": "refund-0",
            "operation_id": "op_1234_req0",
            "min_attempts": 2,
            "max_attempts": 2,
            "require_same_operation_id": True,
        }],
    }
    contract_data["fault_observations"] = {
        "hardened": {
            "required": [{
                "tool": "issue_refund",
                "type": "timeout_after_success",
                "occurrence": 1,
            }],
            "allowed_unobserved": [],
        },
    }
    contract = validate_contract_manifest(
        build_manifest([scenario]),
        {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [contract_data]},
        expected_train_count=1,
        expected_held_out_count=0,
    )[scenario.scenario_id]
    result = EpisodeRunner.run_episode(scenario, patch=PATCH, agent=NaiveRefundAgent(model_client=None))
    observation = validate_episode_contract(contract, scenario, result, "hardened")
    assert observation["valid"] is True
    assert observation["operation_observation"]["per_operation"]["op_1234_req0"] == {
        "request_count": 2,
        "mutation_count": 1,
        "mutation_amounts": [50.0],
    }
    assert observation["fault_observation"]["observed"] == [["issue_refund", "timeout_after_success", 1]]


@pytest.mark.parametrize("operation_update", [
    {"operations": [{
        "operation_id": "op_1234_req0",
        "amount": 49.0,
        "kind": "partial_refund",
        "expected_mutation_count": 1,
    }]},
    {"operations": [{
        "operation_id": "op_1234_req0",
        "amount": 50.0,
        "kind": "partial_refund",
        "expected_mutation_count": 2,
    }]},
])
def test_operation_binding_schema_rejects_wrong_amount_or_mutation_contract(operation_update):
    scenario = _numeric_order_scenario("sc_operation_schema", [])
    contract_data = _contract(scenario)
    contract_data["contract_features"] = ["operation_bindings"]
    contract_data["requested_operation"] = {
        **contract_data["requested_operation"],
        **operation_update,
    }
    if operation_update["operations"][0]["amount"] == 49.0:
        with pytest.raises(ContractValidationError, match="operation amounts conflict"):
            validate_contract_manifest(
                build_manifest([scenario]),
                {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [contract_data]},
                expected_train_count=1,
                expected_held_out_count=0,
            )
    else:
        contract = validate_contract_manifest(
            build_manifest([scenario]),
            {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [contract_data]},
            expected_train_count=1,
            expected_held_out_count=0,
        )[scenario.scenario_id]
        from src.silentfail.runner import EpisodeRunner
        result = EpisodeRunner.run_episode(scenario, agent=NaiveRefundAgent(model_client=None))
        observation = validate_episode_contract(contract, scenario, result)
        assert observation["valid"] is False
        assert any("mutation count mismatch" in error for error in observation["errors"])


def test_fault_observations_allow_short_circuited_unconsumed_fault():
    scenario = _numeric_order_scenario(
        "sc_fault_short_circuit",
        [
            {"tool": "issue_refund", "call_index": 0, "type": "timeout_after_success"},
            {"tool": "issue_refund", "call_index": 1, "type": "duplicate_webhook"},
        ],
        category="compound_faults",
    )
    contract_data = _contract(scenario)
    contract_data["contract_features"] = ["operation_bindings", "fault_observations"]
    contract_data["requested_operation"] = {
        **contract_data["requested_operation"],
        "operations": [{
            "operation_id": "op_1234_req0",
            "amount": 50.0,
            "kind": "partial_refund",
            "expected_mutation_count": 1,
        }],
        "retry_groups": [{
            "group_id": "refund-0",
            "operation_id": "op_1234_req0",
            "min_attempts": 2,
            "max_attempts": 2,
            "require_same_operation_id": True,
        }],
    }
    contract_data["fault_observations"] = {
        "hardened": {
            "required": [{
                "tool": "issue_refund",
                "type": "timeout_after_success",
                "occurrence": 1,
            }],
            "allowed_unobserved": [{
                "tool": "issue_refund",
                "type": "duplicate_webhook",
                "occurrence": 1,
            }],
        },
    }
    contract = validate_contract_manifest(
        build_manifest([scenario]),
        {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [contract_data]},
        expected_train_count=1,
        expected_held_out_count=0,
    )[scenario.scenario_id]
    from src.silentfail.runner import EpisodeRunner
    result = EpisodeRunner.run_episode(scenario, patch=PATCH, agent=NaiveRefundAgent(model_client=None))
    observation = validate_episode_contract(contract, scenario, result, "hardened")
    assert observation["valid"] is True
    assert observation["fault_observation"]["observed"] == [["issue_refund", "timeout_after_success", 1]]


def test_operation_bindings_reject_duplicate_ids_and_invalid_retry_references():
    scenario = _numeric_order_scenario("sc_operation_schema_errors", [])
    duplicate = _contract(scenario)
    duplicate["contract_features"] = ["operation_bindings"]
    duplicate["requested_operation"] = {
        **duplicate["requested_operation"],
        "operations": [
            {
                "operation_id": "op_1234_req0",
                "amount": 25.0,
                "kind": "partial_refund",
                "expected_mutation_count": 1,
            },
            {
                "operation_id": "op_1234_req0",
                "amount": 25.0,
                "kind": "partial_refund",
                "expected_mutation_count": 0,
            },
        ],
    }
    with pytest.raises(ContractValidationError, match="duplicate operation_id"):
        validate_contract_manifest(
            build_manifest([scenario]),
            {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [duplicate]},
            expected_train_count=1,
            expected_held_out_count=0,
        )

    invalid_retry = _contract(scenario)
    invalid_retry["contract_features"] = ["operation_bindings"]
    invalid_retry["requested_operation"] = {
        **invalid_retry["requested_operation"],
        "operations": [{
            "operation_id": "op_1234_req0",
            "amount": 50.0,
            "kind": "partial_refund",
            "expected_mutation_count": 1,
        }],
        "retry_groups": [{
            "group_id": "refund-0",
            "operation_id": "missing-op",
            "min_attempts": 0,
            "max_attempts": 1,
            "require_same_operation_id": True,
        }],
    }
    with pytest.raises(ContractValidationError, match="undeclared"):
        validate_contract_manifest(
            build_manifest([scenario]),
            {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [invalid_retry]},
            expected_train_count=1,
            expected_held_out_count=0,
        )


def test_operation_bindings_reject_undeclared_ledger_operation_id():
    scenario = _numeric_order_scenario("sc_operation_undeclared", [])
    contract_data = _contract(scenario)
    contract_data["contract_features"] = ["operation_bindings"]
    contract_data["requested_operation"] = {
        **contract_data["requested_operation"],
        "operations": [{
            "operation_id": "declared-operation",
            "amount": 50.0,
            "kind": "partial_refund",
            "expected_mutation_count": 1,
        }],
    }
    contract = validate_contract_manifest(
        build_manifest([scenario]),
        {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [contract_data]},
        expected_train_count=1,
        expected_held_out_count=0,
    )[scenario.scenario_id]
    from src.silentfail.runner import EpisodeRunner
    result = EpisodeRunner.run_episode(scenario, agent=NaiveRefundAgent(model_client=None))
    observation = validate_episode_contract(contract, scenario, result)
    assert observation["valid"] is False
    assert any("undeclared operation_id" in error for error in observation["errors"])


def test_fault_observations_fail_for_required_or_unaccounted_faults():
    scenario = _numeric_order_scenario(
        "sc_fault_observation_errors",
        [{"tool": "issue_refund", "call_index": 1, "type": "timeout_after_success"}],
    )
    contract_data = _contract(scenario)
    contract_data["contract_features"] = ["fault_observations"]
    contract_data["fault_observations"] = {
        "hardened": {
            "required": [{
                "tool": "send_email",
                "type": "timeout_after_success",
                "occurrence": 1,
            }],
            "allowed_unobserved": [],
        },
    }
    contract = validate_contract_manifest(
        build_manifest([scenario]),
        {"contract_schema_version": CONTRACT_SCHEMA_VERSION, "contracts": [contract_data]},
        expected_train_count=1,
        expected_held_out_count=0,
    )[scenario.scenario_id]
    from src.silentfail.runner import EpisodeRunner
    result = EpisodeRunner.run_episode(scenario, patch=PATCH, agent=NaiveRefundAgent(model_client=None))
    observation = validate_episode_contract(contract, scenario, result, "hardened")
    assert observation["valid"] is False
    assert any("does not match scheduled fault" in error for error in observation["errors"])
    assert any("neither observed nor allowed" in error for error in observation["errors"])
