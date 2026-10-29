import json

import pytest

import src.silentfail.offline_harness as offline_harness
from src.silentfail.contract_validation import CONTRACT_SCHEMA_VERSION
from src.silentfail.models import Scenario
from src.silentfail.offline_harness import (
    VALIDATION_MANIFEST_FILENAME,
    _write_json,
    build_manifest,
    save_manifest,
    validate_validation_manifest_gate,
    write_validation_manifest,
)


def _scenario(
    scenario_id,
    split,
    *,
    category,
    faults,
    order_status="delivered",
    message="Please refund $50 for order 1234.",
):
    return Scenario.model_validate({
        "scenario_id": scenario_id,
        "split": split,
        "seed": 10,
        "category": category,
        "initial_state": {
            "orders": [{
                "id": "1234",
                "status": order_status,
                "total": 50.0,
                "refunded_amount": 0.0,
                "items": [{"item_id": "item_1", "price": 50.0}],
            }],
            "refunds": [],
            "emails": [],
            "tickets": [],
        },
        "customer_message": message,
        "faults": faults,
        "expected_outcome": {
            "refund_count": 1,
            "refund_total": 50.0,
            "email_count": 1,
            "ticket_count": 0,
            "allow_refusal": False,
        },
    })


def _contract(scenario):
    return {
        "scenario_id": scenario.scenario_id,
        "requested_operation": {
            "kind": "full_refund",
            "amount": 50.0,
            "operation_count": 1,
        },
        "refund": {
            "expected_count": 1,
            "expected_total": 50.0,
            "amount_tolerance": 0.01,
        },
        "refusal": {"mode": "forbidden"},
        "completion_claim": {
            "kind": "success",
            "amount": 50.0,
            "amount_tolerance": 0.01,
        },
        "email": {"required_count": 1, "allowed_count": 1},
        "detectors": {},
    }


def _valid_inputs():
    scenarios = []
    for index in range(65):
        split = "train" if index < 39 else "held_out"
        category = "timeout_after_success" if index < 39 else "empty_200_applied"
        fault = (
            {"tool": "issue_refund", "call_index": 0, "type": "timeout_after_success"}
            if category == "timeout_after_success"
            else {
                "tool": "issue_refund",
                "call_index": 0,
                "type": "empty_200",
                "effect_applied": True,
            }
        )
        scenarios.append(
            _scenario(
                f"{split}-{index:03d}",
                split,
                category=category,
                faults=[fault],
                message=f"Please refund $50 for order 1234, {split} fixture token.",
            )
        )
    manifest = build_manifest(scenarios)
    contracts = {
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "contracts": [_contract(scenario) for scenario in scenarios],
    }
    return manifest, contracts


def _statistical_input():
    def report(total, correct):
        counts = {"total": total, "correct": correct}
        intervals = {
            name: {
                "numerator": value,
                "denominator": total,
                "point_estimate": value / total if total else None,
                "lower_95": 0.0 if total else None,
                "upper_95": 1.0 if total else None,
            }
            for name, value in counts.items()
        }
        transitions = {
            "failed_to_corrected": 0,
            "failed_to_still_failed": total - correct,
            "correct_to_regressed": 0,
            "correct_to_unchanged": correct,
            "invalid_pairs": 0,
            "inconclusive_pairs": 0,
            "total_paired_scenarios": total,
            "transition_denominator": total,
            "baseline_failure_denominator": total - correct,
            "baseline_correct_denominator": correct,
            "paired_correctness_difference_denominator": total,
        }
        return {
            "counts": counts,
            "wilson_intervals": intervals,
            "transitions": transitions,
            "detector_metrics": {
                "duplicate_action": {
                    "numerator": 0,
                    "denominator": total,
                    "interval": {
                        "numerator": 0,
                        "denominator": total,
                        "point_estimate": 0.0 if total else None,
                        "lower_95": 0.0 if total else None,
                        "upper_95": 0.0 if total else None,
                    },
                }
            },
            "email_metrics": {
                "successful_mutation": {
                    "numerator": correct,
                    "denominator": total,
                    "interval": intervals["correct"],
                }
            },
        }
    return {
        "overall": report(65, 65),
        "splits": {
            "train": report(39, 39),
            "held_out": report(26, 26),
        },
    }


def test_legacy_writers_reject_v2_filename(tmp_path):
    manifest, _ = _valid_inputs()
    path = tmp_path / VALIDATION_MANIFEST_FILENAME
    with pytest.raises(ValueError, match="mandatory pre-manifest validation gate"):
        save_manifest(path, manifest)
    with pytest.raises(ValueError, match="mandatory pre-manifest validation gate"):
        _write_json(path, manifest)
    assert not path.exists()


@pytest.mark.parametrize("failure", ["structural", "contract", "statistical"])
def test_each_required_gate_failure_blocks_writer(tmp_path, failure, monkeypatch):
    manifest, contracts = _valid_inputs()
    calls = []

    def writer(path, payload):
        calls.append((path, payload))

    original = offline_harness._mandatory_pre_manifest_validators()
    if failure == "structural":
        original["structural"] = lambda *args, **kwargs: {
            "valid": False,
            "errors": ["leakage"],
        }
    elif failure == "contract":
        original["contract"] = lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("bad contract")
        )
    else:
        original["statistical"] = lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("bad statistics")
        )
    monkeypatch.setattr(
        offline_harness,
        "_mandatory_pre_manifest_validators",
        lambda: original,
    )
    with pytest.raises(ValueError, match="pre-manifest"):
        write_validation_manifest(
            tmp_path / VALIDATION_MANIFEST_FILENAME,
            manifest,
            contracts,
            statistical_input=_statistical_input(),
            writer=writer,
        )
    assert calls == []


def test_missing_validator_and_validator_exception_fail_closed(tmp_path, monkeypatch):
    manifest, contracts = _valid_inputs()
    missing = {
        "structural": offline_harness.validate_structural_leakage,
        "contract": offline_harness.validate_contract_manifest,
    }
    monkeypatch.setattr(
        offline_harness,
        "_mandatory_pre_manifest_validators",
        lambda: missing,
    )
    with pytest.raises(ValueError, match="missing validators"):
        write_validation_manifest(
            tmp_path / VALIDATION_MANIFEST_FILENAME,
            manifest,
            contracts,
            writer=lambda *_: None,
        )

    raising = {
        "structural": offline_harness.validate_structural_leakage,
        "contract": offline_harness.validate_contract_manifest,
        "statistical": offline_harness._validate_statistical_gate_input,
    }
    raising["structural"] = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("boom")
    )
    monkeypatch.setattr(
        offline_harness,
        "_mandatory_pre_manifest_validators",
        lambda: raising,
    )
    with pytest.raises(ValueError, match="structural validation failed"):
        write_validation_manifest(
            tmp_path / VALIDATION_MANIFEST_FILENAME,
            manifest,
            contracts,
            writer=lambda *_: None,
        )


def test_successful_gate_allows_only_one_intended_write(tmp_path):
    manifest, contracts = _valid_inputs()
    writes = []

    def writer(path, payload):
        writes.append((path.name, json.loads(payload)))

    result = write_validation_manifest(
        tmp_path / VALIDATION_MANIFEST_FILENAME,
        manifest,
        contracts,
        statistical_input=_statistical_input(),
        writer=writer,
    )
    assert result["gate"]["valid"] is True
    assert writes == [(VALIDATION_MANIFEST_FILENAME, manifest)]


def test_failed_gate_does_not_overwrite_existing_path(tmp_path):
    manifest, contracts = _valid_inputs()
    path = tmp_path / VALIDATION_MANIFEST_FILENAME
    original = b"existing"
    path.write_bytes(original)
    with pytest.raises(FileExistsError):
        write_validation_manifest(
            path,
            manifest,
            contracts,
            statistical_input=_statistical_input(),
        )
    assert path.read_bytes() == original


def test_legacy_manifest_generation_and_save_remain_unchanged(tmp_path):
    manifest, _ = _valid_inputs()
    destination = tmp_path / "legacy-manifest.json"
    digest = save_manifest(destination, manifest)
    assert destination.exists()
    assert digest


def test_v2_filename_check_is_case_insensitive_for_relative_and_absolute_paths(tmp_path):
    manifest, _ = _valid_inputs()
    for name in ("OFFLINE-VALIDATION-V2.JSON", "Offline-Validation-V2.Json"):
        path = tmp_path / name
        with pytest.raises(ValueError, match="mandatory pre-manifest validation gate"):
            save_manifest(path, manifest)
        with pytest.raises(ValueError, match="mandatory pre-manifest validation gate"):
            _write_json(path, manifest)
        assert not path.exists()


def test_v2_gate_requires_statistical_input_and_rejects_incomplete_reports():
    manifest, contracts = _valid_inputs()
    with pytest.raises(ValueError, match="statistical validation failed"):
        validate_validation_manifest_gate(manifest, contracts)
    with pytest.raises(ValueError, match="statistical validation failed"):
        validate_validation_manifest_gate(
            manifest,
            contracts,
            statistical_input={"overall": {"counts": {}}, "splits": {}},
        )


def test_gate_rejects_malformed_structural_report_even_when_marked_valid(monkeypatch):
    manifest, contracts = _valid_inputs()
    original = offline_harness._mandatory_pre_manifest_validators()
    original["structural"] = lambda *args, **kwargs: {
        "valid": True,
        "errors": [],
        "signatures": {
            scenario["scenario_id"]: {
                "scenario_id": scenario["scenario_id"],
                "split": scenario["split"],
                "components": {},
                "sha256": 123,
            }
            for scenario in manifest["scenarios"]
        },
        "exact_structural_duplicates": [],
        "message_template_reuse": [],
        "nearest_training": [],
    }
    monkeypatch.setattr(
        offline_harness,
        "_mandatory_pre_manifest_validators",
        lambda: original,
    )
    with pytest.raises(ValueError, match="structural signature is incomplete"):
        validate_validation_manifest_gate(
            manifest,
            contracts,
            statistical_input=_statistical_input(),
        )


def test_gate_rejects_missing_contract_class_and_stale_contract_report():
    manifest, contracts = _valid_inputs()
    incomplete = {
        **contracts,
        "contracts": contracts["contracts"][:1],
    }
    with pytest.raises(ValueError, match="contract validation failed"):
        validate_validation_manifest_gate(
            manifest,
            incomplete,
            statistical_input=_statistical_input(),
        )


@pytest.mark.parametrize(
    "mutator, expected",
    [
        (lambda scenarios: scenarios[:64], "exactly 65"),
        (lambda scenarios: scenarios + [scenarios[0]], "duplicate scenario IDs"),
        (
            lambda scenarios: [
                {**scenario, "split": "train"} for scenario in scenarios
            ],
            "39 train and 26 held_out",
        ),
    ],
)
def test_v2_gate_rejects_invalid_population_and_split_allocation(mutator, expected):
    manifest, contracts = _valid_inputs()
    manifest["scenarios"] = mutator(manifest["scenarios"])
    with pytest.raises(ValueError, match=expected):
        validate_validation_manifest_gate(
            manifest,
            contracts,
            statistical_input=_statistical_input(),
        )


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda report: report["overall"].pop("wilson_intervals"), "wilson_intervals"),
        (lambda report: report["overall"]["transitions"].update(
            {"transition_denominator": 1}
        ), "does not reconcile"),
        (lambda report: report["overall"]["counts"].update({"correct": True}), "non-negative integer"),
    ],
)
def test_v2_gate_rejects_incomplete_or_inconsistent_statistics(mutation, expected):
    manifest, contracts = _valid_inputs()
    statistical = _statistical_input()
    mutation(statistical)
    with pytest.raises(ValueError, match=expected):
        validate_validation_manifest_gate(
            manifest,
            contracts,
            statistical_input=statistical,
        )
