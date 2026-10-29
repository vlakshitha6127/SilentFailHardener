"""Strict, opt-in validation for offline evaluation contracts."""

from __future__ import annotations

import math
from math import comb, sqrt
from typing import Any, Dict, Iterable, Mapping

from src.silentfail.detectors import ClaimExtractor
from src.silentfail.models import Scenario


CONTRACT_SCHEMA_VERSION = "offline-contract-v1"
DEFAULT_TRAIN_COUNT = 39
DEFAULT_HELD_OUT_COUNT = 26
SUPPORTED_DETECTOR_ASSERTIONS = {
    "duplicate_action",
    "excess_mutation",
    "ineligible_refund_mutation",
    "incorrect_refund_amount",
    "missed_action",
    "incorrect_completion",
    "over_suppression",
    "silent_failure_false_success",
    "legitimate_completion",
    "claim_uncertain",
}
REFUSAL_MODES = {"required", "permitted", "forbidden"}
CLAIM_KINDS = {"success", "refusal", "success_or_refusal"}
SUPPORTED_OPERATION_KINDS = {"refund", "partial_refund", "full_refund"}
REQUIRED_CONTRACT_FIELDS = {
    "scenario_id",
    "requested_operation",
    "refund",
    "refusal",
    "completion_claim",
    "email",
    "detectors",
}
SUPPORTED_FAULT_TYPES = {"timeout_after_success", "empty_200", "duplicate_webhook"}
SUPPORTED_CONTRACT_FEATURES = {
    "operation_bindings",
    "fault_observations",
    "configuration_expectations",
}
SUPPORTED_FAULT_TOOLS = {"get_order", "issue_refund", "send_email", "create_ticket"}
CONFIGURATIONS = {"baseline", "hardened"}
EXTENSION_FIELDS = {
    "contract_features",
    "detectors_by_configuration",
    "fault_observations",
}


class ContractValidationError(ValueError):
    """Raised when a contract definition is unsupported or contradictory."""


def wilson_interval(
    successes: int,
    denominator: int,
    *,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Return a deterministic Wilson interval with explicit count metadata."""
    if type(successes) is not int or type(denominator) is not int:
        raise ContractValidationError("Wilson counts must be integers")
    if successes < 0 or denominator < 0 or successes > denominator:
        raise ContractValidationError("Wilson counts are inconsistent")
    if confidence != 0.95:
        raise ContractValidationError("only deterministic 95% Wilson intervals are supported")
    if denominator == 0:
        return {
            "numerator": successes,
            "denominator": 0,
            "point_estimate": None,
            "lower_95": None,
            "upper_95": None,
        }
    z = 1.959963984540054
    estimate = successes / denominator
    z_squared = z * z
    centre = (estimate + z_squared / (2 * denominator)) / (1 + z_squared / denominator)
    margin = (
        z
        * sqrt(
            estimate * (1 - estimate) / denominator
            + z_squared / (4 * denominator * denominator)
        )
        / (1 + z_squared / denominator)
    )
    return {
        "numerator": successes,
        "denominator": denominator,
        "point_estimate": estimate,
        "lower_95": max(0.0, centre - margin),
        "upper_95": min(1.0, centre + margin),
    }


def _exact_mcnemar_p_value(failed_to_corrected: int, correct_to_regressed: int) -> float | None:
    discordant = failed_to_corrected + correct_to_regressed
    if discordant == 0:
        return None
    tail = sum(comb(discordant, k) for k in range(0, min(failed_to_corrected, correct_to_regressed) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def _strict_keys(value: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ContractValidationError(f"{location} contains unsupported fields: {sorted(unknown)}")


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{location} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ContractValidationError(f"{location} must be finite")
    return number


def _required_object(value: Any, location: str, fields: set[str]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractValidationError(f"{location} must be an object")
    missing = fields - set(value)
    if missing:
        raise ContractValidationError(f"{location} missing required fields: {sorted(missing)}")
    _strict_keys(value, fields, location)
    return value


def _validate_feature_declarations(
    raw: Mapping[str, Any],
    location: str,
) -> set[str]:
    features = raw.get("contract_features", [])
    if not isinstance(features, list) or any(
        not isinstance(feature, str) for feature in features
    ):
        raise ContractValidationError(f"{location}.contract_features must be a list of strings")
    if len(features) != len(set(features)):
        raise ContractValidationError(f"{location}.contract_features must be unique")
    unknown = set(features) - SUPPORTED_CONTRACT_FEATURES
    if unknown:
        raise ContractValidationError(
            f"{location}.contract_features contains unsupported features: {sorted(unknown)}"
        )
    required = set()
    requested = raw.get("requested_operation")
    if isinstance(requested, Mapping) and (
        "operations" in requested or "retry_groups" in requested
    ):
        required.add("operation_bindings")
    if "fault_observations" in raw:
        required.add("fault_observations")
    if "detectors_by_configuration" in raw:
        required.add("configuration_expectations")
    missing = required - set(features)
    if missing:
        raise ContractValidationError(
            f"{location} extension fields require features: {sorted(missing)}"
        )
    return set(features)


def _validate_operation_extensions(
    requested: Mapping[str, Any],
    location: str,
) -> None:
    base = {"kind", "amount", "operation_count"}
    allowed = base | {"operations", "retry_groups"}
    _strict_keys(requested, allowed, location)
    operations = requested.get("operations")
    retry_groups = requested.get("retry_groups")
    operation_ids: set[str] = set()
    if operations is not None:
        if not isinstance(operations, list) or not operations:
            raise ContractValidationError(f"{location}.operations must be a non-empty list")
        for index, operation in enumerate(operations):
            op_location = f"{location}.operations[{index}]"
            if not isinstance(operation, Mapping):
                raise ContractValidationError(f"{op_location} must be an object")
            fields = {
                "operation_id",
                "amount",
                "kind",
                "expected_mutation_count",
            }
            _strict_keys(operation, fields, op_location)
            if set(operation) != fields:
                raise ContractValidationError(
                    f"{op_location} must contain exactly: {sorted(fields)}"
                )
            operation_id = operation["operation_id"]
            if not isinstance(operation_id, str) or not operation_id.strip():
                raise ContractValidationError(f"{op_location}.operation_id must be non-empty")
            if operation_id in operation_ids:
                raise ContractValidationError(f"duplicate operation_id {operation_id!r}")
            operation_ids.add(operation_id)
            if (
                not isinstance(operation["kind"], str)
                or operation["kind"] not in SUPPORTED_OPERATION_KINDS
            ):
                raise ContractValidationError(f"{op_location}.kind is unsupported")
            _number(operation["amount"], f"{op_location}.amount")
            if (
                type(operation["expected_mutation_count"]) is not int
                or operation["expected_mutation_count"] < 0
            ):
                raise ContractValidationError(
                    f"{op_location}.expected_mutation_count must be non-negative"
                )
        if requested["operation_count"] != len(operations):
            raise ContractValidationError(
                f"{location}.operation_count conflicts with operations"
            )
    elif retry_groups is not None:
        raise ContractValidationError(
            f"{location}.retry_groups requires operations"
        )

    if retry_groups is not None:
        if not isinstance(retry_groups, list):
            raise ContractValidationError(f"{location}.retry_groups must be a list")
        group_ids: set[str] = set()
        grouped_operations: set[str] = set()
        for index, group in enumerate(retry_groups):
            group_location = f"{location}.retry_groups[{index}]"
            if not isinstance(group, Mapping):
                raise ContractValidationError(f"{group_location} must be an object")
            fields = {
                "group_id",
                "operation_id",
                "min_attempts",
                "max_attempts",
                "require_same_operation_id",
            }
            _strict_keys(group, fields, group_location)
            if set(group) != fields:
                raise ContractValidationError(
                    f"{group_location} must contain exactly: {sorted(fields)}"
                )
            group_id = group["group_id"]
            operation_id = group["operation_id"]
            if not isinstance(group_id, str) or not group_id.strip():
                raise ContractValidationError(f"{group_location}.group_id must be non-empty")
            if group_id in group_ids:
                raise ContractValidationError(f"duplicate retry group_id {group_id!r}")
            group_ids.add(group_id)
            if not isinstance(operation_id, str) or operation_id not in operation_ids:
                raise ContractValidationError(
                    f"{group_location}.operation_id is undeclared"
                )
            if operation_id in grouped_operations:
                raise ContractValidationError(
                    f"operation_id {operation_id!r} appears in multiple retry groups"
                )
            grouped_operations.add(operation_id)
            if (
                type(group["min_attempts"]) is not int
                or type(group["max_attempts"]) is not int
                or group["min_attempts"] < 1
                or group["max_attempts"] < group["min_attempts"]
            ):
                raise ContractValidationError(
                    f"{group_location} attempt bounds are invalid"
                )
            if type(group["require_same_operation_id"]) is not bool:
                raise ContractValidationError(
                    f"{group_location}.require_same_operation_id must be boolean"
                )


def _validate_detector_map(value: Any, location: str) -> None:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{location} must be an object")
    for name, expected in value.items():
        if name not in SUPPORTED_DETECTOR_ASSERTIONS or type(expected) is not bool:
            raise ContractValidationError(
                f"{location} has unsupported assertion {name!r}"
            )


def _validate_fault_expectations(value: Any, location: str) -> None:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{location} must be an object")
    unknown_configs = set(value) - CONFIGURATIONS
    if unknown_configs:
        raise ContractValidationError(
            f"{location} contains unsupported configurations: {sorted(unknown_configs)}"
        )
    for configuration, expectations in value.items():
        if not isinstance(expectations, Mapping):
            raise ContractValidationError(f"{location}.{configuration} must be an object")
        _strict_keys(expectations, {"required", "allowed_unobserved"}, f"{location}.{configuration}")
        for field in ("required", "allowed_unobserved"):
            entries = expectations.get(field, [])
            if not isinstance(entries, list):
                raise ContractValidationError(
                    f"{location}.{configuration}.{field} must be a list"
                )
            for index, entry in enumerate(entries):
                entry_location = f"{location}.{configuration}.{field}[{index}]"
                if not isinstance(entry, Mapping):
                    raise ContractValidationError(f"{entry_location} must be an object")
                fields = {"tool", "type", "occurrence"}
                _strict_keys(entry, fields, entry_location)
                if set(entry) != fields:
                    raise ContractValidationError(
                        f"{entry_location} must contain exactly: {sorted(fields)}"
                    )
                if (
                    not isinstance(entry["tool"], str)
                    or entry["tool"] not in SUPPORTED_FAULT_TOOLS
                ):
                    raise ContractValidationError(f"{entry_location}.tool is unsupported")
                if (
                    not isinstance(entry["type"], str)
                    or entry["type"] not in SUPPORTED_FAULT_TYPES
                ):
                    raise ContractValidationError(f"{entry_location}.type is unsupported")
                if type(entry["occurrence"]) is not int or entry["occurrence"] < 1:
                    raise ContractValidationError(
                        f"{entry_location}.occurrence must be positive"
                    )


def validate_contract_manifest(
    manifest: Mapping[str, Any],
    contract_manifest: Mapping[str, Any],
    *,
    expected_train_count: int = DEFAULT_TRAIN_COUNT,
    expected_held_out_count: int = DEFAULT_HELD_OUT_COUNT,
) -> Dict[str, Dict[str, Any]]:
    """Validate an opt-in contract manifest before any episode can run."""
    if not isinstance(contract_manifest, Mapping):
        raise ContractValidationError("contract manifest must be an object")
    if contract_manifest.get("contract_schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ContractValidationError("unsupported contract schema version")
    _strict_keys(contract_manifest, {"contract_schema_version", "contracts"}, "contract manifest")
    if not isinstance(contract_manifest.get("contracts"), list):
        raise ContractValidationError("contracts must be a list")

    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ContractValidationError("scenario manifest has no scenarios")
    scenario_ids = [s.get("scenario_id") for s in scenarios if isinstance(s, dict)]
    if len(scenario_ids) != len(scenarios) or len(set(scenario_ids)) != len(scenario_ids):
        raise ContractValidationError("scenario IDs must be unique")
    scenario_map = {s["scenario_id"]: Scenario.model_validate(s) for s in scenarios}
    train_count = sum(s.split == "train" for s in scenario_map.values())
    held_out_count = sum(s.split == "held_out" for s in scenario_map.values())
    if train_count != expected_train_count or held_out_count != expected_held_out_count:
        raise ContractValidationError(
            f"expected {expected_train_count} train and {expected_held_out_count} held_out scenarios, "
            f"found {train_count} train and {held_out_count} held_out"
        )

    contracts: Dict[str, Dict[str, Any]] = {}
    for index, raw in enumerate(contract_manifest["contracts"]):
        location = f"contracts[{index}]"
        if not isinstance(raw, dict):
            raise ContractValidationError(f"{location} must be an object")
        _strict_keys(raw, REQUIRED_CONTRACT_FIELDS | EXTENSION_FIELDS, location)
        missing = REQUIRED_CONTRACT_FIELDS - set(raw)
        if missing:
            raise ContractValidationError(f"{location} missing required fields: {sorted(missing)}")
        scenario_id = raw["scenario_id"]
        if scenario_id not in scenario_map:
            raise ContractValidationError(f"{location} references unknown scenario {scenario_id!r}")
        if scenario_id in contracts:
            raise ContractValidationError(f"duplicate contract for scenario {scenario_id!r}")

        requested = raw["requested_operation"]
        if not isinstance(requested, Mapping):
            raise ContractValidationError(f"{location}.requested_operation must be an object")
        requested_required = {"kind", "amount", "operation_count"}
        missing_requested = requested_required - set(requested)
        if missing_requested:
            raise ContractValidationError(
                f"{location}.requested_operation missing required fields: "
                f"{sorted(missing_requested)}"
            )
        _validate_feature_declarations(raw, location)
        _validate_operation_extensions(requested, f"{location}.requested_operation")
        if (
            not isinstance(requested["kind"], str)
            or requested["kind"] not in SUPPORTED_OPERATION_KINDS
        ):
            raise ContractValidationError(f"{location}.requested_operation.kind is unsupported")
        _number(requested["amount"], f"{location}.requested_operation.amount")
        if type(requested["operation_count"]) is not int or requested["operation_count"] < 1:
            raise ContractValidationError(f"{location}.requested_operation.operation_count must be positive")

        refund = _required_object(raw["refund"], f"{location}.refund", {
            "expected_count", "expected_total", "amount_tolerance",
        })
        if not isinstance(refund["expected_count"], int) or refund["expected_count"] < 0:
            raise ContractValidationError(f"{location}.refund.expected_count must be non-negative")
        expected_total = _number(refund["expected_total"], f"{location}.refund.expected_total")
        tolerance = _number(refund["amount_tolerance"], f"{location}.refund.amount_tolerance")
        if expected_total < 0 or tolerance < 0:
            raise ContractValidationError(f"{location}.refund values must be non-negative")
        operations = requested.get("operations")
        if operations is not None:
            operation_total = sum(float(operation["amount"]) for operation in operations)
            if abs(operation_total - expected_total) > tolerance:
                raise ContractValidationError(
                    f"{location} operation amounts conflict with expected refund total"
                )

        refusal = _required_object(raw["refusal"], f"{location}.refusal", {"mode"})
        if refusal["mode"] not in REFUSAL_MODES:
            raise ContractValidationError(f"{location}.refusal.mode is unsupported")

        claim = _required_object(raw["completion_claim"], f"{location}.completion_claim", {
            "kind", "amount", "amount_tolerance",
        })
        if claim["kind"] not in CLAIM_KINDS:
            raise ContractValidationError(f"{location}.completion_claim.kind is unsupported")
        if claim["kind"] in {"success", "success_or_refusal"}:
            if claim["amount"] is None:
                if claim["kind"] == "success":
                    raise ContractValidationError(f"{location}.completion_claim.amount is required for success")
            else:
                _number(claim["amount"], f"{location}.completion_claim.amount")
        elif claim["amount"] is not None:
            raise ContractValidationError(f"{location}.completion_claim.amount must be null for refusal")
        claim_tolerance = _number(claim["amount_tolerance"], f"{location}.completion_claim.amount_tolerance")
        if claim_tolerance < 0:
            raise ContractValidationError(f"{location}.completion_claim.amount_tolerance must be non-negative")

        email = _required_object(raw["email"], f"{location}.email", {"required_count", "allowed_count"})
        if not all(isinstance(email[key], int) and email[key] >= 0 for key in ("required_count", "allowed_count")):
            raise ContractValidationError(f"{location}.email counts must be non-negative integers")
        if email["allowed_count"] < email["required_count"]:
            raise ContractValidationError(f"{location}.email.allowed_count cannot be below required_count")

        detectors = raw["detectors"]
        _validate_detector_map(detectors, f"{location}.detectors")
        if "detectors_by_configuration" in raw:
            by_configuration = raw["detectors_by_configuration"]
            if not isinstance(by_configuration, Mapping):
                raise ContractValidationError(
                    f"{location}.detectors_by_configuration must be an object"
                )
            unknown_configs = set(by_configuration) - CONFIGURATIONS
            if unknown_configs:
                raise ContractValidationError(
                    f"{location}.detectors_by_configuration contains unsupported configurations: "
                    f"{sorted(unknown_configs)}"
                )
            for configuration, values in by_configuration.items():
                _validate_detector_map(
                    values,
                    f"{location}.detectors_by_configuration.{configuration}",
                )
        if "fault_observations" in raw:
            _validate_fault_expectations(
                raw["fault_observations"],
                f"{location}.fault_observations",
            )

        scenario = scenario_map[scenario_id]
        for fault in scenario.faults:
            if fault.type not in SUPPORTED_FAULT_TYPES:
                raise ContractValidationError(f"{location} uses unsupported fault type {fault.type!r}")
        if scenario.expected_outcome.refund_count != refund["expected_count"]:
            raise ContractValidationError(f"{location} refund count conflicts with Scenario.expected_outcome")
        if abs(float(scenario.expected_outcome.refund_total) - expected_total) > tolerance:
            raise ContractValidationError(f"{location} refund total conflicts with Scenario.expected_outcome")
        if scenario.expected_outcome.email_count != email["required_count"]:
            raise ContractValidationError(f"{location} email count conflicts with Scenario.expected_outcome")
        if refusal["mode"] == "required":
            if not scenario.expected_outcome.allow_refusal or refund["expected_count"] != 0:
                raise ContractValidationError(f"{location} required refusal is contradictory")
            if claim["kind"] != "refusal":
                raise ContractValidationError(f"{location} required refusal must use a refusal claim")
        elif refusal["mode"] == "permitted":
            if not scenario.expected_outcome.allow_refusal:
                raise ContractValidationError(f"{location} permitted refusal conflicts with allow_refusal")
        else:
            if scenario.expected_outcome.allow_refusal or refund["expected_count"] == 0:
                raise ContractValidationError(f"{location} forbidden refusal is contradictory")
        if claim["kind"] == "success" and abs(float(claim["amount"]) - expected_total) > max(claim_tolerance, tolerance):
            raise ContractValidationError(f"{location} completion claim conflicts with expected refund total")
        contracts[scenario_id] = raw

    if set(contracts) != set(scenario_map):
        raise ContractValidationError(f"missing contracts for scenarios: {sorted(set(scenario_map) - set(contracts))}")
    return contracts


def _claim_status(message: str) -> Dict[str, Any]:
    claims = ClaimExtractor.extract_rule_based(message)
    has_success_signal = bool(claims.get("claims_refund_issued"))
    has_refusal_signal = bool(claims.get("is_refusal"))
    amount = claims.get("claimed_amount")
    if has_success_signal and not has_refusal_signal and amount is not None:
        status = "success"
    elif has_refusal_signal and not has_success_signal:
        status = "refusal"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "claims_refund_issued": has_success_signal,
        "is_refusal": has_refusal_signal,
        "claimed_amount": amount,
        "uncertain": status == "inconclusive",
    }


def classify_email(record: Any) -> Dict[str, Any]:
    successful = sum(
        event.type == "world_mutation" and event.effect == "email_sent"
        for event in record.events
    )
    failures = 0
    unknown = 0
    for event in record.events:
        if event.type != "tool_response_to_agent":
            continue
        observed = event.observed or {}
        is_supported_failure = (
            event.fault in {"timeout_after_success", "empty_200"}
            or observed.get("error") is not None
            or observed.get("status") in {"error", "skipped"}
            or (observed.get("status") == 200 and observed.get("data") is None)
        )
        if event.tool == "send_email":
            if is_supported_failure:
                failures += 1
        elif event.tool is None and is_supported_failure:
            unknown += 1
    return {
        "successful_mutation_count": int(successful),
        "tool_failure_count": failures,
        "tool_failure": failures > 0,
        "unknown_response_count": unknown,
        "unknown_attribution": unknown > 0,
    }


def _mutation_observation(record: Any) -> Dict[str, Any]:
    mutations = [
        event for event in record.events
        if event.type == "world_mutation" and event.effect == "refund_created"
    ]
    operation_ids = [
        event.extra.get("operation_id") if event.extra else None
        for event in mutations
    ]
    missing_ids = sum(operation_id is None for operation_id in operation_ids)
    return {
        "refund_mutation_count": len(mutations),
        "operation_ids": operation_ids,
        "missing_operation_id_count": missing_ids,
        "unique_operation_id_count": len({x for x in operation_ids if x is not None}),
    }


def _operation_observation(
    record: Any,
    requested: Mapping[str, Any],
) -> Dict[str, Any]:
    operations = requested.get("operations")
    if operations is None:
        return {"enabled": False}
    declared = {operation["operation_id"]: operation for operation in operations}
    requests = [
        event for event in record.events
        if event.type == "tool_request" and event.tool == "issue_refund"
    ]
    mutations = [
        event for event in record.events
        if event.type == "world_mutation" and event.effect == "refund_created"
    ]
    errors: list[str] = []
    request_ids = [
        event.args.get("operation_id") if event.args else None
        for event in requests
    ]
    for operation_id in request_ids:
        if operation_id not in declared:
            errors.append(f"undeclared operation_id in request: {operation_id!r}")
    mutation_ids = [
        event.extra.get("operation_id") if event.extra else None
        for event in mutations
    ]
    for operation_id in mutation_ids:
        if operation_id not in declared:
            errors.append(f"undeclared operation_id in mutation: {operation_id!r}")
    if any(operation_id is None for operation_id in request_ids):
        errors.append("operation request is missing operation_id")
    if any(operation_id is None for operation_id in mutation_ids):
        errors.append("operation mutation is missing operation_id")

    retry_by_operation = {
        group["operation_id"]: group
        for group in requested.get("retry_groups", [])
    }
    per_operation: Dict[str, Any] = {}
    for operation_id, operation in declared.items():
        operation_requests = [
            event for event in requests
            if event.args and event.args.get("operation_id") == operation_id
        ]
        operation_mutations = [
            event for event in mutations
            if event.extra and event.extra.get("operation_id") == operation_id
        ]
        retry_group = retry_by_operation.get(operation_id)
        if retry_group is None:
            if len(operation_requests) != 1:
                errors.append(
                    f"operation {operation_id!r} expected exactly one request, "
                    f"observed {len(operation_requests)}"
                )
        elif not (
            retry_group["min_attempts"]
            <= len(operation_requests)
            <= retry_group["max_attempts"]
        ):
            errors.append(
                f"operation {operation_id!r} request attempts outside retry bounds"
            )
        for request in operation_requests:
            request_amount = request.args.get("amount") if request.args else None
            if request_amount is None or abs(float(request_amount) - float(operation["amount"])) > 0.01:
                errors.append(f"operation {operation_id!r} request amount mismatch")
        mutation_amounts = [event.amount for event in operation_mutations]
        if len(operation_mutations) != operation["expected_mutation_count"]:
            errors.append(
                f"operation {operation_id!r} mutation count mismatch"
            )
        for amount in mutation_amounts:
            if amount is None or abs(float(amount) - float(operation["amount"])) > 0.01:
                errors.append(f"operation {operation_id!r} mutation amount mismatch")
        per_operation[operation_id] = {
            "request_count": len(operation_requests),
            "mutation_count": len(operation_mutations),
            "mutation_amounts": mutation_amounts,
        }
    return {
        "enabled": True,
        "valid": not errors,
        "errors": errors,
        "per_operation": per_operation,
        "request_operation_ids": request_ids,
        "mutation_operation_ids": mutation_ids,
    }


def _fault_observation(
    record: Any,
    scenario: Scenario,
    contract: Mapping[str, Any],
    configuration: str,
) -> Dict[str, Any]:
    expectations = contract.get("fault_observations", {}).get(configuration)
    if expectations is None:
        return {"enabled": False}
    scheduled: Dict[tuple[str, str, int], bool] = {}
    occurrences: Dict[tuple[str, str], int] = {}
    for fault in scenario.faults:
        key_base = (fault.tool, fault.type)
        occurrences[key_base] = occurrences.get(key_base, 0) + 1
        scheduled[(*key_base, occurrences[key_base])] = False
    observed: Dict[tuple[str, str, int], bool] = {}
    observed_occurrences: Dict[tuple[str, str], int] = {}
    for event in record.events:
        if event.type != "tool_response_to_agent" or event.fault is None:
            continue
        key_base = (event.tool or "", event.fault)
        observed_occurrences[key_base] = observed_occurrences.get(key_base, 0) + 1
        observed[(*key_base, observed_occurrences[key_base])] = True

    errors: list[str] = []
    required = {
        (entry["tool"], entry["type"], entry["occurrence"])
        for entry in expectations.get("required", [])
    }
    allowed_unobserved = {
        (entry["tool"], entry["type"], entry["occurrence"])
        for entry in expectations.get("allowed_unobserved", [])
    }
    for key in required | allowed_unobserved:
        if key not in scheduled:
            errors.append(f"fault expectation does not match scheduled fault: {key}")
    for key in required:
        if not observed.get(key, False):
            errors.append(f"required fault was not observed: {key}")
    for key in scheduled:
        if not observed.get(key, False) and key not in allowed_unobserved:
            errors.append(f"scheduled fault was neither observed nor allowed unobserved: {key}")
    for key in observed:
        if key not in scheduled:
            errors.append(f"unexpected observed fault: {key}")
    return {
        "enabled": True,
        "valid": not errors,
        "errors": errors,
        "scheduled": [list(key) for key in sorted(scheduled)],
        "observed": [list(key) for key in sorted(observed)],
    }


def validate_episode_contract(
    contract: Mapping[str, Any],
    scenario: Scenario,
    result: Mapping[str, Any],
    configuration: str = "baseline",
) -> Dict[str, Any]:
    """Return observations and validation errors for one raw episode."""
    record = result["record"]
    detection = result["detection"]
    state = record.final_state
    actual_count = int(state.get("refund_count", 0))
    actual_total = float(state.get("refund_total", 0.0))
    expected_count = int(contract["refund"]["expected_count"])
    expected_total = float(contract["refund"]["expected_total"])
    tolerance = float(contract["refund"]["amount_tolerance"])
    expected_operations = int(contract["requested_operation"]["operation_count"])
    claim = _claim_status(record.final_message)
    email = classify_email(record)
    mutations = _mutation_observation(record)
    errors = []

    mode = contract["refusal"]["mode"]
    refusal_claim = claim["status"] == "refusal"
    success_claim = claim["status"] == "success"
    successful_path = actual_count == expected_count and abs(actual_total - expected_total) <= tolerance
    if mode == "required":
        if actual_count != 0 or actual_total != 0 or not refusal_claim:
            errors.append("required refusal contract violated")
    elif mode == "forbidden":
        if not successful_path or not success_claim:
            errors.append("forbidden refusal contract violated")
    else:  # permitted: either truthful refusal or the expected successful completion.
        if refusal_claim:
            if actual_count != 0 or actual_total != 0:
                errors.append("permitted refusal contains refund mutations")
        elif not successful_path or not success_claim:
            errors.append("permitted contract is neither a valid refusal nor successful completion")

    expected_claim = contract["completion_claim"]
    if expected_claim["kind"] == "success":
        claimed = claim["claimed_amount"]
        if claim["uncertain"] or not success_claim or claimed is None:
            errors.append("success claim is missing or inconclusive")
        elif abs(float(claimed) - float(expected_claim["amount"])) > float(expected_claim["amount_tolerance"]):
            errors.append("claimed amount mismatch")
    elif expected_claim["kind"] == "refusal":
        if claim["uncertain"] or not refusal_claim:
            errors.append("refusal claim is missing or inconclusive")
    elif claim["uncertain"] or (not refusal_claim and not success_claim):
        errors.append("completion claim is missing or inconclusive")
    if expected_claim["kind"] == "success_or_refusal" and success_claim:
        expected_amount = expected_claim["amount"]
        if expected_amount is not None and abs(float(claim["claimed_amount"] or 0) - float(expected_amount)) > float(expected_claim["amount_tolerance"]):
            errors.append("claimed amount mismatch")
    elif expected_claim["kind"] not in {"success", "refusal", "success_or_refusal"}:
        errors.append("refusal claim is missing or inconclusive")

    if mutations["missing_operation_id_count"]:
        errors.append("refund mutation is missing operation_id")
    if refusal_claim and mode == "permitted":
        if mutations["refund_mutation_count"] != 0:
            errors.append("permitted refusal contains refund mutation events")
    elif mutations["refund_mutation_count"] != expected_operations:
        errors.append("refund mutation count does not match requested operation count")
    elif mutations["unique_operation_id_count"] != expected_operations:
        errors.append("operation_ids are not distinct for requested operations")

    if not (mode == "permitted" and refusal_claim):
        if email["successful_mutation_count"] < contract["email"]["required_count"]:
            errors.append("missing required email")
        if email["successful_mutation_count"] > contract["email"]["allowed_count"]:
            errors.append("excess email")

    if configuration not in CONFIGURATIONS:
        raise ContractValidationError(f"unsupported configuration {configuration!r}")
    expected_detectors = dict(contract["detectors"])
    expected_detectors.update(
        contract.get("detectors_by_configuration", {}).get(configuration, {})
    )
    detector_values = {}
    for name, expected in expected_detectors.items():
        actual = bool(getattr(detection, name))
        detector_values[name] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            errors.append(f"detector assertion {name} mismatch")
    operation_observation = _operation_observation(
        record,
        contract["requested_operation"],
    )
    if operation_observation.get("enabled") and not operation_observation["valid"]:
        errors.extend(operation_observation["errors"])
    fault_observation = _fault_observation(
        record,
        scenario,
        contract,
        configuration,
    )
    if fault_observation.get("enabled") and not fault_observation["valid"]:
        errors.extend(fault_observation["errors"])

    return {
        "valid": not errors,
        "errors": errors,
        "claim_status": claim["status"],
        "claim_uncertain": claim["uncertain"],
        "claimed_amount": claim["claimed_amount"],
        "refund_count": actual_count,
        "refund_total": actual_total,
        "operation_ids": mutations["operation_ids"],
        "missing_operation_id_count": mutations["missing_operation_id_count"],
        "detectors": detector_values,
        "email_successful_mutation_count": email["successful_mutation_count"],
        "email_tool_failure_count": email["tool_failure_count"],
        "email_tool_failure": email["tool_failure"],
        "email_missing_required": email["successful_mutation_count"] < contract["email"]["required_count"],
        "email_excess": email["successful_mutation_count"] > contract["email"]["allowed_count"],
        "email_unknown_attribution": email["unknown_attribution"],
        "configuration": configuration,
        "operation_observation": operation_observation,
        "fault_observation": fault_observation,
    }


def validate_paired_results(
    results_by_configuration: Mapping[str, Iterable[Mapping[str, Any]]],
    scenarios: Mapping[str, Scenario],
) -> Dict[str, Any]:
    """Validate exact baseline/hardened pairing and report split results."""
    if set(results_by_configuration) != {"baseline", "hardened"}:
        raise ContractValidationError("paired results must contain baseline and hardened")
    if not isinstance(scenarios, Mapping) or not scenarios:
        raise ContractValidationError("scenarios must be a non-empty mapping")
    for scenario_id, scenario in scenarios.items():
        if not isinstance(scenario_id, str) or not isinstance(scenario, Scenario):
            raise ContractValidationError("paired scenarios must be Scenario objects keyed by ID")
        if scenario.scenario_id != scenario_id:
            raise ContractValidationError(f"scenario key does not match scenario_id: {scenario_id}")
        if scenario.split not in {"train", "held_out"}:
            raise ContractValidationError(f"unsupported scenario split: {scenario.split!r}")
    expected_ids = set(scenarios)
    by_config = {}
    for configuration, results in results_by_configuration.items():
        items = list(results)
        ids = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ContractValidationError(f"{configuration}[{index}] must be an object")
            record = item.get("record")
            if record is None or not hasattr(record, "scenario_id"):
                raise ContractValidationError(
                    f"{configuration}[{index}] is missing a ledger record"
                )
            scenario_id = record.scenario_id
            if not isinstance(scenario_id, str) or scenario_id not in expected_ids:
                raise ContractValidationError(
                    f"{configuration}[{index}] has an unexpected scenario ID: {scenario_id!r}"
                )
            observation = item.get("contract_observation")
            if not isinstance(observation, Mapping):
                raise ContractValidationError(
                    f"{configuration}[{index}] is missing contract_observation"
                )
            for field in ("valid", "claim_uncertain"):
                if type(observation.get(field)) is not bool:
                    raise ContractValidationError(
                        f"{configuration}[{index}].contract_observation.{field} "
                        "must be a boolean"
                    )
            if type(item.get("is_correct")) is not bool:
                raise ContractValidationError(
                    f"{configuration}[{index}].is_correct must be a boolean"
                )
            ids.append(scenario_id)
        if len(ids) != len(set(ids)) or set(ids) != expected_ids:
            raise ContractValidationError(f"{configuration}: scenario pairing mismatch")
        by_config[configuration] = {item["record"].scenario_id: item for item in items}
    splits = {}
    for split in ("train", "held_out"):
        split_ids = {sid for sid, scenario in scenarios.items() if scenario.split == split}
        splits[split] = {
            "scenario_ids": sorted(split_ids),
            "baseline_count": sum(sid in by_config["baseline"] for sid in split_ids),
            "hardened_count": sum(sid in by_config["hardened"] for sid in split_ids),
        }
        if splits[split]["baseline_count"] != len(split_ids) or splits[split]["hardened_count"] != len(split_ids):
            raise ContractValidationError(f"{split}: paired result count mismatch")
    def transition_report(ids: set[str]) -> Dict[str, Any]:
        transitions = {
        "failed_to_corrected": 0,
        "failed_to_still_failed": 0,
        "correct_to_regressed": 0,
        "correct_to_unchanged": 0,
        "invalid_pairs": 0,
        "inconclusive_pairs": 0,
        }
        for sid in ids:
            baseline_obs = by_config["baseline"][sid]["contract_observation"]
            hardened_obs = by_config["hardened"][sid]["contract_observation"]
            baseline_ok = (
                by_config["baseline"][sid]["is_correct"]
                and baseline_obs["valid"]
            )
            hardened_ok = (
                by_config["hardened"][sid]["is_correct"]
                and hardened_obs["valid"]
            )
            if not baseline_obs["valid"] or not hardened_obs["valid"]:
                transitions["invalid_pairs"] += 1
                continue
            if baseline_obs["claim_uncertain"] or hardened_obs["claim_uncertain"]:
                transitions["inconclusive_pairs"] += 1
                continue
            if not baseline_ok and hardened_ok:
                transitions["failed_to_corrected"] += 1
            elif not baseline_ok and not hardened_ok:
                transitions["failed_to_still_failed"] += 1
            elif baseline_ok and not hardened_ok:
                transitions["correct_to_regressed"] += 1
            else:
                transitions["correct_to_unchanged"] += 1
        transitions["total_paired_scenarios"] = len(ids)
        transitions["transition_denominator"] = sum(
            transitions[name]
            for name in (
                "failed_to_corrected",
                "failed_to_still_failed",
                "correct_to_regressed",
                "correct_to_unchanged",
            )
        )
        transitions["invalid_or_inconclusive_denominator"] = (
            transitions["invalid_pairs"] + transitions["inconclusive_pairs"]
        )
        eligible = transitions["transition_denominator"]
        transitions["correction_rate_among_baseline_failures"] = (
            transitions["failed_to_corrected"]
            / (transitions["failed_to_corrected"] + transitions["failed_to_still_failed"])
            if transitions["failed_to_corrected"] + transitions["failed_to_still_failed"]
            else None
        )
        transitions["baseline_failure_denominator"] = (
            transitions["failed_to_corrected"] + transitions["failed_to_still_failed"]
        )
        transitions["regression_rate_among_baseline_correct"] = (
            transitions["correct_to_regressed"]
            / (transitions["correct_to_regressed"] + transitions["correct_to_unchanged"])
            if transitions["correct_to_regressed"] + transitions["correct_to_unchanged"]
            else None
        )
        transitions["baseline_correct_denominator"] = (
            transitions["correct_to_regressed"] + transitions["correct_to_unchanged"]
        )
        transitions["paired_correctness_difference"] = (
            (transitions["failed_to_corrected"] - transitions["correct_to_regressed"]) / eligible
            if eligible
            else None
        )
        transitions["paired_correctness_difference_denominator"] = eligible
        transitions["mcnemar_exact_p_value"] = _exact_mcnemar_p_value(
            transitions["failed_to_corrected"],
            transitions["correct_to_regressed"],
        )
        transitions["denominator"] = len(ids)
        return transitions

    for split in splits:
        splits[split]["transitions"] = transition_report(set(splits[split]["scenario_ids"]))
    overall = transition_report(expected_ids)
    split_transition_totals = {
        name: sum(splits[split]["transitions"][name] for split in splits)
        for name in (
            "failed_to_corrected",
            "failed_to_still_failed",
            "correct_to_regressed",
            "correct_to_unchanged",
            "invalid_pairs",
            "inconclusive_pairs",
            "total_paired_scenarios",
            "transition_denominator",
        )
    }
    for name, value in split_transition_totals.items():
        if overall[name] != value:
            raise ContractValidationError(f"paired transition {name} does not reconcile across splits")
    return {"splits": splits, "transitions": overall}


def validate_aggregate_reconciliation(
    results: Iterable[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    observations: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Reconcile all detector, email, claim, and denominator metrics."""
    items = list(results)
    obs = list(observations)
    if len(items) != len(obs):
        raise ContractValidationError("raw results and contract observations differ in length")
    counts = aggregate.get("counts")
    if not isinstance(counts, Mapping):
        raise ContractValidationError("aggregate is missing counts")
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ContractValidationError(f"result[{index}] must be an object")
        if type(item.get("is_correct")) is not bool:
            raise ContractValidationError(f"result[{index}].is_correct must be a boolean")
        detection = item.get("detection")
        if detection is None or any(
            not hasattr(detection, name)
            for name in SUPPORTED_DETECTOR_ASSERTIONS
        ):
            raise ContractValidationError(
                f"result[{index}] is missing supported detector observations"
            )
    detector_names = sorted(SUPPORTED_DETECTOR_ASSERTIONS - {"claim_uncertain"})
    detector_counts = {
        name: sum(bool(item["detection"].__getattribute__(name)) for item in items)
        for name in detector_names
    }
    expected_counts = {
        "total": len(items),
        "correct": sum(bool(item["is_correct"]) for item in items),
        "claim_uncertain": sum(bool(o["claim_uncertain"]) for o in obs),
    }
    episode_counts = episode_accounting(items, obs)
    raw_episode_count = episode_counts["raw_episode_count"]
    aggregate_required = {
        "correct", "total", "claim_uncertain", *detector_counts, *episode_counts
    }
    missing = sorted(aggregate_required - set(counts))
    if missing:
        raise ContractValidationError(f"aggregate is missing detector fields: {missing}")
    for name, value in detector_counts.items():
        legacy_name = {
            "duplicate_action": "duplicate",
            "ineligible_refund_mutation": "ineligible_refund_mutation",
            "incorrect_refund_amount": "incorrect_refund_amount",
        }.get(name, name)
        if legacy_name not in counts:
            raise ContractValidationError(f"aggregate is missing detector fields: [{legacy_name!r}]")
        if counts[legacy_name] != value:
            raise ContractValidationError(f"aggregate count {legacy_name} does not reconcile")
    if counts["correct"] != expected_counts["correct"] or counts["total"] != expected_counts["total"]:
        raise ContractValidationError("aggregate correctness counts do not reconcile")
    if counts["claim_uncertain"] != expected_counts["claim_uncertain"]:
        raise ContractValidationError("aggregate claim uncertainty does not reconcile")
    for name, value in episode_counts.items():
        if counts[name] != value:
            raise ContractValidationError(f"aggregate episode count {name} does not reconcile")
    email = {
        "successful_mutation": sum(o["email_successful_mutation_count"] > 0 for o in obs),
        "tool_failure": sum(o["email_tool_failure"] for o in obs),
        "missing_required": sum(o["email_missing_required"] for o in obs),
        "excess": sum(o["email_excess"] for o in obs),
    }
    aggregate_email = {
        "email_successful_mutation": email["successful_mutation"],
        "email_tool_failure": email["tool_failure"],
        "email_missing_required": email["missing_required"],
        "email_excess": email["excess"],
    }
    missing_email = sorted(set(aggregate_email) - set(counts))
    if missing_email:
        raise ContractValidationError(f"aggregate is missing email fields: {missing_email}")
    for name, value in aggregate_email.items():
        if counts[name] != value:
            raise ContractValidationError(f"aggregate email count {name} does not reconcile")
    rate_values = {**detector_counts, **aggregate_email}
    interval_counts = {
        "correct": expected_counts["correct"],
        "valid_episode": episode_counts["valid_episode_count"],
        "invalid_episode": episode_counts["invalid_episode_count"],
        "inconclusive_episode": episode_counts["inconclusive_episode_count"],
        **rate_values,
    }
    intervals = {
        name: wilson_interval(value, raw_episode_count)
        for name, value in interval_counts.items()
    }
    return {
        "counts": {**expected_counts, **detector_counts, **aggregate_email, **episode_counts},
        "rates": {
            "correct": expected_counts["correct"] / raw_episode_count if raw_episode_count else 0.0,
            **{
                name: (value / raw_episode_count if raw_episode_count else 0.0)
                for name, value in rate_values.items()
            },
        },
        "rate_denominators": {
            "correct": raw_episode_count,
            **{name: raw_episode_count for name in rate_values},
        },
        "denominators": {
            "all_raw_episodes": raw_episode_count,
            "detectors": raw_episode_count,
            "email": raw_episode_count,
            "correctness": raw_episode_count,
        },
        "episode_counts": episode_counts,
        "wilson_intervals": intervals,
    }


def validate_split_reconciliation(
    overall: Mapping[str, Any],
    split_reports: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Require overall aggregate counts to equal train plus held-out counts."""
    if set(split_reports) != {"train", "held_out"}:
        raise ContractValidationError("split reports must contain train and held_out")
    overall_counts = overall.get("counts")
    if not isinstance(overall_counts, Mapping):
        raise ContractValidationError("overall report is missing counts")
    train_counts = split_reports["train"].get("counts")
    held_out_counts = split_reports["held_out"].get("counts")
    if not isinstance(train_counts, Mapping) or not isinstance(held_out_counts, Mapping):
        raise ContractValidationError("split report is missing counts")
    if set(train_counts) != set(overall_counts) or set(held_out_counts) != set(overall_counts):
        raise ContractValidationError(
            "split count keys do not match overall count keys"
        )
    for label, counts in (
        ("overall", overall_counts),
        ("train", train_counts),
        ("held_out", held_out_counts),
    ):
        for key, value in counts.items():
            if type(value) is not int or value < 0:
                raise ContractValidationError(
                    f"{label} count {key!r} must be a non-negative integer"
                )
    split_counts = {
        key: train_counts[key] + held_out_counts[key]
        for key in overall_counts
    }
    mismatches = {
        key: (overall_counts[key], split_counts[key])
        for key in overall_counts
        if overall_counts[key] != split_counts[key]
    }
    if mismatches:
        raise ContractValidationError(
            f"overall and split counts do not reconcile: {mismatches}"
        )
    return {
        "valid": True,
        "checked_count_keys": sorted(overall_counts),
    }


def episode_accounting(
    results: Iterable[Mapping[str, Any]],
    observations: Iterable[Mapping[str, Any]],
) -> Dict[str, int]:
    """Return mutually exclusive raw, valid, invalid, and inconclusive counts."""
    items = list(results)
    obs = list(observations)
    if len(items) != len(obs):
        raise ContractValidationError("raw results and contract observations differ in length")
    required_fields = {
        "valid",
        "claim_uncertain",
        "email_successful_mutation_count",
        "email_tool_failure",
        "email_missing_required",
        "email_excess",
    }
    for index, observation in enumerate(obs):
        if not isinstance(observation, Mapping):
            raise ContractValidationError(f"observation[{index}] must be an object")
        missing = required_fields - set(observation)
        if missing:
            raise ContractValidationError(
                f"observation[{index}] missing required fields: {sorted(missing)}"
            )
        for field in ("valid", "claim_uncertain", "email_tool_failure", "email_missing_required", "email_excess"):
            if type(observation[field]) is not bool:
                raise ContractValidationError(
                    f"observation[{index}].{field} must be a boolean"
                )
        email_count = observation["email_successful_mutation_count"]
        if type(email_count) is not int or email_count < 0:
            raise ContractValidationError(
                f"observation[{index}].email_successful_mutation_count must be a non-negative integer"
            )
    raw_episode_count = len(items)
    inconclusive_episode_count = sum(bool(o.get("claim_uncertain")) for o in obs)
    valid_episode_count = sum(
        bool(o.get("valid")) and not bool(o.get("claim_uncertain"))
        for o in obs
    )
    invalid_episode_count = raw_episode_count - valid_episode_count - inconclusive_episode_count
    if invalid_episode_count < 0:
        raise ContractValidationError("episode accounting counts are inconsistent")
    return {
        "raw_episode_count": raw_episode_count,
        "valid_episode_count": valid_episode_count,
        "invalid_episode_count": invalid_episode_count,
        "inconclusive_episode_count": inconclusive_episode_count,
        "invalid_or_inconclusive_episode_count": invalid_episode_count + inconclusive_episode_count,
        "execution_count": raw_episode_count,
        "scenario_count": len({item["record"].scenario_id for item in items}),
    }
