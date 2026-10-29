"""Deterministic structural validation for offline scenario cohorts."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from src.silentfail.models import Scenario


STRUCTURAL_DIMENSIONS = (
    "category",
    "message_template",
    "fault_sequence",
    "operation_structure",
    "initial_state_shape",
    "expected_outcome_shape",
    "contract_assertion_structure",
)
VALID_SPLITS = {"train", "held_out"}


class StructuralValidationError(ValueError):
    """Raised when a scenario cohort cannot be safely compared."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _normalize_message(message: Any) -> str:
    if not isinstance(message, str) or not message:
        raise StructuralValidationError("customer_message must be a non-empty string")
    normalized = message
    normalized = re.sub(r"\$\s*\d+(?:\.\d{1,2})?", "<AMOUNT>", normalized)
    normalized = re.sub(
        r"\b(?:order\s+)?(?:ORD[-_]?)?\d{3,}\b",
        lambda match: re.sub(r"\d", "<ID>", match.group(0)),
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\b\d+\b", "<NUM>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _normalize_identifier(value: Any, prefix: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructuralValidationError(f"{prefix} must be a non-empty string")
    return re.sub(r"\d+", "<ID>", value.strip().lower())


def _amount_pattern(amount: float, total: float, count: int) -> str:
    if count == 0 or amount == 0:
        return "none"
    if total <= 0:
        raise StructuralValidationError("positive operation amount requires positive order total")
    ratio = amount / total
    if abs(ratio - 1.0) <= 0.01:
        return "full"
    if 0 < ratio < 1:
        return "partial"
    return "excess"


def _scenario_value(scenario: Scenario | Mapping[str, Any]) -> Scenario:
    try:
        return scenario if isinstance(scenario, Scenario) else Scenario.model_validate(scenario)
    except Exception as exc:
        raise StructuralValidationError(f"malformed scenario: {exc}") from exc


def _contract_shape(contract: Mapping[str, Any] | None) -> dict[str, Any]:
    if contract is None:
        return {"present": False}
    if not isinstance(contract, Mapping):
        raise StructuralValidationError("contract must be an object")
    requested = contract.get("requested_operation", {})
    if not isinstance(requested, Mapping):
        raise StructuralValidationError("contract.requested_operation must be an object")
    operations = requested.get("operations", [])
    if operations is not None and not isinstance(operations, list):
        raise StructuralValidationError("contract requested operations must be a list")
    operation_shape = []
    for operation in operations or []:
        if not isinstance(operation, Mapping):
            raise StructuralValidationError("contract operation must be an object")
        operation_shape.append({
            "kind": operation.get("kind"),
            "amount_pattern": operation.get("amount_pattern", "declared"),
            "expected_mutation_count": operation.get("expected_mutation_count"),
        })
    retry_shape = []
    for group in requested.get("retry_groups", []) or []:
        if not isinstance(group, Mapping):
            raise StructuralValidationError("contract retry group must be an object")
        retry_shape.append({
            "min_attempts": group.get("min_attempts"),
            "max_attempts": group.get("max_attempts"),
            "require_same_operation_id": group.get("require_same_operation_id"),
        })
    detectors = contract.get("detectors", {})
    by_config = contract.get("detectors_by_configuration", {})
    if not isinstance(detectors, Mapping) or not isinstance(by_config, Mapping):
        raise StructuralValidationError("contract detector assertions must be objects")
    return {
        "present": True,
        "requested_kind": requested.get("kind"),
        "requested_amount": requested.get("amount"),
        "operation_count": requested.get("operation_count"),
        "operations": operation_shape,
        "retry_groups": retry_shape,
        "detectors": sorted((str(k), bool(v)) for k, v in detectors.items()),
        "detectors_by_configuration": {
            str(configuration): sorted((str(k), bool(v)) for k, v in values.items())
            for configuration, values in sorted(by_config.items())
            if isinstance(values, Mapping)
        },
        "refund_shape": {
            "expected_count": contract.get("refund", {}).get("expected_count")
            if isinstance(contract.get("refund"), Mapping)
            else None,
            "expected_total": contract.get("refund", {}).get("expected_total")
            if isinstance(contract.get("refund"), Mapping)
            else None,
            "amount_tolerance": contract.get("refund", {}).get("amount_tolerance")
            if isinstance(contract.get("refund"), Mapping)
            else None,
        },
        "refusal_mode": contract.get("refusal", {}).get("mode")
        if isinstance(contract.get("refusal"), Mapping)
        else None,
        "claim_kind": contract.get("completion_claim", {}).get("kind")
        if isinstance(contract.get("completion_claim"), Mapping)
        else None,
        "email_shape": {
            "required_count": contract.get("email", {}).get("required_count")
            if isinstance(contract.get("email"), Mapping)
            else None,
            "allowed_count": contract.get("email", {}).get("allowed_count")
            if isinstance(contract.get("email"), Mapping)
            else None,
        },
        "completion_claim_shape": {
            "amount": contract.get("completion_claim", {}).get("amount")
            if isinstance(contract.get("completion_claim"), Mapping)
            else None,
            "amount_tolerance": contract.get("completion_claim", {}).get(
                "amount_tolerance"
            )
            if isinstance(contract.get("completion_claim"), Mapping)
            else None,
        },
        "fault_observations": contract.get("fault_observations"),
    }


def scenario_structural_signature(
    scenario: Scenario | Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return component signatures and a canonical SHA-256 structural hash."""
    value = _scenario_value(scenario)
    orders = value.initial_state.get("orders")
    if not isinstance(orders, list) or not orders:
        raise StructuralValidationError("initial_state.orders must be a non-empty list")
    if any(not isinstance(order, Mapping) for order in orders):
        raise StructuralValidationError("initial_state.orders entries must be objects")
    first_order = orders[0]
    total = first_order.get("total")
    try:
        total = float(total)
    except (TypeError, ValueError) as exc:
        raise StructuralValidationError("initial order total must be numeric") from exc
    faults = []
    for fault in value.faults:
        faults.append({
            "tool": fault.tool,
            "type": fault.type,
            "call_index": fault.call_index,
            "effect_applied": fault.effect_applied,
        })
    operation = {
        "requested_kind": "full_refund"
        if value.expected_outcome.refund_total == total and total > 0
        else "partial_refund",
        "operation_count": value.expected_outcome.refund_count,
        "amount_pattern": _amount_pattern(
            value.expected_outcome.refund_total,
            total,
            value.expected_outcome.refund_count,
        ),
    }
    if contract is not None and isinstance(contract.get("requested_operation"), Mapping):
        requested = contract["requested_operation"]
        operation["declared_operations"] = [
            {
                "kind": item.get("kind"),
                "expected_mutation_count": item.get("expected_mutation_count"),
                "amount_pattern": item.get("amount_pattern", "declared"),
            }
            for item in requested.get("operations", []) or []
            if isinstance(item, Mapping)
        ]
        operation["retry_groups"] = [
            {
                "min_attempts": item.get("min_attempts"),
                "max_attempts": item.get("max_attempts"),
                "require_same_operation_id": item.get("require_same_operation_id"),
            }
            for item in requested.get("retry_groups", []) or []
            if isinstance(item, Mapping)
        ]
    if contract is not None:
        requested = contract.get("requested_operation", {})
        if isinstance(requested, Mapping):
            operation["requested_kind"] = requested.get("kind", operation["requested_kind"])
            operation["operation_count"] = requested.get(
                "operation_count", operation["operation_count"]
            )
            if "amount" in requested:
                operation["amount_pattern"] = _amount_pattern(
                    float(requested["amount"]),
                    total,
                    int(requested["operation_count"]),
                )
    initial_shape = {
        "order_count": len(orders),
        "orders": [
            {
                "status": order.get("status"),
                "item_count": len(order.get("items", []))
                if isinstance(order.get("items", []), list)
                else None,
                "refund_history_count": len(value.initial_state.get("refunds", []))
                if isinstance(value.initial_state.get("refunds", []), list)
                else None,
            }
            for order in orders
        ],
        "existing_email_count": len(value.initial_state.get("emails", []))
        if isinstance(value.initial_state.get("emails", []), list)
        else None,
        "existing_ticket_count": len(value.initial_state.get("tickets", []))
        if isinstance(value.initial_state.get("tickets", []), list)
        else None,
    }
    components = {
        "category": value.category,
        "message_template": _normalize_message(value.customer_message),
        "fault_sequence": faults,
        "operation_structure": operation,
        "initial_state_shape": initial_shape,
        "expected_outcome_shape": {
            "refund_count": value.expected_outcome.refund_count,
            "refund_amount_pattern": _amount_pattern(
                value.expected_outcome.refund_total,
                total,
                value.expected_outcome.refund_count,
            ),
            "email_count": value.expected_outcome.email_count,
            "ticket_count": value.expected_outcome.ticket_count,
            "allow_refusal": value.expected_outcome.allow_refusal,
        },
        "contract_assertion_structure": _contract_shape(contract),
    }
    return {
        "scenario_id": value.scenario_id,
        "normalized_scenario_id": _normalize_identifier(value.scenario_id, "scenario_id"),
        "split": value.split,
        "components": components,
        "sha256": _hash(components),
    }


def validate_structural_leakage(
    scenarios: Sequence[Scenario | Mapping[str, Any]],
    contracts: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    overlap_exemptions: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Fail closed on train/held-out structural leakage and return a review report."""
    if not isinstance(scenarios, Sequence) or isinstance(scenarios, (str, bytes)):
        raise StructuralValidationError("scenarios must be a sequence")
    parsed = [_scenario_value(item) for item in scenarios]
    ids = [item.scenario_id for item in parsed]
    if len(ids) != len(set(ids)):
        raise StructuralValidationError("duplicate scenario IDs")
    if any(item.split not in VALID_SPLITS for item in parsed):
        raise StructuralValidationError("scenario split must be train or held_out")
    contracts = contracts or {}
    exemptions: dict[str, set[str]] = {}
    for category, dimensions in (overlap_exemptions or {}).items():
        if not isinstance(category, str) or not category.strip():
            raise StructuralValidationError("overlap exemption category must be non-empty")
        if isinstance(dimensions, (str, bytes)):
            raise StructuralValidationError(
                f"overlap exemption for {category!r} must be a sequence of dimensions"
            )
        normalized_dimensions = set(dimensions)
        unknown_dimensions = normalized_dimensions - set(STRUCTURAL_DIMENSIONS)
        if unknown_dimensions:
            raise StructuralValidationError(
                f"overlap exemption for {category!r} contains unsupported dimensions: "
                f"{sorted(unknown_dimensions)}"
            )
        exemptions[category] = normalized_dimensions
    signatures = {
        item.scenario_id: scenario_structural_signature(
            item, contracts.get(item.scenario_id)
        )
        for item in parsed
    }
    train = [item for item in parsed if item.split == "train"]
    held_out = [item for item in parsed if item.split == "held_out"]
    exact_duplicates = []
    message_reuse = []
    nearest = []
    errors = []
    for held in sorted(held_out, key=lambda item: item.scenario_id):
        held_components = signatures[held.scenario_id]["components"]
        candidates = []
        for candidate in train:
            candidate_components = signatures[candidate.scenario_id]["components"]
            matching = [
                dimension
                for dimension in STRUCTURAL_DIMENSIONS
                if held_components[dimension] == candidate_components[dimension]
            ]
            differing = [
                dimension for dimension in STRUCTURAL_DIMENSIONS if dimension not in matching
            ]
            effective_differing = [
                dimension
                for dimension in differing
                if dimension not in exemptions.get(held.category, set())
            ]
            candidates.append((len(effective_differing), candidate.scenario_id, matching, differing))
            if held_components == candidate_components:
                exact_duplicates.append((held.scenario_id, candidate.scenario_id))
            if held_components["message_template"] == candidate_components["message_template"]:
                message_reuse.append((held.scenario_id, candidate.scenario_id))
        if candidates:
            difference_count, candidate_id, matching, differing = min(
                candidates, key=lambda item: (item[0], item[1])
            )
            nearest.append({
                "held_out_scenario_id": held.scenario_id,
                "training_scenario_id": candidate_id,
                "matching_dimensions": matching,
                "differing_dimensions": differing,
                "independent_difference_count": difference_count,
            })
            if difference_count < 2:
                errors.append(
                    f"{held.scenario_id} has fewer than two independent structural differences "
                    f"from nearest training scenario {candidate_id}"
                )
    if exact_duplicates:
        errors.append(f"exact structural duplicates across splits: {sorted(exact_duplicates)}")
    unexpected_message_reuse = [
        pair for pair in message_reuse
        if "message_template" not in exemptions.get(
            next(item.category for item in held_out if item.scenario_id == pair[0]),
            set(),
        )
    ]
    if unexpected_message_reuse:
        errors.append(
            "normalized message-template reuse across splits: "
            f"{sorted(unexpected_message_reuse)}"
        )
    return {
        "valid": not errors,
        "errors": sorted(errors),
        "signatures": {
            scenario_id: signatures[scenario_id]
            for scenario_id in sorted(signatures)
        },
        "exact_structural_duplicates": sorted(exact_duplicates),
        "message_template_reuse": sorted(message_reuse),
        "nearest_training": nearest,
        "overlap_exemptions": {
            key: sorted(value) for key, value in sorted(exemptions.items())
        },
    }
