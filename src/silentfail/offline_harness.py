"""Strictly offline evaluation harness.

This module deliberately bypasses ``Orchestrator``.  It replays one reviewed
scenario manifest with the deterministic local agent under baseline and one
explicitly selected patch, writing only to a new run directory.
"""

from __future__ import annotations

import hashlib
import json
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.silentfail.agent import NaiveRefundAgent
from src.silentfail.contract_validation import (
    SUPPORTED_DETECTOR_ASSERTIONS,
    ContractValidationError,
    episode_accounting,
    validate_aggregate_reconciliation,
    validate_contract_manifest,
    validate_episode_contract,
    validate_paired_results,
    validate_split_reconciliation,
    validate_split_reconciliation,
)
from src.silentfail.models import PatchSpec, Scenario
from src.silentfail.runner import BenchmarkAggregator, EpisodeRunner
from src.silentfail.scenarios import ScenarioGenerator, ScenarioValidator
from src.silentfail.structural_validation import (
    StructuralValidationError,
    validate_structural_leakage,
)


MANIFEST_SCHEMA_VERSION = "offline-pilot-manifest-v1"
RUN_SCHEMA_VERSION = "offline-pilot-run-v1"
VALIDATION_MANIFEST_FILENAME = "offline-validation-v2.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_PATHS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "data" / "latest_pipeline_run.json",
    REPO_ROOT / "data" / "candidate_pipeline_run.json",
)

def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def patch_provenance(
    patch: Optional[PatchSpec],
    *,
    configuration: str,
    evaluation_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return deterministic provenance for one effective evaluation configuration."""
    if configuration not in {"baseline", "hardened"}:
        raise ValueError("configuration must be baseline or hardened")
    serialized = {
        "configuration": configuration,
        "patch": patch.model_dump(mode="json") if patch is not None else None,
        "evaluation_config": evaluation_config or {},
    }
    return {
        "serialization": serialized,
        "sha256": hashlib.sha256(_canonical_bytes(serialized)).hexdigest(),
    }


def patch_fingerprint(
    patch: Optional[PatchSpec],
    *,
    configuration: str,
    evaluation_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Return only the deterministic fingerprint for an effective configuration."""
    return patch_provenance(
        patch,
        configuration=configuration,
        evaluation_config=evaluation_config,
    )["sha256"]


def verify_expected_patch_fingerprint(
    expected: str,
    actual: Mapping[str, Any],
) -> None:
    """Fail closed when a supplied provenance fingerprint does not match."""
    if not isinstance(expected, str) or expected != actual.get("sha256"):
        raise ValueError("expected patch fingerprint does not match effective configuration")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    if path.name.casefold() == VALIDATION_MANIFEST_FILENAME.casefold():
        raise ValueError(
            f"{VALIDATION_MANIFEST_FILENAME} requires the mandatory pre-manifest validation gate"
        )
    path.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str).encode("utf-8") + b"\n")


def _scenario_dump(scenario: Scenario) -> Dict[str, Any]:
    return scenario.model_dump(mode="json")


def build_manifest(scenarios: Iterable[Scenario], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a versioned manifest from already reviewed scenarios."""
    items = [_scenario_dump(scenario) for scenario in scenarios]
    _validate_manifest({"schema_version": MANIFEST_SCHEMA_VERSION, "scenarios": items, "metadata": metadata or {}})
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "scenarios": items,
        "metadata": metadata or {},
    }


def generate_manifest(count: int, base_seed: int, split: str = "train", seed_offset: int = 0) -> Dict[str, Any]:
    """Generate and validate a deterministic manifest without running episodes."""
    if count < 1:
        raise ValueError("count must be positive")
    scenarios = ScenarioGenerator(base_seed=base_seed).generate_suite(
        count=count,
        split=split,
        seed_offset=seed_offset,
    )
    return build_manifest(
        scenarios,
        metadata={
            "generator": "ScenarioGenerator",
            "base_seed": base_seed,
            "split": split,
            "seed_offset": seed_offset,
            "count": count,
        },
    )


def _validate_scenario_fields(raw: Dict[str, Any], location: str) -> None:
    expected = {
        "scenario_id",
        "split",
        "seed",
        "category",
        "initial_state",
        "customer_message",
        "faults",
        "expected_outcome",
    }
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if missing:
        raise ValueError(f"{location} missing required fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{location} contains unsupported fields: {sorted(unknown)}")


def _validate_manifest(manifest: Dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema_version')!r}")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("manifest.scenarios must be a non-empty list")
    ids = []
    for index, raw in enumerate(scenarios):
        location = f"manifest.scenarios[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{location} must be an object")
        _validate_scenario_fields(raw, location)
        try:
            scenario = Scenario.model_validate(raw)
        except Exception as exc:
            raise ValueError(f"{location} is invalid: {exc}") from exc
        valid, message = ScenarioValidator.validate_and_compute_outcome(scenario)
        if not valid:
            raise ValueError(f"{location} failed scenario validation: {message}")
        ids.append(scenario.scenario_id)
    if len(ids) != len(set(ids)):
        raise ValueError("manifest contains duplicate scenario IDs")


def load_manifest(path: Path | str) -> Dict[str, Any]:
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load manifest {manifest_path}: {exc}") from exc
    _validate_manifest(manifest)
    return manifest


def save_manifest(path: Path | str, manifest: Dict[str, Any]) -> str:
    _validate_manifest(manifest)
    destination = Path(path)
    if destination.name.casefold() == VALIDATION_MANIFEST_FILENAME.casefold():
        raise ValueError(
            f"{VALIDATION_MANIFEST_FILENAME} requires the mandatory pre-manifest validation gate"
        )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_canonical_bytes(manifest) + b"\n")
    return _sha256_file(destination)


def _validate_statistical_gate_input(
    statistical_input: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if statistical_input is None:
        raise ContractValidationError(
            "statistical validation input is required for the v2 manifest"
        )
    if not isinstance(statistical_input, Mapping):
        raise ContractValidationError("statistical gate input must be an object")
    required = {"overall", "splits"}
    missing = required - set(statistical_input)
    if missing:
        raise ContractValidationError(
            f"statistical gate input missing required fields: {sorted(missing)}"
        )
    overall = statistical_input["overall"]
    splits = statistical_input["splits"]
    if not isinstance(overall, Mapping) or not isinstance(splits, Mapping):
        raise ContractValidationError("statistical gate reports must be objects")
    if set(splits) != {"train", "held_out"}:
        raise ContractValidationError(
            "statistical gate splits must contain exactly train and held_out"
        )
    required_report_fields = {
        "counts",
        "wilson_intervals",
        "transitions",
        "detector_metrics",
        "email_metrics",
    }
    required_transition_fields = {
        "failed_to_corrected",
        "failed_to_still_failed",
        "correct_to_regressed",
        "correct_to_unchanged",
        "invalid_pairs",
        "inconclusive_pairs",
        "total_paired_scenarios",
        "transition_denominator",
        "baseline_failure_denominator",
        "baseline_correct_denominator",
        "paired_correctness_difference_denominator",
    }

    def validate_report(report: Mapping[str, Any], label: str) -> None:
        missing_fields = required_report_fields - set(report)
        if missing_fields:
            raise ContractValidationError(
                f"{label} statistical report missing required fields: "
                f"{sorted(missing_fields)}"
            )
        counts = report["counts"]
        if not isinstance(counts, Mapping) or not counts:
            raise ContractValidationError(f"{label}.counts must be a non-empty object")
        for name, value in counts.items():
            if type(value) is not int or value < 0:
                raise ContractValidationError(
                    f"{label}.counts[{name!r}] must be a non-negative integer"
                )
        intervals = report["wilson_intervals"]
        if not isinstance(intervals, Mapping) or set(intervals) != set(counts):
            raise ContractValidationError(
                f"{label}.wilson_intervals must cover exactly the count metrics"
            )
        for name, interval in intervals.items():
            if not isinstance(interval, Mapping):
                raise ContractValidationError(
                    f"{label}.wilson_intervals[{name!r}] must be an object"
                )
            required_interval = {
                "numerator",
                "denominator",
                "point_estimate",
                "lower_95",
                "upper_95",
            }
            if set(interval) != required_interval:
                raise ContractValidationError(
                    f"{label}.wilson_intervals[{name!r}] has an invalid schema"
                )
            numerator = interval["numerator"]
            denominator = interval["denominator"]
            if (
                type(numerator) is not int
                or numerator < 0
                or type(denominator) is not int
                or denominator < 0
                or numerator > denominator
            ):
                raise ContractValidationError(
                    f"{label}.wilson_intervals[{name!r}] has invalid counts"
                )
            if numerator != counts[name]:
                raise ContractValidationError(
                    f"{label}.wilson_intervals[{name!r}] numerator does not match count"
                )
            if denominator == 0:
                if any(interval[field] is not None for field in (
                    "point_estimate", "lower_95", "upper_95"
                )):
                    raise ContractValidationError(
                        f"{label}.wilson_intervals[{name!r}] must be undefined at zero denominator"
                    )
            elif any(
                not isinstance(interval[field], (int, float))
                or isinstance(interval[field], bool)
                for field in ("point_estimate", "lower_95", "upper_95")
            ):
                raise ContractValidationError(
                    f"{label}.wilson_intervals[{name!r}] has invalid bounds"
                )
        transitions = report["transitions"]
        if not isinstance(transitions, Mapping) or not required_transition_fields.issubset(
            transitions
        ):
            raise ContractValidationError(
                f"{label}.transitions is incomplete"
            )
        for name in required_transition_fields:
            if type(transitions[name]) is not int or transitions[name] < 0:
                raise ContractValidationError(
                    f"{label}.transitions[{name!r}] must be a non-negative integer"
                )
        if transitions["transition_denominator"] != sum(
            transitions[name]
            for name in (
                "failed_to_corrected",
                "failed_to_still_failed",
                "correct_to_regressed",
                "correct_to_unchanged",
            )
        ):
            raise ContractValidationError(
                f"{label}.transitions transition denominator does not reconcile"
            )
        if transitions["total_paired_scenarios"] != (
            transitions["transition_denominator"]
            + transitions["invalid_pairs"]
            + transitions["inconclusive_pairs"]
        ):
            raise ContractValidationError(
                f"{label}.transitions total paired scenarios does not reconcile"
            )
        for metric_name in ("detector_metrics", "email_metrics"):
            metrics = report[metric_name]
            if not isinstance(metrics, Mapping) or not metrics:
                raise ContractValidationError(
                    f"{label}.{metric_name} must be a non-empty object"
                )
            for name, metric in metrics.items():
                if not isinstance(metric, Mapping) or set(metric) != {
                    "numerator", "denominator", "interval"
                }:
                    raise ContractValidationError(
                        f"{label}.{metric_name}[{name!r}] has an invalid schema"
                    )
                if (
                    type(metric["numerator"]) is not int
                    or metric["numerator"] < 0
                    or type(metric["denominator"]) is not int
                    or metric["denominator"] < 0
                    or metric["numerator"] > metric["denominator"]
                    or not isinstance(metric["interval"], Mapping)
                ):
                    raise ContractValidationError(
                        f"{label}.{metric_name}[{name!r}] has invalid counts"
                    )
                metric_interval = metric["interval"]
                if set(metric_interval) != {
                    "numerator",
                    "denominator",
                    "point_estimate",
                    "lower_95",
                    "upper_95",
                }:
                    raise ContractValidationError(
                        f"{label}.{metric_name}[{name!r}] interval is malformed"
                    )
                if (
                    metric_interval["numerator"] != metric["numerator"]
                    or type(metric_interval["denominator"]) is not int
                    or metric_interval["denominator"] != metric["denominator"]
                ):
                    raise ContractValidationError(
                        f"{label}.{metric_name}[{name!r}] interval counts do not reconcile"
                    )
                if metric["denominator"] == 0 and any(
                    metric_interval[field] is not None
                    for field in ("point_estimate", "lower_95", "upper_95")
                ):
                    raise ContractValidationError(
                        f"{label}.{metric_name}[{name!r}] zero denominator must be undefined"
                    )

    validate_report(overall, "overall")
    for split, report in splits.items():
        if not isinstance(report, Mapping):
            raise ContractValidationError(f"{split} statistical report must be an object")
        validate_report(report, split)
    for name in required_transition_fields:
        split_total = sum(splits[split]["transitions"][name] for split in splits)
        if overall["transitions"][name] != split_total:
            raise ContractValidationError(
                f"overall and split transitions do not reconcile for {name!r}"
            )
    reconciliation = validate_split_reconciliation(overall, splits)
    return {
        "valid": True,
        "errors": [],
        "provided": True,
        "split_reconciliation": reconciliation,
    }


def _require_mapping_report(
    report: Any,
    *,
    name: str,
    required_keys: set[str],
) -> Mapping[str, Any]:
    if not isinstance(report, Mapping):
        raise ValueError(f"pre-manifest {name} validator returned a non-object report")
    if type(report.get("valid")) is not bool:
        raise ValueError(f"pre-manifest {name} report has invalid valid field")
    if not isinstance(report.get("errors"), list):
        raise ValueError(f"pre-manifest {name} report has invalid errors field")
    missing = required_keys - set(report)
    if missing:
        raise ValueError(
            f"pre-manifest {name} report is incomplete: {sorted(missing)}"
        )
    if report["valid"] is not True:
        raise ValueError(
            f"pre-manifest {name} validation rejected manifest: {report['errors']}"
        )
    return report


def _mandatory_pre_manifest_validators() -> Dict[str, Any]:
    return {
        "structural": validate_structural_leakage,
        "contract": validate_contract_manifest,
        "statistical": _validate_statistical_gate_input,
    }


def validate_validation_manifest_gate(
    manifest: Mapping[str, Any],
    contract_manifest: Mapping[str, Any],
    *,
    overlap_exemptions: Optional[Mapping[str, Iterable[str]]] = None,
    statistical_input: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run every mandatory pre-manifest validator and fail closed."""
    if not isinstance(manifest, Mapping):
        raise ValueError("pre-manifest gate requires a manifest object")
    if not isinstance(contract_manifest, Mapping):
        raise ContractValidationError("pre-manifest gate requires a contract manifest")
    _validate_manifest(dict(manifest))
    scenarios = [Scenario.model_validate(raw) for raw in manifest["scenarios"]]
    if len(scenarios) != 65:
        raise ValueError(
            f"v2 manifest requires exactly 65 scenarios; found {len(scenarios)}"
        )
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("v2 manifest scenario IDs must be unique")
    split_counts = {
        "train": sum(scenario.split == "train" for scenario in scenarios),
        "held_out": sum(scenario.split == "held_out" for scenario in scenarios),
    }
    if split_counts != {"train": 39, "held_out": 26}:
        raise ValueError(
            "v2 manifest requires exactly 39 train and 26 held_out scenarios; "
            f"found {split_counts['train']} train and {split_counts['held_out']} held_out"
        )
    validator_map = _mandatory_pre_manifest_validators()
    required_validators = {"structural", "contract", "statistical"}
    missing_validators = sorted(required_validators - set(validator_map))
    if missing_validators:
        raise ValueError(
            f"pre-manifest gate missing validators: {missing_validators}"
        )
    if any(not callable(validator_map[name]) for name in required_validators):
        raise ValueError("pre-manifest gate validators must be callable")

    try:
        structural_report = validator_map["structural"](
            scenarios,
            contracts={
                str(contract["scenario_id"]): contract
                for contract in contract_manifest.get("contracts", [])
                if isinstance(contract, Mapping) and "scenario_id" in contract
            },
            overlap_exemptions=overlap_exemptions,
        )
    except Exception as exc:
        raise ValueError(f"pre-manifest structural validation failed: {exc}") from exc
    structural_report = _require_mapping_report(
        structural_report,
        name="structural",
        required_keys={
            "signatures",
            "exact_structural_duplicates",
            "message_template_reuse",
            "nearest_training",
        },
    )
    expected_ids = {scenario.scenario_id for scenario in scenarios}
    signatures = structural_report["signatures"]
    if not isinstance(signatures, Mapping) or set(signatures) != expected_ids:
        raise ValueError(
            "pre-manifest structural report does not cover exactly the written scenarios"
        )
    for scenario_id, signature in signatures.items():
        if not isinstance(signature, Mapping):
            raise ValueError(
                f"pre-manifest structural signature is malformed for {scenario_id}"
            )
        if (
            signature.get("scenario_id") != scenario_id
            or signature.get("split")
            not in {"train", "held_out"}
            or not isinstance(signature.get("components"), Mapping)
            or not isinstance(signature.get("sha256"), str)
        ):
            raise ValueError(
                f"pre-manifest structural signature is incomplete for {scenario_id}"
            )

    expected_train_count = 39
    expected_held_out_count = 26
    try:
        contract_report = validator_map["contract"](
            dict(manifest),
            contract_manifest,
            expected_train_count=expected_train_count,
            expected_held_out_count=expected_held_out_count,
        )
    except Exception as exc:
        raise ValueError(f"pre-manifest contract validation failed: {exc}") from exc
    if not isinstance(contract_report, Mapping) or set(contract_report) != expected_ids:
        raise ValueError("pre-manifest contract validation returned incomplete coverage")
    raw_contracts = contract_manifest.get("contracts")
    if not isinstance(raw_contracts, list):
        raise ValueError("pre-manifest contract manifest contracts must be a list")
    expected_contracts = {
        str(contract["scenario_id"]): contract
        for contract in raw_contracts
        if isinstance(contract, Mapping) and "scenario_id" in contract
    }
    if set(expected_contracts) != expected_ids or len(expected_contracts) != len(raw_contracts):
        raise ValueError(
            "pre-manifest contracts must cover exactly the written scenarios once"
        )
    for scenario_id, report in contract_report.items():
        if not isinstance(report, Mapping) or report != expected_contracts.get(scenario_id):
            raise ValueError(
                f"pre-manifest contract report does not match written contract for {scenario_id}"
            )

    try:
        statistical_report = validator_map["statistical"](statistical_input)
    except Exception as exc:
        raise ValueError(f"pre-manifest statistical validation failed: {exc}") from exc
    statistical_report = _require_mapping_report(
        statistical_report,
        name="statistical",
        required_keys={"provided", "split_reconciliation"},
    )
    if statistical_report["provided"] is not True:
        raise ValueError("pre-manifest statistical validation was not executed")
    if not isinstance(statistical_report["split_reconciliation"], Mapping):
        raise ValueError(
            "pre-manifest statistical report has malformed reconciliation"
        )
    return {
        "valid": True,
        "scenario_count": len(scenarios),
        "train_count": expected_train_count,
        "held_out_count": expected_held_out_count,
        "structural": structural_report,
        "contract_scenario_ids": sorted(contract_report),
        "statistical": statistical_report,
    }


def write_validation_manifest(
    path: Path | str,
    manifest: Mapping[str, Any],
    contract_manifest: Mapping[str, Any],
    *,
    overlap_exemptions: Optional[Mapping[str, Iterable[str]]] = None,
    statistical_input: Optional[Mapping[str, Any]] = None,
    writer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Validate and write the opt-in v2 manifest exactly once."""
    destination = Path(path)
    if destination.name.casefold() != VALIDATION_MANIFEST_FILENAME.casefold():
        raise ValueError(
            f"validation manifest must be named {VALIDATION_MANIFEST_FILENAME}"
        )
    gate_report = validate_validation_manifest_gate(
        manifest,
        contract_manifest,
        overlap_exemptions=overlap_exemptions,
        statistical_input=statistical_input,
    )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {destination}")
    payload = _canonical_bytes(dict(manifest)) + b"\n"
    if writer is None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    else:
        if not callable(writer):
            raise TypeError("manifest writer must be callable")
        writer(destination, payload)
    return {
        "path": str(destination),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "gate": gate_report,
    }


def _git_output(args: List[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _source_hashes() -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for path in sorted((REPO_ROOT / "src" / "silentfail").rglob("*.py")):
        hashes[str(path.relative_to(REPO_ROOT))] = _sha256_file(path)
    return hashes


def _reject_unsafe_output(path: Path) -> None:
    resolved = path.resolve(strict=False)
    for protected in PROTECTED_PATHS:
        protected_resolved = protected.resolve(strict=False)
        if resolved == protected_resolved or resolved in protected_resolved.parents or protected_resolved in resolved.parents:
            raise ValueError(f"output directory overlaps protected evidence: {path}")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {path}")
    if not resolved.name or resolved == resolved.parent:
        raise ValueError(f"unsafe output directory: {path}")


def _assert_offline_agent(agent: Any) -> None:
    if getattr(agent, "model_client", None) is not None:
        raise RuntimeError("offline harness refuses an agent with a live model client")
    if not callable(getattr(agent, "run", None)):
        raise TypeError("offline agent must provide a callable run method")


class OfflineAgentProvider:
    """Explicit provider for a preconstructed, locally controlled offline agent.

    The harness accepts this exact provider type only. It never accepts or
    invokes arbitrary factories. The provider is a boundary for this module,
    not a Python sandbox: callers remain responsible for supplying an agent
    whose ``run`` method is offline-safe.
    """

    def __init__(self, agent: Any):
        _assert_offline_agent(agent)
        self._agent = agent

    def get_agent(self) -> Any:
        return self._agent


def _default_agent_provider() -> OfflineAgentProvider:
    return OfflineAgentProvider(NaiveRefundAgent(model_client=None))


def _episode_payload(
    scenario: Scenario,
    configuration: str,
    patch: Optional[PatchSpec],
    result: Dict[str, Any],
    contract_observation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record = result["record"]
    detection = result["detection"]
    operation_ids = sorted({
        event.extra.get("operation_id")
        for event in record.events
        if event.extra and event.extra.get("operation_id")
    })
    payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "scenario_id": scenario.scenario_id,
        "scenario_seed": scenario.seed,
        "category": scenario.category,
        "configuration": configuration,
        "patch": patch.model_dump(mode="json") if patch else None,
        "customer_message": scenario.customer_message,
        "logical_operation_ids": operation_ids,
        "record": record.model_dump(mode="json"),
        "detection": detection.model_dump(mode="json"),
        "is_correct": result["is_correct"],
        "local_elapsed_seconds": result["elapsed_seconds"],
        "simulated_cost": record.cost.model_dump(mode="json"),
    }
    if contract_observation is not None:
        payload["contract_observation"] = contract_observation
    return payload


def _hash_output_files(destination: Path) -> Dict[str, str]:
    return {
        str(path.relative_to(destination)): _sha256_file(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "hashes.json"
    }


def _validate_output_hashes(destination: Path, hashes: Dict[str, Any]) -> None:
    if hashes.get("output_sha256") != _hash_output_files(destination):
        raise ValueError("output hash validation failed before publication")


def _add_contract_counts(
    aggregate: Dict[str, Any],
    results: List[Dict[str, Any]],
    observations: Optional[List[Dict[str, Any]]] = None,
) -> None:
    counts = aggregate["counts"]
    for name in SUPPORTED_DETECTOR_ASSERTIONS:
        if name == "claim_uncertain":
            counts[name] = sum(bool(result["detection"].claim_uncertain) for result in results)
        else:
            counts[name] = sum(bool(getattr(result["detection"], name)) for result in results)
    if observations is not None:
        counts.update(episode_accounting(results, observations))
        counts.update({
            "email_successful_mutation": sum(
                observation["email_successful_mutation_count"] > 0 for observation in observations
            ),
            "email_tool_failure": sum(observation["email_tool_failure"] for observation in observations),
            "email_missing_required": sum(observation["email_missing_required"] for observation in observations),
            "email_excess": sum(observation["email_excess"] for observation in observations),
        })


def _contract_aggregate(
    results: List[Dict[str, Any]],
    patch_id: str,
    observations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    aggregate = BenchmarkAggregator.aggregate(results, patch_id)
    _add_contract_counts(aggregate, results, observations)
    return aggregate


def run_offline_evaluation(
    manifest: Dict[str, Any] | Path | str,
    output_dir: Path | str,
    hardened_patch: PatchSpec,
    *,
    offline_only: bool = False,
    agent_providers: Optional[Dict[str, OfflineAgentProvider]] = None,
    contract_manifest: Optional[Dict[str, Any]] = None,
    expected_patch_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    """Run baseline and one explicit patch over exactly one manifest."""
    if not offline_only:
        raise RuntimeError("offline_only=True is required")
    if not isinstance(hardened_patch, PatchSpec):
        raise TypeError("hardened_patch must be a PatchSpec")

    loaded = load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
    _validate_manifest(loaded)
    manifest_fingerprint = hashlib.sha256(_canonical_bytes(loaded) + b"\n").hexdigest()
    contracts = (
        validate_contract_manifest(loaded, contract_manifest)
        if contract_manifest is not None
        else None
    )
    evaluation_config = {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "scenario_manifest_sha256": manifest_fingerprint,
        "contract_schema_version": (
            contract_manifest.get("contract_schema_version")
            if contract_manifest is not None
            else None
        ),
        "contract_manifest_sha256": (
            hashlib.sha256(_canonical_bytes(contract_manifest)).hexdigest()
            if contract_manifest is not None
            else None
        ),
        "offline_only": True,
    }
    hardened_provenance = patch_provenance(
        hardened_patch,
        configuration="hardened",
        evaluation_config=evaluation_config,
    )
    if expected_patch_fingerprint is not None:
        verify_expected_patch_fingerprint(expected_patch_fingerprint, hardened_provenance)
    destination = Path(output_dir)
    _reject_unsafe_output(destination)

    scenarios = [Scenario.model_validate(raw) for raw in loaded["scenarios"]]
    configurations = (("baseline", None), ("hardened", hardened_patch))
    providers = agent_providers or {
        "baseline": _default_agent_provider(),
        "hardened": _default_agent_provider(),
    }
    if set(providers) != {"baseline", "hardened"} or not all(
        type(provider) is OfflineAgentProvider for provider in providers.values()
    ):
        raise TypeError("agent_providers must contain exactly two OfflineAgentProvider instances")
    for provider in providers.values():
        _assert_offline_agent(provider.get_agent())

    parent = destination.parent.resolve(strict=False)
    parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent))
    try:
        (temp_path / "episodes").mkdir()
        all_payloads: Dict[str, List[Dict[str, Any]]] = {}
        aggregate_metrics: Dict[str, Dict[str, Any]] = {}
        raw_results: Dict[str, List[Dict[str, Any]]] = {}
        contract_observations: Dict[str, List[Dict[str, Any]]] = {}
        started = time.perf_counter()

        manifest_path = temp_path / "scenario_manifest.json"
        save_manifest(manifest_path, loaded)

        for configuration, patch in configurations:
            results = []
            payloads = []
            observations = []
            agent = providers[configuration].get_agent()
            _assert_offline_agent(agent)
            for scenario in scenarios:
                result = EpisodeRunner.run_episode(scenario, patch=patch, agent=agent)
                contract_observation = (
                    validate_episode_contract(
                        contracts[scenario.scenario_id],
                        scenario,
                        result,
                        configuration=configuration,
                    )
                    if contracts is not None
                    else None
                )
                if contract_observation is not None:
                    result = dict(result)
                    result["is_correct"] = bool(result["is_correct"]) and contract_observation["valid"]
                    observations.append(contract_observation)
                results.append(result)
                payload = _episode_payload(scenario, configuration, patch, result, contract_observation)
                payloads.append(payload)
                _write_json(temp_path / "episodes" / f"{configuration}__{scenario.scenario_id}.json", payload)
            all_payloads[configuration] = payloads
            raw_results[configuration] = results
            contract_observations[configuration] = observations
            aggregate_metrics[configuration] = BenchmarkAggregator.aggregate(results, configuration)
            if contracts is not None:
                _add_contract_counts(aggregate_metrics[configuration], results, observations)
                aggregate_metrics[configuration]["contract_validation"] = validate_aggregate_reconciliation(
                    results, aggregate_metrics[configuration], observations
                )
                aggregate_metrics[configuration]["contract_validation"]["splits"] = {}
                for split in ("train", "held_out"):
                    split_results = [
                        result for result in results
                        if next(
                            scenario.split
                            for scenario in scenarios
                            if scenario.scenario_id == result["record"].scenario_id
                        ) == split
                    ]
                    split_observations = [
                        observation
                        for result, observation in zip(results, observations)
                        if next(
                            scenario.split
                            for scenario in scenarios
                            if scenario.scenario_id == result["record"].scenario_id
                        ) == split
                    ]
                    aggregate_metrics[configuration]["contract_validation"]["splits"][split] = (
                        validate_aggregate_reconciliation(
                            split_results,
                            _contract_aggregate(split_results, configuration, split_observations),
                            split_observations,
                        )
                    )
                aggregate_metrics[configuration]["contract_validation"]["split_reconciliation"] = (
                    validate_split_reconciliation(
                        aggregate_metrics[configuration]["contract_validation"],
                        aggregate_metrics[configuration]["contract_validation"]["splits"],
                    )
                )
            aggregate_metrics[configuration]["patch_provenance"] = patch_provenance(
                patch,
                configuration=configuration,
                evaluation_config=evaluation_config,
            )
        if contracts is not None:
            pairing = validate_paired_results(
                {
                    configuration: [
                        {**result, "contract_observation": observation}
                        for result, observation in zip(raw_results[configuration], contract_observations[configuration])
                    ]
                    for configuration in ("baseline", "hardened")
                },
                {scenario.scenario_id: scenario for scenario in scenarios},
            )
        else:
            pairing = None

        config_manifest = {
            "schema_version": RUN_SCHEMA_VERSION,
            "mode": "local_simulation",
            "external_calls_allowed": False,
            "offline_only": True,
            "scenario_manifest_sha256": _sha256_file(manifest_path),
            "baseline": "deterministic local agent",
            "hardened_patch": hardened_patch.model_dump(mode="json"),
            "evaluation_config": evaluation_config,
            "patch_provenance": {
                "baseline": aggregate_metrics["baseline"]["patch_provenance"],
                "hardened": aggregate_metrics["hardened"]["patch_provenance"],
            },
        }
        if contract_manifest is not None:
            config_manifest["contract_schema_version"] = contract_manifest["contract_schema_version"]
            config_manifest["contract_pairing"] = pairing
        _write_json(temp_path / "run_configuration.json", config_manifest)
        _write_json(temp_path / "aggregate_metrics.json", aggregate_metrics)
        summary = {
            "schema_version": RUN_SCHEMA_VERSION,
            "mode": "local_simulation",
            "scenario_count": len(scenarios),
            "configurations": ["baseline", "hardened"],
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "note": "Local elapsed time and simulated costs are not provider latency or billing.",
        }
        _write_json(temp_path / "summary.json", summary)

        hashes = {
            "hash_procedure": {
                "scenario_manifest": "SHA-256 of UTF-8 canonical JSON with sorted keys, compact separators, ensure_ascii=false, plus one trailing newline",
                "configuration_and_outputs": "SHA-256 of exact bytes written to each file; hashes.json is excluded from output_sha256",
            },
            "git_commit": _git_output(["rev-parse", "HEAD"]),
            "git_status_short": _git_output(["status", "--short"]),
            "source_sha256": _source_hashes(),
            "scenario_manifest_sha256": _sha256_file(manifest_path),
            "configuration_sha256": _sha256_file(temp_path / "run_configuration.json"),
            "output_sha256": _hash_output_files(temp_path),
        }
        _validate_output_hashes(temp_path, hashes)
        _write_json(temp_path / "hashes.json", hashes)
        os.rename(temp_path, destination)
        return {
            "output_dir": str(destination.resolve()),
            "summary": summary,
            "aggregate_metrics": aggregate_metrics,
            "hashes": hashes,
            "episodes": all_payloads,
        }
    except Exception:
        if temp_path.exists():
            shutil.rmtree(temp_path)
        raise


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point requiring an explicit offline-only switch."""
    parser = argparse.ArgumentParser(description="Run a strictly offline SilentFail evaluation.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--patch-id", required=True)
    parser.add_argument("--patch-name", required=True)
    parser.add_argument("--patch-type", required=True, choices=[
        "wrapper.idempotency_key",
        "wrapper.read_back_verify",
        "wrapper.retry_policy",
        "combo.idempotency_and_read_back",
        "control.refuse_all",
        "control.blind_dedupe",
    ])
    parser.add_argument("--patch-description", required=True)
    args = parser.parse_args(argv)
    patch = PatchSpec(
        patch_id=args.patch_id,
        name=args.patch_name,
        type=args.patch_type,
        description=args.patch_description,
    )
    try:
        result = run_offline_evaluation(
            args.manifest,
            args.output_dir,
            patch,
            offline_only=args.offline_only,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"offline evaluation refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
