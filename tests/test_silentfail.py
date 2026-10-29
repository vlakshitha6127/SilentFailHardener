"""
Automated unit and integration test suite for SilentFail Hardener.
Verifies:
1. World state isolation and mutations
2. Tool operations and fault injection
3. Ledger causal timeline recording
4. Failure detectors (duplicate, false success, missed action, over-suppression)
5. Scenario generator distributions and schema validation
6. Candidate patches and guard enforcement
"""

import pytest
from src.silentfail.models import Scenario, ExpectedOutcome, FaultConfig
from src.silentfail.ledger import WorldState, CausalLedger
from src.silentfail.tools import ToolRegistry, FaultInjector
from src.silentfail.agent import NaiveRefundAgent
from src.silentfail.detectors import FailureDetector, ClaimExtractor
from src.silentfail.scenarios import ScenarioGenerator
from src.silentfail.patches import PatchedToolRegistry, get_default_candidate_patches
from src.silentfail.runner import EpisodeRunner, BenchmarkAggregator, VerifierAndSelector


def test_world_state_and_tools():
    initial = {
        "orders": [{"id": "101", "status": "delivered", "total": 50.0, "refunded_amount": 0.0}]
    }
    ws = WorldState(initial)
    ledger = CausalLedger(scenario_id="sc_test_1")
    tools = ToolRegistry(world_state=ws, ledger=ledger)

    # get_order
    order = tools.get_order("101")
    assert order["id"] == "101"
    assert order["total"] == 50.0

    # issue_refund
    res = tools.issue_refund("101", 30.0)
    assert res["status"] == "ok"
    assert ws.orders["101"].refunded_amount == 30.0
    assert len(ws.refunds) == 1

    # Over-refund check
    res2 = tools.issue_refund("101", 30.0)
    assert "error" in res2
    assert ws.orders["101"].refunded_amount == 30.0


def test_fault_injection_timeout_after_success():
    """Verify that timeout_after_success mutates world state but returns timeout error."""
    initial = {
        "orders": [{"id": "102", "status": "delivered", "total": 40.0, "refunded_amount": 0.0}]
    }
    ws = WorldState(initial)
    ledger = CausalLedger(scenario_id="sc_test_2")
    faults = [FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success")]
    injector = FaultInjector(faults)
    tools = ToolRegistry(world_state=ws, ledger=ledger, fault_injector=injector)

    res = tools.issue_refund("102", 40.0)
    # Agent observes timeout error
    assert res.get("error") == "timeout"
    # Ground truth world state was mutated!
    assert ws.orders["102"].refunded_amount == 40.0
    assert len(ws.refunds) == 1

    # Verify ledger captures both
    mutation_events = [e for e in ledger.events if e.type == "world_mutation"]
    response_events = [e for e in ledger.events if e.type == "tool_response_to_agent"]
    assert len(mutation_events) == 1
    assert len(response_events) == 1
    assert response_events[0].observed.get("error") == "timeout"


def test_generated_legitimate_repeat_scenarios_seed_two_items_matching_expected_total():
    scenarios = ScenarioGenerator(base_seed=1234).generate_suite(50, split="train")
    repeats = [scenario for scenario in scenarios if scenario.category == "legitimate_repeat"]

    assert {scenario.scenario_id for scenario in repeats} == {
        "sc_train_00008", "sc_train_00011", "sc_train_00013",
        "sc_train_00037", "sc_train_00047", "sc_train_00049",
    }
    for scenario in repeats:
        order = scenario.initial_state["orders"][0]
        assert [item["item_id"] for item in order["items"]] == ["item_1", "item_2"]
        assert round(sum(item["price"] for item in order["items"]), 2) == order["total"]
        assert scenario.expected_outcome.refund_count == 2
        assert scenario.expected_outcome.refund_total == order["total"]


def test_gate_g1_agent_uses_operation_id_to_prevent_duplicate_refund():
    """
    Gate G1 check:
    Verify that a timeout retry with the same operation_id produces one refund
    and does not claim a duplicate side effect.
    """
    sc = Scenario(
        scenario_id="sc_g1_test",
        split="train",
        category="timeout_after_success",
        initial_state={"orders": [{"id": "2001", "status": "delivered", "total": 100.0, "refunded_amount": 0.0}]},
        customer_message="Please refund $50 for order 2001",
        faults=[FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success")],
        expected_outcome=ExpectedOutcome(refund_count=1, refund_total=50.0, email_count=1)
    )

    result = EpisodeRunner.run_episode(sc)
    record = result["record"]
    detection = result["detection"]

    assert detection.duplicate_action is False
    assert record.final_state["refund_count"] == 1
    assert detection.silent_failure_false_success is False
    assert result["is_correct"] is True


def test_claim_extractor_and_crosscheck():
    msg1 = "Your refund of $49.99 has been issued."
    ext1 = ClaimExtractor.extract_rule_based(msg1)
    assert ext1["claims_refund_issued"] is True
    assert ext1["claimed_amount"] == 49.99

    msg2 = "Order 1042 has already been fully refunded. No additional refund can be issued."
    ext2 = ClaimExtractor.extract_rule_based(msg2)
    assert ext2["claims_refund_issued"] is False
    assert ext2["is_refusal"] is True


def test_verifier_guards_reject_controls():
    """
    Verify that verifier guards strictly reject:
    1. control.refuse_all (fails regression guard)
    2. control.blind_dedupe (fails legitimate-completion guard)
    """
    gen = ScenarioGenerator(base_seed=999)
    scenarios = gen.generate_suite(30, split="train")

    base_results = [EpisodeRunner.run_episode(sc) for sc in scenarios]
    base_metrics = BenchmarkAggregator.aggregate(base_results, "baseline")

    candidates = get_default_candidate_patches()
    cand_metrics = {}
    for cand in candidates:
        res = [EpisodeRunner.run_episode(sc, patch=cand) for sc in scenarios]
        cand_metrics[cand.patch_id] = BenchmarkAggregator.aggregate(res, cand.patch_id)

    verifier = VerifierAndSelector(tolerance=0.02)
    winner, race = verifier.select_winner(base_metrics, cand_metrics, candidates)

    assert winner is not None
    assert winner.is_control is False
    # Check that refuse_all was rejected
    assert race["control_refuse_all"]["eligible"] is False
    # Check that blind_dedupe was rejected
    assert race["control_blind_dedupe"]["eligible"] is False
