"""
Tests for Nemotron model-driven patch proposal integration:
1. Successful model proposals (JSON array parsed and validated)
2. Malformed JSON handling (captured in parse_error, fallback returned)
3. Unavailable model or API error (captured in error, fallback returned)
4. Fallback labeling and metadata reporting
5. Rejection of invalid patch types and fields
6. Preservation of deterministic guards (model output enters evaluation, doesn't decide pass/fail)
7. Exclusion of credentials from metadata and error reports
8. Specific refund intent and duplicate side-effect demonstration tests
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from src.silentfail.nebius import NebiusClient, ALLOWED_PATCH_TYPES, PATCH_PROPOSAL_SYSTEM_PROMPT
from src.silentfail.models import PatchSpec
from src.silentfail.orchestrator import Orchestrator
from src.silentfail.runner import EpisodeRunner, BenchmarkAggregator, VerifierAndSelector


class MockMessage:
    def __init__(self, content: str):
        self.content = content


class MockChoice:
    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockCompletionResponse:
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]


def test_fallback_labeling_when_offline():
    client = NebiusClient(api_key="")
    assert not client.is_live
    candidates, metadata = client.diagnose_and_propose_patches([{"dummy": "ledger"}])

    assert metadata["source"] == "deterministic_fallback"
    assert metadata["model_outcome"] == "not_attempted"
    assert metadata["model"] is None
    assert metadata["error"] is None
    assert metadata["parse_error"] is None
    assert metadata["fallback_candidates_count"] >= 4
    assert metadata["raw_proposals_count"] == 0
    assert metadata["accepted_candidates_count"] == 0
    assert len(candidates) >= 4
    assert all(isinstance(c, PatchSpec) for c in candidates)


def test_successful_model_proposals_parsed_and_labeled():
    client = NebiusClient(api_key="mock_key")
    client.is_live = True
    client.client = MagicMock()

    model_proposed_json = json.dumps([
        {
            "patch_id": "patch_model_idemp",
            "name": "Model Idempotency",
            "type": "wrapper.idempotency_key",
            "description": "Model-proposed idempotency wrapper",
            "is_control": False,
            "config": {}
        },
        {
            "patch_id": "patch_model_readback",
            "name": "Model Readback",
            "type": "wrapper.read_back_verify",
            "description": "Model-proposed read-back wrapper",
            "is_control": False,
            "config": {}
        }
    ])
    client.client.chat.completions.create.return_value = MockCompletionResponse(model_proposed_json)

    candidates, metadata = client.diagnose_and_propose_patches([{"seq": 1, "fault": "empty_200"}])

    assert metadata["source"] == "nemotron_ultra"
    assert metadata["model_outcome"] == "accepted"
    assert metadata["model"] == client.ultra_model
    assert metadata["raw_proposals_count"] == 2
    assert metadata["accepted_candidates_count"] == 2
    assert metadata["fallback_candidates_count"] == 0
    assert metadata["error"] is None
    assert metadata["parse_error"] is None
    assert len(metadata["rejected_proposals"]) == 0

    candidate_ids = {c.patch_id for c in candidates}
    assert "patch_model_idemp" in candidate_ids
    assert "patch_model_readback" in candidate_ids
    # Verifies controls were automatically retained for guard evaluation
    assert "control_refuse_all" in candidate_ids
    assert "control_blind_dedupe" in candidate_ids


def test_malformed_json_handling():
    client = NebiusClient(api_key="mock_key")
    client.is_live = True
    client.client = MagicMock()

    client.client.chat.completions.create.return_value = MockCompletionResponse("Here are your patches: [{not valid json]]")

    candidates, metadata = client.diagnose_and_propose_patches([{"seq": 1}])

    assert metadata["source"] == "deterministic_fallback"
    assert metadata["model_outcome"] == "parse_error"
    assert metadata["parse_error"] is not None
    assert "JSON parse error" in metadata["parse_error"]
    assert metadata["fallback_candidates_count"] >= 4
    assert len(candidates) >= 4


def test_model_api_error_handling():
    client = NebiusClient(api_key="mock_key")
    client.is_live = True
    client.client = MagicMock()

    client.client.chat.completions.create.side_effect = ConnectionError("Remote connection refused 503")

    candidates, metadata = client.diagnose_and_propose_patches([{"seq": 1}])

    assert metadata["source"] == "deterministic_fallback"
    assert metadata["model_outcome"] == "request_error"
    assert metadata["error"] is not None
    assert "ConnectionError" in metadata["error"]
    assert "mock_key" not in metadata["error"]
    assert metadata["fallback_candidates_count"] >= 4
    assert len(candidates) >= 4


def test_invalid_patch_type_rejection():
    client = NebiusClient(api_key="mock_key")
    client.is_live = True
    client.client = MagicMock()

    # Contains one invalid patch type and one valid
    model_proposed_json = json.dumps([
        {
            "patch_id": "patch_invalid",
            "name": "Invalid Patch",
            "type": "unsupported_magic_fix",
            "description": "Invalid type",
            "is_control": False
        },
        {
            "patch_id": "patch_valid",
            "name": "Valid Retry",
            "type": "wrapper.retry_policy",
            "description": "Valid retry policy",
            "is_control": False
        }
    ])
    client.client.chat.completions.create.return_value = MockCompletionResponse(model_proposed_json)

    candidates, metadata = client.diagnose_and_propose_patches([{"seq": 1}])

    assert metadata["source"] == "nemotron_ultra"
    assert metadata["model_outcome"] == "accepted"
    assert metadata["raw_proposals_count"] == 2
    assert metadata["accepted_candidates_count"] == 1
    assert metadata["fallback_candidates_count"] == 0
    assert len(metadata["rejected_proposals"]) == 1
    assert metadata["rejected_proposals"][0]["patch_id"] == "patch_invalid"
    assert "unsupported_magic_fix" in metadata["rejected_proposals"][0]["reason"]

    candidate_ids = {c.patch_id for c in candidates}
    assert "patch_valid" in candidate_ids
    assert "patch_invalid" not in candidate_ids


def test_pipeline_records_proposal_metadata_and_evaluates_guards(tmp_path):
    orch = Orchestrator(data_dir=str(tmp_path / "data"), tests_dir=str(tmp_path / "tests"))
    
    # Mock NebiusClient so diagnose_and_propose_patches returns model-proposed candidates
    fake_proposed_patch = PatchSpec(
        patch_id="patch_model_idemp",
        name="Model Idempotency",
        type="wrapper.idempotency_key",
        description="Model proposed",
        is_control=False
    )
    fake_control = PatchSpec(
        patch_id="control_refuse_all",
        name="Control Refuse",
        type="control.refuse_all",
        description="Control",
        is_control=True
    )

    orch.nebius_client.diagnose_and_propose_patches = MagicMock(return_value=(
        [fake_proposed_patch, fake_control],
        {
            "source": "nemotron_ultra",
            "model": "nvidia/Nemotron-3-Ultra-550b-a55b",
            "raw_proposals_count": 1,
            "accepted_candidates_count": 1,
            "fallback_candidates_count": 0,
            "rejected_proposals": [],
            "error": None,
            "parse_error": None
        }
    ))

    result = orch.run_full_pipeline(train_count=10, held_out_count=6)

    # 1. Gate G0 reported
    assert "gate_g0" in result
    # 2. Proposal metadata reported
    assert result["proposal_metadata"]["source"] == "nemotron_ultra"
    assert result["proposal_metadata"]["model"] == "nvidia/Nemotron-3-Ultra-550b-a55b"
    assert result["proposal_metadata"]["raw_proposals_count"] == 1
    assert result["proposal_metadata"]["accepted_candidates_count"] == 1
    # 3. Guard rejection evaluated (control_refuse_all rejected by regression guard)
    assert "control_refuse_all" in result["race_summary"]
    assert result["race_summary"]["control_refuse_all"]["eligible"] is False
    assert len(result["race_summary"]["control_refuse_all"]["rejection_reasons"]) > 0
    # 4. Winner selected by deterministic guard evaluation, not by model
    assert result["winning_patch"] is not None
    assert result["winning_patch"]["patch_id"] == "patch_model_idemp"
    # 5. Held-out metrics evaluated
    assert "heldout_metrics_before" in result
    assert "heldout_metrics_after" in result
    assert result["heldout_metrics_after"]["counts"]["total"] == 6


def test_system_prompt_includes_all_allowed_patch_types_and_examples():
    """Verify system prompt accurately contains all 6 allowed patch types and strict contract rules."""
    for patch_type in ALLOWED_PATCH_TYPES:
        assert patch_type in PATCH_PROPOSAL_SYSTEM_PROMPT
    assert "MINIMAL VALID EXAMPLE:" in PATCH_PROPOSAL_SYSTEM_PROMPT
    assert "wrapper.idempotency_key" in PATCH_PROPOSAL_SYSTEM_PROMPT
    assert "PROHIBITED:" in PATCH_PROPOSAL_SYSTEM_PROMPT
    assert "tool_contract" in PATCH_PROPOSAL_SYSTEM_PROMPT
    assert "empty JSON array: []" in PATCH_PROPOSAL_SYSTEM_PROMPT


def test_prompt_sent_to_model_contains_system_prompt():
    client = NebiusClient(api_key="mock_key")
    client.is_live = True
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = MockCompletionResponse("[]")

    client.diagnose_and_propose_patches([{"seq": 1, "test": "val"}])

    call_kwargs = client.client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert any(m["role"] == "system" and "ALLOWED PATCH TYPES" in m["content"] for m in messages)
    assert any(m["role"] == "user" and "failing execution ledgers" in m["content"] for m in messages)


def test_extra_fields_rejection():
    client = NebiusClient(api_key="mock_key")
    client.is_live = True
    client.client = MagicMock()

    model_proposed_json = json.dumps([
        {
            "patch_id": "patch_with_extra",
            "name": "Extra Field Patch",
            "type": "wrapper.idempotency_key",
            "description": "Valid type but prohibited extra fields",
            "is_control": False,
            "executable_code": "def run(): pass",
            "unsupported_extra_field": 123
        }
    ])
    client.client.chat.completions.create.return_value = MockCompletionResponse(model_proposed_json)

    candidates, metadata = client.diagnose_and_propose_patches([{"seq": 1}])

    assert metadata["source"] == "deterministic_fallback"
    assert len(metadata["rejected_proposals"]) == 1
    assert "Prohibited extra fields" in metadata["rejected_proposals"][0]["reason"]
    assert "executable_code" in metadata["rejected_proposals"][0]["reason"]


def test_model_proposals_must_pass_guards_and_cannot_bypass_evaluation(tmp_path):
    """
    Verify that even if the model proposes a syntactically valid patch,
    it cannot bypass the regression guard or declare itself winner.
    """
    orch = Orchestrator(data_dir=str(tmp_path / "data"), tests_dir=str(tmp_path / "tests"))

    # Model proposes a control patch that refuses all operations
    failing_candidate = PatchSpec(
        patch_id="patch_model_refuse_bad",
        name="Bad Refusal Patch",
        type="control.refuse_all",
        description="Fails regression guard",
        is_control=False
    )

    orch.nebius_client.diagnose_and_propose_patches = MagicMock(return_value=(
        [failing_candidate],
        {
            "source": "nemotron_ultra",
            "model": "nvidia/Nemotron-3-Ultra-550b-a55b",
            "raw_proposals_count": 1,
            "accepted_candidates_count": 1,
            "fallback_candidates_count": 0,
            "rejected_proposals": [],
            "error": None,
            "parse_error": None
        }
    ))

    result = orch.run_full_pipeline(train_count=10, held_out_count=6)

    # Candidate must be rejected by guards
    assert "patch_model_refuse_bad" in result["race_summary"]
    assert result["race_summary"]["patch_model_refuse_bad"]["eligible"] is False
    assert any("Regression Guard" in r or "Legitimate-Completion Guard" in r for r in result["race_summary"]["patch_model_refuse_bad"]["rejection_reasons"])
    # No winner should be fabricated
    assert result["winning_patch"] is None
    assert result["heldout_metrics_after"]["status"] == "no_candidate_passed_guards"


def test_empty_json_proposals_handling():
    """When the model legitimately returns [], it should produce model_outcome='empty' and fall back."""
    client = NebiusClient(api_key="mock_key")
    client.is_live = True
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = MockCompletionResponse("[]")

    candidates, metadata = client.diagnose_and_propose_patches([{"seq": 1}])

    assert metadata["source"] == "deterministic_fallback"
    assert metadata["model_outcome"] == "empty"
    assert metadata["raw_proposals_count"] == 0
    assert metadata["accepted_candidates_count"] == 0
    assert metadata["fallback_candidates_count"] >= 4
    assert metadata["parse_error"] is None
    assert metadata["error"] is None
    assert len(candidates) >= 4


def test_non_empty_config_rejection():
    """Proposals with non-empty config dicts must produce model_outcome='rejected' and fall back."""
    client = NebiusClient(api_key="mock_key")
    client.is_live = True
    client.client = MagicMock()

    model_proposed_json = json.dumps([
        {
            "patch_id": "patch_with_config",
            "name": "Config Patch",
            "type": "wrapper.idempotency_key",
            "description": "Has unsupported non-empty config",
            "is_control": False,
            "config": {"arbitrary_key": "some_value"}
        }
    ])
    client.client.chat.completions.create.return_value = MockCompletionResponse(model_proposed_json)

    candidates, metadata = client.diagnose_and_propose_patches([{"seq": 1}])

    assert metadata["source"] == "deterministic_fallback"
    assert metadata["model_outcome"] == "rejected"
    assert metadata["raw_proposals_count"] == 1
    assert metadata["accepted_candidates_count"] == 0
    assert metadata["fallback_candidates_count"] >= 4
    assert len(metadata["rejected_proposals"]) == 1
    assert "Prohibited non-empty config" in metadata["rejected_proposals"][0]["reason"]
    assert "arbitrary_key" in metadata["rejected_proposals"][0]["reason"]


def test_all_invalid_proposals_produces_model_outcome_rejected():
    """When all proposed patches are rejected, model_outcome must be 'rejected'."""
    client = NebiusClient(api_key="mock_key")
    client.is_live = True
    client.client = MagicMock()

    model_proposed_json = json.dumps([
        {
            "patch_id": "patch_invalid_type",
            "name": "Invalid Type Patch",
            "type": "tool_contract",
            "description": "Unsupported type",
            "is_control": False
        }
    ])
    client.client.chat.completions.create.return_value = MockCompletionResponse(model_proposed_json)

    candidates, metadata = client.diagnose_and_propose_patches([{"seq": 1}])

    assert metadata["source"] == "deterministic_fallback"
    assert metadata["model_outcome"] == "rejected"
    assert metadata["raw_proposals_count"] == 1
    assert metadata["accepted_candidates_count"] == 0
    assert metadata["fallback_candidates_count"] >= 4
    assert len(metadata["rejected_proposals"]) == 1
    assert "Unsupported or missing patch type" in metadata["rejected_proposals"][0]["reason"]


def test_empty_string_model_response():
    """When the model returns empty string content '', model_outcome must be 'parse_error' and fall back."""
    client = NebiusClient(api_key="mock_key")
    client.is_live = True
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = MockCompletionResponse("")

    candidates, metadata = client.diagnose_and_propose_patches([{"seq": 1}])

    assert metadata["source"] == "deterministic_fallback"
    assert metadata["model_outcome"] == "parse_error"
    assert metadata["parse_error"] == "Empty or null response content received from model."
    assert metadata["raw_proposals_count"] == 0
    assert metadata["accepted_candidates_count"] == 0
    assert metadata["fallback_candidates_count"] >= 4
    assert len(candidates) >= 4


def test_full_refund_intent_below_and_above_50():
    """
    Full-refund correctness:
    Verify that general full refund requests receive the full remaining balance
    regardless of whether order total >= $50 (eliminating the arbitrary half-refund heuristic).
    """
    from src.silentfail.tools import WorldState, ToolRegistry, CausalLedger
    from src.silentfail.agent import NaiveRefundAgent

    # 1. Order below $50 ($35.00)
    ws_below = WorldState({"orders": [{"id": "1001", "total": 35.0, "refunded_amount": 0.0}]})
    tools_below = ToolRegistry(ws_below, CausalLedger("sc_below", "baseline"))
    res_below = NaiveRefundAgent().run("Please issue a full refund for order 1001.", tools_below)
    assert ws_below.get_summary()["refund_count"] == 1
    assert ws_below.get_summary()["refund_total"] == 35.0

    # 2. Order at or above $50 ($80.00)
    ws_above = WorldState({"orders": [{"id": "1002", "total": 80.0, "refunded_amount": 0.0}]})
    tools_above = ToolRegistry(ws_above, CausalLedger("sc_above", "baseline"))
    res_above = NaiveRefundAgent().run("I need a full refund for order 1002.", tools_above)
    assert ws_above.get_summary()["refund_count"] == 1
    assert ws_above.get_summary()["refund_total"] == 80.0


def test_explicit_partial_refund_intent():
    """
    Partial-refund correctness:
    Verify that explicit dollar amounts (e.g. $42.50) in customer messages are honored,
    and refunded exactly once.
    """
    from src.silentfail.tools import WorldState, ToolRegistry, CausalLedger
    from src.silentfail.agent import NaiveRefundAgent

    ws = WorldState({"orders": [{"id": "1003", "total": 120.0, "refunded_amount": 0.0}]})
    tools = ToolRegistry(ws, CausalLedger("sc_partial", "baseline"))
    res = NaiveRefundAgent().run("Please refund $42.50 for order 1003, one item was damaged.", tools)
    assert ws.get_summary()["refund_count"] == 1
    assert ws.get_summary()["refund_total"] == 42.50


def test_order_id_not_confused_for_refund_amount():
    """
    Verify that order IDs (e.g. order 5000 or 1234) are not parsed as requested refund amounts.
    """
    from src.silentfail.tools import WorldState, ToolRegistry, CausalLedger
    from src.silentfail.agent import NaiveRefundAgent

    ws = WorldState({"orders": [{"id": "5000", "total": 85.0, "refunded_amount": 0.0}]})
    tools = ToolRegistry(ws, CausalLedger("sc_order_id", "baseline"))
    res = NaiveRefundAgent().run("Please issue refund for order 5000", tools)
    assert ws.get_summary()["refund_count"] == 1
    assert ws.get_summary()["refund_total"] == 85.0  # Full balance, not $5000


def test_refund_parsing_dollars_word_and_multiple_dollar_amounts():
    """
    Boundary tests for refund parser:
    1. 'refund 40 dollars' / '40 dollars' phrasing.
    2. Multiple dollar amounts in a message (e.g., total price, damage amount, shipping)
       where the refund amount is disambiguated and honored correctly.
    """
    from src.silentfail.tools import WorldState, ToolRegistry, CausalLedger
    from src.silentfail.agent import NaiveRefundAgent

    # 1. Word 'dollars' phrasing
    ws1 = WorldState({"orders": [{"id": "9002", "total": 100.0, "refunded_amount": 0.0}]})
    tools1 = ToolRegistry(ws1, CausalLedger("sc_dollars_word", "baseline"))
    res1 = NaiveRefundAgent().run("Please refund 40 dollars for order 9002.", tools1)
    assert ws1.get_summary()["refund_count"] == 1
    assert ws1.get_summary()["refund_total"] == 40.0

    # 2. Multiple dollar amounts in customer message (item $40, shipping $10, total $100)
    ws2 = WorldState({"orders": [{"id": "9003", "total": 100.0, "refunded_amount": 0.0}]})
    tools2 = ToolRegistry(ws2, CausalLedger("sc_multi_amount", "baseline"))
    res2 = NaiveRefundAgent().run("Total was $100 with $10 shipping, but please refund $40 for order 9003.", tools2)
    assert ws2.get_summary()["refund_count"] == 1
    assert ws2.get_summary()["refund_total"] == 40.0


def test_amount_greater_than_remaining_balance_is_not_issued():
    """
    Verify that if a customer message requests an amount greater than the remaining balance,
    the agent falls back to the full eligible remaining balance rather than over-refunding or crashing,
    and tool balance validation strictly blocks any overdraft.
    """
    from src.silentfail.tools import WorldState, ToolRegistry, CausalLedger
    from src.silentfail.agent import NaiveRefundAgent

    ws = WorldState({"orders": [{"id": "1005", "total": 50.0, "refunded_amount": 10.0}]})
    tools = ToolRegistry(ws, CausalLedger("sc_over_balance", "baseline"))
    # Customer asks for $100 on an order with $40 remaining balance
    res = NaiveRefundAgent().run("Please refund $100 for order 1005", tools)
    assert ws.get_summary()["refund_count"] == 1
    assert ws.get_summary()["refund_total"] == 40.0  # Capped at remaining balance (50 - 10)


def test_ambiguous_request_intent_behavior():
    """
    Ambiguous request intent behavior:
    When a customer request is general or ambiguous without specifying an explicit partial amount
    (e.g., 'Please help with order 1004, package arrived damaged'),
    the agent follows the documented standard service contract and refunds the eligible remaining balance
    rather than inventing an arbitrary partial refund.
    """
    from src.silentfail.tools import WorldState, ToolRegistry, CausalLedger
    from src.silentfail.agent import NaiveRefundAgent

    ws = WorldState({"orders": [{"id": "1004", "total": 65.0, "refunded_amount": 0.0}]})
    tools = ToolRegistry(ws, CausalLedger("sc_ambig", "baseline"))
    res = NaiveRefundAgent().run("Please help with order 1004, package arrived damaged.", tools)
    assert ws.get_summary()["refund_count"] == 1
    assert ws.get_summary()["refund_total"] == 65.0


def test_duplicate_retry_regression_timeout_after_success():
    """
    Duplicate-retry regression:
    Order with $100 balance and explicit $40 refund request with timeout_after_success fault.
    1. Raw operation-ID deduplication lets the baseline retry without a second mutation.
    2. Hardened wrapper prevents the second mutation, producing exactly one mutation and a $40 total.
    3. Tool balance validation is preserved (not overdrafted).
    """
    from src.silentfail.models import Scenario, ExpectedOutcome, FaultConfig, PatchSpec
    from src.silentfail.runner import EpisodeRunner

    scenario = Scenario(
        scenario_id="sc_duplicate_retry_test",
        split="train",
        category="timeout_after_success",
        initial_state={"orders": [{"id": "9001", "status": "delivered", "total": 100.0, "refunded_amount": 0.0}]},
        customer_message="Please refund $40 for order 9001.",
        faults=[FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success")],
        expected_outcome=ExpectedOutcome(refund_count=1, refund_total=40.0, email_count=1)
    )

    # 1. Unpatched baseline: stable operation_id must prevent duplicate mutation.
    base = EpisodeRunner.run_episode(scenario, patch=None)
    assert base["detection"].duplicate_action is False
    assert base["record"].final_state["refund_count"] == 1
    assert base["record"].final_state["refund_total"] == 40.0
    assert base["detection"].silent_failure_false_success is False
    assert base["is_correct"] is True

    # Confirm mutations from causal ledger events
    mutations_base = [e for e in base["record"].events if e.type == "world_mutation" and e.effect == "refund_created"]
    assert len(mutations_base) == 1
    assert mutations_base[0].amount == 40.0

    # 2. Hardened wrapper: must prevent the duplicate retry mutation
    patch = PatchSpec(
        patch_id="patch_retry_policy",
        name="State-Checked Retry Policy Wrapper",
        type="wrapper.retry_policy",
        description="On timeout, inspects world state before repeating mutating actions.",
        is_control=False
    )
    hardened = EpisodeRunner.run_episode(scenario, patch=patch)
    assert hardened["detection"].duplicate_action is False
    assert hardened["record"].final_state["refund_count"] == 1
    assert hardened["record"].final_state["refund_total"] == 40.0
    assert hardened["record"].final_state["email_count"] == 1
    assert hardened["is_correct"] is True

    mutations_hardened = [e for e in hardened["record"].events if e.type == "world_mutation" and e.effect == "refund_created"]
    assert len(mutations_hardened) == 1
    assert mutations_hardened[0].amount == 40.0


def test_sanitized_error_metadata():
    """
    Check proposal metadata handling:
    Verify that sanitize_error_message redacts sensitive tokens, API keys,
    bearer tokens, and authorization headers from error messages before storing in proposal metadata.
    """
    from src.silentfail.nebius import sanitize_error_message, NebiusClient

    # 1. Direct sanitization test
    fake_key = "v1.CmQKHHN0YXRpY2tleS1lMDB4azR0Mndza2RzZGt0ZnISIXNlcnZpY2VhY2NvdW50LWUwMGExaGEyOXJxcnY2bnpldDIMCObg7dYGENfI4rMDOgwI3uOFogcQgNm6uwNAAloDZTAw"
    raw_error = f"HTTP 401 Unauthorized: Authorization: Bearer {fake_key} with api_key={fake_key}"
    sanitized = sanitize_error_message(raw_error, sensitive_tokens=[fake_key])
    assert fake_key not in sanitized
    assert "[REDACTED" in sanitized

    # 2. Integration test in client.diagnose_and_propose_patches
    client = NebiusClient(api_key=fake_key)
    client.is_live = True
    client.client = MagicMock()
    client.client.chat.completions.create.side_effect = RuntimeError(f"Connection failed for key {fake_key}")

    candidates, metadata = client.diagnose_and_propose_patches([{"seq": 1}])
    assert metadata["model_outcome"] == "request_error"
    assert fake_key not in metadata["error"]
    assert "[REDACTED" in metadata["error"]


def test_structured_diagnosis_with_valid_citations():
    """
    Test Phase C: Structured diagnosis object with valid ledger sequence citations.
    """
    from src.silentfail.nebius import NebiusClient, validate_ledger_citations

    failing_ledgers = [
        {
            "run_id": "run_test_1",
            "events": [
                {"seq": 1, "type": "tool_request", "tool": "issue_refund", "args": {"order_id": "1001", "amount": 50.0}},
                {"seq": 2, "type": "world_mutation", "effect": "refund_created", "amount": 50.0},
                {"seq": 3, "type": "tool_response_to_agent", "observed": {"error": "timeout"}, "fault": "timeout_after_success"}
            ]
        }
    ]

    proposal = {
        "root_cause": "Agent received timeout after mutation and blindly retried",
        "cited_seqs": [2, 3],
        "patches": [
            {
                "patch_id": "patch_cited_idemp",
                "name": "Cited Idempotency",
                "type": "wrapper.idempotency_key",
                "description": "Fixes timeout duplicate",
                "is_control": False,
                "config": {}
            }
        ]
    }

    is_valid, errors = validate_ledger_citations(failing_ledgers, proposal)
    assert is_valid is True
    assert len(errors) == 0

    # Integration test through client
    client = NebiusClient(api_key="mock_key")
    client.is_live = True
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = MockCompletionResponse(json.dumps(proposal))

    candidates, meta = client.diagnose_and_propose_patches(failing_ledgers)
    assert meta["source"] == "nemotron_ultra"
    assert meta["model_outcome"] == "accepted"
    assert meta["diagnosis"]["citation_valid"] is True
    assert meta["diagnosis"]["cited_seqs"] == [2, 3]
    assert any(c.patch_id == "patch_cited_idemp" for c in candidates)


def test_structured_diagnosis_with_invalid_citations_rejected():
    """
    Test Phase C: Rejection of diagnosis proposing citations not in supplied ledgers
    or citing non-evidence events.
    """
    from src.silentfail.nebius import NebiusClient, validate_ledger_citations

    failing_ledgers = [
        {
            "run_id": "run_test_2",
            "events": [
                {"seq": 1, "type": "tool_request", "tool": "get_order", "args": {"order_id": "1002"}},
                {"seq": 2, "type": "tool_response_to_agent", "observed": {"id": "1002", "status": "delivered"}}
            ]
        }
    ]

    # Nonexistent sequence citation (99)
    bad_proposal_missing = {
        "root_cause": "Timeout failure",
        "cited_seqs": [99],
        "patches": [{"patch_id": "p1", "name": "p1", "type": "wrapper.idempotency_key", "description": "d", "is_control": False}]
    }
    is_valid_miss, errors_miss = validate_ledger_citations(failing_ledgers, bad_proposal_missing)
    assert is_valid_miss is False
    assert any("does not exist" in e for e in errors_miss)

    # Empty sequence citation
    bad_proposal_empty = {
        "root_cause": "Timeout failure",
        "cited_seqs": [],
        "patches": [{"patch_id": "p1", "name": "p1", "type": "wrapper.idempotency_key", "description": "d", "is_control": False}]
    }
    is_valid_emp, errors_emp = validate_ledger_citations(failing_ledgers, bad_proposal_empty)
    assert is_valid_emp is False
    assert any("must cite at least one" in e for e in errors_emp)

    # Integration: client should reject proposal and fall back to labeled deterministic defaults
    client = NebiusClient(api_key="mock_key")
    client.is_live = True
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = MockCompletionResponse(json.dumps(bad_proposal_missing))

    candidates, meta = client.diagnose_and_propose_patches(failing_ledgers)
    assert meta["source"] == "deterministic_fallback"
    assert meta["model_outcome"] == "rejected"
    assert "Ledger citation validation failed" in meta["parse_error"]
    assert meta["diagnosis"]["citation_valid"] is False
    assert meta["fallback_candidates_count"] >= 4



def test_none_content_model_response():
    """When the model returns None content, model_outcome must be 'parse_error' and fall back."""
    client = NebiusClient(api_key="mock_key")
    client.is_live = True
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = MockCompletionResponse(None)

    candidates, metadata = client.diagnose_and_propose_patches([{"seq": 1}])

    assert metadata["source"] == "deterministic_fallback"
    assert metadata["model_outcome"] == "parse_error"
    assert metadata["parse_error"] == "Empty or null response content received from model."
    assert metadata["raw_proposals_count"] == 0
    assert metadata["accepted_candidates_count"] == 0
    assert metadata["fallback_candidates_count"] >= 4
    assert len(candidates) >= 4


def test_full_refund_intent_below_and_above_50():
    """
    Full-refund correctness:
    Verify that general full refund requests receive the full remaining balance
    regardless of whether order total >= $50 (eliminating the arbitrary half-refund heuristic).
    """
    from src.silentfail.tools import WorldState, ToolRegistry, CausalLedger
    from src.silentfail.agent import NaiveRefundAgent

    # 1. Order below $50 ($35.00)
    ws_below = WorldState({"orders": [{"id": "1001", "total": 35.0, "refunded_amount": 0.0}]})
    tools_below = ToolRegistry(ws_below, CausalLedger("sc_below", "baseline"))
    res_below = NaiveRefundAgent().run("Please issue a full refund for order 1001.", tools_below)
    assert ws_below.get_summary()["refund_count"] == 1
    assert ws_below.get_summary()["refund_total"] == 35.0

    # 2. Order at or above $50 ($80.00)
    ws_above = WorldState({"orders": [{"id": "1002", "total": 80.0, "refunded_amount": 0.0}]})
    tools_above = ToolRegistry(ws_above, CausalLedger("sc_above", "baseline"))
    res_above = NaiveRefundAgent().run("I need a full refund for order 1002.", tools_above)
    assert ws_above.get_summary()["refund_count"] == 1
    assert ws_above.get_summary()["refund_total"] == 80.0


def test_explicit_partial_refund_intent():
    """
    Partial-refund correctness:
    Verify that explicit dollar amounts (e.g. $42.50) in customer messages are honored,
    and refunded exactly once.
    """
    from src.silentfail.tools import WorldState, ToolRegistry, CausalLedger
    from src.silentfail.agent import NaiveRefundAgent

    ws = WorldState({"orders": [{"id": "1003", "total": 120.0, "refunded_amount": 0.0}]})
    tools = ToolRegistry(ws, CausalLedger("sc_partial", "baseline"))
    res = NaiveRefundAgent().run("Please refund $42.50 for order 1003, one item was damaged.", tools)
    assert ws.get_summary()["refund_count"] == 1
    assert ws.get_summary()["refund_total"] == 42.50


def test_order_id_not_confused_for_refund_amount():
    """
    Verify that order IDs (e.g. order 5000 or 1234) are not parsed as requested refund amounts.
    """
    from src.silentfail.tools import WorldState, ToolRegistry, CausalLedger
    from src.silentfail.agent import NaiveRefundAgent

    ws = WorldState({"orders": [{"id": "5000", "total": 85.0, "refunded_amount": 0.0}]})
    tools = ToolRegistry(ws, CausalLedger("sc_order_id", "baseline"))
    res = NaiveRefundAgent().run("Please issue refund for order 5000", tools)
    assert ws.get_summary()["refund_count"] == 1
    assert ws.get_summary()["refund_total"] == 85.0  # Full balance, not $5000


def test_refund_parsing_dollars_word_and_multiple_dollar_amounts():
    """
    Boundary tests for refund parser:
    1. 'refund 40 dollars' / '40 dollars' phrasing.
    2. Multiple dollar amounts in a message (e.g., total price, damage amount, shipping)
       where the refund amount is disambiguated and honored correctly.
    """
    from src.silentfail.tools import WorldState, ToolRegistry, CausalLedger
    from src.silentfail.agent import NaiveRefundAgent

    # 1. Word 'dollars' phrasing
    ws1 = WorldState({"orders": [{"id": "9002", "total": 100.0, "refunded_amount": 0.0}]})
    tools1 = ToolRegistry(ws1, CausalLedger("sc_dollars_word", "baseline"))
    res1 = NaiveRefundAgent().run("Please refund 40 dollars for order 9002.", tools1)
    assert ws1.get_summary()["refund_count"] == 1
    assert ws1.get_summary()["refund_total"] == 40.0

    # 2. Multiple dollar amounts in customer message (item $40, shipping $10, total $100)
    ws2 = WorldState({"orders": [{"id": "9003", "total": 100.0, "refunded_amount": 0.0}]})
    tools2 = ToolRegistry(ws2, CausalLedger("sc_multi_amount", "baseline"))
    res2 = NaiveRefundAgent().run("Total was $100 with $10 shipping, but please refund $40 for order 9003.", tools2)
    assert ws2.get_summary()["refund_count"] == 1
    assert ws2.get_summary()["refund_total"] == 40.0


def test_amount_greater_than_remaining_balance_is_not_issued():
    """
    Verify that if a customer message requests an amount greater than the remaining balance,
    the agent falls back to the full eligible remaining balance rather than over-refunding or crashing,
    and tool balance validation strictly blocks any overdraft.
    """
    from src.silentfail.tools import WorldState, ToolRegistry, CausalLedger
    from src.silentfail.agent import NaiveRefundAgent

    ws = WorldState({"orders": [{"id": "1005", "total": 50.0, "refunded_amount": 10.0}]})
    tools = ToolRegistry(ws, CausalLedger("sc_over_balance", "baseline"))
    # Customer asks for $100 on an order with $40 remaining balance
    res = NaiveRefundAgent().run("Please refund $100 for order 1005", tools)
    assert ws.get_summary()["refund_count"] == 1
    assert ws.get_summary()["refund_total"] == 40.0  # Capped at remaining balance (50 - 10)


def test_ambiguous_request_intent_behavior():
    """
    Ambiguous request intent behavior:
    When a customer request is general or ambiguous without specifying an explicit partial amount
    (e.g., 'Please help with order 1004, package arrived damaged'),
    the agent follows the documented standard service contract and refunds the eligible remaining balance
    rather than inventing an arbitrary partial refund.
    """
    from src.silentfail.tools import WorldState, ToolRegistry, CausalLedger
    from src.silentfail.agent import NaiveRefundAgent

    ws = WorldState({"orders": [{"id": "1004", "total": 65.0, "refunded_amount": 0.0}]})
    tools = ToolRegistry(ws, CausalLedger("sc_ambig", "baseline"))
    res = NaiveRefundAgent().run("Please help with order 1004, package arrived damaged.", tools)
    assert ws.get_summary()["refund_count"] == 1
    assert ws.get_summary()["refund_total"] == 65.0


def test_duplicate_retry_regression_timeout_after_success():
    """
    Duplicate-retry regression:
    Order with $100 balance and explicit $40 refund request with timeout_after_success fault.
    1. Raw operation-ID deduplication lets the baseline retry without a second mutation.
    2. Hardened wrapper prevents the second mutation, producing exactly one mutation and a $40 total.
    3. Tool balance validation is preserved (not overdrafted).
    """
    from src.silentfail.models import Scenario, ExpectedOutcome, FaultConfig, PatchSpec
    from src.silentfail.runner import EpisodeRunner

    scenario = Scenario(
        scenario_id="sc_duplicate_retry_test",
        split="train",
        category="timeout_after_success",
        initial_state={"orders": [{"id": "9001", "status": "delivered", "total": 100.0, "refunded_amount": 0.0}]},
        customer_message="Please refund $40 for order 9001.",
        faults=[FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success")],
        expected_outcome=ExpectedOutcome(refund_count=1, refund_total=40.0, email_count=1)
    )

    # 1. Unpatched baseline: stable operation_id must prevent duplicate mutation.
    base = EpisodeRunner.run_episode(scenario, patch=None)
    assert base["detection"].duplicate_action is False
    assert base["record"].final_state["refund_count"] == 1
    assert base["record"].final_state["refund_total"] == 40.0
    assert base["detection"].silent_failure_false_success is False
    assert base["is_correct"] is True

    # Confirm mutations from causal ledger events
    mutations_base = [e for e in base["record"].events if e.type == "world_mutation" and e.effect == "refund_created"]
    assert len(mutations_base) == 1
    assert mutations_base[0].amount == 40.0

    # 2. Hardened wrapper: must prevent the duplicate retry mutation
    patch = PatchSpec(
        patch_id="patch_retry_policy",
        name="State-Checked Retry Policy Wrapper",
        type="wrapper.retry_policy",
        description="On timeout, inspects world state before repeating mutating actions.",
        is_control=False
    )
    hardened = EpisodeRunner.run_episode(scenario, patch=patch)
    assert hardened["detection"].duplicate_action is False
    assert hardened["record"].final_state["refund_count"] == 1
    assert hardened["record"].final_state["refund_total"] == 40.0
    assert hardened["is_correct"] is True

    mutations_hardened = [e for e in hardened["record"].events if e.type == "world_mutation" and e.effect == "refund_created"]
    assert len(mutations_hardened) == 1
    assert mutations_hardened[0].amount == 40.0


def test_sanitized_error_metadata():
    """
    Check proposal metadata handling:
    Verify that sanitize_error_message redacts sensitive tokens, API keys,
    bearer tokens, and authorization headers from error messages before storing in proposal metadata.
    """
    from src.silentfail.nebius import sanitize_error_message, NebiusClient

    # 1. Direct sanitization test
    fake_key = "v1.CmQKHHN0YXRpY2tleS1lMDB4azR0Mndza2RzZGt0ZnISIXNlcnZpY2VhY2NvdW50LWUwMGExaGEyOXJxcnY2bnpldDIMCObg7dYGENfI4rMDOgwI3uOFogcQgNm6uwNAAloDZTAw"
    raw_error = f"HTTP 401 Unauthorized: Authorization: Bearer {fake_key} with api_key={fake_key}"
    sanitized = sanitize_error_message(raw_error, sensitive_tokens=[fake_key])
    assert fake_key not in sanitized
    assert "[REDACTED" in sanitized

    # 2. Integration test in client.diagnose_and_propose_patches
    client = NebiusClient(api_key=fake_key)
    client.is_live = True
    client.client = MagicMock()
    client.client.chat.completions.create.side_effect = RuntimeError(f"Connection failed for key {fake_key}")

    candidates, metadata = client.diagnose_and_propose_patches([{"seq": 1}])
    assert metadata["model_outcome"] == "request_error"
    assert fake_key not in metadata["error"]
    assert "[REDACTED" in metadata["error"]


def test_structured_diagnosis_with_valid_citations():
    """
    Test Phase C: Structured diagnosis object with valid ledger sequence citations.
    """
    from src.silentfail.nebius import NebiusClient, validate_ledger_citations

    failing_ledgers = [
        {
            "run_id": "run_test_1",
            "events": [
                {"seq": 1, "type": "tool_request", "tool": "issue_refund", "args": {"order_id": "1001", "amount": 50.0}},
                {"seq": 2, "type": "world_mutation", "effect": "refund_created", "amount": 50.0},
                {"seq": 3, "type": "tool_response_to_agent", "observed": {"error": "timeout"}, "fault": "timeout_after_success"}
            ]
        }
    ]

    proposal = {
        "root_cause": "Agent received timeout after mutation and blindly retried",
        "cited_seqs": [2, 3],
        "patches": [
            {
                "patch_id": "patch_cited_idemp",
                "name": "Cited Idempotency",
                "type": "wrapper.idempotency_key",
                "description": "Fixes timeout duplicate",
                "is_control": False,
                "config": {}
            }
        ]
    }

    is_valid, errors = validate_ledger_citations(failing_ledgers, proposal)
    assert is_valid is True
    assert len(errors) == 0

    # Integration test through client
    client = NebiusClient(api_key="mock_key")
    client.is_live = True
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = MockCompletionResponse(json.dumps(proposal))

    candidates, meta = client.diagnose_and_propose_patches(failing_ledgers)
    assert meta["source"] == "nemotron_ultra"
    assert meta["model_outcome"] == "accepted"
    assert meta["diagnosis"]["citation_valid"] is True
    assert meta["diagnosis"]["cited_seqs"] == [2, 3]
    assert any(c.patch_id == "patch_cited_idemp" for c in candidates)


def test_structured_diagnosis_with_invalid_citations_rejected():
    """
    Test Phase C: Rejection of diagnosis proposing citations not in supplied ledgers
    or citing non-evidence events.
    """
    from src.silentfail.nebius import NebiusClient, validate_ledger_citations

    failing_ledgers = [
        {
            "run_id": "run_test_2",
            "events": [
                {"seq": 1, "type": "tool_request", "tool": "get_order", "args": {"order_id": "1002"}},
                {"seq": 2, "type": "tool_response_to_agent", "observed": {"id": "1002", "status": "delivered"}}
            ]
        }
    ]

    # Nonexistent sequence citation (99)
    bad_proposal_missing = {
        "root_cause": "Timeout failure",
        "cited_seqs": [99],
        "patches": [{"patch_id": "p1", "name": "p1", "type": "wrapper.idempotency_key", "description": "d", "is_control": False}]
    }
    is_valid_miss, errors_miss = validate_ledger_citations(failing_ledgers, bad_proposal_missing)
    assert is_valid_miss is False
    assert any("does not exist" in e for e in errors_miss)

    # Empty sequence citation
    bad_proposal_empty = {
        "root_cause": "Timeout failure",
        "cited_seqs": [],
        "patches": [{"patch_id": "p1", "name": "p1", "type": "wrapper.idempotency_key", "description": "d", "is_control": False}]
    }
    is_valid_emp, errors_emp = validate_ledger_citations(failing_ledgers, bad_proposal_empty)
    assert is_valid_emp is False
    assert any("must cite at least one" in e for e in errors_emp)

    # Integration: client should reject proposal and fall back to labeled deterministic defaults
    client = NebiusClient(api_key="mock_key")
    client.is_live = True
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = MockCompletionResponse(json.dumps(bad_proposal_missing))

    candidates, meta = client.diagnose_and_propose_patches(failing_ledgers)
    assert meta["source"] == "deterministic_fallback"
    assert meta["model_outcome"] == "rejected"
    assert "Ledger citation validation failed" in meta["parse_error"]
    assert meta["diagnosis"]["citation_valid"] is False
    assert meta["fallback_candidates_count"] >= 4

