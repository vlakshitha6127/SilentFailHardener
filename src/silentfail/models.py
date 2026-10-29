"""
Core domain models for SilentFail Hardener.
Defines schemas for World State, Scenarios, Ledger Events, Patches, and Metrics.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
import time


class Order(BaseModel):
    id: str
    status: str = "delivered"
    total: float
    refunded_amount: float = 0.0
    items: List[Dict[str, Any]] = Field(default_factory=list)


class Refund(BaseModel):
    id: str
    order_id: str
    amount: float
    idempotency_key: Optional[str] = None
    created_at: float = Field(default_factory=time.time)


class Email(BaseModel):
    id: str
    to: str
    subject: str
    body: str


class Ticket(BaseModel):
    id: str
    order_id: str
    summary: str


class FaultConfig(BaseModel):
    tool: str
    call_index: int = 0
    type: Literal["timeout_after_success", "empty_200", "duplicate_webhook"]
    effect_applied: Optional[bool] = None  # Relevant for empty_200


class ExpectedOutcome(BaseModel):
    refund_count: int = 0
    refund_total: float = 0.0
    email_count: int = 0
    ticket_count: int = 0
    allow_refusal: bool = False


class Scenario(BaseModel):
    scenario_id: str
    split: Literal["train", "held_out"] = "train"
    seed: int = 1234
    category: str
    initial_state: Dict[str, Any]
    customer_message: str
    faults: List[FaultConfig] = Field(default_factory=list)
    expected_outcome: ExpectedOutcome


class LedgerEvent(BaseModel):
    seq: int
    type: Literal["tool_request", "world_mutation", "tool_response_to_agent"]
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    effect: Optional[str] = None
    refund_id: Optional[str] = None
    amount: Optional[float] = None
    observed: Optional[Dict[str, Any]] = None
    fault: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class ExecutionCost(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: int = 0
    steps: int = 0


class LedgerRecord(BaseModel):
    run_id: str
    scenario_id: str
    agent_version: str = "baseline"
    patch_id: Optional[str] = None
    events: List[LedgerEvent] = Field(default_factory=list)
    final_message: str = ""
    final_state: Dict[str, Any] = Field(default_factory=dict)
    cost: ExecutionCost = Field(default_factory=ExecutionCost)


class PatchSpec(BaseModel):
    patch_id: str
    name: str
    type: Literal[
        "wrapper.idempotency_key",
        "wrapper.read_back_verify",
        "wrapper.retry_policy",
        "combo.idempotency_and_read_back",
        "control.refuse_all",
        "control.blind_dedupe"
    ]
    description: str
    is_control: bool = False
    config: Dict[str, Any] = Field(default_factory=dict)


class DetectionResult(BaseModel):
    duplicate_action: bool = False
    excess_mutation: bool = False
    ineligible_refund_mutation: bool = False
    incorrect_refund_amount: bool = False
    silent_failure_false_success: bool = False
    missed_action: bool = False
    incorrect_completion: bool = False
    over_suppression: bool = False
    legitimate_completion: bool = False
    claim_uncertain: bool = False
    claims_refund_issued: Optional[bool] = None
    claimed_amount: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    # Granular contract tracking (Phase 3.8 / v2.1)
    duplicate_mutation: bool = False
    ineligible_refund_issued: bool = False
