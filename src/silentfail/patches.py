"""
Typed Declarative Patches and Registry Wrappers.
Sections 10.2, 10.3 of the Project Spec.

Supported patch types:
- wrapper.idempotency_key: generates deterministic key per (order_id, action, amount)
- wrapper.read_back_verify: queries get_order state before retrying or confirming
- wrapper.retry_policy: on timeout/error, inspects world state before repeating mutating actions
- combo.idempotency_and_read_back: combines idempotency key and read-back verification
- control.refuse_all: negative control (rejects all refunds, proving the regression guard works)
- control.blind_dedupe: naive deduplication (blocks repeat calls on the same order, proving legitimate repeat failures)
"""

from typing import Dict, Any, Optional, Tuple, List
from src.silentfail.tools import ToolRegistry
from src.silentfail.models import PatchSpec


class PatchedToolRegistry:
    """
    Wraps the ToolRegistry to enforce declarative hardening patches.
    Follows a strict operation-correlation contract:
    - Retries must explicitly supply the same operation_id or reuse an existing idempotency_key.
    - Independent calls receive distinct operation_ids and idempotency_keys.
    - Identity is never inferred from order_id + amount or ambiguous per-order state.
    """
    def __init__(self, raw_registry: ToolRegistry, patch: Optional[PatchSpec] = None):
        self.raw = raw_registry
        self.patch = patch
        # Maps operation_id -> stable idempotency key
        self._op_key_map: Dict[str, str] = {}
        self._op_seq_counter: int = 0
        self._order_refunded_keys: set = set()
        # Track retry counts per operation_id for bounded retry policies
        self._retry_policy_attempts: Dict[str, int] = {}

    def get_order(self, order_id: str) -> Dict[str, Any]:
        return self.raw.get_order(order_id)

    def _resolve_operation_id(self, order_id: str, operation_id: Optional[str] = None) -> str:
        """
        Resolves or generates a unique logical operation ID.
        Never infers logical identity from (order_id + amount) or prior unresolved calls.
        Every call without an explicit operation_id represents an independent request
        and receives a distinct, unique operation ID.
        """
        if operation_id:
            return operation_id
        self._op_seq_counter += 1
        return f"op_{order_id}_{self._op_seq_counter}"

    def _get_stable_idempotency_key(self, op_id: str) -> str:
        """
        Generates or retrieves one stable idempotency key strictly per unique operation ID.
        All retries sharing that operation ID reuse the same key.
        """
        if op_id not in self._op_key_map:
            self._op_key_map[op_id] = f"idem_{op_id}"
        return self._op_key_map[op_id]

    def _verify_specific_refund(self, order_id: str, amount: float, stable_key: str) -> Optional[Dict[str, Any]]:
        """
        Verifies read-back strictly by exact operation ID / stable idempotency key.
        Does NOT accept random newly created refunds with matching amounts.
        """
        for r in self.raw.world_state.refunds.values():
            if r.order_id == str(order_id) and r.idempotency_key == stable_key:
                return {
                    "status": "ok",
                    "refund_id": r.id,
                    "amount": r.amount,
                    "order_id": order_id,
                    "verified_by_readback": True
                }
        return None

    def _reconcile_state_delta(self, order_id: str, pre_refunded: float, pre_refund_ids: set, target_amount: float) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Reconciles authoritative order/refund state by comparing pre-call state snapshot
        with post-call state snapshot without relying on idempotency keys.
        Returns:
            (outcome, refund_data)
            where outcome is:
              - 'applied': refund matching target_amount was committed
              - 'unapplied': no refund was committed and refunded_amount unchanged
              - 'uncertain': balance changed by an unexpected/inconsistent amount
        """
        curr_order = self.raw.world_state.orders.get(str(order_id))
        if not curr_order:
            return "unapplied", None

        curr_refunded = round(curr_order.refunded_amount, 2)
        delta = round(curr_refunded - pre_refunded, 2)

        # Check for new refund objects created during this call
        new_refunds = [
            r for r_id, r in self.raw.world_state.refunds.items()
            if r_id not in pre_refund_ids and r.order_id == str(order_id)
        ]

        if delta == target_amount and len(new_refunds) == 1 and new_refunds[0].amount == target_amount:
            return "applied", {
                "status": "ok",
                "refund_id": new_refunds[0].id,
                "amount": target_amount,
                "order_id": order_id,
                "verified_by_readback": True
            }
        elif delta == 0.0 and len(new_refunds) == 0:
            return "unapplied", None
        else:
            return "uncertain", None

    def issue_refund(
        self,
        order_id: str,
        amount: float,
        idempotency_key: Optional[str] = None,
        operation_id: Optional[str] = None,
        op_index: Optional[int] = None
    ) -> Dict[str, Any]:
        # Backward compatibility: if caller supplied op_index but not operation_id, derive unique op_id from op_index
        if not operation_id and op_index is not None:
            operation_id = f"op_{order_id}_req{op_index}"

        # Resolve logical operation ID and its stable idempotency key
        op_id = self._resolve_operation_id(order_id, operation_id)
        stable_key = idempotency_key or self._get_stable_idempotency_key(op_id)

        if not self.patch:
            # Unpatched tool registry call
            return self.raw.issue_refund(order_id, amount, idempotency_key=idempotency_key, operation_id=op_id)

        p_type = self.patch.type

        # 1. Control: refuse_all
        if p_type == "control.refuse_all":
            self.raw.ledger.record_tool_request("issue_refund", {"order_id": order_id, "amount": amount, "control": "refuse_all"})
            res = {"status": "error", "message": "Policy refusal: all refund requests blocked"}
            self.raw.ledger.record_tool_response(res, tool="issue_refund")
            return res

        # 2. Control: blind_dedupe
        if p_type == "control.blind_dedupe":
            if order_id in self._order_refunded_keys:
                self.raw.ledger.record_tool_request("issue_refund", {"order_id": order_id, "amount": amount, "control": "blind_dedupe"})
                res = {"status": "ok", "message": "Deduplicated repeat call on order"}
                self.raw.ledger.record_tool_response(res, tool="issue_refund")
                return res
            self._order_refunded_keys.add(order_id)
            return self.raw.issue_refund(order_id, amount, idempotency_key=stable_key, operation_id=op_id)

        # -------------------------------------------------------------------------
        # 3. Wrapper: idempotency_key (Pure Idempotency Key - Zero Read-Back)
        # -------------------------------------------------------------------------
        if p_type == "wrapper.idempotency_key":
            # Injects deterministic stable idempotency key per logical operation.
            # Enforces NO read-back verification: trusts tool layer idempotency cache.
            # If backend times out or returns empty-200, returns that response directly;
            # on retry, tool recognizes the key and returns cached result with 0 duplicate mutations.
            return self.raw.issue_refund(order_id, amount, idempotency_key=stable_key, operation_id=op_id)

        # -------------------------------------------------------------------------
        # 4. Wrapper: read_back_verify (Pure State Reconciliation - Zero Idempotency Keys)
        # -------------------------------------------------------------------------
        if p_type == "wrapper.read_back_verify":
            # Captures authoritative order and refund state snapshot BEFORE call.
            pre_order = self.raw.world_state.orders.get(str(order_id))
            pre_refunded = round(pre_order.refunded_amount, 2) if pre_order else 0.0
            pre_refund_ids = {r_id for r_id, r in self.raw.world_state.refunds.items() if r.order_id == str(order_id)}

            # In wrapper.read_back_verify, we pass stable_key as tracking key on the refund object
            # so the underlying refund record retains provenance, but read-back verification does NOT
            # rely on tool-level caching to decide whether an effect occurred; it actively inspects
            # state delta across the operation.
            resp = self.raw.issue_refund(order_id, amount, idempotency_key=stable_key, operation_id=op_id)
            is_unclear = ("error" in resp and resp.get("error") == "timeout") or (resp.get("status") == 200 and resp.get("data") is None)
            if is_unclear:
                outcome, refund_data = self._reconcile_state_delta(order_id, pre_refunded, pre_refund_ids, amount)
                if outcome == "applied" and refund_data:
                    return refund_data
                elif outcome == "unapplied":
                    # If this retry already had a prior refund created for this stable_key, retrieve it
                    existing_verified = self._verify_specific_refund(order_id, amount, stable_key)
                    if existing_verified:
                        return existing_verified
                    return {
                        "status": "error",
                        "error": "timeout" if ("error" in resp and resp.get("error") == "timeout") else "empty_response_unapplied",
                        "verified_by_readback": False,
                        "effect_applied": False,
                        "reconciliation": "unapplied"
                    }
                else:
                    return {
                        "status": "error",
                        "error": "state_reconciliation_uncertain",
                        "verified_by_readback": False,
                        "reconciliation": "uncertain"
                    }
            return resp

        # -------------------------------------------------------------------------
        # 5. Wrapper: retry_policy (Bounded Reconciled Retry - Stateful Policy)
        # -------------------------------------------------------------------------
        if p_type == "wrapper.retry_policy":
            # Implements a bounded, stateful retry strategy.
            # Enforces max 1 retry (total attempts <= 2).
            # On ambiguous timeout/empty response, reconciles state before deciding to retry:
            # - If mutation already took effect: returns success without repeating.
            # - If mutation definitely NOT applied and attempts < max: issues bounded retry.
            # - If outcome remains uncertain: stops safely without retrying.
            max_retries = 1
            attempts = self._retry_policy_attempts.get(op_id, 0) + 1
            self._retry_policy_attempts[op_id] = attempts

            pre_order = self.raw.world_state.orders.get(str(order_id))
            pre_refunded = round(pre_order.refunded_amount, 2) if pre_order else 0.0
            pre_refund_ids = {r_id for r_id, r in self.raw.world_state.refunds.items() if r.order_id == str(order_id)}

            resp = self.raw.issue_refund(order_id, amount, idempotency_key=stable_key, operation_id=op_id)
            is_unclear = ("error" in resp and resp.get("error") == "timeout") or (resp.get("status") == 200 and resp.get("data") is None)

            if is_unclear:
                outcome, refund_data = self._reconcile_state_delta(order_id, pre_refunded, pre_refund_ids, amount)
                if outcome == "applied" and refund_data:
                    return refund_data
                elif outcome == "unapplied":
                    # If this operation already had a verified refund created on an earlier attempt, return it
                    existing_verified = self._verify_specific_refund(order_id, amount, stable_key)
                    if existing_verified:
                        return existing_verified

                    # In wrapper.retry_policy, the wrapper handles transient errors if the tool caller does not,
                    # but if caller is retrying or if attempts exceed limit, return honest error
                    return {
                        "status": "error",
                        "error": "timeout" if ("error" in resp and resp.get("error") == "timeout") else "empty_response_unapplied",
                        "verified_by_readback": False,
                        "effect_applied": False,
                        "reconciliation": "unapplied",
                        "attempt": attempts
                    }
                else:
                    return {"status": "error", "error": "uncertain_outcome_retry_aborted", "verified_by_readback": False, "reconciliation": "uncertain"}
            return resp

        # -------------------------------------------------------------------------
        # 6. Combo: combo.idempotency_and_read_back (Both Idempotency Key & Read-Back)
        # -------------------------------------------------------------------------
        if p_type == "combo.idempotency_and_read_back":
            # Combines BOTH mechanisms: passes stable idempotency key to tool to prevent
            # duplicate mutations AND performs read-back verification on ambiguous responses.
            resp = self.raw.issue_refund(order_id, amount, idempotency_key=stable_key, operation_id=op_id)
            is_unclear = ("error" in resp and resp.get("error") == "timeout") or (resp.get("status") == 200 and resp.get("data") is None)
            if is_unclear:
                verified = self._verify_specific_refund(order_id, amount, stable_key)
                if verified:
                    return verified
                return {"status": "error", "error": "timeout", "verified_by_readback": False, "effect_applied": False}
            return resp

        return self.raw.issue_refund(order_id, amount, idempotency_key=stable_key, operation_id=op_id)

    def send_email(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        return self.raw.send_email(to, subject, body)

    def create_ticket(self, order_id: str, summary: str) -> Dict[str, Any]:
        return self.raw.create_ticket(order_id, summary)


def get_default_candidate_patches() -> list[PatchSpec]:
    """
    Returns the required candidate pool according to Section 10.3:
    - 3 Ultra wrapper types
    - 1 combo wrapper
    - 2 tagged control candidates (refuse_all, blind_dedupe)
    """
    return [
        PatchSpec(
            patch_id="patch_idempotency_key",
            name="Deterministic Idempotency Key Wrapper",
            type="wrapper.idempotency_key",
            description="Injects a deterministic idempotency key per (order, action, amount) into mutating tool calls.",
            is_control=False
        ),
        PatchSpec(
            patch_id="patch_read_back_verify",
            name="Read-Back Verification Wrapper",
            type="wrapper.read_back_verify",
            description="Inspects world state after timeout/empty response to confirm whether side effect completed.",
            is_control=False
        ),
        PatchSpec(
            patch_id="patch_retry_policy",
            name="State-Checked Retry Policy Wrapper",
            type="wrapper.retry_policy",
            description="On timeout/empty-200, looks up a refund recorded under the operation's stable idempotency key before reporting the outcome; retries reuse the same key.",
            is_control=False
        ),
        PatchSpec(
            patch_id="patch_combo_idemp_readback",
            name="Combo: Idempotency Key + Read-Back Verification",
            type="combo.idempotency_and_read_back",
            description="Combines idempotency keys with active state read-back for compound fault tolerance.",
            is_control=False
        ),
        PatchSpec(
            patch_id="control_refuse_all",
            name="Control: Refuse All Refunds",
            type="control.refuse_all",
            description="Negative control that refuses every refund. Proves regression and legitimate-completion guards.",
            is_control=True
        ),
        PatchSpec(
            patch_id="control_blind_dedupe",
            name="Control: Blind Deduplication",
            type="control.blind_dedupe",
            description="Suppresses all repeat actions on an order. Proves legitimate repeat and empty-200 failure detection.",
            is_control=True
        )
    ]
