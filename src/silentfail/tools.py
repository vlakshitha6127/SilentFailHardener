from typing import Dict, Any, List, Optional
from src.silentfail.models import FaultConfig, Refund, Email, Ticket
from src.silentfail.ledger import WorldState, CausalLedger
import uuid


class FaultInjector:
    def __init__(self, faults: Optional[List[FaultConfig]] = None):
        self.faults = faults or []
        self.call_counters: Dict[str, int] = {}

    def get_fault_for_call(self, tool_name: str) -> Optional[FaultConfig]:
        idx = self.call_counters.get(tool_name, 0)
        self.call_counters[tool_name] = idx + 1
        for f in self.faults:
            if f.tool == tool_name and f.call_index == idx:
                return f
        return None


class ToolRegistry:
    def __init__(self, world_state: WorldState, ledger: CausalLedger, fault_injector: Optional[FaultInjector] = None):
        self.world_state = world_state
        self.ledger = ledger
        self.fault_injector = fault_injector or FaultInjector()
        self._refund_key_cache: Dict[str, Dict[str, Any]] = {}

    def get_order(self, order_id: str) -> Dict[str, Any]:
        self.ledger.record_tool_request("get_order", {"order_id": order_id})
        fault = self.fault_injector.get_fault_for_call("get_order")
        order = self.world_state.orders.get(str(order_id))
        real_result = order.model_dump() if order else {"error": f"Order {order_id} not found"}
        if fault and fault.type == "timeout_after_success":
            observed = {"error": "timeout"}
            self.ledger.record_tool_response(observed, fault="timeout_after_success", tool="get_order")
            return observed
        elif fault and fault.type == "empty_200":
            observed = {"status": 200, "data": None}
            self.ledger.record_tool_response(observed, fault="empty_200", tool="get_order")
            return observed
        self.ledger.record_tool_response(real_result, tool="get_order")
        return real_result

    def issue_refund(
        self,
        order_id: str,
        amount: float,
        idempotency_key: Optional[str] = None,
        operation_id: Optional[str] = None,
        op_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.ledger.record_tool_request(
            "issue_refund",
            {
                "order_id": order_id,
                "amount": amount,
                "idempotency_key": idempotency_key,
                "operation_id": operation_id,
            },
        )
        fault = self.fault_injector.get_fault_for_call("issue_refund")

        # operation_id is the caller's stable identity for a logical refund.
        # Preserve the legacy idempotency_key when supplied, and use the
        # operation_id as its fallback for callers that follow the agent contract.
        refund_key = idempotency_key or operation_id

        # Idempotency check at tool layer (if caller supplied either identity)
        if refund_key and refund_key in self._refund_key_cache:
            cached_result = self._refund_key_cache[refund_key]
            # No world mutation performed for duplicate key!
            self.ledger.record_tool_response(cached_result, tool="issue_refund")
            return cached_result

        # Determine real effect
        order = self.world_state.orders.get(str(order_id))
        if not order:
            err = {"error": f"Order {order_id} not found"}
            self.ledger.record_tool_response(err, tool="issue_refund")
            return err

        pre_refunded_amount = order.refunded_amount
        pre_refund_ids = set(self.world_state.refunds)

        remaining_balance = round(order.total - order.refunded_amount, 2)
        if amount > remaining_balance or remaining_balance <= 0:
            err = {"error": "Order already fully refunded or requested amount exceeds remaining balance", "remaining_balance": remaining_balance}
            self.ledger.record_tool_response(err, tool="issue_refund")
            return err

        # Check fault conditions that suppress mutation.
        # - empty_200 with effect_applied=False: the effect was never applied; suppress mutation.
        # - empty_200 with effect_applied=True: the effect WAS applied; we must still apply the
        #   mutation to materialize it in world state, then return verified_by_readback=True.
        # - duplicate_webhook: replay event only, no new mutation.
        apply_mutation = True
        if fault and fault.type == "empty_200" and fault.effect_applied is False:
            apply_mutation = False
        if fault and fault.type == "duplicate_webhook":
            apply_mutation = False

        if apply_mutation:
            ref_id = f"rf_{uuid.uuid4().hex[:6]}"
            new_refund = Refund(id=ref_id, order_id=str(order_id), amount=amount, idempotency_key=refund_key)
            self.world_state.refunds[ref_id] = new_refund
            order.refunded_amount = round(order.refunded_amount + amount, 2)
            if order.refunded_amount >= order.total:
                order.status = "refunded"
            else:
                order.status = "partially_refunded"
            self.ledger.record_world_mutation(
                "refund_created",
                details={"order_id": order_id, "amount": amount, "operation_id": operation_id},
                refund_id=ref_id,
                amount=amount,
            )
            real_result = {"status": "ok", "refund_id": ref_id, "amount": amount, "order_id": order_id}
            if refund_key:
                self._refund_key_cache[refund_key] = real_result
        else:
            real_result = {"status": "skipped", "reason": "fault_injected_no_effect"}

        # Determine what the agent observes
        if fault:
            if fault.type == "timeout_after_success":
                observed = {"error": "timeout"}
                self.ledger.record_tool_response(observed, fault="timeout_after_success", tool="issue_refund")
                return observed
            elif fault.type == "empty_200":
                # Determine observed result based on whether the effect was applied.
                if fault.effect_applied is True:
                    # The mutation was already applied earlier; locate the original refund.
                    existing_refund_id = None
                    # Candidates are refunds that existed before this call for the same order.
                    candidates = [
                        rid for rid in pre_refund_ids
                        if (r := self.world_state.refunds.get(rid)) and r.order_id == str(order_id)
                    ]
                    if candidates:
                        # Prefer matching amount if possible; choose most recent by created_at.
                        amount_match = [rid for rid in candidates if self.world_state.refunds[rid].amount == amount]
                        if amount_match:
                            existing_refund_id = max(amount_match, key=lambda rid: self.world_state.refunds[rid].created_at)
                        else:
                            # No amount match -> do not reuse any refund.
                            existing_refund_id = None
                    observed = {
                        "status": "ok",
                        "refund_id": existing_refund_id,
                        "verified_by_readback": True,
                    }
                    self.ledger.record_tool_response(observed, fault="empty_200", tool="issue_refund")
                    return observed
                else:
                    # Read-back verification for empty_200 where effect may not have been applied.
                    new_refunds = [
                        refund
                        for refund_id, refund in self.world_state.refunds.items()
                        if refund_id not in pre_refund_ids and refund.order_id == str(order_id)
                    ]
                    refunded_delta = round(order.refunded_amount - pre_refunded_amount, 2)
                    if (
                        len(new_refunds) == 1
                        and new_refunds[0].amount == amount
                        and refunded_delta == amount
                    ):
                        observed = {**real_result, "verified_by_readback": True}
                    elif not new_refunds and refunded_delta == 0:
                        observed = {
                            "status": "error",
                            "error": "refund_not_applied",
                            "verified_by_readback": False,
                            "state_checked": True,
                            "effect_applied": False,
                        }
                    else:
                        observed = {
                            "status": "error",
                            "error": "refund_state_uncertain",
                            "verified_by_readback": False,
                            "state_checked": True,
                        }
                    self.ledger.record_tool_response(observed, fault="empty_200", tool="issue_refund")
                    return observed
            elif fault.type == "duplicate_webhook":
                # Simulate a duplicate webhook replay: find the original refund for this order
                original_refund_id = None
                for rid, refund in self.world_state.refunds.items():
                    if refund.order_id == str(order_id) and refund.amount == amount:
                        original_refund_id = rid
                        break
                observed = {
                    "status": "ok",
                    "event": "webhook.refund.completed",
                    "refund_id": original_refund_id,
                    "order_id": order_id,
                    "amount": amount,
                }
                self.ledger.record_tool_response(observed, fault="duplicate_webhook", tool="issue_refund")
                return observed
        self.ledger.record_tool_response(real_result, tool="issue_refund")
        return real_result

    def send_email(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        self.ledger.record_tool_request("send_email", {"to": to, "subject": subject, "body": body})
        fault = self.fault_injector.get_fault_for_call("send_email")
        apply_mutation = True
        if fault and fault.type in ("empty_200", "duplicate_webhook"):
            apply_mutation = False
        em_id = f"em_{uuid.uuid4().hex[:6]}"
        if apply_mutation:
            new_em = Email(id=em_id, to=to, subject=subject, body=body)
            self.world_state.emails.append(new_em)
            self.ledger.record_world_mutation("email_sent", details={"to": to, "subject": subject})
            real_result = {"status": "ok", "email_id": em_id}
        else:
            real_result = {"status": "skipped", "reason": "fault_injected_no_effect"}
        if fault:
            if fault.type == "timeout_after_success":
                observed = {"error": "timeout"}
                self.ledger.record_tool_response(observed, fault="timeout_after_success", tool="send_email")
                return observed
            elif fault.type == "empty_200":
                observed = {"status": 200, "data": None}
                self.ledger.record_tool_response(observed, fault="empty_200", tool="send_email")
                return observed
            elif fault.type == "duplicate_webhook":
                observed = {"status": "skipped", "reason": "duplicate_webhook"}
                self.ledger.record_tool_response(observed, fault="duplicate_webhook", tool="send_email")
                return observed
        self.ledger.record_tool_response(real_result, tool="send_email")
        return real_result

    def create_ticket(self, order_id: str, summary: str) -> Dict[str, Any]:
        self.ledger.record_tool_request("create_ticket", {"order_id": order_id, "summary": summary})
        fault = self.fault_injector.get_fault_for_call("create_ticket")
        tk_id = f"tk_{uuid.uuid4().hex[:6]}"
        new_tk = Ticket(id=tk_id, order_id=str(order_id), summary=summary)
        self.world_state.tickets.append(new_tk)
        self.ledger.record_world_mutation("ticket_created", details={"order_id": order_id, "summary": summary})
        real_result = {"status": "ok", "ticket_id": tk_id}
        if fault and fault.type == "timeout_after_success":
            observed = {"error": "timeout"}
            self.ledger.record_tool_response(observed, fault="timeout_after_success", tool="create_ticket")
            return observed
        self.ledger.record_tool_response(real_result, tool="create_ticket")
        return real_result