"""
Causal Side-Effect Ledger and World State implementation.
Sections 5.3, 5.4, 7 of the Project Spec.

World state is ground truth.
Every mutation goes through the mock tool layer and is logged to the ledger
with a monotonically increasing sequence number.
"""

from typing import Dict, Any, List, Optional
from src.silentfail.models import Order, Refund, Email, Ticket, LedgerEvent, LedgerRecord, ExecutionCost
import uuid
import copy


class WorldState:
    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self.orders: Dict[str, Order] = {}
        self.refunds: Dict[str, Refund] = {}
        self.emails: List[Email] = []
        self.tickets: List[Ticket] = []
        
        if initial_state:
            for ord_data in initial_state.get("orders", []):
                self.orders[ord_data["id"]] = Order(**ord_data)
            for ref_data in initial_state.get("refunds", []):
                self.refunds[ref_data["id"]] = Refund(**ref_data)
            for em_data in initial_state.get("emails", []):
                self.emails.append(Email(**em_data))
            for tk_data in initial_state.get("tickets", []):
                self.tickets.append(Ticket(**tk_data))

    def get_summary(self) -> Dict[str, Any]:
        refund_count = len(self.refunds)
        refund_total = sum(r.amount for r in self.refunds.values())
        email_count = len(self.emails)
        ticket_count = len(self.tickets)
        return {
            "refund_count": refund_count,
            "refund_total": round(refund_total, 2),
            "email_count": email_count,
            "ticket_count": ticket_count,
            "orders": {k: v.model_dump() for k, v in self.orders.items()},
            "refunds": [r.model_dump() for r in self.refunds.values()]
        }


class CausalLedger:
    def __init__(self, run_id: Optional[str] = None, scenario_id: str = "unknown", agent_version: str = "baseline", patch_id: Optional[str] = None):
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        self.scenario_id = scenario_id
        self.agent_version = agent_version
        self.patch_id = patch_id
        self.events: List[LedgerEvent] = []
        self._seq = 1

    def record_tool_request(self, tool: str, args: Dict[str, Any]) -> int:
        req_seq = self._seq
        self.events.append(LedgerEvent(
            seq=self._seq,
            type="tool_request",
            tool=tool,
            args=copy.deepcopy(args)
        ))
        self._seq += 1
        return req_seq

    def record_world_mutation(self, effect: str, details: Optional[Dict[str, Any]] = None, refund_id: Optional[str] = None, amount: Optional[float] = None) -> int:
        mut_seq = self._seq
        self.events.append(LedgerEvent(
            seq=self._seq,
            type="world_mutation",
            effect=effect,
            refund_id=refund_id,
            amount=amount,
            extra=copy.deepcopy(details)
        ))
        self._seq += 1
        return mut_seq

    def record_tool_response(self, observed: Dict[str, Any], fault: Optional[str] = None, tool: Optional[str] = None) -> int:
        resp_seq = self._seq
        self.events.append(LedgerEvent(
            seq=self._seq,
            type="tool_response_to_agent",
            tool=tool,
            observed=copy.deepcopy(observed),
            fault=fault
        ))
        self._seq += 1
        return resp_seq

    def to_record(self, final_message: str, final_state: Dict[str, Any], cost: Optional[ExecutionCost] = None) -> LedgerRecord:
        return LedgerRecord(
            run_id=self.run_id,
            scenario_id=self.scenario_id,
            agent_version=self.agent_version,
            patch_id=self.patch_id,
            events=copy.deepcopy(self.events),
            final_message=final_message,
            final_state=copy.deepcopy(final_state),
            cost=cost or ExecutionCost()
        )
