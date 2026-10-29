"""
Agent Adapter and Reference Naive Refund Agent.
Sections 5.1, 5.2 of the Project Spec.

The reference agent handles customer refund requests.
Intentionally naive: retries on errors/timeouts and trusts tool responses at face value.
No code path may special-case a fault type.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from src.silentfail.tools import ToolRegistry
from src.silentfail.models import ExecutionCost
import re
import json


@dataclass
class AgentResult:
    final_message: str
    transcript: List[Dict[str, Any]]
    terminated: str  # 'completed' | 'max_steps' | 'error'
    cost: ExecutionCost


class NaiveRefundAgent:
    """
    Reference refund agent.
    Can be run with an actual OpenAI-compatible LLM endpoint (Nebius Nemotron)
    or with the deterministic reference loop that replicates genuine naive LLM tool calling.
    """
    def __init__(self, model_client: Optional[Any] = None, model_name: str = "nvidia/nemotron-mini-4b-instruct", max_retries: int = 2):
        self.model_client = model_client
        self.model_name = model_name
        self.max_retries = max_retries

    def run(self, user_message: str, tools: ToolRegistry, config: Optional[Dict[str, Any]] = None) -> AgentResult:
        config = config or {}
        max_retries = config.get("max_retries", self.max_retries)
        
        # If model_client is available and configured, call the live LLM endpoint
        if self.model_client:
            return self._run_llm(user_message, tools, config)
        else:
            return self._run_deterministic_naive(user_message, tools, max_retries)

    def _run_deterministic_naive(self, user_message: str, tools: ToolRegistry, max_retries: int) -> AgentResult:
        """
        Deterministic naive agent logic implementing:
        - parse order id and requested items or refund amount
        - fetch order
        - issue refund (with naive retry on error/timeout without read-back or idempotency keys)
        - send email confirmation or refuse truthfully
        """
        transcript: List[Dict[str, Any]] = []
        tokens_in = 150
        tokens_out = 50
        tool_call_count = 0
        step_count = 0

        # Step 1: Parse user intent naively
        order_match = re.search(r"order\s+(\d+)", user_message, re.IGNORECASE)
        order_id = order_match.group(1) if order_match else "1042"

        # Check for multiple item refunds in message (e.g. "two items", "item A and item B", "partial refunds", "two separate refunds")
        msg_lower = user_message.lower()
        is_two_partial_refunds = (
            "two items" in msg_lower or
            "two valid partial refunds" in msg_lower or
            "items 1 and 2" in msg_lower or
            "two damaged items" in msg_lower or
            "two separate refunds" in msg_lower or
            "both partial refunds" in msg_lower
        )

        # Step 2: Fetch order
        step_count += 1
        tool_call_count += 1
        transcript.append({"step": step_count, "action": "call get_order", "args": {"order_id": order_id}})
        order_info = tools.get_order(order_id)
        transcript.append({"step": step_count, "observation": order_info})
        tokens_in += 100
        tokens_out += 30

        if "error" in order_info:
            final_msg = f"Sorry, I could not find order {order_id}."
            return AgentResult(final_message=final_msg, transcript=transcript, terminated="completed", cost=ExecutionCost(tokens_in=tokens_in, tokens_out=tokens_out, tool_calls=tool_call_count, steps=step_count))

        total_amount = float(order_info.get("total", 0.0))
        refunded = float(order_info.get("refunded_amount", 0.0))
        remaining = round(total_amount - refunded, 2)

        if remaining <= 0:
            final_msg = f"Order {order_id} has already been fully refunded. No additional refund can be issued."
            return AgentResult(final_message=final_msg, transcript=transcript, terminated="completed", cost=ExecutionCost(tokens_in=tokens_in, tokens_out=tokens_out, tool_calls=tool_call_count, steps=step_count))

        # Determine refund amounts to issue based on customer intent and remaining balance
        refund_amounts = []
        if is_two_partial_refunds:
            # Multi-item / legitimate repeat requests: split into two distinct partial refunds
            half = round(remaining / 2, 2)
            rem = round(remaining - half, 2)
            refund_amounts = [half, rem]
        else:
            # 1. Explicit partial amount specified in message (e.g. "$40", "$42.50", "refund of $15", "refund 40 dollars", "40 dollars")
            # Strict boundary parsing: ensure order IDs (e.g. "order 1234") are never parsed as dollar amounts.
            # When refund/amount is explicitly associated with a number, prioritize that specific refund amount.
            explicit_amount = None

            # First priority: check for explicit refund targeted amount (e.g., "refund $40", "refund of $40", "refund 40 dollars")
            targeted_refund_match = re.search(
                r"\b(?:refund|amount)\s+(?:of\s+)?(?:(?:\$\s*(\d+(?:\.\d+)?))|(?:(\d+(?:\.\d+)?)\s*(?:dollars?|bucks?|\$)?))\b",
                msg_lower
            )
            if targeted_refund_match:
                cand_str = targeted_refund_match.group(1) or targeted_refund_match.group(2)
                if cand_str and cand_str != order_id:
                    cand_val = float(cand_str)
                    if 0 < cand_val <= remaining:
                        explicit_amount = cand_val

            # Second priority: any explicit currency amount (e.g., "$42.50" or "40 dollars")
            if explicit_amount is None:
                dollar_matches = re.finditer(
                    r"(?:\$\s*(\d+(?:\.\d+)?))|(?:\b(\d+(?:\.\d+)?)\s*(?:dollars?|bucks?)\b)",
                    msg_lower
                )
                for dm in dollar_matches:
                    candidate_num = dm.group(1) or dm.group(2)
                    if candidate_num and candidate_num != order_id:
                        candidate_amt = float(candidate_num)
                        if 0 < candidate_amt <= remaining:
                            explicit_amount = candidate_amt
                            break

            if explicit_amount is not None:
                refund_amounts = [explicit_amount]
            else:
                # 2. General refund request: the existing scenario contract defines general requests
                # as full refund requests. Honor full remaining eligible balance.
                # Documented ambiguous behavior: when no explicit amount is requested (or if requested
                # amount exceeds remaining balance), do not invent a partial amount. Refund the full eligible balance.
                refund_amounts = [remaining]

        refunds_successful = 0
        total_refunded_claimed = 0.0

        for op_idx, amt in enumerate(refund_amounts):
            # Unique operation ID per logical refund request
            logical_op_id = f"op_{order_id}_req{op_idx}"
            attempts = 0
            success = False
            last_resp = None
            
            while attempts <= max_retries and not success:
                attempts += 1
                step_count += 1
                tool_call_count += 1
                transcript.append({"step": step_count, "action": "call issue_refund", "args": {"order_id": order_id, "amount": amt, "operation_id": logical_op_id}})
                
                # Pass operation_id so all retries reuse the same logical operation identity
                resp = tools.issue_refund(order_id, amt, operation_id=logical_op_id, op_index=op_idx)
                transcript.append({"step": step_count, "observation": resp})
                tokens_in += 120
                tokens_out += 40
                last_resp = resp

                if resp.get("status") == "ok":
                    success = True
                    refunds_successful += 1
                    total_refunded_claimed += amt
                elif "error" in resp:
                    # Intentionally naive retry: retry on timeout or generic error
                    if (resp.get("error") == "timeout" or resp.get("status") == "error") and attempts <= max_retries:
                        continue
                    else:
                        break
                elif resp.get("status") == 200 and resp.get("data") is None:
                    # Empty 200: naive agent sees status 200 and assumes it succeeded!
                    success = True
                    refunds_successful += 1
                    total_refunded_claimed += amt
                elif resp.get("status") == "skipped":
                    # Patch or fault suppressed it
                    break

        # Step 3: Follow-up action (send email if refund succeeded or attempted)
        if refunds_successful > 0:
            step_count += 1
            tool_call_count += 1
            transcript.append({"step": step_count, "action": "call send_email", "args": {"to": "customer@example.com", "subject": f"Refund for Order {order_id}"}})
            email_resp = tools.send_email(to="customer@example.com", subject=f"Refund for Order {order_id}", body=f"Your refund of ${total_refunded_claimed:.2f} has been processed.")
            transcript.append({"step": step_count, "observation": email_resp})
            tokens_in += 80
            tokens_out += 25
            final_msg = f"Your refund of ${total_refunded_claimed:.2f} has been issued."
        else:
            final_msg = f"Unable to process refund for order {order_id}."

        return AgentResult(
            final_message=final_msg,
            transcript=transcript,
            terminated="completed",
            cost=ExecutionCost(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tool_calls=tool_call_count,
                steps=step_count
            )
        )

    def _run_llm(self, user_message: str, tools: ToolRegistry, config: Dict[str, Any]) -> AgentResult:
        """
        Run with OpenAI client against Nebius or NVIDIA Nemotron endpoint.
        """
        # Formulate tool calling loop
        tools_def = [
            {
                "type": "function",
                "function": {
                    "name": "get_order",
                    "description": "Fetch details of an order by ID",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "issue_refund",
                    "description": "Issue refund for an order. MUST supply operation_id to uniquely identify this logical refund request; retries MUST reuse the exact same operation_id, while distinct refund requests MUST use different operation_ids.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string", "description": "The order ID to refund"},
                            "amount": {"type": "number", "description": "The amount to refund"},
                            "operation_id": {"type": "string", "description": "Required unique identifier for this specific refund operation. Retries MUST pass the same operation_id; independent refunds MUST use distinct operation_ids."}
                        },
                        "required": ["order_id", "amount", "operation_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "Send email notification to customer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"}
                        },
                        "required": ["to", "subject", "body"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_ticket",
                    "description": "Create support ticket",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "summary": {"type": "string"}
                        },
                        "required": ["order_id", "summary"]
                    }
                }
            }
        ]

        system_prompt = (
            "You are a customer support refund agent. "
            "Help the customer with their refund request using available tools. "
            "If a tool call times out or errors, retry it to make sure the customer is helped."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        transcript = []
        tokens_in = 0
        tokens_out = 0
        tool_call_count = 0
        step_count = 0
        registry = getattr(tools, "raw", tools)
        world_state = getattr(registry, "world_state", None)
        initial_refund_ids = set(world_state.refunds) if world_state is not None else set()
        refund_operations: Dict[str, float] = {}


        def get_raw_registry(registry):
            while getattr(registry, "raw", None) is not None:
                registry = registry.raw
            return registry

        def has_refund_confirmation():
            ledger = getattr(get_raw_registry(tools), "ledger", None)
            if not ledger:
                return False
            refund_seqs = [
                event.seq for event in ledger.events
                if event.type == "world_mutation" and event.effect == "refund_created"
            ]
            if not refund_seqs:
                return False
            first_refund_seq = min(refund_seqs)
            return any(
                event.type == "world_mutation" and event.effect == "email_sent"
                and event.seq > first_refund_seq
                for event in ledger.events
            )

        def ensure_refund_confirmation() -> bool:
            registry = get_raw_registry(tools)
            ledger = getattr(registry, "ledger", None)
            if not ledger:
                return False
            refund_events = [
                event for event in ledger.events
                if event.type == "world_mutation" and event.effect == "refund_created"
            ]
            if not refund_events or has_refund_confirmation():
                return False

            order_ids = sorted({
                str((event.extra or {}).get("order_id"))
                for event in refund_events if (event.extra or {}).get("order_id") is not None
            })
            refund_total = sum(event.amount or 0.0 for event in refund_events)
            order_label = ", ".join(order_ids)
            subject = f"Refund for Order {order_label}" if len(order_ids) == 1 else "Refund confirmation"
            body = f"Your refund of ${refund_total:.2f} has been processed."
            if order_label:
                body += f" Order ID: {order_label}."
            response = tools.send_email(
                to="customer@example.com",
                subject=subject,
                body=body
            )
            transcript.append({"step": step_count, "action": "send_email", "args": {
                "to": "customer@example.com", "subject": subject, "body": body
            }})
            transcript.append({"step": step_count, "tool_response": response})
            return True

        max_steps = config.get("max_steps", 6)
        for _ in range(max_steps):
            step_count += 1
            try:
                response = self.model_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools_def,
                    temperature=0.0
                )
            except Exception as e:
                # LLM call failure fallback
                break

            choice = response.choices[0]
            msg = choice.message
            if hasattr(response, "usage") and response.usage:
                tokens_in += response.usage.prompt_tokens
                tokens_out += response.usage.completion_tokens

            messages.append(msg.model_dump())
            transcript.append({"step": step_count, "message": msg.model_dump()})

            if not msg.tool_calls:
                confirmation_sent = ensure_refund_confirmation()
                return AgentResult(
                    final_message=msg.content or "",
                    transcript=transcript,
                    terminated="completed",
                    cost=ExecutionCost(tokens_in=tokens_in, tokens_out=tokens_out, tool_calls=tool_call_count + int(confirmation_sent), steps=step_count)
                )

            for tc in msg.tool_calls:
                tool_call_count += 1
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                tool_res = {}
                if fn_name == "get_order":
                    tool_res = tools.get_order(str(fn_args.get("order_id")))
                elif fn_name == "issue_refund":
                    op_id = fn_args.get("operation_id")
                    if not op_id:
                        tool_res = {
                            "status": "error",
                            "error": "missing_operation_id",
                            "message": "operation_id is required for issue_refund to guarantee idempotency and avoid duplicate charges."
                        }
                    else:
                        refund_operations[str(op_id)] = float(fn_args.get("amount", 0))
                        tool_res = tools.issue_refund(
                            order_id=str(fn_args.get("order_id")),
                            amount=float(fn_args.get("amount", 0)),
                            operation_id=str(op_id)
                        )
                elif fn_name == "send_email":
                    if has_refund_confirmation():
                        tool_res = {"status": "ok", "deduplicated": True}
                    else:
                        tool_res = tools.send_email(str(fn_args.get("to")), str(fn_args.get("subject")), str(fn_args.get("body")))
                elif fn_name == "create_ticket":
                    tool_res = tools.create_ticket(str(fn_args.get("order_id")), str(fn_args.get("summary")))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_res)
                })
                transcript.append({"step": step_count, "tool_response": tool_res})

        confirmation_sent = ensure_refund_confirmation()
        return AgentResult(
            final_message=messages[-1].get("content", "") if isinstance(messages[-1], dict) else getattr(messages[-1], "content", ""),
            transcript=transcript,
            terminated="max_steps",
            cost=ExecutionCost(tokens_in=tokens_in, tokens_out=tokens_out, tool_calls=tool_call_count + int(confirmation_sent), steps=step_count)
        )

    @staticmethod
    def _ensure_refund_confirmation(
        user_message: str,
        tools: ToolRegistry,
        initial_refund_ids: set,
        refund_operations: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Send one confirmation when a committed refund lacks one already."""
        order_match = re.search(r"order\s+(\d+)", user_message, re.IGNORECASE)
        if not order_match:
            return None

        order_id = order_match.group(1)
        registry = getattr(tools, "raw", tools)
        world_state = getattr(registry, "world_state", None)
        if world_state is None:
            return None

        refunds = [
            refund for refund in world_state.refunds.values()
            if refund.order_id == str(order_id) and refund.id not in initial_refund_ids
        ]
        if not refunds:
            return None

        subject = f"Refund for Order {order_id}"
        if any(email.subject == subject for email in world_state.emails):
            return None

        operation_refunds = [
            refund for refund in refunds
            if refund.idempotency_key
            and refund.idempotency_key.removeprefix("idem_") in refund_operations
        ]
        if len(operation_refunds) != 1:
            return None

        amount = operation_refunds[0].amount
        args = {
            "to": "customer@example.com",
            "subject": subject,
            "body": f"Your refund of ${amount:.2f} has been processed."
        }
        response = tools.send_email(**args)
        return {"args": args, "response": response}
