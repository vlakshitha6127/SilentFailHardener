with open('src/silentfail/tools.py', 'r') as f:
    lines = f.readlines()

# Lines 129-216 (0-indexed: 128-215) are the broken section.
# We keep lines 1-128 (0-indexed 0-127) and lines 217+ (0-indexed 216+).
before = lines[:128]  # up to and including line 128 (return observed for timeout)
after = lines[216:]   # from line 217 onward (self.ledger.record_tool_response(real_result...))

replacement = [
    '            elif fault.type == "empty_200":\n',
    '                # Determine observed result based on whether the effect was applied.\n',
    '                if fault.effect_applied is True:\n',
    '                    # The mutation was already applied earlier; locate the original refund.\n',
    '                    existing_refund_id = None\n',
    '                    # Candidates are refunds that existed before this call for the same order.\n',
    '                    candidates = [\n',
    '                        rid for rid in pre_refund_ids\n',
    '                        if (r := self.world_state.refunds.get(rid)) and r.order_id == str(order_id)\n',
    '                    ]\n',
    '                    if candidates:\n',
    '                        # Prefer matching amount if possible; choose most recent by created_at.\n',
    '                        amount_match = [rid for rid in candidates if self.world_state.refunds[rid].amount == amount]\n',
    '                        if amount_match:\n',
    '                            existing_refund_id = max(amount_match, key=lambda rid: self.world_state.refunds[rid].created_at)\n',
    '                        else:\n',
    '                            # No amount match -> do not reuse any refund.\n',
    '                            existing_refund_id = None\n',
    '                    observed = {\n',
    '                        "status": "ok",\n',
    '                        "refund_id": existing_refund_id,\n',
    '                        "verified_by_readback": True,\n',
    '                    }\n',
    '                    self.ledger.record_tool_response(observed, fault="empty_200", tool="issue_refund")\n',
    '                    return observed\n',
    '                else:\n',
    '                    # Read-back verification for empty_200 where effect may not have been applied.\n',
    '                    new_refunds = [\n',
    '                        refund\n',
    '                        for refund_id, refund in self.world_state.refunds.items()\n',
    '                        if refund_id not in pre_refund_ids and refund.order_id == str(order_id)\n',
    '                    ]\n',
    '                    refunded_delta = round(order.refunded_amount - pre_refunded_amount, 2)\n',
    '                    if (\n',
    '                        len(new_refunds) == 1\n',
    '                        and new_refunds[0].amount == amount\n',
    '                        and refunded_delta == amount\n',
    '                    ):\n',
    '                        observed = {**real_result, "verified_by_readback": True}\n',
    '                    elif not new_refunds and refunded_delta == 0:\n',
    '                        observed = {\n',
    '                            "status": "error",\n',
    '                            "error": "refund_not_applied",\n',
    '                            "verified_by_readback": False,\n',
    '                            "state_checked": True,\n',
    '                            "effect_applied": False,\n',
    '                        }\n',
    '                    else:\n',
    '                        observed = {\n',
    '                            "status": "error",\n',
    '                            "error": "refund_state_uncertain",\n',
    '                            "verified_by_readback": False,\n',
    '                            "state_checked": True,\n',
    '                        }\n',
    '                    self.ledger.record_tool_response(observed, fault="empty_200", tool="issue_refund")\n',
    '                    return observed\n',
    '            elif fault.type == "duplicate_webhook":\n',
    '                # Simulate a duplicate webhook replay: find the original refund for this order\n',
    '                original_refund_id = None\n',
    '                for rid, refund in self.world_state.refunds.items():\n',
    '                    if refund.order_id == str(order_id) and refund.amount == amount:\n',
    '                        original_refund_id = rid\n',
    '                        break\n',
    '                observed = {\n',
    '                    "status": "ok",\n',
    '                    "event": "webhook.refund.completed",\n',
    '                    "refund_id": original_refund_id,\n',
    '                    "order_id": order_id,\n',
    '                    "amount": amount,\n',
    '                }\n',
    '                self.ledger.record_tool_response(observed, fault="duplicate_webhook", tool="issue_refund")\n',
    '                return observed\n',
]

new_content = before + replacement + after

with open('src/silentfail/tools.py', 'w') as f:
    f.writelines(new_content)

print(f"Done. Old file had {len(lines)} lines, new file has {len(new_content)} lines.")
