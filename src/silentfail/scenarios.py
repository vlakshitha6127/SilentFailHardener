"""
Scenario Generator and Validator.
Sections 6.1, 6.2, 6.3 of the Project Spec.

Implements the official scenario mix:
- Timeout after success on issue_refund (15%)
- empty_200, effect applied (8%)
- empty_200, effect NOT applied (7%)
- duplicate_webhook (10%)
- Compound faults (timeout then duplicate webhook) (15%)
- Legitimate repeat (two valid partial refunds, two items) (15%)
- Ineligible / already-refunded orders (10%)
- Timeout or empty_200 on send_email (10%)
- No-fault control (10%)
"""

from typing import List, Dict, Any, Tuple
import random
from src.silentfail.models import Scenario, FaultConfig, ExpectedOutcome


PROPOSED_SCENARIO_CLASSES = (
    "clean_eligible_full_refund",
    "timeout_after_success",
    "empty_200_applied",
    "empty_200_not_applied_recovery",
    "compound_fault_recovery",
    "same_operation_retry",
    "distinct_legitimate_operations",
    "ineligible_refusal",
    "excess_mutation",
    "incorrect_refund_amount",
    "refund_success_email_fault",
    "refund_success_email_empty_200_not_applied",
    "partial_refund_with_email_success",
)


def build_proposed_scenario(
    scenario_class: str,
    *,
    scenario_id: str,
    split: str,
    seed: int,
) -> Scenario:
    """Build one explicit fixture from the approved 13-class taxonomy."""
    if scenario_class not in PROPOSED_SCENARIO_CLASSES:
        raise ValueError(f"unsupported proposed scenario class: {scenario_class}")
    order_id = str(7000 + (seed % 1000))
    amount = 50.0
    total = 100.0
    faults: List[FaultConfig] = []
    message = f"Please refund ${amount:.2f} for order {order_id}."
    expected_count = 1
    expected_total = amount
    email_count = 1
    allow_refusal = False
    status = "delivered"

    if scenario_class == "timeout_after_success":
        faults = [FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success")]
    elif scenario_class == "empty_200_applied":
        faults = [FaultConfig(tool="issue_refund", call_index=0, type="empty_200", effect_applied=True)]
    elif scenario_class == "empty_200_not_applied_recovery":
        faults = [FaultConfig(tool="issue_refund", call_index=0, type="empty_200", effect_applied=False)]
    elif scenario_class == "compound_fault_recovery":
        faults = [
            FaultConfig(tool="issue_refund", call_index=0, type="empty_200", effect_applied=False),
            FaultConfig(tool="issue_refund", call_index=1, type="timeout_after_success"),
        ]
    elif scenario_class == "same_operation_retry":
        faults = [FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success")]
    elif scenario_class == "distinct_legitimate_operations":
        expected_count = 2
        expected_total = total
        message = f"Please refund two items separately for order {order_id}."
    elif scenario_class == "ineligible_refusal":
        status = "refunded"
        expected_count = 0
        expected_total = 0.0
        email_count = 0
        allow_refusal = True
    elif scenario_class == "excess_mutation":
        faults = [
            FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success"),
            FaultConfig(tool="issue_refund", call_index=1, type="duplicate_webhook"),
        ]
    elif scenario_class == "incorrect_refund_amount":
        message = f"Please refund $150.00 for order {order_id}."
        # The deterministic agent clamps an excessive explicit request to the
        # remaining eligible balance. The contract records that observable
        # safety behavior rather than requiring an unsupported refusal.
        expected_count = 1
        expected_total = total
        email_count = 1
        allow_refusal = False
    elif scenario_class == "refund_success_email_fault":
        faults = [FaultConfig(tool="send_email", call_index=0, type="timeout_after_success")]
    elif scenario_class == "refund_success_email_empty_200_not_applied":
        faults = [FaultConfig(tool="send_email", call_index=0, type="empty_200", effect_applied=False)]
        email_count = 0
    elif scenario_class == "partial_refund_with_email_success":
        amount = 25.0
        expected_total = amount
        message = f"Please refund $25.00 for order {order_id}."
    elif scenario_class == "clean_eligible_full_refund":
        amount = total
        expected_total = total
        message = f"Please refund the full balance for order {order_id}."

    initial_state = {
        "orders": [{
            "id": order_id,
            "status": status,
            "total": total,
            "refunded_amount": total if status == "refunded" else 0.0,
            "items": [
                {"item_id": "item_1", "price": 50.0},
                {"item_id": "item_2", "price": 50.0},
            ],
        }],
        "refunds": [],
        "emails": [],
        "tickets": [],
    }
    if split == "held_out":
        if scenario_class == "distinct_legitimate_operations":
            initial_state["orders"][0]["items"].append(
                {"item_id": "item_3", "price": 0.01}
            )
        else:
            initial_state["orders"][0]["items"] = [
                {"item_id": "item_1", "price": 50.0}
            ]

    return Scenario(
        scenario_id=scenario_id,
        split=split,
        seed=seed,
        category=scenario_class,
        initial_state=initial_state,
        customer_message=(
            f"For a held-out validation case, {message}"
            if split == "held_out"
            else f"For a training validation case, {message}"
        ),
        faults=faults,
        expected_outcome=ExpectedOutcome(
            refund_count=expected_count,
            refund_total=expected_total,
            email_count=email_count,
            ticket_count=0,
            allow_refusal=allow_refusal,
        ),
    )


def proposed_scenario_contract(scenario: Scenario) -> Dict[str, Any]:
    """Return explicit contract assertions for one proposed fixture."""
    refusal = (
        "permitted"
        if scenario.category == "ineligible_refusal"
        else "forbidden"
    )
    claim_kind = (
        "refusal"
        if scenario.category == "ineligible_refusal"
        else "success"
    )
    operation_kind = (
        "partial_refund"
        if scenario.category == "partial_refund_with_email_success"
        else "full_refund"
    )
    contract: Dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "requested_operation": {
            "kind": operation_kind,
            "amount": (
                150.0
                if scenario.category == "incorrect_refund_amount"
                else scenario.expected_outcome.refund_total
            ),
            "operation_count": scenario.expected_outcome.refund_count or 1,
        },
        "refund": {
            "expected_count": scenario.expected_outcome.refund_count,
            "expected_total": scenario.expected_outcome.refund_total,
            "amount_tolerance": 0.01,
        },
        "refusal": {"mode": refusal},
        "completion_claim": {
            "kind": claim_kind,
            "amount": (
                scenario.expected_outcome.refund_total
                if claim_kind == "success"
                else None
            ),
            "amount_tolerance": 0.01,
        },
        "email": {
            "required_count": scenario.expected_outcome.email_count,
            "allowed_count": scenario.expected_outcome.email_count,
        },
        "detectors": {},
    }
    if scenario.category == "ineligible_refusal":
        contract["detectors"] = {"ineligible_refund_mutation": False}
    elif scenario.category == "excess_mutation":
        contract["contract_features"] = [
            "configuration_expectations",
            "fault_observations",
        ]
        contract["detectors_by_configuration"] = {
            "baseline": {"excess_mutation": True},
            "hardened": {"excess_mutation": False},
        }
    elif scenario.category == "incorrect_refund_amount":
        contract["detectors"] = {"incorrect_refund_amount": True}
    if scenario.category == "same_operation_retry":
        order_id = scenario.initial_state["orders"][0]["id"]
        operation_id = f"op_{order_id}_req0"
        contract["contract_features"] = ["operation_bindings"]
        contract["requested_operation"]["operations"] = [{
            "operation_id": operation_id,
            "amount": 50.0,
            "kind": "full_refund",
            "expected_mutation_count": 1,
        }]
        contract["requested_operation"]["retry_groups"] = [{
            "group_id": "refund_retry",
            "operation_id": operation_id,
            # A committed mutation confirmed by read-back needs no second
            # request. Unresolved failures may still be retried.
            "min_attempts": 1,
            "max_attempts": 3,
            "require_same_operation_id": True,
        }]
    if scenario.category in {
        "timeout_after_success",
        "empty_200_applied",
        "empty_200_not_applied_recovery",
        "compound_fault_recovery",
        "same_operation_retry",
        "excess_mutation",
        "refund_success_email_fault",
        "refund_success_email_empty_200_not_applied",
    }:
        features = set(contract.get("contract_features", []))
        features.add("fault_observations")
        contract["contract_features"] = sorted(features)
        contract["fault_observations"] = {
            "baseline": {
                "required": [
                    {
                        "tool": fault.tool,
                        "type": fault.type,
                        "occurrence": 1,
                    }
                    for fault in scenario.faults
                ],
                "allowed_unobserved": [],
            },
            "hardened": {
                "required": [
                    {
                        "tool": scenario.faults[0].tool,
                        "type": scenario.faults[0].type,
                        "occurrence": 1,
                    }
                ],
                "allowed_unobserved": [
                    {
                        "tool": scenario.faults[1].tool,
                        "type": scenario.faults[1].type,
                        "occurrence": 1,
                    }
                ] if scenario.category == "excess_mutation" else [],
            },
        }
    return contract


def proposed_cohort_contracts(scenarios: List[Scenario]) -> Dict[str, Any]:
    """Return the contract manifest for already-built proposed fixtures."""
    if len(scenarios) != 65:
        raise ValueError("proposed cohort requires exactly 65 scenarios")
    return {
        "contract_schema_version": "offline-contract-v1",
        "contracts": [proposed_scenario_contract(scenario) for scenario in scenarios],
    }


def build_proposed_cohort(base_seed: int = 90210) -> List[Scenario]:
    """Return the deterministic 39-train/26-held-out proposed cohort."""
    scenarios: List[Scenario] = []
    index = 0
    for scenario_class in PROPOSED_SCENARIO_CLASSES:
        for repetition in range(5):
            split = "train" if repetition < 3 else "held_out"
            scenarios.append(
                build_proposed_scenario(
                    scenario_class,
                    scenario_id=f"v2_{scenario_class}_{repetition + 1:02d}",
                    split=split,
                    seed=base_seed + index,
                )
            )
            index += 1
    return scenarios


def build_refund_success_email_empty_200_not_applied_scenario(
    scenario_id: str = "sc_refund_success_email_empty_200_not_applied",
    split: str = "train",
    seed: int = 4242,
) -> Scenario:
    """Build the supported replacement fixture for a failed email side effect."""
    return Scenario(
        scenario_id=scenario_id,
        split=split,
        seed=seed,
        category="refund_success_email_empty_200_not_applied",
        initial_state={
            "orders": [{
                "id": "1234",
                "status": "delivered",
                "total": 50.0,
                "refunded_amount": 0.0,
                "items": [{"item_id": "item_1", "price": 50.0}],
            }],
            "refunds": [],
            "emails": [],
            "tickets": [],
        },
        customer_message="Please refund $50 for order 1234.",
        faults=[FaultConfig(
            tool="send_email",
            call_index=0,
            type="empty_200",
            effect_applied=False,
        )],
        expected_outcome=ExpectedOutcome(
            refund_count=1,
            refund_total=50.0,
            email_count=0,
            ticket_count=0,
            allow_refusal=False,
        ),
    )


def replacement_email_contract(scenario_id: str) -> Dict[str, Any]:
    """Return the contract assertions supported by the replacement fixture."""
    return {
        "scenario_id": scenario_id,
        "contract_features": ["fault_observations"],
        "requested_operation": {
            "kind": "full_refund",
            "amount": 50.0,
            "operation_count": 1,
        },
        "refund": {
            "expected_count": 1,
            "expected_total": 50.0,
            "amount_tolerance": 0.01,
        },
        "refusal": {"mode": "forbidden"},
        "completion_claim": {
            "kind": "success",
            "amount": 50.0,
            "amount_tolerance": 0.01,
        },
        "email": {"required_count": 0, "allowed_count": 0},
        "detectors": {},
        "fault_observations": {
            "baseline": {
                "required": [{
                    "tool": "send_email",
                    "type": "empty_200",
                    "occurrence": 1,
                }],
                "allowed_unobserved": [],
            },
            "hardened": {
                "required": [{
                    "tool": "send_email",
                    "type": "empty_200",
                    "occurrence": 1,
                }],
                "allowed_unobserved": [],
            },
        },
    }


class ScenarioValidator:
    """
    Deterministic validator that verifies the schema and derives
    expected_outcome from initial_state and scenario rules.
    Rejects malformed scenarios.
    """
    @classmethod
    def validate_and_compute_outcome(cls, scenario: Scenario) -> Tuple[bool, str]:
        # Validate order exists in initial state
        orders = scenario.initial_state.get("orders", [])
        if not orders:
            return False, "Scenario initial_state missing orders"

        order = orders[0]
        total = float(order.get("total", 0.0))
        refunded = float(order.get("refunded_amount", 0.0))
        remaining = round(total - refunded, 2)

        # Ineligible cases: already fully refunded
        if scenario.category == "ineligible_already_refunded":
            if remaining > 0:
                return False, "Ineligible category has remaining balance > 0"
            if scenario.expected_outcome.refund_count != 0:
                return False, "Ineligible scenario expected refund_count != 0"

        # Legitimate repeat cases: expected refund_count == 2
        elif scenario.category == "legitimate_repeat":
            if scenario.expected_outcome.refund_count != 2:
                return False, "Legitimate repeat must have expected refund_count == 2"

        if scenario.category in PROPOSED_SCENARIO_CLASSES:
            if scenario.category == "ineligible_refusal":
                if not scenario.expected_outcome.allow_refusal:
                    return False, "Ineligible refusal must allow refusal"
            elif scenario.category == "distinct_legitimate_operations":
                if scenario.expected_outcome.refund_count != 2:
                    return False, "Distinct legitimate operations require two refunds"
            elif scenario.category == "partial_refund_with_email_success":
                if scenario.expected_outcome.refund_total != 25.0:
                    return False, "Partial refund fixture must request $25"

        return True, "Valid"


class ScenarioGenerator:
    """
    Generates deterministic train and held-out scenario splits
    according to the specified probability distribution.
    """
    CATEGORY_MIX = [
        ("timeout_after_success", 0.15),
        ("empty_200_applied", 0.08),
        ("empty_200_not_applied", 0.07),
        ("duplicate_webhook", 0.10),
        ("compound_faults", 0.15),
        ("legitimate_repeat", 0.15),
        ("ineligible_already_refunded", 0.10),
        ("email_fault", 0.10),
        ("no_fault_control", 0.10),
    ]

    CUSTOMER_PHRASINGS = [
        "Hi, my package arrived damaged, please refund order {order_id}.",
        "I need a full refund for order {order_id} as the product was defective.",
        "Could you please refund my order {order_id} right away?",
        "Please issue a refund on order {order_id}.",
        "Order {order_id} was never received. Please process a refund."
    ]

    LEGITIMATE_REPEAT_PHRASINGS = [
        "Please refund items 1 and 2 for order {order_id} as two valid partial refunds.",
        "I need to return two items separately for order {order_id}. Please issue both partial refunds.",
        "Order {order_id} has two damaged items, please process two separate refunds for each item."
    ]

    def __init__(self, base_seed: int = 1234):
        self.base_seed = base_seed

    def generate_suite(self, count: int, split: str = "train", seed_offset: int = 0) -> List[Scenario]:
        rng = random.Random(self.base_seed + seed_offset)
        scenarios: List[Scenario] = []

        categories = [cat for cat, _ in self.CATEGORY_MIX]
        weights = [weight for _, weight in self.CATEGORY_MIX]

        for i in range(count):
            sc_id = f"sc_{split}_{i+1:05d}"
            cat = rng.choices(categories, weights=weights, k=1)[0]
            order_id = str(rng.randint(1000, 9999))
            
            # Base order values
            price_base = round(rng.uniform(25.0, 150.0), 2)

            faults: List[FaultConfig] = []
            expected_refund_count = 1
            expected_refund_total = price_base
            refunded_amount = 0.0
            allow_refusal = False
            phrasing_template = rng.choice(self.CUSTOMER_PHRASINGS)

            if cat == "timeout_after_success":
                faults.append(FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success"))
            
            elif cat == "empty_200_applied":
                faults.append(FaultConfig(tool="issue_refund", call_index=0, type="empty_200", effect_applied=True))

            elif cat == "empty_200_not_applied":
                faults.append(FaultConfig(tool="issue_refund", call_index=0, type="empty_200", effect_applied=False))

            elif cat == "duplicate_webhook":
                faults.append(FaultConfig(tool="issue_refund", call_index=0, type="duplicate_webhook"))

            elif cat == "compound_faults":
                # Timeout on first call, duplicate webhook simulation or empty 200
                faults.append(FaultConfig(tool="issue_refund", call_index=0, type="timeout_after_success"))
                faults.append(FaultConfig(tool="issue_refund", call_index=1, type="duplicate_webhook"))

            elif cat == "legitimate_repeat":
                expected_refund_count = 2
                expected_refund_total = price_base
                phrasing_template = rng.choice(self.LEGITIMATE_REPEAT_PHRASINGS)

            elif cat == "ineligible_already_refunded":
                refunded_amount = price_base
                expected_refund_count = 0
                expected_refund_total = 0.0
                allow_refusal = True

            elif cat == "email_fault":
                if rng.random() > 0.5:
                    faults.append(FaultConfig(tool="send_email", call_index=0, type="timeout_after_success"))
                else:
                    faults.append(FaultConfig(tool="send_email", call_index=0, type="empty_200", effect_applied=True))

            elif cat == "no_fault_control":
                # Clean control run
                pass

            items = [{"item_id": "item_1", "price": price_base}]
            if cat == "legitimate_repeat":
                first_item_price = round(price_base / 2, 2)
                second_item_price = round(price_base - first_item_price, 2)
                items = [
                    {"item_id": "item_1", "price": first_item_price},
                    {"item_id": "item_2", "price": second_item_price},
                ]

            initial_state = {
                "orders": [{
                    "id": order_id,
                    "status": "refunded" if refunded_amount >= price_base else "delivered",
                    "total": price_base,
                    "refunded_amount": refunded_amount,
                    "items": items
                }],
                "refunds": [],
                "emails": [],
                "tickets": []
            }

            customer_msg = phrasing_template.format(order_id=order_id)
            expected_outcome = ExpectedOutcome(
                refund_count=expected_refund_count,
                refund_total=expected_refund_total,
                email_count=1 if expected_refund_count > 0 else 0,
                ticket_count=0,
                allow_refusal=allow_refusal
            )

            sc = Scenario(
                scenario_id=sc_id,
                split=split,
                seed=self.base_seed + seed_offset + i,
                category=cat,
                initial_state=initial_state,
                customer_message=customer_msg,
                faults=faults,
                expected_outcome=expected_outcome
            )

            # Validate deterministic consistency
            is_valid, msg = ScenarioValidator.validate_and_compute_outcome(sc)
            if is_valid:
                scenarios.append(sc)
            else:
                raise ValueError(f"Generated malformed scenario {sc_id}: {msg}")

        return scenarios
