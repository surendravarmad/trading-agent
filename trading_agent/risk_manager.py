"""
Phase IV (pre-flight) — RISK VALIDATION
Enforces all non-negotiable risk guardrails before any order is placed.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from trading_agent.strategy import SpreadPlan

logger = logging.getLogger(__name__)


@dataclass
class RiskVerdict:
    """Result of the risk-check pipeline."""
    approved: bool
    plan: SpreadPlan
    account_balance: float
    max_allowed_loss: float
    checks_passed: list
    checks_failed: list
    summary: str


class RiskManager:
    """
    Non-negotiable pre-trade risk checks:

    1. Credit-to-Width ratio  ≥  min_credit_ratio  (~0.25)
    2. Sold-strike |Delta|    ≤  max_delta          (~0.25)
    3. Max loss               ≤  max_risk_pct × account_balance
    4. Account type           == "paper"
    5. Market is open
    """

    def __init__(self, max_risk_pct: float = 0.02,
                 min_credit_ratio: float = 0.25,  # updated default
                 max_delta: float = 0.25):  # updated default
        self.max_risk_pct = max_risk_pct
        self.min_credit_ratio = min_credit_ratio
        self.max_delta = max_delta

    def evaluate(self, plan: SpreadPlan,
                 account_balance: float,
                 account_type: str = "paper",
                 market_open: bool = True,
                 force_market_open: bool = False) -> RiskVerdict:
        """
        Run all guardrails against *plan*.  Returns a RiskVerdict.
        """
        passed, failed = [], []

        # --- Check 1: plan validity (strategy planner already pre-checks) ---
        if plan.valid:
            passed.append("Plan is structurally valid")
        else:
            failed.append(f"Plan invalid: {plan.rejection_reason}")

        # --- Check 2: credit-to-width ratio ---
        if plan.credit_to_width_ratio >= self.min_credit_ratio:
            passed.append(
                f"Credit/Width ratio {plan.credit_to_width_ratio:.4f} "
                f"≥ {self.min_credit_ratio}")
        else:
            failed.append(
                f"Credit/Width ratio {plan.credit_to_width_ratio:.4f} "
                f"< {self.min_credit_ratio}")

        # --- Check 3: sold-strike delta ---
        sold_legs = [l for l in plan.legs if l.action == "sell"]
        for leg in sold_legs:
            if abs(leg.delta) <= self.max_delta:
                passed.append(
                    f"Sold {leg.strike} |Δ|={abs(leg.delta):.3f} ≤ {self.max_delta}")
            else:
                failed.append(
                    f"Sold {leg.strike} |Δ|={abs(leg.delta):.3f} > {self.max_delta}")

        # --- Check 4: max loss vs account ---
        max_allowed = round(account_balance * self.max_risk_pct, 2)
        if plan.max_loss <= max_allowed:
            passed.append(
                f"Max loss ${plan.max_loss} ≤ {self.max_risk_pct*100:.0f}% "
                f"of ${account_balance:,.2f} (=${max_allowed})")
        else:
            failed.append(
                f"Max loss ${plan.max_loss} > {self.max_risk_pct*100:.0f}% "
                f"of ${account_balance:,.2f} (=${max_allowed})")

        # --- Check 5: paper account ---
        if account_type.lower() == "paper":
            passed.append("Account type is PAPER")
        else:
            failed.append(f"Account type is '{account_type}' — must be 'paper'")

        # --- Check 6: market hours ---
        if market_open or force_market_open:
            passed.append("Market is currently OPEN")
        else:
            failed.append("Market is currently CLOSED")

        approved = len(failed) == 0
        summary = (f"{'APPROVED' if approved else 'REJECTED'}: "
                   f"{len(passed)} passed, {len(failed)} failed")

        verdict = RiskVerdict(
            approved=approved,
            plan=plan,
            account_balance=account_balance,
            max_allowed_loss=max_allowed,
            checks_passed=passed,
            checks_failed=failed,
            summary=summary,
        )

        level = logging.INFO if approved else logging.WARNING
        logger.log(level, "[%s] Risk verdict: %s", plan.ticker, summary)
        for msg in failed:
            logger.warning("  FAIL: %s", msg)

        return verdict
