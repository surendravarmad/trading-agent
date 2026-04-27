"""
Phase IV (pre-flight) — RISK VALIDATION
Enforces all non-negotiable risk guardrails before any order is placed.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from trading_agent.strategy.strategy import SpreadPlan

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

    1. Plan structural validity
    2. Credit-to-Width ratio  ≥  min_credit_ratio
    3. Sold-strike |Delta|    ≤  max_delta
    4. Max loss               ≤  max_risk_pct × account_balance
    5. Account type           == "paper"
    6. Market is open
    7. Underlying liquidity   — bid/ask spread < max(liquidity_max_spread,
                                  liquidity_bps_of_mid × mid_price);
                                  if relative spread > stale_spread_pct
                                  the quote is treated as stale and the
                                  check is downgraded to a soft warning.
    8. Buying power           — available BP ≥ (1 - max_buying_power_pct) × (equity × margin_multiplier)
    """

    def __init__(self, max_risk_pct: float = 0.02,
                 min_credit_ratio: float = 0.33,
                 max_delta: float = 0.20,
                 liquidity_max_spread: float = 0.05,
                 liquidity_bps_of_mid: float = 0.0005,
                 stale_spread_pct: float = 0.01,
                 max_buying_power_pct: float = 0.80,
                 margin_multiplier: float = 2.0):
        self.max_risk_pct = max_risk_pct
        self.min_credit_ratio = min_credit_ratio
        self.max_delta = max_delta
        # Floor of the bid/ask gate, in dollars.  For low-priced
        # underlyings ($20-$100) this is the binding constraint.
        self.liquidity_max_spread = liquidity_max_spread
        # Slope of the bid/ask gate as a fraction of mid price.  5 bps
        # (0.0005) is a reasonable bound for liquid SPY/QQQ-class names
        # whose 1-2 cent absolute spread translates to ~0.2 bps, but
        # whose +$500 / +$700 names would otherwise fail a flat $0.05
        # absolute cap.  Effective threshold = max(absolute, bps × mid).
        self.liquidity_bps_of_mid = liquidity_bps_of_mid
        # Above this relative spread (default 1%) the quote is treated
        # as stale rather than illiquid — common on free-tier IEX feeds
        # outside RTH or right at the open.  Stale quotes don't fail
        # the trade, they just emit a WARNING so the operator can audit.
        self.stale_spread_pct = stale_spread_pct
        self.max_buying_power_pct = max_buying_power_pct
        self.margin_multiplier = margin_multiplier

    def evaluate(self, plan: SpreadPlan,
                 account_balance: float,
                 account_type: str = "paper",
                 market_open: bool = True,
                 force_market_open: bool = False,
                 underlying_bid_ask: Optional[Tuple[float, float]] = None,
                 account_buying_power: Optional[float] = None) -> RiskVerdict:
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

        # --- Check 7: underlying liquidity (bid/ask spread) ---
        # Scaled threshold: max($0.05 floor, 5 bps of mid).  A flat
        # 5-cent gate over-rejects high-priced underlyings (GOOG ~$170,
        # SPY ~$520) where a 1-2 cent absolute spread is normal but
        # 5 cents is also normal.  Stale-quote heuristic: when the
        # relative spread blows past stale_spread_pct (1%) the quote is
        # almost certainly delayed/stale on the free IEX feed; downgrade
        # to a WARN soft-pass rather than hard-fail the trade.
        if underlying_bid_ask is not None:
            bid, ask = underlying_bid_ask
            spread = ask - bid
            mid = (bid + ask) / 2.0 if (bid + ask) > 0 else 0.0
            scaled_floor = max(
                self.liquidity_max_spread,
                self.liquidity_bps_of_mid * mid,
            )
            rel_spread = (spread / mid) if mid > 0 else float("inf")
            if mid > 0 and rel_spread > self.stale_spread_pct:
                passed.append(
                    f"Underlying spread ${spread:.4f} / mid ${mid:.2f} "
                    f"= {rel_spread*100:.2f}% > stale threshold "
                    f"{self.stale_spread_pct*100:.1f}% — treating as "
                    f"STALE quote (soft-pass)"
                )
                logger.warning(
                    "[%s] Stale-quote soft-pass: spread=$%.4f rel=%.2f%% "
                    "(threshold %.1f%%) bid=$%.4f ask=$%.4f",
                    plan.ticker, spread, rel_spread * 100,
                    self.stale_spread_pct * 100, bid, ask,
                )
            elif spread < scaled_floor:
                passed.append(
                    f"Underlying bid/ask spread ${spread:.4f} "
                    f"< ${scaled_floor:.4f} (mid=${mid:.2f}, liquid)")
            else:
                failed.append(
                    f"Underlying bid/ask spread ${spread:.4f} "
                    f">= ${scaled_floor:.4f} "
                    f"(floor=${self.liquidity_max_spread:.2f}, "
                    f"{self.liquidity_bps_of_mid*1e4:.1f}bps×mid="
                    f"${self.liquidity_bps_of_mid*mid:.4f}, illiquid)")

        # --- Check 8: buying power availability ---
        if account_buying_power is not None and account_balance > 0:
            initial_bp = account_balance * self.margin_multiplier
            pct_used = 1.0 - (account_buying_power / initial_bp)
            if pct_used <= self.max_buying_power_pct:
                passed.append(
                    f"Buying power {pct_used*100:.1f}% used "
                    f"≤ {self.max_buying_power_pct*100:.0f}% limit")
            else:
                failed.append(
                    f"Buying power {pct_used*100:.1f}% used "
                    f"> {self.max_buying_power_pct*100:.0f}% limit "
                    f"— enter Liquidation Mode")

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
