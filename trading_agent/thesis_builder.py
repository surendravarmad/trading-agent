"""
thesis_builder — structured trade-thesis formatter
==================================================

Extracted from agent.py during week 3-4 modularization.

The JournalKB raw_signal schema includes a ``thesis`` sub-dict with
three fields:

  • ``why``      — what the market is doing right now (regime, SMAs,
                   RSI, Bollinger width)
  • ``why_now``  — the specific trigger that caused this entry
                   (mean-reversion band touch, relative-strength edge,
                   or the plain classifier reasoning as fallback)
  • ``exit_plan``— expiration, profit target, and max-loss cap

Keeping this formatter separate from the agent core means:
  • The thesis format can evolve without touching orchestration
  • Tests can assert on thesis structure from inputs alone
  • Future callers (e.g. a "thesis preview" Streamlit widget) can
    reuse the same logic
"""

from __future__ import annotations

from typing import Dict


def build_thesis(analysis, plan, verdict) -> Dict[str, str]:
    """
    Build the three-part thesis dict.

    Parameters are duck-typed to match ``RegimeAnalysis``, ``SpreadPlan``,
    and ``RiskVerdict`` respectively — we don't import the classes here
    to keep this module dependency-free and easy to test.
    """
    mr_signal = getattr(analysis, "mean_reversion_signal", False)
    rs_vs_spy = getattr(analysis, "relative_strength_vs_spy", 0.0)

    why = (
        f"{analysis.regime.value.upper()} regime — "
        f"price={analysis.current_price:.2f}, "
        f"SMA50={analysis.sma_50:.2f}, SMA200={analysis.sma_200:.2f}, "
        f"RSI={analysis.rsi_14:.1f}, BB_width={analysis.bollinger_width:.4f}"
    )

    if mr_signal:
        direction = getattr(analysis, "mean_reversion_direction", "")
        why_now = (
            f"Price touched {direction} 3-std Bollinger Band — "
            f"mean reversion trade triggered"
        )
    elif rs_vs_spy > 0.001:
        why_now = (
            f"Ticker outperforming SPY by {rs_vs_spy * 100:.2f}% "
            f"in 5-min window — relative strength bias"
        )
    else:
        why_now = (analysis.reasoning or "")[:200]

    if plan.valid:
        profit_target = round(plan.net_credit * 0.5 * 100, 2)
        exit_plan = (
            f"Expiry {plan.expiration} | "
            f"Profit target: 50% of credit "
            f"(${profit_target:.2f}/contract) | "
            f"Max loss: ${plan.max_loss:.2f} | "
            f"Close if regime shifts adversely"
        )
    else:
        exit_plan = f"No trade — plan rejected: {plan.rejection_reason}"

    return {"why": why, "why_now": why_now, "exit_plan": exit_plan}
