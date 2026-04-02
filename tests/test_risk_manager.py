"""Tests for the risk management guardrails."""

import pytest
from trading_agent.risk_manager import RiskManager
from trading_agent.strategy import SpreadPlan, SpreadLeg


def _make_plan(net_credit=1.70, width=5.0, max_loss=330.0,
               ratio=0.34, delta=-0.18, valid=True) -> SpreadPlan:
    """Helper to create a plan with controllable parameters."""
    return SpreadPlan(
        ticker="SPY",
        strategy_name="Bull Put Spread",
        regime="bullish",
        legs=[
            SpreadLeg("SPY250425P00480000", 480.0, "sell", "put",
                      delta, -0.05, 1.80, 2.00, net_credit + 0.4),
            SpreadLeg("SPY250425P00475000", 475.0, "buy", "put",
                      -0.10, -0.03, 0.50, 0.70, 0.4),
        ],
        spread_width=width,
        net_credit=net_credit,
        max_loss=max_loss,
        credit_to_width_ratio=ratio,
        expiration="2025-04-25",
        reasoning="test",
        valid=valid,
    )


class TestRiskChecks:
    def setup_method(self):
        self.rm = RiskManager(max_risk_pct=0.02,
                              min_credit_ratio=0.25,
                              max_delta=0.25)

    def test_all_checks_pass(self):
        plan = _make_plan(ratio=0.34, delta=-0.18, max_loss=300)
        verdict = self.rm.evaluate(plan, account_balance=100_000,
                                    account_type="paper", market_open=True, force_market_open=False)
        assert verdict.approved is True
        assert len(verdict.checks_failed) == 0

    def test_reject_low_credit_ratio(self):
        plan = _make_plan(ratio=0.20)  # below 0.25 threshold
        verdict = self.rm.evaluate(plan, 100_000, "paper", True, False)
        assert verdict.approved is False
        assert any("Credit/Width" in f for f in verdict.checks_failed)

    def test_reject_high_delta(self):
        plan = _make_plan(delta=-0.30)  # above 0.25 threshold
        verdict = self.rm.evaluate(plan, 100_000, "paper", True, False)
        assert verdict.approved is False
        assert any("Δ" in f or "delta" in f.lower() for f in verdict.checks_failed)

    def test_reject_excessive_loss(self):
        plan = _make_plan(max_loss=5000)  # 2% of 100k = 2000
        verdict = self.rm.evaluate(plan, 100_000, "paper", True, False)
        assert verdict.approved is False
        assert any("Max loss" in f for f in verdict.checks_failed)

    def test_reject_live_account(self):
        plan = _make_plan(ratio=0.34, max_loss=300)
        verdict = self.rm.evaluate(plan, 100_000, "live", True, False)
        assert verdict.approved is False
        assert any("paper" in f.lower() for f in verdict.checks_failed)

    def test_reject_market_closed(self):
        plan = _make_plan(ratio=0.34, max_loss=300)
        verdict = self.rm.evaluate(plan, 100_000, "paper", False, False)
        assert verdict.approved is False
        assert any("CLOSED" in f for f in verdict.checks_failed)

    def test_reject_invalid_plan(self):
        plan = _make_plan(ratio=0.34, max_loss=300, valid=False)
        plan.rejection_reason = f"No call with |delta| ≤ {self.rm.max_delta}"
        verdict = self.rm.evaluate(plan, 100_000, "paper", True, False)
        assert verdict.approved is False

    def test_max_allowed_loss_calculation(self):
        plan = _make_plan(ratio=0.34, max_loss=300)
        verdict = self.rm.evaluate(plan, 50_000, "paper", True, False)
        assert verdict.max_allowed_loss == 1000.0  # 2% of 50k
