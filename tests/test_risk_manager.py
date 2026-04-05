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


class TestLiquidityCheck:
    def setup_method(self):
        self.rm = RiskManager(max_risk_pct=0.02, min_credit_ratio=0.25,
                              max_delta=0.25, liquidity_max_spread=0.05)

    def test_liquid_underlying_passes(self):
        plan = _make_plan(ratio=0.34, max_loss=300)
        verdict = self.rm.evaluate(plan, 100_000, "paper", True, False,
                                   underlying_bid_ask=(499.98, 500.01))  # $0.03 spread
        assert verdict.approved is True
        assert any("liquid" in p.lower() for p in verdict.checks_passed)

    def test_illiquid_underlying_fails(self):
        plan = _make_plan(ratio=0.34, max_loss=300)
        verdict = self.rm.evaluate(plan, 100_000, "paper", True, False,
                                   underlying_bid_ask=(499.90, 499.96))  # $0.06 spread
        assert verdict.approved is False
        assert any("illiquid" in f.lower() for f in verdict.checks_failed)

    def test_no_bid_ask_skips_check(self):
        """When underlying_bid_ask is not provided, the check is skipped."""
        plan = _make_plan(ratio=0.34, max_loss=300)
        verdict = self.rm.evaluate(plan, 100_000, "paper", True, False,
                                   underlying_bid_ask=None)
        assert verdict.approved is True
        assert not any("liquid" in p.lower() for p in verdict.checks_passed)


class TestBuyingPowerCheck:
    def setup_method(self):
        self.rm = RiskManager(max_risk_pct=0.02, min_credit_ratio=0.25,
                              max_delta=0.25, max_buying_power_pct=0.80)

    def test_sufficient_buying_power_passes(self):
        plan = _make_plan(ratio=0.34, max_loss=300)
        # 50% buying power remaining (50% used)
        verdict = self.rm.evaluate(plan, 100_000, "paper", True, False,
                                   account_buying_power=50_000)
        assert verdict.approved is True
        assert any("buying power" in p.lower() for p in verdict.checks_passed)

    def test_exhausted_buying_power_fails(self):
        plan = _make_plan(ratio=0.34, max_loss=300)
        # Only 5% buying power remaining (95% used > 80% limit)
        verdict = self.rm.evaluate(plan, 100_000, "paper", True, False,
                                   account_buying_power=5_000)
        assert verdict.approved is False
        assert any("liquidation" in f.lower() for f in verdict.checks_failed)

    def test_no_buying_power_skips_check(self):
        plan = _make_plan(ratio=0.34, max_loss=300)
        verdict = self.rm.evaluate(plan, 100_000, "paper", True, False,
                                   account_buying_power=None)
        assert verdict.approved is True
        assert not any("buying power" in p.lower() for p in verdict.checks_passed)

    def test_check8_fresh_margin_passes(self):
        """Fresh margin account (bp = 2x equity, 0% deployed) must pass."""
        plan = _make_plan(ratio=0.34, max_loss=300)
        verdict = self.rm.evaluate(plan, 100_000, "paper", True, False,
                                   account_buying_power=200_000)
        assert any("buying power" in p.lower() for p in verdict.checks_passed)

    def test_check8_margin_70pct_used_passes(self):
        """70% of initial buying power deployed (< 80% limit) must pass."""
        plan = _make_plan(ratio=0.34, max_loss=300)
        # 70% used: remaining = 200k * 0.30 = $60k
        verdict = self.rm.evaluate(plan, 100_000, "paper", True, False,
                                   account_buying_power=60_000)
        assert any("buying power" in p.lower() for p in verdict.checks_passed)

    def test_check8_margin_85pct_used_fails(self):
        """85% of initial buying power deployed (> 80% limit) must fail."""
        plan = _make_plan(ratio=0.34, max_loss=300)
        # 85% used: remaining = 200k * 0.15 = $30k
        verdict = self.rm.evaluate(plan, 100_000, "paper", True, False,
                                   account_buying_power=30_000)
        assert any("liquidation" in f.lower() for f in verdict.checks_failed)
