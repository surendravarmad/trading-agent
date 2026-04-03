"""Tests for the strategy planner — spread selection and strike picking."""

import pytest
from unittest.mock import MagicMock

from trading_agent.strategy import StrategyPlanner, SpreadPlan
from trading_agent.regime import Regime, RegimeAnalysis
from trading_agent.market_data import MarketDataProvider


def _make_analysis(regime: Regime) -> RegimeAnalysis:
    return RegimeAnalysis(
        regime=regime,
        current_price=500.0,
        sma_50=498.0,
        sma_200=490.0,
        sma_50_slope=0.5,
        rsi_14=55.0,
        bollinger_width=0.06,
        reasoning="Test analysis",
    )


class TestStrategySelection:

    def _make_planner(self, put_chain=None, call_chain=None):
        provider = MagicMock(spec=MarketDataProvider)
        provider.fetch_option_chain.side_effect = lambda ticker, exp, opt_type: (
            put_chain if opt_type == "put" else call_chain
        )
        return StrategyPlanner(provider, max_delta=0.20, min_credit_ratio=0.33)

    def test_bullish_selects_bull_put(self, sample_put_contracts):
        planner = self._make_planner(put_chain=sample_put_contracts)
        plan = planner.plan("SPY", _make_analysis(Regime.BULLISH))
        assert plan.strategy_name == "Bull Put Spread"
        assert len(plan.legs) == 2
        # Sold leg should have higher strike
        sold = [l for l in plan.legs if l.action == "sell"]
        bought = [l for l in plan.legs if l.action == "buy"]
        assert len(sold) == 1
        assert len(bought) == 1
        assert sold[0].strike > bought[0].strike

    def test_bearish_selects_bear_call(self, sample_call_contracts):
        planner = self._make_planner(call_chain=sample_call_contracts)
        plan = planner.plan("SPY", _make_analysis(Regime.BEARISH))
        assert plan.strategy_name == "Bear Call Spread"
        assert len(plan.legs) == 2
        sold = [l for l in plan.legs if l.action == "sell"]
        bought = [l for l in plan.legs if l.action == "buy"]
        assert sold[0].strike < bought[0].strike

    def test_sideways_selects_iron_condor(self, sample_put_contracts,
                                           sample_call_contracts):
        planner = self._make_planner(
            put_chain=sample_put_contracts,
            call_chain=sample_call_contracts)
        plan = planner.plan("SPY", _make_analysis(Regime.SIDEWAYS))
        assert plan.strategy_name == "Iron Condor"
        assert len(plan.legs) == 4

    def test_empty_chain_returns_invalid_plan(self):
        planner = self._make_planner(put_chain=None, call_chain=None)
        plan = planner.plan("SPY", _make_analysis(Regime.BULLISH))
        assert plan.valid is False
        assert "unavailable" in plan.rejection_reason.lower() or \
               "no" in plan.rejection_reason.lower()


class TestPickExpiration:
    """Verify _pick_expiration always returns a Friday within DTE_RANGE."""

    def _planner(self):
        return StrategyPlanner(MagicMock(), max_delta=0.20, min_credit_ratio=0.33)

    def _dte(self, expiry_str, from_date):
        from datetime import datetime
        exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        return (exp - from_date).days

    def test_result_is_always_friday(self):
        from datetime import datetime
        planner = self._planner()
        exp = planner._pick_expiration()
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        assert exp_date.weekday() == 4, f"{exp} is not a Friday"

    def test_result_within_dte_range(self):
        from datetime import datetime
        planner = self._planner()
        exp = planner._pick_expiration()
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        from datetime import date
        dte = (exp_date - date.today()).days
        lo, hi = planner.DTE_RANGE
        assert lo <= dte <= hi, f"DTE={dte} is outside range {lo}-{hi}"

    def test_both_valid_fridays_picks_further_one(self):
        """When both adjacent Fridays are within DTE_RANGE, pick the further one
        (higher DTE = less gamma risk) — capital retainment DTE bias update."""
        from datetime import datetime, date
        from unittest.mock import patch

        planner = self._planner()
        # fake_today = April 2, 2026 → TARGET_DTE=45 → target = May 17 (Sunday)
        # next_friday = May 22 (DTE=50, at max boundary) — VALID
        # prev_friday = May 15 (DTE=43)                 — VALID
        # Both in DTE_RANGE (35, 50) → prefer further = May 22
        fake_today = date(2026, 4, 2)
        with patch("trading_agent.strategy.datetime") as mock_dt:
            mock_dt.now.return_value.date.return_value = fake_today
            exp = planner._pick_expiration()

        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        assert exp_date.weekday() == 4, "Expiry must be a Friday"
        assert exp_date == date(2026, 5, 22), "Should prefer the higher-DTE Friday"
        assert (exp_date - fake_today).days == 50  # exactly at DTE_RANGE upper bound

    def test_only_prev_friday_valid_picks_prev(self):
        """When next_friday exceeds max DTE, fall back to prev_friday."""
        from datetime import datetime, date
        from unittest.mock import patch

        planner = self._planner()
        # fake_today = April 1, 2026 → target = May 16 (Saturday)
        # next_friday = May 22 (DTE=51 > 50) — INVALID (exceeds max)
        # prev_friday = May 15 (DTE=44)       — VALID
        fake_today = date(2026, 4, 1)
        with patch("trading_agent.strategy.datetime") as mock_dt:
            mock_dt.now.return_value.date.return_value = fake_today
            exp = planner._pick_expiration()

        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        assert exp_date.weekday() == 4, "Expiry must be a Friday"
        assert exp_date == date(2026, 5, 15), "Should pick prev_friday when next is out of range"
        dte = (exp_date - fake_today).days
        assert planner.DTE_RANGE[0] <= dte <= planner.DTE_RANGE[1]

    def test_old_utcnow_bug_would_have_returned_outside_range(self):
        """Regression: old utcnow roll-forward logic returned May 22 (51 DTE)."""
        from datetime import date, timedelta
        fake_today = date(2026, 4, 2)   # UTC date that caused the failure
        target = fake_today + timedelta(days=44)   # May 16 (Saturday)
        days_until_friday = (4 - target.weekday()) % 7
        broken_expiry = target + timedelta(days=days_until_friday)
        broken_dte = (broken_expiry - fake_today).days
        # Confirm the old code would have been outside range
        assert broken_dte > 45


class TestMeanReversionStrategy:
    def _make_planner(self, put_chain=None, call_chain=None):
        provider = MagicMock(spec=MarketDataProvider)
        provider.fetch_option_chain.side_effect = lambda ticker, exp, opt_type: (
            put_chain if opt_type == "put" else call_chain
        )
        return StrategyPlanner(provider, max_delta=0.20, min_credit_ratio=0.33)

    def _make_mr_analysis(self, direction: str) -> RegimeAnalysis:
        from trading_agent.regime import Regime
        return RegimeAnalysis(
            regime=Regime.MEAN_REVERSION,
            current_price=500.0,
            sma_50=498.0,
            sma_200=490.0,
            sma_50_slope=0.5,
            rsi_14=55.0,
            bollinger_width=0.06,
            reasoning="3-std BB touch",
            mean_reversion_signal=True,
            mean_reversion_direction=direction,
        )

    def test_upper_band_touch_sells_bear_call(self, sample_call_contracts):
        planner = self._make_planner(call_chain=sample_call_contracts)
        plan = planner.plan("SPY", self._make_mr_analysis("upper"))
        assert plan.strategy_name == "Mean Reversion Spread"
        assert "upper" in plan.reasoning.lower() or "reversion" in plan.reasoning.lower()
        sold = [l for l in plan.legs if l.action == "sell"]
        bought = [l for l in plan.legs if l.action == "buy"]
        assert sold[0].strike < bought[0].strike  # bear call structure

    def test_lower_band_touch_sells_bull_put(self, sample_put_contracts):
        planner = self._make_planner(put_chain=sample_put_contracts)
        plan = planner.plan("SPY", self._make_mr_analysis("lower"))
        assert plan.strategy_name == "Mean Reversion Spread"
        sold = [l for l in plan.legs if l.action == "sell"]
        bought = [l for l in plan.legs if l.action == "buy"]
        assert sold[0].strike > bought[0].strike  # bull put structure


class TestRelativeStrengthBias:
    def _make_planner(self, put_chain=None, call_chain=None):
        provider = MagicMock(spec=MarketDataProvider)
        provider.fetch_option_chain.side_effect = lambda ticker, exp, opt_type: (
            put_chain if opt_type == "put" else call_chain
        )
        return StrategyPlanner(provider, max_delta=0.20, min_credit_ratio=0.33)

    def test_sideways_with_rs_picks_bull_put(self, sample_put_contracts):
        """SIDEWAYS regime + relative strength outperforming → Bull Put Spread."""
        from trading_agent.regime import Regime
        planner = self._make_planner(put_chain=sample_put_contracts)
        analysis = RegimeAnalysis(
            regime=Regime.SIDEWAYS,
            current_price=500.0, sma_50=498.0, sma_200=490.0,
            sma_50_slope=0.5, rsi_14=55.0, bollinger_width=0.06,
            reasoning="sideways",
            relative_strength_vs_spy=0.002,   # +0.2% outperforming
        )
        plan = planner.plan("SPY", analysis)
        assert plan.strategy_name == "Bull Put Spread"

    def test_sideways_without_rs_picks_iron_condor(self, sample_put_contracts,
                                                     sample_call_contracts):
        """SIDEWAYS regime with no RS signal → Iron Condor."""
        from trading_agent.regime import Regime
        planner = self._make_planner(put_chain=sample_put_contracts,
                                      call_chain=sample_call_contracts)
        analysis = RegimeAnalysis(
            regime=Regime.SIDEWAYS,
            current_price=500.0, sma_50=498.0, sma_200=490.0,
            sma_50_slope=0.0, rsi_14=50.0, bollinger_width=0.06,
            reasoning="sideways",
            relative_strength_vs_spy=0.0,
            relative_strength_vs_qqq=0.0,
        )
        plan = planner.plan("SPY", analysis)
        assert plan.strategy_name == "Iron Condor"


class TestPlanSerialization:
    def test_to_dict_roundtrip(self, valid_spread_plan):
        d = valid_spread_plan.to_dict()
        assert d["ticker"] == "SPY"
        assert d["strategy"] == "Bull Put Spread"
        assert len(d["legs"]) == 2
        assert isinstance(d["max_loss"], (int, float))
