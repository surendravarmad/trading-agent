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


class TestPlanSerialization:
    def test_to_dict_roundtrip(self, valid_spread_plan):
        d = valid_spread_plan.to_dict()
        assert d["ticker"] == "SPY"
        assert d["strategy"] == "Bull Put Spread"
        assert len(d["legs"]) == 2
        assert isinstance(d["max_loss"], (int, float))
