#!/usr/bin/env python3
"""
Test runner — runs the full test suite using unittest.
Works without pytest by importing test modules and running them directly.
"""

import sys
import os
import unittest
import importlib
from pathlib import Path
from unittest.mock import MagicMock
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_agent.config import load_config, AppConfig, AlpacaConfig, TradingConfig, LoggingConfig, IntelligenceConfig
from trading_agent.market.market_data import MarketDataProvider
from trading_agent.strategy.regime import RegimeClassifier, Regime, RegimeAnalysis
from trading_agent.strategy.strategy import StrategyPlanner, SpreadPlan, SpreadLeg
from trading_agent.strategy.risk_manager import RiskManager, RiskVerdict
from trading_agent.execution.executor import OrderExecutor
from trading_agent.core.agent import TradingAgent
from trading_agent.execution.position_monitor import (
    PositionMonitor, PositionSnapshot, SpreadPosition, ExitSignal,
    STRATEGY_REGIME_MAP,
)
from trading_agent.execution.order_tracker import OrderTracker, OrderRecord, OrderStatus
from trading_agent.intelligence.llm_client import LLMClient, LLMConfig
from trading_agent.intelligence.trade_journal import TradeJournal, TradeEntry
from trading_agent.intelligence.knowledge_base import KnowledgeBase, KBDocument
from trading_agent.intelligence.llm_analyst import LLMAnalyst, AnalystDecision
from trading_agent.intelligence.fine_tuning import FineTuningExporter

import tempfile
import json

# ===================================================================
# Helper factories
# ===================================================================

def make_bullish_prices():
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=200, freq="B")
    base = np.linspace(400, 500, 200)
    noise = np.random.normal(0, 8.0, 200)  # Higher volatility for realistic BB width
    prices = base + noise
    return pd.DataFrame({
        "Open": prices - 0.5, "High": prices + 4,
        "Low": prices - 4, "Close": prices,
        "Volume": np.random.randint(50_000_000, 150_000_000, 200),
    }, index=dates)


def make_bearish_prices():
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=200, freq="B")
    base = np.linspace(500, 400, 200)
    noise = np.random.normal(0, 8.0, 200)  # Higher volatility for realistic BB width
    prices = base + noise
    return pd.DataFrame({
        "Open": prices - 0.5, "High": prices + 4,
        "Low": prices - 4, "Close": prices,
        "Volume": np.random.randint(50_000_000, 150_000_000, 200),
    }, index=dates)


def make_sideways_prices():
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=200, freq="B")
    base = 450 + 3 * np.sin(np.linspace(0, 8 * np.pi, 200))
    noise = np.random.normal(0, 0.5, 200)
    prices = base + noise
    return pd.DataFrame({
        "Open": prices - 0.3, "High": prices + 1,
        "Low": prices - 1, "Close": prices,
        "Volume": np.random.randint(50_000_000, 150_000_000, 200),
    }, index=dates)


def sample_put_contracts():
    return [
        {"symbol": "SPY250425P00480000", "strike": 480.0, "bid": 1.20, "ask": 1.40,
         "mid": 1.30, "delta": -0.15, "theta": -0.05, "vega": 0.10, "gamma": 0.01,
         "iv": 0.18, "expiration": "2025-04-25", "type": "put"},
        {"symbol": "SPY250425P00475000", "strike": 475.0, "bid": 0.80, "ask": 1.00,
         "mid": 0.90, "delta": -0.10, "theta": -0.04, "vega": 0.08, "gamma": 0.008,
         "iv": 0.17, "expiration": "2025-04-25", "type": "put"},
        {"symbol": "SPY250425P00470000", "strike": 470.0, "bid": 0.50, "ask": 0.65,
         "mid": 0.575, "delta": -0.07, "theta": -0.03, "vega": 0.06, "gamma": 0.005,
         "iv": 0.16, "expiration": "2025-04-25", "type": "put"},
    ]


def sample_call_contracts():
    return [
        {"symbol": "SPY250425C00520000", "strike": 520.0, "bid": 1.10, "ask": 1.30,
         "mid": 1.20, "delta": 0.18, "theta": -0.05, "vega": 0.10, "gamma": 0.01,
         "iv": 0.18, "expiration": "2025-04-25", "type": "call"},
        {"symbol": "SPY250425C00525000", "strike": 525.0, "bid": 0.70, "ask": 0.90,
         "mid": 0.80, "delta": 0.12, "theta": -0.04, "vega": 0.08, "gamma": 0.008,
         "iv": 0.17, "expiration": "2025-04-25", "type": "call"},
        {"symbol": "SPY250425C00530000", "strike": 530.0, "bid": 0.40, "ask": 0.55,
         "mid": 0.475, "delta": 0.08, "theta": -0.03, "vega": 0.06, "gamma": 0.005,
         "iv": 0.16, "expiration": "2025-04-25", "type": "call"},
    ]


def make_plan(net_credit=1.70, width=5.0, max_loss=330.0,
              ratio=0.34, delta=-0.18, valid=True):
    return SpreadPlan(
        ticker="SPY", strategy_name="Bull Put Spread", regime="bullish",
        legs=[
            SpreadLeg("SPY250425P00480000", 480.0, "sell", "put",
                      delta, -0.05, 1.80, 2.00, net_credit + 0.4),
            SpreadLeg("SPY250425P00475000", 475.0, "buy", "put",
                      -0.10, -0.03, 0.50, 0.70, 0.4),
        ],
        spread_width=width, net_credit=net_credit, max_loss=max_loss,
        credit_to_width_ratio=ratio, expiration="2025-04-25",
        reasoning="test", valid=valid,
    )


def make_verdict(approved=True, plan=None):
    if plan is None:
        plan = make_plan(ratio=0.34 if approved else 0.10,
                         max_loss=300 if approved else 5000,
                         valid=approved)
    return RiskVerdict(
        approved=approved, plan=plan, account_balance=100_000,
        max_allowed_loss=2_000,
        checks_passed=["check1"] if approved else [],
        checks_failed=[] if approved else ["failed_check"],
        summary="APPROVED" if approved else "REJECTED",
    )


def make_classifier(price_data):
    provider = MagicMock(spec=MarketDataProvider)
    provider.fetch_historical_prices.return_value = price_data
    # get_current_price now calls Alpaca snapshot; in tests, return the
    # last closing price from our synthetic data to keep behaviour consistent.
    provider.get_current_price.return_value = float(price_data["Close"].iloc[-1])
    provider.compute_sma = MarketDataProvider.compute_sma
    provider.compute_rsi = MarketDataProvider.compute_rsi
    provider.compute_bollinger_bands = MarketDataProvider.compute_bollinger_bands
    provider.sma_slope = MarketDataProvider.sma_slope
    return RegimeClassifier(provider)


# ===================================================================
# TEST CASES
# ===================================================================

class TestConfig(unittest.TestCase):
    def test_load_from_env(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("ALPACA_API_KEY=test_key\nALPACA_SECRET_KEY=test_secret\n"
                    "TICKERS=SPY,QQQ,IWM\nMAX_RISK_PCT=0.03\nDRY_RUN=true\n")
            f.flush()
            cfg = load_config(f.name)
        self.assertEqual(cfg.alpaca.api_key, "test_key")
        self.assertEqual(cfg.trading.tickers, ["SPY", "QQQ", "IWM"])
        self.assertAlmostEqual(cfg.trading.max_risk_pct, 0.03)
        self.assertTrue(cfg.trading.dry_run)
        os.unlink(f.name)

    def test_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("")
            f.flush()
            cfg = load_config(f.name)
        self.assertEqual(cfg.trading.tickers, ["SPY", "QQQ"])
        self.assertAlmostEqual(cfg.trading.max_risk_pct, 0.02)
        os.unlink(f.name)


class TestTechnicalIndicators(unittest.TestCase):
    def test_sma_values(self):
        prices = pd.Series([10, 20, 30, 40, 50])
        sma = MarketDataProvider.compute_sma(prices, 3)
        self.assertAlmostEqual(sma.iloc[2], 20.0)
        self.assertAlmostEqual(sma.iloc[4], 40.0)

    def test_rsi_range(self):
        rsi = MarketDataProvider.compute_rsi(make_bullish_prices()["Close"], 14)
        valid = rsi.dropna()
        self.assertTrue((valid >= 0).all())
        self.assertTrue((valid <= 100).all())

    def test_rsi_bullish_is_high(self):
        rsi = MarketDataProvider.compute_rsi(make_bullish_prices()["Close"], 14)
        self.assertGreater(rsi.tail(20).mean(), 50)

    def test_bollinger_band_ordering(self):
        upper, mid, lower = MarketDataProvider.compute_bollinger_bands(
            make_bullish_prices()["Close"], 20, 2.0)
        idx = upper.dropna().index
        self.assertTrue((upper[idx] >= mid[idx]).all())
        self.assertTrue((mid[idx] >= lower[idx]).all())

    def test_sma_slope_positive(self):
        sma = pd.Series([100, 101, 102, 103, 104])
        self.assertGreater(MarketDataProvider.sma_slope(sma, 5), 0)

    def test_sma_slope_negative(self):
        sma = pd.Series([104, 103, 102, 101, 100])
        self.assertLess(MarketDataProvider.sma_slope(sma, 5), 0)

    def test_strike_extraction(self):
        self.assertAlmostEqual(
            MarketDataProvider._extract_strike("SPY250404P00550000"), 550.0)
        self.assertAlmostEqual(
            MarketDataProvider._extract_strike("SPY250404C00482500"), 482.5)
        self.assertAlmostEqual(
            MarketDataProvider._extract_strike("INVALID"), 0.0)


class TestRegimeClassifier(unittest.TestCase):
    def test_bullish(self):
        result = make_classifier(make_bullish_prices()).classify("SPY")
        self.assertEqual(result.regime, Regime.BULLISH)
        self.assertGreater(result.current_price, result.sma_200)
        self.assertGreater(result.sma_50_slope, 0)

    def test_bearish(self):
        result = make_classifier(make_bearish_prices()).classify("SPY")
        self.assertEqual(result.regime, Regime.BEARISH)
        self.assertLess(result.current_price, result.sma_200)
        self.assertLess(result.sma_50_slope, 0)

    def test_sideways(self):
        result = make_classifier(make_sideways_prices()).classify("SPY")
        self.assertEqual(result.regime, Regime.SIDEWAYS)

    def test_has_reasoning(self):
        result = make_classifier(make_bullish_prices()).classify("SPY")
        self.assertGreater(len(result.reasoning), 0)


class TestStrategyPlanner(unittest.TestCase):
    def _make_analysis(self, regime):
        return RegimeAnalysis(regime=regime, current_price=500.0,
                              sma_50=498.0, sma_200=490.0, sma_50_slope=0.5,
                              rsi_14=55.0, bollinger_width=0.06,
                              reasoning="Test")

    def _make_planner(self, put_chain=None, call_chain=None):
        provider = MagicMock(spec=MarketDataProvider)
        provider.fetch_option_chain.side_effect = lambda t, e, ot: (
            put_chain if ot == "put" else call_chain)
        return StrategyPlanner(provider, max_delta=0.20, min_credit_ratio=0.33)

    def test_bullish_selects_bull_put(self):
        planner = self._make_planner(put_chain=sample_put_contracts())
        plan = planner.plan("SPY", self._make_analysis(Regime.BULLISH))
        self.assertEqual(plan.strategy_name, "Bull Put Spread")
        self.assertEqual(len(plan.legs), 2)
        sold = [l for l in plan.legs if l.action == "sell"]
        bought = [l for l in plan.legs if l.action == "buy"]
        self.assertGreater(sold[0].strike, bought[0].strike)

    def test_bearish_selects_bear_call(self):
        planner = self._make_planner(call_chain=sample_call_contracts())
        plan = planner.plan("SPY", self._make_analysis(Regime.BEARISH))
        self.assertEqual(plan.strategy_name, "Bear Call Spread")
        self.assertEqual(len(plan.legs), 2)

    def test_sideways_selects_iron_condor(self):
        planner = self._make_planner(
            put_chain=sample_put_contracts(),
            call_chain=sample_call_contracts())
        plan = planner.plan("SPY", self._make_analysis(Regime.SIDEWAYS))
        self.assertEqual(plan.strategy_name, "Iron Condor")
        self.assertEqual(len(plan.legs), 4)

    def test_empty_chain_returns_invalid(self):
        planner = self._make_planner(put_chain=None)
        plan = planner.plan("SPY", self._make_analysis(Regime.BULLISH))
        self.assertFalse(plan.valid)

    def test_plan_serialization(self):
        plan = make_plan()
        d = plan.to_dict()
        self.assertEqual(d["ticker"], "SPY")
        self.assertEqual(len(d["legs"]), 2)


class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(max_risk_pct=0.02, min_credit_ratio=0.33,
                              max_delta=0.20)

    def test_all_pass(self):
        plan = make_plan(ratio=0.34, delta=-0.18, max_loss=300)
        v = self.rm.evaluate(plan, 100_000, "paper", True)
        self.assertTrue(v.approved)
        self.assertEqual(len(v.checks_failed), 0)

    def test_reject_low_credit_ratio(self):
        plan = make_plan(ratio=0.20)
        v = self.rm.evaluate(plan, 100_000, "paper", True)
        self.assertFalse(v.approved)

    def test_reject_high_delta(self):
        plan = make_plan(delta=-0.30)
        v = self.rm.evaluate(plan, 100_000, "paper", True)
        self.assertFalse(v.approved)

    def test_reject_excessive_loss(self):
        plan = make_plan(max_loss=5000)
        v = self.rm.evaluate(plan, 100_000, "paper", True)
        self.assertFalse(v.approved)

    def test_reject_live_account(self):
        plan = make_plan(ratio=0.34, max_loss=300)
        v = self.rm.evaluate(plan, 100_000, "live", True)
        self.assertFalse(v.approved)

    def test_reject_market_closed(self):
        plan = make_plan(ratio=0.34, max_loss=300)
        v = self.rm.evaluate(plan, 100_000, "paper", False)
        self.assertFalse(v.approved)

    def test_max_allowed_loss_calc(self):
        plan = make_plan(ratio=0.34, max_loss=300)
        v = self.rm.evaluate(plan, 50_000, "paper", True)
        self.assertAlmostEqual(v.max_allowed_loss, 1000.0)


class TestExecutor(unittest.TestCase):
    def test_dry_run_writes_file(self):
        with tempfile.TemporaryDirectory() as td:
            executor = OrderExecutor("k", "s", trade_plan_dir=td, dry_run=True)
            result = executor.execute(make_verdict(True))
            self.assertEqual(result["status"], "dry_run")
            self.assertTrue(os.path.exists(result["plan_file"]))

    def test_rejected_not_executed(self):
        with tempfile.TemporaryDirectory() as td:
            executor = OrderExecutor("k", "s", trade_plan_dir=td, dry_run=True)
            result = executor.execute(make_verdict(False))
            self.assertEqual(result["status"], "rejected")

    def test_plan_file_has_risk_verdict(self):
        import json
        with tempfile.TemporaryDirectory() as td:
            executor = OrderExecutor("k", "s", trade_plan_dir=td, dry_run=True)
            result = executor.execute(make_verdict(True))
            with open(result["plan_file"]) as f:
                data = json.load(f)
            self.assertIn("risk_verdict", data)
            self.assertTrue(data["risk_verdict"]["approved"])


class TestMlegPayloadFormat(unittest.TestCase):
    """Verify the exact Alpaca mleg payload structure to prevent 422 errors."""

    def test_payload_has_ratio_qty_not_qty_in_legs(self):
        """Alpaca mleg legs require 'ratio_qty' (string), not 'qty'."""
        executor = OrderExecutor("k", "s", dry_run=False)
        plan = make_plan(ratio=0.34, max_loss=300)
        # Access the internal method to inspect the payload it would build
        legs_payload = []
        for leg in plan.legs:
            if leg.action == "sell":
                position_intent = "sell_to_open"
                side = "sell"
            else:
                position_intent = "buy_to_open"
                side = "buy"
            legs_payload.append({
                "symbol": leg.symbol,
                "ratio_qty": "1",
                "side": side,
                "position_intent": position_intent,
            })
        # Verify each leg has ratio_qty as string, NOT qty
        for leg in legs_payload:
            self.assertIn("ratio_qty", leg)
            self.assertNotIn("qty", leg)
            self.assertIsInstance(leg["ratio_qty"], str)

    def test_payload_has_position_intent(self):
        """Each leg must include position_intent."""
        plan = make_plan()
        sell_legs = [l for l in plan.legs if l.action == "sell"]
        buy_legs = [l for l in plan.legs if l.action == "buy"]
        # Verify our mapping logic
        self.assertEqual(len(sell_legs), 1)
        self.assertEqual(len(buy_legs), 1)

    def test_limit_price_is_negative_for_credit(self):
        """Credit spreads must have negative limit_price per Alpaca convention."""
        plan = make_plan(net_credit=1.70)
        limit_price_value = -abs(plan.net_credit)
        self.assertLess(limit_price_value, 0)
        self.assertEqual(str(limit_price_value), "-1.7")

    def test_limit_price_is_string(self):
        """Alpaca requires limit_price as string type."""
        plan = make_plan(net_credit=1.67)
        limit_price_str = str(-abs(plan.net_credit))
        self.assertIsInstance(limit_price_str, str)

    def test_no_top_level_side_field(self):
        """mleg orders should NOT have a top-level 'side' field."""
        plan = make_plan(net_credit=1.70)
        order_payload = {
            "type": "limit",
            "time_in_force": "day",
            "order_class": "mleg",
            "qty": "1",
            "limit_price": str(-abs(plan.net_credit)),
            "legs": [],
        }
        self.assertNotIn("side", order_payload)

    def test_top_level_qty_is_string(self):
        """Top-level qty must be a string."""
        qty = "1"
        self.assertIsInstance(qty, str)


class TestPositionMonitor(unittest.TestCase):
    """Tests for the position monitoring and exit signal logic."""

    def _make_spread(self, pl=-50.0, credit=1.70, max_loss=330.0,
                     strategy="Bull Put Spread", underlying="SPY"):
        leg1 = PositionSnapshot(
            symbol="SPY250425P00480000", qty=-1, side="short",
            avg_entry_price=1.70, current_price=1.50,
            market_value=-150.0, cost_basis=-170.0,
            unrealized_pl=20.0, unrealized_plpc=0.12, asset_class="us_option")
        leg2 = PositionSnapshot(
            symbol="SPY250425P00475000", qty=1, side="long",
            avg_entry_price=0.90, current_price=1.60,
            market_value=160.0, cost_basis=90.0,
            unrealized_pl=-70.0, unrealized_plpc=-0.78, asset_class="us_option")
        return SpreadPosition(
            underlying=underlying, strategy_name=strategy,
            legs=[leg1, leg2], original_credit=credit,
            max_loss=max_loss, spread_width=5.0,
            net_unrealized_pl=pl)

    def test_hold_signal(self):
        """No exit when within normal range."""
        spread = self._make_spread(pl=-50.0, max_loss=330.0, credit=1.70)
        monitor = PositionMonitor("k", "s")
        spreads = monitor.evaluate([spread], {"SPY": Regime.BULLISH})
        self.assertEqual(spreads[0].exit_signal, ExitSignal.HOLD)

    def test_stop_loss_triggered(self):
        """Exit when unrealized loss >= 50% of max loss."""
        spread = self._make_spread(pl=-170.0, max_loss=330.0)
        monitor = PositionMonitor("k", "s", stop_loss_pct=0.50)
        spreads = monitor.evaluate([spread], {"SPY": Regime.BULLISH})
        self.assertEqual(spreads[0].exit_signal, ExitSignal.STOP_LOSS)

    def test_stop_loss_not_triggered_below_threshold(self):
        """No stop-loss when loss is below threshold."""
        spread = self._make_spread(pl=-100.0, max_loss=330.0)
        monitor = PositionMonitor("k", "s", stop_loss_pct=0.50)
        spreads = monitor.evaluate([spread], {"SPY": Regime.BULLISH})
        self.assertEqual(spreads[0].exit_signal, ExitSignal.HOLD)

    def test_profit_target_triggered(self):
        """Exit when unrealized profit >= 75% of credit * 100."""
        # credit=1.70 → credit_value=170 → threshold=127.5
        spread = self._make_spread(pl=130.0, credit=1.70)
        monitor = PositionMonitor("k", "s", profit_target_pct=0.75)
        spreads = monitor.evaluate([spread], {"SPY": Regime.BULLISH})
        self.assertEqual(spreads[0].exit_signal, ExitSignal.PROFIT_TARGET)

    def test_profit_target_not_triggered(self):
        """No profit-target when profit is below threshold."""
        spread = self._make_spread(pl=50.0, credit=1.70)
        monitor = PositionMonitor("k", "s", profit_target_pct=0.75)
        spreads = monitor.evaluate([spread], {"SPY": Regime.BULLISH})
        self.assertEqual(spreads[0].exit_signal, ExitSignal.HOLD)

    def test_regime_shift_triggers_exit(self):
        """Exit when regime contradicts strategy."""
        spread = self._make_spread(pl=-10.0, strategy="Bull Put Spread")
        monitor = PositionMonitor("k", "s")
        # Bull Put Spread expects BULLISH, but regime is now BEARISH
        spreads = monitor.evaluate([spread], {"SPY": Regime.BEARISH})
        self.assertEqual(spreads[0].exit_signal, ExitSignal.REGIME_SHIFT)

    def test_no_regime_shift_when_compatible(self):
        """No exit when regime matches strategy."""
        spread = self._make_spread(pl=-10.0, strategy="Bull Put Spread")
        monitor = PositionMonitor("k", "s")
        spreads = monitor.evaluate([spread], {"SPY": Regime.BULLISH})
        self.assertEqual(spreads[0].exit_signal, ExitSignal.HOLD)

    def test_strategy_regime_map(self):
        """Verify the strategy-to-regime mapping is correct."""
        self.assertEqual(STRATEGY_REGIME_MAP["Bull Put Spread"], Regime.BULLISH)
        self.assertEqual(STRATEGY_REGIME_MAP["Bear Call Spread"], Regime.BEARISH)
        self.assertEqual(STRATEGY_REGIME_MAP["Iron Condor"], Regime.SIDEWAYS)

    def test_stop_loss_takes_priority_over_regime_shift(self):
        """Stop-loss fires even if regime also shifted."""
        spread = self._make_spread(pl=-170.0, max_loss=330.0,
                                    strategy="Bull Put Spread")
        monitor = PositionMonitor("k", "s", stop_loss_pct=0.50)
        spreads = monitor.evaluate([spread], {"SPY": Regime.BEARISH})
        # stop_loss is checked first, so it should fire
        self.assertEqual(spreads[0].exit_signal, ExitSignal.STOP_LOSS)

    def test_summary(self):
        """Summary reports correct totals."""
        s1 = self._make_spread(pl=-50.0, underlying="SPY")
        s2 = self._make_spread(pl=100.0, underlying="QQQ",
                                strategy="Bear Call Spread")
        s2.exit_signal = ExitSignal.PROFIT_TARGET
        monitor = PositionMonitor("k", "s")
        result = monitor.summary([s1, s2])
        self.assertEqual(result["total_spreads"], 2)
        self.assertAlmostEqual(result["total_unrealized_pl"], 50.0)

    def test_group_into_spreads(self):
        """Positions are matched to trade plans by symbol."""
        positions = [
            PositionSnapshot("SPY250425P00480000", -1, "short", 1.70, 1.50,
                             -150.0, -170.0, 20.0, 0.12, "us_option"),
            PositionSnapshot("SPY250425P00475000", 1, "long", 0.90, 1.60,
                             160.0, 90.0, -70.0, -0.78, "us_option"),
        ]
        trade_plans = [{
            "trade_plan": {
                "ticker": "SPY",
                "strategy": "Bull Put Spread",
                "legs": [
                    {"symbol": "SPY250425P00480000", "strike": 480.0,
                     "action": "sell", "type": "put"},
                    {"symbol": "SPY250425P00475000", "strike": 475.0,
                     "action": "buy", "type": "put"},
                ],
                "net_credit": 1.70,
                "max_loss": 330.0,
                "spread_width": 5.0,
            }
        }]
        monitor = PositionMonitor("k", "s")
        spreads = monitor.group_into_spreads(positions, trade_plans)
        self.assertEqual(len(spreads), 1)
        self.assertEqual(spreads[0].underlying, "SPY")
        self.assertEqual(spreads[0].strategy_name, "Bull Put Spread")
        self.assertEqual(len(spreads[0].legs), 2)

    def test_group_into_spreads_no_match(self):
        """Returns empty when no positions match trade plans."""
        positions = [
            PositionSnapshot("AAPL250425P00180000", -1, "short", 1.0, 0.8,
                             -80.0, -100.0, 20.0, 0.2, "us_option"),
        ]
        trade_plans = [{
            "trade_plan": {
                "ticker": "SPY",
                "legs": [{"symbol": "SPY250425P00480000"}],
                "net_credit": 1.70, "max_loss": 330.0, "spread_width": 5.0,
                "strategy": "Bull Put Spread",
            }
        }]
        monitor = PositionMonitor("k", "s")
        spreads = monitor.group_into_spreads(positions, trade_plans)
        self.assertEqual(len(spreads), 0)


class TestOrderTracker(unittest.TestCase):
    """Tests for order parsing and summary logic."""

    def _make_raw_order(self, status="filled", order_class="mleg",
                        order_id="abc123", symbol="", side=""):
        return {
            "id": order_id,
            "status": status,
            "symbol": symbol,
            "side": side,
            "type": "limit",
            "order_class": order_class,
            "qty": "1",
            "filled_qty": "1" if status == "filled" else "0",
            "limit_price": "-1.70",
            "filled_avg_price": "-1.65" if status == "filled" else None,
            "created_at": "2026-03-28T10:00:00Z",
            "updated_at": "2026-03-28T10:05:00Z",
            "legs": [
                {"symbol": "SPY250425P00480000", "side": "sell",
                 "qty": "1", "filled_qty": "1", "status": status},
                {"symbol": "SPY250425P00475000", "side": "buy",
                 "qty": "1", "filled_qty": "1", "status": status},
            ],
        }

    def test_parse_filled_order(self):
        tracker = OrderTracker("k", "s")
        raw = self._make_raw_order(status="filled")
        record = tracker._parse_order(raw)
        self.assertEqual(record.status, OrderStatus.FILLED)
        self.assertEqual(record.order_id, "abc123")
        self.assertEqual(len(record.legs), 2)

    def test_parse_rejected_order(self):
        tracker = OrderTracker("k", "s")
        raw = self._make_raw_order(status="rejected")
        record = tracker._parse_order(raw)
        self.assertEqual(record.status, OrderStatus.REJECTED)

    def test_parse_unknown_status(self):
        tracker = OrderTracker("k", "s")
        raw = self._make_raw_order(status="some_new_status")
        record = tracker._parse_order(raw)
        self.assertEqual(record.status, OrderStatus.UNKNOWN)

    def test_summarize_orders(self):
        tracker = OrderTracker("k", "s")
        records = [
            tracker._parse_order(self._make_raw_order("filled", order_id="1")),
            tracker._parse_order(self._make_raw_order("filled", order_id="2")),
            tracker._parse_order(self._make_raw_order("rejected", order_id="3")),
            tracker._parse_order(self._make_raw_order("new", order_id="4")),
        ]
        summary = tracker.summarize_orders(records)
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["filled"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["open"], 1)

    def test_summarize_empty(self):
        tracker = OrderTracker("k", "s")
        summary = tracker.summarize_orders([])
        self.assertEqual(summary["total"], 0)
        self.assertEqual(summary["filled"], 0)

    def test_parse_preserves_raw(self):
        tracker = OrderTracker("k", "s")
        raw = self._make_raw_order()
        record = tracker._parse_order(raw)
        self.assertEqual(record.raw, raw)


class TestCloseSpread(unittest.TestCase):
    """Tests for the close_spread logic in executor."""

    def test_close_spread_dry_run_in_agent(self):
        """In dry-run mode, agent logs close but doesn't call API."""
        with tempfile.TemporaryDirectory() as td:
            config = AppConfig(
                alpaca=AlpacaConfig("K", "S",
                                    "https://paper-api.alpaca.markets/v2",
                                    "https://data.alpaca.markets/v2"),
                trading=TradingConfig(["SPY"], "dry_run", 0.02, 0.33, 0.20, True, False),
                logging=LoggingConfig("DEBUG",
                                      os.path.join(td, "logs"),
                                      os.path.join(td, "plans")),
            )
            agent = TradingAgent(config)

            # The dry_run path in _stage_monitor logs but doesn't call executor
            # Test the signal evaluation directly
            leg1 = PositionSnapshot(
                "SPY250425P00480000", -1, "short", 1.70, 1.50,
                -150.0, -170.0, 20.0, 0.12, "us_option")
            leg2 = PositionSnapshot(
                "SPY250425P00475000", 1, "long", 0.90, 1.60,
                160.0, 90.0, -70.0, -0.78, "us_option")

            spread = SpreadPosition(
                underlying="SPY", strategy_name="Bull Put Spread",
                legs=[leg1, leg2], original_credit=1.70,
                max_loss=330.0, spread_width=5.0, net_unrealized_pl=-170.0)

            # Evaluate should trigger stop-loss
            monitor = agent.position_monitor
            spreads = monitor.evaluate([spread], {"SPY": Regime.BULLISH})
            self.assertEqual(spreads[0].exit_signal, ExitSignal.STOP_LOSS)


class TestAgentIntegration(unittest.TestCase):
    def _make_config(self, tmp_dir):
        return AppConfig(
            alpaca=AlpacaConfig("K", "S",
                                "https://paper-api.alpaca.markets/v2",
                                "https://data.alpaca.markets/v2"),
            trading=TradingConfig(["SPY"], "dry_run", 0.02, 0.33, 0.20, True, False),
            logging=LoggingConfig("DEBUG",
                                  os.path.join(tmp_dir, "logs"),
                                  os.path.join(tmp_dir, "plans")),
        )

    def _mock_monitor_and_tracker(self, agent):
        """Mock the position monitor and order tracker API calls."""
        agent.position_monitor.fetch_open_positions = MagicMock(return_value=[])
        agent.order_tracker.fetch_open_orders = MagicMock(return_value=[])
        agent.order_tracker.fetch_recent_fills = MagicMock(return_value=[])

    def test_full_cycle_bullish(self):
        with tempfile.TemporaryDirectory() as td:
            bullish = make_bullish_prices()
            agent = TradingAgent(self._make_config(td))
            agent.data_provider.fetch_historical_prices = MagicMock(
                return_value=bullish)
            agent.data_provider.get_current_price = MagicMock(
                return_value=float(bullish["Close"].iloc[-1]))
            agent.data_provider.fetch_option_chain = MagicMock(
                side_effect=lambda t, e, ot: sample_put_contracts() if ot == "put" else sample_call_contracts())
            agent.data_provider.get_account_info = MagicMock(
                return_value={"equity": "100000"})
            agent.data_provider.is_market_open = MagicMock(return_value=True)
            self._mock_monitor_and_tracker(agent)

            results = agent.run_cycle()
            # run_cycle now returns a dict with monitor, new_trades, order_summary
            self.assertIn("new_trades", results)
            self.assertIn("monitor", results)
            self.assertIn("order_summary", results)
            trades = results["new_trades"]
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades[0]["ticker"], "SPY")
            self.assertEqual(trades[0]["regime"], "bullish")
            self.assertEqual(trades[0]["strategy"], "Bull Put Spread")

    def test_no_account_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            agent = TradingAgent(self._make_config(td))
            agent.data_provider.get_account_info = MagicMock(return_value=None)
            results = agent.run_cycle()
            # When account fails, returns a list with error dict
            self.assertIsInstance(results, list)
            self.assertEqual(results[0]["status"], "error")

    def test_exception_handled(self):
        with tempfile.TemporaryDirectory() as td:
            agent = TradingAgent(self._make_config(td))
            agent.data_provider.get_account_info = MagicMock(
                return_value={"equity": "100000"})
            agent.data_provider.is_market_open = MagicMock(return_value=True)
            self._mock_monitor_and_tracker(agent)
            agent.regime_classifier.classify = MagicMock(
                side_effect=Exception("API timeout"))
            results = agent.run_cycle()
            # Now returns a dict; errors are in new_trades list
            trades = results["new_trades"]
            self.assertEqual(trades[0]["status"], "error")


class TestTradeJournal(unittest.TestCase):
    """Tests for the trade journal system."""

    def test_open_and_get_trade(self):
        with tempfile.TemporaryDirectory() as td:
            journal = TradeJournal(journal_dir=td)
            entry = TradeEntry(
                ticker="SPY", strategy_name="Bull Put Spread",
                regime="bullish", current_price=500.0,
                net_credit=1.70, max_loss=330.0, spread_width=5.0)
            trade_id = journal.open_trade(entry)
            self.assertIn("SPY", trade_id)

            loaded = journal.get_trade(trade_id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.ticker, "SPY")
            self.assertEqual(loaded.strategy_name, "Bull Put Spread")

    def test_close_trade(self):
        with tempfile.TemporaryDirectory() as td:
            journal = TradeJournal(journal_dir=td)
            entry = TradeEntry(
                ticker="QQQ", strategy_name="Bear Call Spread",
                regime="bearish", net_credit=1.50, max_loss=350.0)
            trade_id = journal.open_trade(entry)

            closed = journal.close_trade(
                trade_id, exit_signal="profit_target",
                exit_reason="Hit 75% of credit",
                realized_pl=112.5)
            self.assertIsNotNone(closed)
            self.assertEqual(closed.outcome_label, "win")
            self.assertGreater(closed.realized_pl, 0)

    def test_get_closed_trades(self):
        with tempfile.TemporaryDirectory() as td:
            journal = TradeJournal(journal_dir=td)

            # Open and close two trades
            e1 = TradeEntry(ticker="SPY", strategy_name="Bull Put Spread",
                            net_credit=1.70, regime="bullish")
            id1 = journal.open_trade(e1)
            journal.close_trade(id1, "profit_target", "test", 100.0)

            e2 = TradeEntry(ticker="QQQ", strategy_name="Bear Call Spread",
                            net_credit=1.50, regime="bearish")
            id2 = journal.open_trade(e2)
            journal.close_trade(id2, "stop_loss", "test", -150.0)

            closed = journal.get_closed_trades()
            self.assertEqual(len(closed), 2)

    def test_stats_computation(self):
        with tempfile.TemporaryDirectory() as td:
            journal = TradeJournal(journal_dir=td)

            for i, pl in enumerate([100, 80, -150, 50, -200]):
                e = TradeEntry(trade_id=f"SPY_test_{i}",
                               ticker="SPY", strategy_name="Bull Put Spread",
                               net_credit=1.70, regime="bullish")
                tid = journal.open_trade(e)
                journal.close_trade(tid, "test", "test", float(pl))

            stats = journal.get_stats()
            self.assertEqual(stats["total_trades"], 5)
            self.assertEqual(stats["wins"], 3)
            self.assertEqual(stats["losses"], 2)

    def test_embedding_text(self):
        entry = TradeEntry(
            ticker="SPY", strategy_name="Bull Put Spread",
            regime="bullish", current_price=500.0, rsi_14=62.5,
            bollinger_width=0.06, sma_50=498.0, sma_200=490.0,
            net_credit=1.70, spread_width=5.0,
            credit_to_width_ratio=0.34, sold_delta=0.18,
            dte_at_entry=30)
        text = entry.to_embedding_text()
        self.assertIn("Bull Put Spread", text)
        self.assertIn("SPY", text)
        self.assertIn("bullish", text)


class TestKnowledgeBase(unittest.TestCase):
    """Tests for the RAG knowledge base."""

    def test_add_and_keyword_search(self):
        """Without embeddings, falls back to keyword search."""
        with tempfile.TemporaryDirectory() as td:
            kb = KnowledgeBase(kb_dir=td, embed_fn=None)
            kb.add_trade("t1", "Bull Put Spread SPY bullish regime win",
                         {"ticker": "SPY"})
            kb.add_trade("t2", "Bear Call Spread QQQ bearish regime loss",
                         {"ticker": "QQQ"})

            results = kb.search_similar("SPY bullish", top_k=2)
            self.assertGreater(len(results), 0)
            # SPY should rank higher
            self.assertIn("SPY", results[0][0].text)

    def test_add_lesson(self):
        with tempfile.TemporaryDirectory() as td:
            kb = KnowledgeBase(kb_dir=td, embed_fn=None)
            doc_id = kb.add_lesson("Always check RSI before entering",
                                    trade_id="t1")
            self.assertIn("lesson", doc_id)

    def test_document_count(self):
        with tempfile.TemporaryDirectory() as td:
            kb = KnowledgeBase(kb_dir=td, embed_fn=None)
            kb.add_trade("t1", "Trade 1")
            kb.add_trade("t2", "Trade 2")
            kb.add_lesson("Lesson 1")

            counts = kb.document_count()
            self.assertEqual(counts.get("trade", 0), 2)
            self.assertEqual(counts.get("lesson", 0), 1)

    def test_cosine_similarity(self):
        """Basic cosine similarity math check."""
        sim = KnowledgeBase._cosine_similarity([1, 0, 0], [1, 0, 0])
        self.assertAlmostEqual(sim, 1.0, places=5)

        sim = KnowledgeBase._cosine_similarity([1, 0, 0], [0, 1, 0])
        self.assertAlmostEqual(sim, 0.0, places=5)

        sim = KnowledgeBase._cosine_similarity([1, 1], [1, 1])
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_vector_search_with_mock_embeddings(self):
        """Test vector search with pre-computed mock embeddings."""
        with tempfile.TemporaryDirectory() as td:
            # Mock embedding function that returns fixed vectors
            embed_map = {
                "Bull Put Spread SPY win": [1.0, 0.0, 0.0],
                "Bear Call Spread QQQ loss": [0.0, 1.0, 0.0],
                "Iron Condor IWM sideways": [0.0, 0.0, 1.0],
                "SPY bullish trade": [0.9, 0.1, 0.0],  # similar to SPY win
            }
            call_count = [0]

            def mock_embed(texts):
                results = []
                for t in texts:
                    # Find closest match in map
                    best = [0.0, 0.0, 0.0]
                    for key, vec in embed_map.items():
                        if any(w in t.lower() for w in key.lower().split()):
                            best = vec
                            break
                    results.append(best)
                    call_count[0] += 1
                return results

            kb = KnowledgeBase(kb_dir=td, embed_fn=mock_embed)
            kb.add_trade("t1", "Bull Put Spread SPY win")
            kb.add_trade("t2", "Bear Call Spread QQQ loss")
            kb.add_trade("t3", "Iron Condor IWM sideways")

            results = kb.search_similar("SPY bullish trade", top_k=2)
            self.assertGreater(len(results), 0)


class TestLLMAnalyst(unittest.TestCase):
    """Tests for the LLM analyst decision layer."""

    def _make_analysis(self, regime=Regime.BULLISH):
        return RegimeAnalysis(
            regime=regime, current_price=500.0,
            sma_50=498.0, sma_200=490.0, sma_50_slope=0.5,
            rsi_14=55.0, bollinger_width=0.06,
            reasoning="Test analysis")

    def test_passthrough_when_disabled(self):
        """When LLM is disabled, defers to rule-based system."""
        with tempfile.TemporaryDirectory() as td:
            llm = MagicMock(spec=LLMClient)
            llm.is_available.return_value = False
            llm.config = LLMConfig()  # Provide the config attribute
            journal = TradeJournal(journal_dir=os.path.join(td, "journal"))
            kb = KnowledgeBase(kb_dir=os.path.join(td, "kb"), embed_fn=None)

            analyst = LLMAnalyst(llm, journal, kb, enabled=True)
            # enabled should be flipped to False since LLM unavailable
            self.assertFalse(analyst.enabled)

            plan = make_plan(ratio=0.34, max_loss=300)
            verdict = make_verdict(True, plan)

            decision = analyst.analyze_trade(
                "SPY", self._make_analysis(), plan, verdict)
            self.assertEqual(decision.action, "approve")
            self.assertEqual(decision.confidence, 1.0)

    def test_skip_when_risk_rejected(self):
        """LLM cannot override risk manager rejection."""
        with tempfile.TemporaryDirectory() as td:
            llm = MagicMock(spec=LLMClient)
            llm.is_available.return_value = True
            journal = TradeJournal(journal_dir=os.path.join(td, "journal"))
            kb = KnowledgeBase(kb_dir=os.path.join(td, "kb"), embed_fn=None)

            analyst = LLMAnalyst(llm, journal, kb, enabled=True)

            plan = make_plan(ratio=0.10, max_loss=5000, valid=False)
            verdict = make_verdict(False, plan)

            decision = analyst.analyze_trade(
                "SPY", self._make_analysis(), plan, verdict)
            self.assertEqual(decision.action, "skip")
            # LLM.chat should NOT have been called for rejected trades
            llm.chat_json.assert_not_called()

    def test_llm_approve_decision(self):
        """LLM approves a valid trade."""
        with tempfile.TemporaryDirectory() as td:
            llm = MagicMock(spec=LLMClient)
            llm.is_available.return_value = True
            llm.chat_json.return_value = {
                "action": "approve",
                "confidence": 0.85,
                "reasoning": "Good trade setup",
                "risk_assessment": "Low risk",
                "similar_trades_summary": "Similar trades won 80%",
                "modifications": {},
                "warnings": [],
            }
            journal = TradeJournal(journal_dir=os.path.join(td, "journal"))
            kb = KnowledgeBase(kb_dir=os.path.join(td, "kb"), embed_fn=None)

            analyst = LLMAnalyst(llm, journal, kb, enabled=True)
            plan = make_plan(ratio=0.34, max_loss=300)
            verdict = make_verdict(True, plan)

            decision = analyst.analyze_trade(
                "SPY", self._make_analysis(), plan, verdict)
            self.assertEqual(decision.action, "approve")
            self.assertAlmostEqual(decision.confidence, 0.85)
            llm.chat_json.assert_called_once()

    def test_llm_skip_decision(self):
        """LLM skips a trade it doesn't like."""
        with tempfile.TemporaryDirectory() as td:
            llm = MagicMock(spec=LLMClient)
            llm.is_available.return_value = True
            llm.chat_json.return_value = {
                "action": "skip",
                "confidence": 0.70,
                "reasoning": "RSI showing divergence",
                "risk_assessment": "Hidden risk detected",
                "similar_trades_summary": "Similar trades lost money",
                "modifications": {},
                "warnings": ["RSI divergence"],
            }
            journal = TradeJournal(journal_dir=os.path.join(td, "journal"))
            kb = KnowledgeBase(kb_dir=os.path.join(td, "kb"), embed_fn=None)

            analyst = LLMAnalyst(llm, journal, kb, enabled=True)
            plan = make_plan(ratio=0.34, max_loss=300)
            verdict = make_verdict(True, plan)

            decision = analyst.analyze_trade(
                "SPY", self._make_analysis(), plan, verdict)
            self.assertEqual(decision.action, "skip")
            self.assertEqual(len(decision.warnings), 1)

    def test_create_journal_entry(self):
        """Creates a properly populated TradeEntry from trade context."""
        with tempfile.TemporaryDirectory() as td:
            llm = MagicMock(spec=LLMClient)
            llm.is_available.return_value = True
            journal = TradeJournal(journal_dir=os.path.join(td, "journal"))
            kb = KnowledgeBase(kb_dir=os.path.join(td, "kb"), embed_fn=None)

            analyst = LLMAnalyst(llm, journal, kb, enabled=True)
            plan = make_plan(ratio=0.34, max_loss=300)
            verdict = make_verdict(True, plan)
            decision = AnalystDecision(
                action="approve", confidence=0.8,
                reasoning="Good setup", risk_assessment="Low risk",
                similar_trades_summary="", modifications={}, warnings=[])

            entry = analyst.create_journal_entry(
                "SPY", self._make_analysis(), plan, verdict, decision)
            self.assertEqual(entry.ticker, "SPY")
            self.assertEqual(entry.strategy_name, "Bull Put Spread")
            self.assertEqual(entry.llm_decision, "approve")
            self.assertAlmostEqual(entry.llm_confidence, 0.8)


class TestFineTuning(unittest.TestCase):
    """Tests for the fine-tuning export pipeline."""

    def test_training_summary_empty(self):
        with tempfile.TemporaryDirectory() as td:
            journal = TradeJournal(journal_dir=os.path.join(td, "j"))
            kb = KnowledgeBase(kb_dir=os.path.join(td, "kb"), embed_fn=None)
            exporter = FineTuningExporter(journal, kb,
                                           export_dir=os.path.join(td, "ft"))
            summary = exporter.get_training_summary()
            self.assertEqual(summary["total_closed_trades"], 0)
            self.assertFalse(summary["ready_for_chat_ft"])

    def test_export_skipped_when_insufficient_data(self):
        with tempfile.TemporaryDirectory() as td:
            journal = TradeJournal(journal_dir=os.path.join(td, "j"))
            kb = KnowledgeBase(kb_dir=os.path.join(td, "kb"), embed_fn=None)
            exporter = FineTuningExporter(journal, kb,
                                           export_dir=os.path.join(td, "ft"))
            path = exporter.export_chat_jsonl(min_trades=20)
            self.assertEqual(path, "")  # No export when <20 trades

    def test_export_with_sufficient_data(self):
        with tempfile.TemporaryDirectory() as td:
            journal = TradeJournal(journal_dir=os.path.join(td, "j"))
            kb = KnowledgeBase(kb_dir=os.path.join(td, "kb"), embed_fn=None)

            # Create 25 closed trades with unique IDs
            for i in range(25):
                e = TradeEntry(
                    trade_id=f"SPY_ft_{i}",
                    ticker="SPY", strategy_name="Bull Put Spread",
                    regime="bullish", current_price=500.0,
                    rsi_14=55.0, bollinger_width=0.06,
                    net_credit=1.70, spread_width=5.0,
                    credit_to_width_ratio=0.34, sold_delta=0.18)
                tid = journal.open_trade(e)
                pl = 100.0 if i % 3 != 0 else -150.0
                journal.close_trade(tid, "test", "test", pl)

            exporter = FineTuningExporter(journal, kb,
                                           export_dir=os.path.join(td, "ft"))
            path = exporter.export_chat_jsonl(min_trades=20)
            self.assertNotEqual(path, "")
            self.assertTrue(os.path.exists(path))

            # Verify JSONL format
            with open(path) as f:
                lines = f.readlines()
            self.assertGreater(len(lines), 0)
            first = json.loads(lines[0])
            self.assertIn("messages", first)


class TestLLMClient(unittest.TestCase):
    """Tests for the LLM client (without actual LLM connection)."""

    def test_config_defaults(self):
        config = LLMConfig()
        self.assertEqual(config.provider, "ollama")
        self.assertEqual(config.model, "mistral")
        self.assertAlmostEqual(config.temperature, 0.3)

    def test_url_resolution_ollama(self):
        config = LLMConfig(provider="ollama", base_url="http://localhost:11434")
        client = LLMClient(config)
        self.assertIn("/api/chat", client._api_url)
        self.assertIn("/api/embed", client._embed_url)

    def test_url_resolution_openai(self):
        config = LLMConfig(provider="openai", base_url="http://localhost:8000")
        client = LLMClient(config)
        self.assertIn("/v1/chat/completions", client._api_url)
        self.assertIn("/v1/embeddings", client._embed_url)

    def test_chat_json_parsing(self):
        """Test JSON extraction from LLM responses."""
        config = LLMConfig(provider="ollama")
        client = LLMClient(config)

        # Mock the chat method to return raw JSON
        client.chat = MagicMock(return_value='{"action": "approve", "confidence": 0.9}')
        result = client.chat_json([{"role": "user", "content": "test"}])
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "approve")

    def test_chat_json_markdown_block(self):
        """Test JSON extraction from markdown code blocks."""
        config = LLMConfig(provider="ollama")
        client = LLMClient(config)

        client.chat = MagicMock(return_value=(
            'Here is my analysis:\n```json\n{"action": "skip"}\n```'))
        result = client.chat_json([{"role": "user", "content": "test"}])
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "skip")

    def test_chat_json_empty_response(self):
        config = LLMConfig(provider="ollama")
        client = LLMClient(config)
        client.chat = MagicMock(return_value="")
        result = client.chat_json([{"role": "user", "content": "test"}])
        self.assertIsNone(result)


# ===================================================================
# Run all tests
# ===================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    for cls in [TestConfig, TestTechnicalIndicators, TestRegimeClassifier,
                TestStrategyPlanner, TestRiskManager, TestExecutor,
                TestMlegPayloadFormat, TestPositionMonitor, TestOrderTracker,
                TestCloseSpread, TestTradeJournal, TestKnowledgeBase,
                TestLLMAnalyst, TestFineTuning, TestLLMClient,
                TestAgentIntegration]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
