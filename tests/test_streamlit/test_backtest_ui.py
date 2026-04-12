"""
Tests for backtest_ui.py — Backtester class and render function.
"""

from datetime import date, timedelta
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from trading_agent.streamlit.backtest_ui import (
    STARTING_EQUITY,
    Backtester,
    BacktestResult,
    SimTrade,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(n: int = 250, start: float = 500.0, drift: float = 0.0003) -> pd.Series:
    """Deterministic price series for testing (no randomness)."""
    np.random.seed(42)
    returns = np.random.normal(drift, 0.01, n)
    closes = start * np.exp(np.cumsum(returns))
    idx = pd.date_range("2025-01-02", periods=n, freq="B")
    return pd.Series(closes, index=idx, name="Close")


def _make_bearish_prices(n: int = 250) -> pd.Series:
    """Prices that trend down to force BEARISH regime."""
    np.random.seed(99)
    returns = np.random.normal(-0.002, 0.008, n)
    closes = 600.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2025-01-02", periods=n, freq="B")
    return pd.Series(closes, index=idx, name="Close")


# ---------------------------------------------------------------------------
# Backtester._classify
# ---------------------------------------------------------------------------

class TestClassify:
    def test_sideways_before_200_bars(self):
        prices = _make_prices(300)
        bt = Backtester()
        assert bt._classify(prices, 50) == "sideways"
        assert bt._classify(prices, 199) == "sideways"

    def test_classifies_at_or_after_200(self):
        prices = _make_prices(300)
        bt = Backtester()
        regime = bt._classify(prices, 200)
        assert regime in ("bullish", "bearish", "sideways")

    def test_bearish_on_downtrend(self):
        prices = _make_bearish_prices(300)
        bt = Backtester()
        # Strong downtrend should classify as bearish somewhere beyond 200
        regimes = {bt._classify(prices, i) for i in range(201, 280)}
        assert "bearish" in regimes


# ---------------------------------------------------------------------------
# Backtester._strategy
# ---------------------------------------------------------------------------

class TestStrategy:
    @pytest.mark.parametrize("regime,expected", [
        ("bullish", "Bull Put Spread"),
        ("bearish", "Bear Call Spread"),
        ("sideways", "Iron Condor"),
        ("mean_reversion", "Iron Condor"),   # fallback
    ])
    def test_strategy_mapping(self, regime, expected):
        assert Backtester._strategy(regime) == expected


# ---------------------------------------------------------------------------
# Backtester._simulate
# ---------------------------------------------------------------------------

class TestSimulate:
    def test_returns_tuple_of_three(self):
        prices = _make_prices(300)
        bt = Backtester()
        result = bt._simulate(prices, 210, "bullish", credit=1.5)
        assert len(result) == 3
        outcome, pnl, hold_days = result
        assert outcome in ("win", "loss")
        assert isinstance(pnl, float)
        assert hold_days >= 0

    def test_win_pnl_positive(self):
        # Flat prices — short strikes never breached → win
        idx = pd.date_range("2025-01-02", periods=300, freq="B")
        flat = pd.Series([500.0] * 300, index=idx)
        bt = Backtester(target_dte=10)
        outcome, pnl, _ = bt._simulate(flat, 210, "bullish", credit=1.5)
        assert outcome == "win"
        assert pnl > 0

    def test_loss_pnl_negative(self):
        # Prices stable then crash AFTER the entry bar → short put breached → loss
        # entry_idx=210, entry_p=500; bar 211 onward = 425 < 500*0.97=485 → breached
        idx = pd.date_range("2025-01-02", periods=300, freq="B")
        start = 500.0
        prices_list = [start] * 211 + [start * 0.85] * 89   # crash starts at bar 211
        crashing = pd.Series(prices_list, index=idx)
        bt = Backtester(target_dte=20)
        outcome, pnl, _ = bt._simulate(crashing, 210, "bullish", credit=1.5)
        assert outcome == "loss"
        assert pnl < 0

    def test_hold_days_capped_at_target_dte(self):
        prices = _make_prices(220)
        bt = Backtester(target_dte=5)
        _, _, hold_days = bt._simulate(prices, 210, "sideways", credit=1.5)
        assert hold_days <= 5


# ---------------------------------------------------------------------------
# Backtester._metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_empty_trades_returns_zeros(self):
        m = Backtester()._metrics([], STARTING_EQUITY)
        assert m["total_trades"] == 0
        assert m["win_rate"] == 0.0
        assert m["profit_factor"] == 0.0

    def test_all_wins(self):
        trades = [
            SimTrade("SPY", "Bull Put Spread", "bullish",
                     date(2025, 1, i + 2), date(2025, 3, 1),
                     1.5, 3.5, "win", 50.0, 30)
            for i in range(10)
        ]
        m = Backtester()._metrics(trades, STARTING_EQUITY)
        assert m["total_trades"] == 10
        assert m["win_rate"] == 100.0
        assert m["profit_factor"] == float("inf")

    def test_all_losses_win_rate_zero(self):
        trades = [
            SimTrade("SPY", "Iron Condor", "sideways",
                     date(2025, 1, i + 2), date(2025, 3, 1),
                     1.5, 3.5, "loss", -250.0, 30)
            for i in range(5)
        ]
        m = Backtester()._metrics(trades, STARTING_EQUITY)
        assert m["win_rate"] == 0.0

    def test_max_drawdown_positive(self):
        trades = [
            SimTrade("SPY", "Bull Put Spread", "bullish",
                     date(2025, 1, 2), date(2025, 2, 1), 1.5, 3.5, "win", 100.0, 20),
            SimTrade("SPY", "Bull Put Spread", "bullish",
                     date(2025, 2, 2), date(2025, 3, 1), 1.5, 3.5, "loss", -400.0, 20),
        ]
        m = Backtester()._metrics(trades, STARTING_EQUITY)
        assert m["max_drawdown_pct"] > 0

    def test_avg_hold_days_correct(self):
        trades = [
            SimTrade("SPY", "Bull Put Spread", "bullish",
                     date(2025, 1, 2), date(2025, 2, 1), 1.5, 3.5, "win", 50.0, 10),
            SimTrade("SPY", "Bull Put Spread", "bullish",
                     date(2025, 2, 2), date(2025, 3, 1), 1.5, 3.5, "win", 50.0, 30),
        ]
        m = Backtester()._metrics(trades, STARTING_EQUITY)
        assert m["avg_hold_days"] == 20.0


# ---------------------------------------------------------------------------
# Backtester._regime_stats
# ---------------------------------------------------------------------------

class TestRegimeStats:
    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame(columns=["ticker", "regime", "pnl"])
        stats = Backtester._regime_stats(df)
        assert stats.empty
        assert "regime" in stats.columns

    def test_groups_by_regime(self):
        df = pd.DataFrame({
            "regime": ["bullish", "bullish", "bearish"],
            "pnl": [100.0, 50.0, -80.0],
        })
        stats = Backtester._regime_stats(df)
        assert len(stats) == 2
        bullish_row = stats[stats["regime"] == "bullish"].iloc[0]
        assert bullish_row["pnl"] == 150.0
        assert bullish_row["trade_count"] == 2


# ---------------------------------------------------------------------------
# Backtester.run (with mocked yfinance)
# ---------------------------------------------------------------------------

class TestBacktesterRun:
    def _mock_yf_download(self, prices: pd.Series):
        """Return a mock yf.download result."""
        df = pd.DataFrame({"Close": prices, "Open": prices, "High": prices,
                           "Low": prices, "Volume": [1_000_000] * len(prices)})
        return df

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_run_returns_backtest_result(self, mock_dl):
        prices = _make_prices(260)
        mock_dl.return_value = self._mock_yf_download(prices)

        bt = Backtester()
        result = bt.run(["SPY"], date(2025, 1, 2), date(2025, 12, 31))

        assert isinstance(result, BacktestResult)
        assert isinstance(result.trades, pd.DataFrame)
        assert isinstance(result.equity_curve, pd.DataFrame)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.regime_stats, pd.DataFrame)

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_run_produces_trades(self, mock_dl):
        prices = _make_prices(260)
        mock_dl.return_value = self._mock_yf_download(prices)

        result = Backtester().run(["SPY"], date(2025, 1, 2), date(2025, 12, 31))
        assert result.metrics["total_trades"] > 0

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_run_two_tickers(self, mock_dl):
        prices = _make_prices(260)
        mock_dl.return_value = self._mock_yf_download(prices)

        result = Backtester().run(["SPY", "QQQ"], date(2025, 1, 2), date(2025, 12, 31))
        if not result.trades.empty:
            assert len(result.trades["ticker"].unique()) <= 2

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_run_empty_price_returns_empty_result(self, mock_dl):
        mock_dl.return_value = pd.DataFrame()

        result = Backtester().run(["SPY"], date(2025, 1, 2), date(2025, 12, 31))
        assert result.trades.empty
        assert result.metrics["total_trades"] == 0

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_equity_curve_starts_at_starting_equity(self, mock_dl):
        prices = _make_prices(260)
        mock_dl.return_value = self._mock_yf_download(prices)

        bt = Backtester(starting_equity=50_000)
        result = bt.run(["SPY"], date(2025, 1, 2), date(2025, 12, 31))
        assert result.equity_curve.iloc[0]["account_balance"] == 50_000

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_equity_curve_has_timestamp_column(self, mock_dl):
        prices = _make_prices(260)
        mock_dl.return_value = self._mock_yf_download(prices)

        result = Backtester().run(["SPY"], date(2025, 1, 2), date(2025, 12, 31))
        assert "timestamp" in result.equity_curve.columns
        assert "account_balance" in result.equity_curve.columns

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_metrics_keys_present(self, mock_dl):
        prices = _make_prices(260)
        mock_dl.return_value = self._mock_yf_download(prices)

        result = Backtester().run(["SPY"], date(2025, 1, 2), date(2025, 12, 31))
        for key in ("total_trades", "win_rate", "profit_factor",
                    "max_drawdown_pct", "sharpe", "avg_hold_days"):
            assert key in result.metrics

    @patch("trading_agent.streamlit.backtest_ui.yf.download", side_effect=Exception("network"))
    def test_run_handles_download_exception(self, _mock_dl):
        result = Backtester().run(["SPY"], date(2025, 1, 2), date(2025, 12, 31))
        assert result.trades.empty


# ---------------------------------------------------------------------------
# Smoke: render_backtest_ui
# ---------------------------------------------------------------------------

class TestRenderBacktestUiSmoke:
    def test_renders_without_exception(self):
        # Before "Run Backtest" is clicked the page shows an info message — no exception.
        # Use lambda-import so the module's globals (including `st`) are available.
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_function(
            lambda: __import__(
                "trading_agent.streamlit.backtest_ui",
                fromlist=["render_backtest_ui"],
            ).render_backtest_ui()
        )
        at.run(timeout=15)
        assert not at.exception
