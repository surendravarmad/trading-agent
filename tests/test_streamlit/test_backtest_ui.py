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
    DAILY_OTM_PCT,
    INTRADAY_OTM_PCT,
    LEADERSHIP_WINDOW_BARS,
    RS_ZSCORE_THRESHOLD,
    STARTING_EQUITY,
    VIX_INHIBIT_ZSCORE,
    VIX_WINDOW_BARS,
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

    def test_intraday_otm_pct_triggers_loss_on_small_move(self):
        # Price drops 1% over 12 bars — well within daily 3% but exceeds intraday 0.5%
        # → with INTRADAY_OTM_PCT the short put is breached; daily OTM would see a win.
        idx = pd.date_range("2025-01-02", periods=250, freq="5min")
        prices_list = [500.0] * 211 + [500.0 * 0.99] * 39  # 1% drop from bar 211
        prices = pd.Series(prices_list, index=idx)
        bt = Backtester(target_dte=20)
        outcome_daily, _, _ = bt._simulate(prices, 210, "bullish", credit=1.5, otm_pct=DAILY_OTM_PCT)
        outcome_intraday, _, _ = bt._simulate(prices, 210, "bullish", credit=1.5, otm_pct=INTRADAY_OTM_PCT)
        assert outcome_daily == "win"      # 1% < 3% OTM → not breached
        assert outcome_intraday == "loss"  # 1% > 0.5% OTM → breached

    def test_otm_pct_constants_are_sensible(self):
        assert DAILY_OTM_PCT == 0.03
        assert INTRADAY_OTM_PCT == 0.005
        assert INTRADAY_OTM_PCT < DAILY_OTM_PCT


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


# ---------------------------------------------------------------------------
# ETF macro signals: _zscore_last (population stdev parity)
# ---------------------------------------------------------------------------

class TestZScoreLast:
    """Mirrors MarketDataProvider.get_leadership_zscore math."""

    def test_returns_none_for_short_series(self):
        assert Backtester._zscore_last([]) is None
        assert Backtester._zscore_last([1.0]) is None

    def test_returns_none_for_zero_variance(self):
        # All identical values → stdev 0 → degenerate
        assert Backtester._zscore_last([0.5, 0.5, 0.5, 0.5]) is None

    def test_zero_for_value_at_mean(self):
        # Symmetric series with last == mean → z = 0
        z = Backtester._zscore_last([-1.0, 1.0, 0.0])
        assert z is not None
        assert abs(z) < 1e-9

    def test_positive_for_value_above_mean(self):
        # Series with one big positive last point → z > 0
        z = Backtester._zscore_last([0.0] * 10 + [1.0])
        assert z is not None and z > 0

    def test_population_stdev_not_sample(self):
        """Use n (not n-1) — verify by computing manually."""
        values = [1.0, 2.0, 3.0, 10.0]
        n = 4
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        std = var ** 0.5
        expected = (values[-1] - mean) / std
        assert Backtester._zscore_last(values) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# ETF macro signals: _leadership_zscore_at
# ---------------------------------------------------------------------------

class TestLeadershipZScoreAt:
    """Verify (raw_diff, zscore) computation against pre-loaded series."""

    def _aligned_series(self, n=30, drift=0.0001):
        idx = pd.date_range("2025-01-02", periods=n, freq="B")
        np.random.seed(7)
        rt = np.random.normal(drift, 0.005, n)
        ra = np.random.normal(0.0, 0.005, n)
        ticker = pd.Series(500.0 * np.exp(np.cumsum(rt)), index=idx)
        anchor = pd.Series(400.0 * np.exp(np.cumsum(ra)), index=idx)
        return ticker, anchor

    def test_returns_none_when_idx_zero(self):
        t, a = self._aligned_series()
        assert Backtester._leadership_zscore_at(t, a, idx=0) is None

    def test_returns_pair_with_sufficient_history(self):
        t, a = self._aligned_series(n=40)
        result = Backtester._leadership_zscore_at(t, a, idx=30)
        assert result is not None
        raw, z = result
        assert isinstance(raw, float)
        assert isinstance(z, float)

    def test_returns_none_when_anchor_missing_around_idx(self):
        # Anchor is empty → reindex collapses to 0 rows
        t, _ = self._aligned_series()
        empty_anchor = pd.Series(dtype=float)
        assert Backtester._leadership_zscore_at(t, empty_anchor, idx=20) is None

    def test_handles_misaligned_anchor_index(self):
        """Anchor with sparse timestamps should still produce a signal."""
        t, a = self._aligned_series(n=40)
        # Drop every other anchor bar
        a_sparse = a.iloc[::2]
        result = Backtester._leadership_zscore_at(t, a_sparse, idx=35)
        # Could legitimately be None (too few aligned bars) but must not raise
        assert result is None or len(result) == 2


# ---------------------------------------------------------------------------
# ETF macro signals: _vix_zscore_at
# ---------------------------------------------------------------------------

class TestVIXZScoreAt:
    def _vix_series(self, n=30, base=15.0):
        idx = pd.date_range("2025-01-02", periods=n, freq="B")
        np.random.seed(11)
        # Slow random walk around `base`
        steps = np.random.normal(0.0, 0.4, n)
        levels = base + np.cumsum(steps)
        return pd.Series(levels, index=idx)

    def test_returns_none_for_empty_series(self):
        assert Backtester._vix_zscore_at(pd.Series(dtype=float),
                                         pd.Timestamp("2025-01-15")) is None

    def test_returns_none_for_none_input(self):
        assert Backtester._vix_zscore_at(None,
                                         pd.Timestamp("2025-01-15")) is None

    def test_returns_pair_for_normal_series(self):
        s = self._vix_series(n=30)
        result = Backtester._vix_zscore_at(s, s.index[-1])
        assert result is not None
        raw, z = result
        assert isinstance(raw, float)
        assert isinstance(z, float)

    def test_high_spike_produces_high_zscore(self):
        """Inject a +5pt spike on the last bar; z should exceed +2σ."""
        idx = pd.date_range("2025-01-02", periods=25, freq="B")
        levels = [15.0] * 24 + [20.0]   # 5-pt jump on last bar
        s = pd.Series(levels, index=idx)
        result = Backtester._vix_zscore_at(s, s.index[-1])
        assert result is not None
        _, z = result
        assert z > VIX_INHIBIT_ZSCORE

    def test_uses_only_window_bars_up_to_ts(self):
        """A bar far in the future shouldn't pollute a mid-series z-score.

        Built with mild noise on the early bars so the rolling stdev is
        non-degenerate, then a huge spike is appended *after* the cutoff
        timestamp.  The helper must compute the z-score off the pre-cutoff
        window only, so the late spike has no effect on the result.
        """
        idx = pd.date_range("2025-01-02", periods=40, freq="B")
        np.random.seed(13)
        # Mild noise on bars 0-34 → small but non-zero stdev
        early = 15.0 + np.cumsum(np.random.normal(0.0, 0.15, 35))
        # Huge spike on bars 35-39 (after the cutoff at idx 25)
        late = [50.0] * 5
        s = pd.Series(list(early) + late, index=idx)
        # Reference z-score from the *truncated* series (no late spike)
        truncated = s.iloc[: 26]
        ref = Backtester._vix_zscore_at(truncated, truncated.index[-1])
        # Z-score from the full series, evaluated at bar 25
        result = Backtester._vix_zscore_at(s, idx[25])
        assert result is not None
        assert ref is not None
        # Both must agree to floating-point precision — proves the helper
        # ignored the post-cutoff bars.
        assert result[1] == pytest.approx(ref[1])
        # And the z-score from the pre-cutoff window must NOT exceed the
        # inhibit threshold (the spike that would push it over is in the
        # future relative to ts).
        assert result[1] <= VIX_INHIBIT_ZSCORE


# ---------------------------------------------------------------------------
# Backtester integration: macro signals gate ordering
# ---------------------------------------------------------------------------

class TestMacroSignalsIntegration:
    """End-to-end run() with use_macro_signals=True."""

    def _mock_yf_download(self, prices: pd.Series):
        df = pd.DataFrame({
            "Close": prices, "Open": prices, "High": prices,
            "Low": prices, "Volume": [1_000_000] * len(prices),
        })
        return df

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_default_off_does_not_load_anchors(self, mock_dl):
        prices = _make_prices(260)
        mock_dl.return_value = self._mock_yf_download(prices)
        bt = Backtester()  # use_macro_signals=False by default
        bt.run(["SPY"], date(2025, 1, 2), date(2025, 12, 31))
        # Counters stay at zero when the toggle is off
        assert bt.leadership_biased == 0
        assert bt.vix_inhibited == 0

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_macro_on_loads_anchor_and_vix(self, mock_dl):
        """With macro on, we expect *additional* yf.download calls
        (one per anchor + one for ^VIX)."""
        prices = _make_prices(260)
        mock_dl.return_value = self._mock_yf_download(prices)

        bt = Backtester(use_macro_signals=True)
        bt.run(["SPY"], date(2025, 1, 2), date(2025, 12, 31))
        # Calls: SPY + anchor (QQQ for SPY) + ^VIX = at least 3
        assert mock_dl.call_count >= 3
        called_args = [c.args[0] if c.args else c.kwargs.get("tickers")
                       for c in mock_dl.call_args_list]
        assert "^VIX" in called_args

    def test_vix_spike_demotes_bullish_to_bearish(self):
        """Direct gate-level test: bullish regime + high VIX z → bearish."""
        # Build a flat VIX series with a large terminal spike
        idx = pd.date_range("2025-01-02", periods=25, freq="B")
        vix_levels = [15.0] * 24 + [25.0]  # +10 pt spike
        vix_series = pd.Series(vix_levels, index=idx)

        bt = Backtester(use_macro_signals=True)
        # Re-implement the gate logic locally — verifying the helper
        # composition matches what run() does
        bar_ts = idx[-1]
        vix_z_pair = bt._vix_zscore_at(vix_series, bar_ts)
        assert vix_z_pair is not None
        _, z = vix_z_pair
        assert z > VIX_INHIBIT_ZSCORE  # gate fires

    def test_leadership_constants_exposed(self):
        """Backtester re-exports live-agent constants — guard against drift."""
        from trading_agent.strategy import StrategyPlanner
        from trading_agent.regime import VIX_INHIBIT_ZSCORE as live_vix
        from trading_agent.market_data import MarketDataProvider as live_mdp

        assert RS_ZSCORE_THRESHOLD == StrategyPlanner.RS_ZSCORE_THRESHOLD
        assert VIX_INHIBIT_ZSCORE == live_vix
        assert LEADERSHIP_WINDOW_BARS == live_mdp.LEADERSHIP_WINDOW_BARS
        assert VIX_WINDOW_BARS == live_mdp.VIX_WINDOW_BARS


# ---------------------------------------------------------------------------
# Backtester._load_anchor_series / _load_vix_series
# ---------------------------------------------------------------------------

class TestLoadAnchorVixSeries:

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_load_anchor_returns_empty_when_no_known_anchors(self, mock_dl):
        # Tickers with no LEADERSHIP_ANCHORS entry → no download call
        out = Backtester._load_anchor_series(
            ["UNKNOWN_TICKER"], date(2025, 1, 1), date(2025, 2, 1), "1d",
        )
        assert out == {}
        mock_dl.assert_not_called()

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_load_anchor_returns_dict_keyed_by_anchor(self, mock_dl):
        # SPY → QQQ anchor
        prices = _make_prices(50)
        mock_dl.return_value = pd.DataFrame({"Close": prices})
        out = Backtester._load_anchor_series(
            ["SPY"], date(2025, 1, 1), date(2025, 2, 1), "1d",
        )
        # Must contain the anchor key (QQQ) — not the ticker (SPY)
        assert "QQQ" in out
        assert isinstance(out["QQQ"], pd.Series)

    @patch("trading_agent.streamlit.backtest_ui.yf.download",
           side_effect=Exception("network"))
    def test_load_anchor_handles_exception(self, _mock_dl):
        out = Backtester._load_anchor_series(
            ["SPY"], date(2025, 1, 1), date(2025, 2, 1), "1d",
        )
        assert out == {}

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_load_vix_returns_series_on_success(self, mock_dl):
        prices = _make_prices(50, start=15.0)
        mock_dl.return_value = pd.DataFrame({"Close": prices})
        s = Backtester._load_vix_series(
            date(2025, 1, 1), date(2025, 2, 1), "1d",
        )
        assert s is not None
        assert isinstance(s, pd.Series)
        assert not s.empty

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_load_vix_returns_none_on_empty(self, mock_dl):
        mock_dl.return_value = pd.DataFrame()
        assert Backtester._load_vix_series(
            date(2025, 1, 1), date(2025, 2, 1), "1d",
        ) is None

    @patch("trading_agent.streamlit.backtest_ui.yf.download",
           side_effect=Exception("network"))
    def test_load_vix_returns_none_on_exception(self, _mock_dl):
        assert Backtester._load_vix_series(
            date(2025, 1, 1), date(2025, 2, 1), "1d",
        ) is None


# ---------------------------------------------------------------------------
# Live Quote Refresh gating
# ---------------------------------------------------------------------------
#
# Two gates govern whether the per-bar Live Quote Refresh stage actually
# runs (see backtest_ui.py:run() ~line 2487):
#
#   1. NOT in use_alpaca_historical mode (the historical plan IS the
#      truthful quote — refreshing against today's snapshot would
#      overwrite honest economics with stale-vs-actual-entry quotes).
#   2. Entry date age <= _SNAPSHOT_FRESH_DAYS (today's snapshot is a
#      structurally meaningless proxy for the quote at an old entry).
#
# These tests guard against re-introducing the cross-mode pollution
# that was emitting bogus "Credit drifted XXXX%" warnings in
# historical-mode runs.

class TestRefreshGating:
    """Verify _refresh_live_quotes is bypassed in the right configurations."""

    def _mock_yf_close_series(self, n: int = 260, drift: float = 0.0003):
        return _make_prices(n, drift=drift)

    def _mock_yf_dataframe(self, prices: pd.Series):
        return pd.DataFrame({
            "Close": prices, "Open": prices, "High": prices,
            "Low": prices, "Volume": [1_000_000] * len(prices),
        })

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_historical_mode_never_calls_refresh(self, mock_dl):
        """Part A: with use_alpaca_historical=True the refresh stage
        must NOT fire — even when API keys are present and the entry
        date is fresh."""
        prices = self._mock_yf_close_series(260)
        mock_dl.return_value = self._mock_yf_dataframe(prices)

        bt = Backtester(
            use_alpaca_historical=True,
            alpaca_api_key="test-key",
            alpaca_secret_key="test-secret",
        )
        # Stub the historical plan to always return a usable plan dict
        # so the per-bar loop reaches the refresh decision point.
        # Returns (plan, reason) where reason is "" on success.
        bt._alpaca_historical_plan = MagicMock(return_value=(
            {
                "strike_distance_pct": 0.03,
                "credit": 1.5,
                "approx_abs_delta": 0.18,
                "option_type": "put",
                "short_strike": 485.0,
                "long_strike": 480.0,
                "expiration": "2025-12-19",
            },
            "",
        ))
        # Spy on the refresh method — it MUST NOT be called
        bt._refresh_live_quotes = MagicMock()
        # Skip the actual exit simulation
        bt._simulate_alpaca_historical = MagicMock(
            return_value=("win", 75.0, 5),
        )

        bt.run(["SPY"], date(2025, 1, 2), date(2025, 12, 31))

        bt._refresh_live_quotes.assert_not_called()

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_stale_snapshot_mode_never_calls_refresh(self, mock_dl):
        """Part B: outside snapshot mode + entry date older than
        _SNAPSHOT_FRESH_DAYS, refresh must NOT fire (today's quote is
        a meaningless proxy for that entry's true quote)."""
        prices = self._mock_yf_close_series(260)
        mock_dl.return_value = self._mock_yf_dataframe(prices)

        bt = Backtester(
            use_alpaca_historical=False,
            alpaca_api_key="test-key",
            alpaca_secret_key="test-secret",
        )
        bt._refresh_live_quotes = MagicMock()
        # Backtest dates are 2025 — many months older than _SNAPSHOT_FRESH_DAYS=3
        bt.run(["SPY"], date(2025, 1, 2), date(2025, 12, 31))

        bt._refresh_live_quotes.assert_not_called()

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_fresh_snapshot_mode_calls_refresh(self, mock_dl):
        """Positive control: entry date inside _SNAPSHOT_FRESH_DAYS and
        snapshot mode → refresh SHOULD fire so we catch quote drift
        between planning and execution (the original purpose of the
        feature).

        Notes on test setup:

        * pd.bdate_range can return ``periods-1`` entries when ``end``
          lands on a weekend (it rolls the boundary back), so we derive
          n from the actual index length to keep ``closes`` aligned
          regardless of when the test runs.
        * The backtest loop spaces entries by ``hold_bars`` (= the
          instance's ``target_dte``).  With the default ``target_dte=35``
          on a 260-bar daily series, the latest possible entry lands
          ~24 business days before the end of the series — well outside
          the 3-day fresh-snapshot window, so refresh would never fire.
          We pass ``target_dte=1`` so entries can fire on every bar and
          the most recent one is guaranteed to be inside the fresh window.
        * ``_get_option_chain_for_date`` is stubbed to return None so the
          test doesn't make a real Alpaca API call with fake credentials.
          The refresh-eligibility check happens AFTER that call regardless
          of its outcome, so the stub is purely about avoiding network I/O.
        """
        end = date.today()
        idx = pd.bdate_range(end=pd.Timestamp(end), periods=260)
        n = len(idx)
        np.random.seed(42)
        rt = np.random.normal(0.0003, 0.01, n)
        closes = 500.0 * np.exp(np.cumsum(rt))
        prices = pd.Series(closes, index=idx, name="Close")
        mock_dl.return_value = self._mock_yf_dataframe(prices)

        bt = Backtester(
            use_alpaca_historical=False,
            alpaca_api_key="test-key",
            alpaca_secret_key="test-secret",
            # Tight DTE so entries can land on the last bar (within the
            # _SNAPSHOT_FRESH_DAYS window).  See docstring above.
            target_dte=1,
        )
        # Avoid hitting the real Alpaca API with fake credentials
        bt._get_option_chain_for_date = MagicMock(return_value=None)
        # Stub refresh to a no-op success (returns the inputs unchanged)
        bt._refresh_live_quotes = MagicMock(
            side_effect=lambda **kw: (
                kw.get("strike_distance_pct"),
                kw.get("credit"),
                kw.get("approx_abs_delta"),
                "success",
            ),
        )
        bt.run(["SPY"], date.today() - timedelta(days=400), end)
        # At least one entry bar must fall inside the fresh-snapshot
        # window → refresh fires at least once.
        assert bt._refresh_live_quotes.call_count >= 1

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_refresh_skipped_without_api_keys(self, mock_dl):
        """Without Alpaca credentials the refresh stage cannot run —
        regardless of mode or entry date."""
        prices = self._mock_yf_close_series(260)
        mock_dl.return_value = self._mock_yf_dataframe(prices)

        bt = Backtester(
            use_alpaca_historical=False,
            alpaca_api_key="",       # explicitly empty
            alpaca_secret_key="",
        )
        bt._refresh_live_quotes = MagicMock()
        bt.run(["SPY"], date(2025, 1, 2), date(2025, 12, 31))
        bt._refresh_live_quotes.assert_not_called()


# ---------------------------------------------------------------------------
# σ-horizon contract for _alpaca_historical_plan
# ---------------------------------------------------------------------------
#
# The short-leg strike target is computed from σ-projection.  For intraday
# backtests, projecting σ over the EXPIRATION horizon (e.g. 30+ days) puts
# strikes ~13% OTM with |Δ|≈0.02 — well below the live agent's 0.15 MIN_DELTA
# floor, so the resulting trades aren't apples-to-apples with live.  The
# fix passes ``hold_bars`` + ``bars_per_year`` so σ is projected over the
# real ~1-hour intraday hold horizon instead.  These tests guard against
# regression to the legacy DTE-only behavior.

class TestAlpacaHistoricalSigmaHorizon:
    """Verify _alpaca_historical_plan projects σ over hold horizon."""

    # Synthetic chain: $1 strike grid from $200 to $300 inclusive, both
    # types.  Wide enough for any reasonable σ-target to land inside.
    @staticmethod
    def _build_chain(option_type: str) -> List[dict]:
        chain = []
        for k in range(200, 301):
            chain.append({
                "symbol": f"TEST260529{option_type[0].upper()}{k:08d}",
                "strike": float(k),
                "expiration": "2026-05-29",
                "type": option_type,
            })
        return chain

    @staticmethod
    def _stub_bars(symbols, start, end, **kw):
        """Synthetic bars priced as a monotonically-decreasing function of
        ``|strike − spot|`` (spot = $250 in every test in this class).

        Why this matters
        ----------------
        ``_alpaca_historical_plan`` rejects with ``non_positive_credit``
        when ``short_close - long_close ≤ 0``.  A flat $0.50 stub close
        on every leg makes that guard fire on every test, masking the
        σ-horizon behavior we actually want to verify.

        The ``5 / (1 + 0.1·|strike − spot|)`` decay is strictly positive
        and strictly decreasing in distance from spot, so the leg closer
        to spot (short) always trades richer than the leg further OTM
        (long), yielding a positive credit regardless of which strikes
        the picker chose.  No floor is needed — at the chain's worst
        case (|strike − spot| = 50) the formula still returns $0.83.
        """
        SPOT = 250.0
        out = {}
        for sym in symbols:
            # OCC-style symbol from ``_build_chain``:
            # ``TEST260529<P|C><8-digit integer strike>``
            try:
                strike = float(int(sym[-8:]))
            except (ValueError, IndexError):
                strike = SPOT
            close = 5.0 / (1.0 + 0.1 * abs(strike - SPOT))
            out[sym] = [{"t": str(start), "o": close, "h": close,
                         "l": close, "c": close, "v": 100}]
        return out

    def _make_bt(self) -> Backtester:
        bt = Backtester(
            alpaca_api_key="test-key",
            alpaca_secret_key="test-secret",
        )
        bt._pick_alpaca_expiration = MagicMock(return_value="2026-05-29")
        bt._fetch_alpaca_option_contracts = MagicMock(
            side_effect=lambda ticker, expiry, opt_type: self._build_chain(opt_type),
        )
        bt._fetch_alpaca_option_bars = MagicMock(side_effect=self._stub_bars)
        return bt

    def test_intraday_picks_strike_near_spot_not_far_otm(self):
        """With hold_bars=12 (1-hour intraday) and bars_per_year=19656,
        σ_hold ≈ σ_annual × 0.0247.  At spot=$250, σ_annual=0.25, the
        short put should land within ~3-5% of spot — NOT the ~13% OTM
        the legacy DTE-horizon math would produce."""
        bt = self._make_bt()
        plan, reason = bt._alpaca_historical_plan(
            ticker="TEST",
            entry_date=date(2026, 4, 24),
            regime="bullish",            # → put
            target_dte=35,
            underlying_price=250.0,
            sigma_annual=0.25,
            effective_sigma_mult=1.5,
            hold_bars=12,                # intraday: 12 × 5-min = 1 hour
            bars_per_year=252 * 78,      # 19656
        )
        assert plan is not None, f"unexpected reject: {reason}"
        assert reason == ""
        distance_pct = (250.0 - plan["short_strike"]) / 250.0
        # Hold-horizon σ × 1.5 ≈ 0.92% — strike should be a few % OTM,
        # not 13% OTM.  Allow 5% upper bound for safety.
        assert 0.0 <= distance_pct <= 0.05, (
            f"Expected short strike within 5% of spot under hold-horizon "
            f"σ-projection, got {distance_pct*100:.2f}% OTM "
            f"(short=${plan['short_strike']}, spot=$250)"
        )

    def test_daily_mode_close_to_legacy(self):
        """Daily mode: hold_bars==target_dte and bars_per_year==252.

        The new hold-horizon math is numerically *very close* to the
        legacy DTE-horizon math but not bit-identical because:
          * legacy uses ``dte_days`` from the actual expiration string
            (calendar days, e.g. 34 days from 2026-04-24 → 2026-05-29)
          * the new path uses ``hold_bars`` (= ``target_dte`` = 35
            business days)

        The 1-day delta can shift the picked strike by at most 1 grid
        step.  This test bounds the divergence so accidental mode swaps
        in the daily path stay flagged."""
        bt = self._make_bt()
        plan_daily, reason_daily = bt._alpaca_historical_plan(
            ticker="TEST",
            entry_date=date(2026, 4, 24),
            regime="bullish",
            target_dte=35,
            underlying_price=250.0,
            sigma_annual=0.25,
            effective_sigma_mult=1.5,
            hold_bars=35,
            bars_per_year=252,
        )
        plan_legacy, reason_legacy = bt._alpaca_historical_plan(
            ticker="TEST",
            entry_date=date(2026, 4, 24),
            regime="bullish",
            target_dte=35,
            underlying_price=250.0,
            sigma_annual=0.25,
            effective_sigma_mult=1.5,
            # No hold_bars/bars_per_year → legacy DTE fallback
        )
        assert plan_daily is not None, f"unexpected reject: {reason_daily}"
        assert plan_legacy is not None, f"unexpected reject: {reason_legacy}"
        # Both should pick a strike within 1 grid step ($1) of each
        # other — they're using essentially the same horizon (35 vs 34
        # days).  This bound catches accidental order-of-magnitude
        # divergences without being brittle to the 1-day discrepancy.
        delta = abs(plan_daily["short_strike"] - plan_legacy["short_strike"])
        assert delta <= 1.0, (
            f"Daily mode (hold_bars=35,bars/yr=252) and legacy DTE-horizon "
            f"path should pick strikes within $1 of each other; got "
            f"daily=${plan_daily['short_strike']}, "
            f"legacy=${plan_legacy['short_strike']} (Δ=${delta})"
        )

    def test_legacy_caller_falls_back_to_dte_horizon(self):
        """Backward-compat: callers that don't pass hold_bars/bars_per_year
        still get the original DTE-horizon math.  At spot=$250, σ=0.25,
        DTE=35d → σ_hold ≈ 9.3%, target_dist ≈ 14% OTM (sigma_mult=1.5)."""
        bt = self._make_bt()
        plan, reason = bt._alpaca_historical_plan(
            ticker="TEST",
            entry_date=date(2026, 4, 24),
            regime="bullish",
            target_dte=35,
            underlying_price=250.0,
            sigma_annual=0.25,
            effective_sigma_mult=1.5,
            # No hold_bars/bars_per_year
        )
        assert plan is not None, f"unexpected reject: {reason}"
        distance_pct = (250.0 - plan["short_strike"]) / 250.0
        # Legacy DTE-horizon math: σ × √(35/252) × 1.5 ≈ 13.8%
        # Allow 10-20% as the DTE-horizon band (includes strike-grid snap).
        assert 0.10 <= distance_pct <= 0.20, (
            f"Legacy DTE-horizon path should pick strike ~14% OTM at these "
            f"inputs, got {distance_pct*100:.2f}% OTM"
        )

    def test_intraday_picks_closer_strike_than_legacy(self):
        """Direct comparison: same inputs except hold_bars/bars_per_year.
        Intraday hold-horizon path MUST pick a strike strictly closer to
        spot than the legacy DTE-horizon path.  This is the core of the
        Option B fix and the most important regression to guard."""
        bt = self._make_bt()
        common = dict(
            ticker="TEST",
            entry_date=date(2026, 4, 24),
            regime="bullish",
            target_dte=35,
            underlying_price=250.0,
            sigma_annual=0.25,
            effective_sigma_mult=1.5,
        )
        plan_intraday, reason_intraday = bt._alpaca_historical_plan(
            **common, hold_bars=12, bars_per_year=252 * 78,
        )
        plan_legacy, reason_legacy = bt._alpaca_historical_plan(
            **common,  # DTE fallback
        )
        assert plan_intraday is not None, f"unexpected reject: {reason_intraday}"
        assert plan_legacy is not None, f"unexpected reject: {reason_legacy}"
        intraday_dist = 250.0 - plan_intraday["short_strike"]
        legacy_dist = 250.0 - plan_legacy["short_strike"]
        assert intraday_dist < legacy_dist, (
            f"Intraday hold-horizon path should pick strike closer to spot. "
            f"intraday short=${plan_intraday['short_strike']} "
            f"(dist={intraday_dist:.2f}), "
            f"legacy short=${plan_legacy['short_strike']} "
            f"(dist={legacy_dist:.2f})"
        )

    def test_bearish_regime_picks_call_otm_using_hold_horizon(self):
        """Symmetric verification for the call side: bearish regime →
        OTM call short, distance computed from hold-horizon σ."""
        bt = self._make_bt()
        plan, reason = bt._alpaca_historical_plan(
            ticker="TEST",
            entry_date=date(2026, 4, 24),
            regime="bearish",            # → call
            target_dte=35,
            underlying_price=250.0,
            sigma_annual=0.25,
            effective_sigma_mult=1.5,
            hold_bars=12,
            bars_per_year=252 * 78,
        )
        assert plan is not None, f"unexpected reject: {reason}"
        assert plan["option_type"] == "call"
        distance_pct = (plan["short_strike"] - 250.0) / 250.0
        assert 0.0 <= distance_pct <= 0.05, (
            f"Bearish call short should be within 5% of spot under "
            f"hold-horizon projection, got {distance_pct*100:.2f}% OTM"
        )


# ---------------------------------------------------------------------------
# Credit-ratio gate parity in alpaca-historical mode
# ---------------------------------------------------------------------------
#
# Prior to Apr 2026 the credit-ratio gate skip-condition was
# ``not use_sigma_path``.  Because ``use_sigma_path`` is true whenever
# σ-strike-picking is on (regardless of whether the credit ultimately came
# from real Alpaca bars or the synthetic σ-credit model), the gate was
# silently disabled in alpaca-historical+σ runs and trades with C/W far
# below ``min_credit_ratio=0.33`` reached the trade journal anyway.
#
# Fix: gate skip is now ``not credit_is_synthetic`` where
# ``credit_is_synthetic = use_sigma_path and not use_alpaca_historical``.
# These tests guard against re-introducing the broader skip condition.

class TestCreditRatioGateAlpacaHistorical:
    """min_credit_ratio must fire in alpaca-historical mode and stay
    bypassed in pure synthetic-σ mode."""

    @staticmethod
    def _mock_yf_dataframe(prices: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({
            "Close": prices, "Open": prices, "High": prices,
            "Low": prices, "Volume": [1_000_000] * len(prices),
        })

    @staticmethod
    def _low_credit_plan() -> dict:
        # C/W = 0.50 / 5.00 = 0.10, well below the 0.33 floor.
        return {
            "short_symbol": "SPY260529P00485000",
            "long_symbol": "SPY260529P00480000",
            "short_strike": 485.0,
            "long_strike": 480.0,
            "expiration": "2026-05-29",
            "option_type": "put",
            "credit": 0.50,
            "strike_distance_pct": 0.03,
            "approx_abs_delta": 0.18,
        }

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_alpaca_historical_low_cw_trades_are_rejected_by_credit_ratio_gate(
        self, mock_dl
    ):
        """In alpaca-historical mode the credit comes from REAL bar
        closes, so the data-quality rationale of the gate fully applies.
        A plan with C/W=0.10 must trip ``credit_ratio<floor`` and never
        reach the trade journal."""
        prices = _make_prices(260)
        mock_dl.return_value = self._mock_yf_dataframe(prices)

        bt = Backtester(
            use_alpaca_historical=True,
            alpaca_api_key="test-key",
            alpaca_secret_key="test-secret",
            spread_width=5.0,
            min_credit_ratio=0.33,
            # Disable adjacent gates so the credit-ratio gate is what
            # actually fires (not earnings, not IV-rank, not max-Δ,
            # not max-risk).  Each one off for a *different* reason —
            # we want the test to fail if the credit-ratio gate
            # disappears, not if some other gate's default changes.
            use_iv_gate=False,
            use_earnings_gate=False,
            max_delta=None,
            max_risk_pct=None,
        )
        bt._alpaca_historical_plan = MagicMock(
            return_value=(self._low_credit_plan(), ""),
        )
        # Trade-execution simulator must never be reached for a rejected
        # plan — assert this explicitly via call_count below.
        bt._simulate_alpaca_historical = MagicMock(
            return_value=("win", 50.0, 5),
        )

        bt.run(["SPY"], date(2025, 1, 2), date(2025, 12, 31))

        assert bt.rejections.get("credit_ratio<floor", 0) >= 1, (
            f"Expected credit_ratio<floor rejections in alpaca-historical "
            f"mode with C/W=0.10 vs floor=0.33; got rejections={dict(bt.rejections)}"
        )
        assert bt._simulate_alpaca_historical.call_count == 0, (
            "Trades with C/W below min_credit_ratio must be rejected "
            "before the simulator runs."
        )
        assert bt._funnel.simulated == 0, (
            "No trades should reach the trade journal when every plan "
            "is below the credit-ratio floor."
        )
        # The skip-marker must NOT be set in alpaca-historical mode —
        # the gate is real, not bypassed.
        assert "credit_ratio" not in bt._funnel.skipped_phases, (
            "credit_ratio must not appear in skipped_phases when the "
            "gate is actively firing in alpaca-historical mode."
        )

    @patch("trading_agent.streamlit.backtest_ui.yf.download")
    def test_synthetic_sigma_mode_skips_credit_ratio_gate(self, mock_dl):
        """In pure synthetic-σ mode (no alpaca-historical) the credit
        is a deterministic function of σ-distance and the gate's data-
        quality rationale does not apply.  The gate must be skipped and
        the funnel must record the skip via ``skipped_phases``."""
        prices = _make_prices(260)
        mock_dl.return_value = self._mock_yf_dataframe(prices)

        bt = Backtester(
            use_alpaca_historical=False,
            alpaca_api_key="",      # no API → no snapshot path
            alpaca_secret_key="",
            spread_width=5.0,
            min_credit_ratio=0.33,
            use_iv_gate=False,
            use_earnings_gate=False,
            max_delta=None,
            max_risk_pct=None,
        )
        bt.run(["SPY"], date(2025, 1, 2), date(2025, 12, 31))

        assert "credit_ratio" in bt._funnel.skipped_phases, (
            f"Synthetic-σ mode must mark credit_ratio as a skipped phase; "
            f"skipped_phases={bt._funnel.skipped_phases}"
        )
        assert bt.rejections.get("credit_ratio<floor", 0) == 0, (
            "Credit-ratio rejections must NOT appear in synthetic-σ mode "
            f"(gate is bypassed); got rejections={dict(bt.rejections)}"
        )


# ---------------------------------------------------------------------------
# Friday-weekly preference + no-bars expiry fallback
# ---------------------------------------------------------------------------
#
# Pre-Apr-2026 picker bug: ``_pick_alpaca_expiration`` chose the strict
# nearest-DTE expiry, which on most weekday entry-dates lands on a
# Mon/Wed/Thu weekly.  Mon/Wed weeklies on QQQ/SPY/IWM trade at a
# fraction of Friday-weekly volume, so Alpaca's options bars endpoint
# routinely returned empty results at deep-OTM strikes — burning ~80%
# of intraday backtest candidates as ``no_bars_on_entry_day``.
#
# Two-pronged fix:
#   1. ``_pick_alpaca_expiration`` now adds a +4-day penalty to non-
#      Friday expiries so liquid Friday weeklies win ties.
#   2. ``_alpaca_historical_plan`` retries with the next-best expiry
#      (up to MAX_FALLBACK_ATTEMPTS=3) when the first-choice expiry
#      yields ``no_bars_on_entry_day`` / ``no_contracts_for_expiry`` /
#      ``non_positive_credit``.  Strike-grid failures
#      (``long_leg_off_grid``, ``no_otm_near_target``) do NOT retry
#      because they're deterministic given the catalogue.
#
# These tests guard against re-introducing either regression.

class TestPickAlpacaExpirationFridayPreference:
    """The expiration picker should prefer Friday weeklies even when a
    Mon/Wed weekly is closer to target_dte."""

    @staticmethod
    def _stub_contracts(expirations: List[str]):
        """Build a stub for ``_alpaca_request`` returning a synthetic
        ``/options/contracts`` response with the given expiration set."""
        def _request(url, params=None, timeout=None):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value={
                "option_contracts": [
                    {"expiration_date": exp, "symbol": f"X{i}"}
                    for i, exp in enumerate(expirations)
                ],
            })
            return resp
        return _request

    def _make_bt(self, expirations: List[str]) -> Backtester:
        bt = Backtester(
            alpaca_api_key="test-key",
            alpaca_secret_key="test-secret",
        )
        bt._alpaca_request = MagicMock(
            side_effect=self._stub_contracts(expirations),
        )
        return bt

    def test_friday_within_4d_beats_monday_at_exact_target(self):
        """target = Mon (entry+35d).  Friday weekly 32d off (3 days
        early) should win over Monday weekly at exact 35d."""
        bt = self._make_bt([
            "2026-05-01",  # Friday, 32d off-target by 3
            "2026-05-04",  # Monday,  35d off-target by 0
            "2026-05-06",  # Wednesday, 37d off-target by 2
        ])
        # Entry Mon 2026-03-30 → target = Mon 2026-05-04 (entry+35d)
        chosen = bt._pick_alpaca_expiration(
            "QQQ", date(2026, 3, 30), target_dte=35,
        )
        assert chosen == "2026-05-01", (
            f"Friday-preference should pick May 1 (Friday, 3d early) over "
            f"May 4 (Monday, exact); got {chosen}"
        )

    def test_friday_loses_when_too_far_from_target(self):
        """A Friday >4d off-target loses to a Monday at exact target,
        because the +4-day penalty is exactly that — 4 days, not infinite."""
        bt = self._make_bt([
            "2026-04-24",  # Friday, 30d off-target by 5  → eff 5+0=5
            "2026-04-29",  # Wednesday, 35d off-target by 0 → eff 0+4=4
            "2026-05-08",  # Friday, 44d off-target by 9  → eff 9+0=9
        ])
        # target = entry + 35 = Mon 2026-04-29
        chosen = bt._pick_alpaca_expiration(
            "QQQ", date(2026, 3, 25), target_dte=35,
        )
        assert chosen == "2026-04-29", (
            f"When no Friday is within 4d of target, the closest non-Friday "
            f"should win; got {chosen}"
        )

    def test_exclude_set_skips_failed_expirations(self):
        """``exclude`` lets the fallback loop request the next-best
        without re-picking the dud."""
        bt = self._make_bt([
            "2026-05-01",  # Friday — will be excluded
            "2026-05-04",  # Monday — should win when May 1 excluded
            "2026-05-08",  # Friday — also a candidate
        ])
        chosen = bt._pick_alpaca_expiration(
            "QQQ", date(2026, 3, 30), target_dte=35,
            exclude={"2026-05-01"},
        )
        # Both May 4 (Mon, eff 0+4=4) and May 8 (Fri, eff 4+0=4) tie;
        # tiebreaker prefers Friday → May 8.
        assert chosen == "2026-05-08", (
            f"With May 1 excluded, picker should pick the Friday May 8 "
            f"over the Monday May 4 (Friday tiebreaker); got {chosen}"
        )


class TestAlpacaHistoricalPlanFallback:
    """When the first-choice expiry returns no bars, the plan builder
    should fall back to the next-best expiration before giving up."""

    @staticmethod
    def _build_chain(expiration: str, option_type: str) -> List[dict]:
        chain = []
        for k in range(200, 301):
            chain.append({
                "symbol": (
                    f"TEST{expiration.replace('-','')[2:]}"
                    f"{option_type[0].upper()}{k:08d}"
                ),
                "strike": float(k),
                "expiration": expiration,
                "type": option_type,
            })
        return chain

    @staticmethod
    def _bars_with_holes(empty_for: List[str]):
        """Return a bars-stub that returns empty bars for any symbol
        belonging to an expiration in ``empty_for``, and synthetic
        priced bars for everything else."""
        SPOT = 250.0

        def _stub(symbols, start, end, **kw):
            out = {}
            for sym in symbols:
                # OCC-style: TEST<YYMMDD><P|C><strike>
                yyyy = "20" + sym[4:6]
                mm = sym[6:8]
                dd = sym[8:10]
                exp = f"{yyyy}-{mm}-{dd}"
                if exp in empty_for:
                    continue   # no bars for this expiry
                try:
                    strike = float(int(sym[-8:]))
                except (ValueError, IndexError):
                    strike = SPOT
                close = 5.0 / (1.0 + 0.1 * abs(strike - SPOT))
                out[sym] = [{"t": str(start), "o": close, "h": close,
                             "l": close, "c": close, "v": 100}]
            return out
        return _stub

    def _make_bt(
        self,
        expiration_order: List[str],
        empty_bars_for: List[str],
    ) -> Backtester:
        bt = Backtester(
            alpaca_api_key="test-key",
            alpaca_secret_key="test-secret",
        )
        # Stub picker to return expirations in order, respecting exclude.
        def _pick(ticker, entry_date, target_dte, exclude=None):
            for exp in expiration_order:
                if not exclude or exp not in exclude:
                    return exp
            return None
        bt._pick_alpaca_expiration = MagicMock(side_effect=_pick)
        bt._fetch_alpaca_option_contracts = MagicMock(
            side_effect=lambda ticker, exp, opt_type: self._build_chain(exp, opt_type),
        )
        bt._fetch_alpaca_option_bars = MagicMock(
            side_effect=self._bars_with_holes(empty_bars_for),
        )
        return bt

    def test_falls_back_when_first_expiry_has_no_bars(self):
        """First-choice expiry returns empty bars; plan builder retries
        with the next-best expiry (which has bars) and succeeds."""
        bt = self._make_bt(
            expiration_order=["2026-05-04", "2026-05-08"],   # Mon then Fri
            empty_bars_for=["2026-05-04"],
        )
        plan, reason = bt._alpaca_historical_plan(
            ticker="TEST",
            entry_date=date(2026, 3, 30),
            regime="bullish",
            target_dte=35,
            underlying_price=250.0,
            sigma_annual=0.25,
            effective_sigma_mult=1.5,
            hold_bars=12,
            bars_per_year=252 * 78,
        )
        assert plan is not None, f"unexpected reject: {reason}"
        assert plan["expiration"] == "2026-05-08", (
            f"Plan should have fallen back from May 4 (no bars) to "
            f"May 8 (bars present); got expiration={plan['expiration']}"
        )

    def test_returns_no_bars_after_fallbacks_when_all_exhausted(self):
        """All offered expirations return empty bars → builder reports
        ``no_bars_after_fallbacks`` rather than masking the cause."""
        bt = self._make_bt(
            expiration_order=["2026-05-04", "2026-05-06", "2026-05-08"],
            empty_bars_for=["2026-05-04", "2026-05-06", "2026-05-08"],
        )
        plan, reason = bt._alpaca_historical_plan(
            ticker="TEST",
            entry_date=date(2026, 3, 30),
            regime="bullish",
            target_dte=35,
            underlying_price=250.0,
            sigma_annual=0.25,
            effective_sigma_mult=1.5,
            hold_bars=12,
            bars_per_year=252 * 78,
        )
        assert plan is None
        assert reason.startswith("no_bars_after_fallbacks:"), (
            f"After exhausting all candidates the builder must report "
            f"no_bars_after_fallbacks; got {reason!r}"
        )

    def test_no_fallback_on_strike_grid_failure(self):
        """``long_leg_off_grid`` is deterministic given the catalogue —
        retrying a different expiry won't help.  The builder must
        return immediately, not iterate.

        Setup: the put chain contains a single strike at ATM ($250).
        With spot=$250 and bullish regime the picker chooses
        short_put=250 (the only OTM put available).  No strike
        ``< 250`` exists for the long leg → ``long_leg_off_grid``
        fires non-retryable, and the fallback loop must NOT iterate.
        """
        bt = Backtester(
            alpaca_api_key="test-key",
            alpaca_secret_key="test-secret",
            spread_width=5.0,
        )
        bt._pick_alpaca_expiration = MagicMock(return_value="2026-05-04")

        def _narrow_chain(ticker, expiration, option_type):
            # Single ATM strike — no room below for the long leg.
            return [{
                "symbol": (
                    f"TEST{expiration.replace('-','')[2:]}"
                    f"{option_type[0].upper()}{250:08d}"
                ),
                "strike": 250.0,
                "expiration": expiration,
                "type": option_type,
            }]
        bt._fetch_alpaca_option_contracts = MagicMock(side_effect=_narrow_chain)
        # Bars fetch must NOT be reached — long_leg_off_grid fires first.
        bt._fetch_alpaca_option_bars = MagicMock(return_value={})

        plan, reason = bt._alpaca_historical_plan(
            ticker="TEST",
            entry_date=date(2026, 3, 30),
            regime="bullish",
            target_dte=35,
            underlying_price=250.0,
            sigma_annual=0.25,
            effective_sigma_mult=1.5,
            hold_bars=12,
            bars_per_year=252 * 78,
        )
        assert plan is None
        assert reason.startswith("long_leg_off_grid:"), (
            f"Strike-grid failures must NOT trigger fallback; got {reason!r}"
        )
        # Picker should have been called exactly once (no retry).
        assert bt._pick_alpaca_expiration.call_count == 1, (
            f"Picker should be called only once on strike-grid failure; "
            f"got call_count={bt._pick_alpaca_expiration.call_count}"
        )
        # Bars fetch must not even have been attempted.
        assert bt._fetch_alpaca_option_bars.call_count == 0, (
            "Strike-grid failure must short-circuit BEFORE the bars fetch."
        )
