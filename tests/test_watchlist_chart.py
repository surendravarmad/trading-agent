"""
Tests for trading_agent/streamlit/watchlist_chart.py.

Most chart-construction code is plotly bookkeeping that can't be deeply
asserted without rendering, but we CAN verify:
  * build_figure runs end-to-end on synthetic OHLCV without errors
  * row layout responds to row-level toggles
  * indicator computation produces every expected key (all in-house,
    no third-party TA library guard needed since we dropped pandas-ta)
  * _adx_series matches the multi_tf_regime.adx_strength last-value
"""

import numpy as np
import pandas as pd
import pytest

from trading_agent.multi_tf_regime import adx_strength
from trading_agent.streamlit.watchlist_chart import (
    CHART_TIMEFRAMES,
    _adx_series,
    _compute_indicators,
    _enabled_rows,
    build_figure,
)


@pytest.fixture
def sample_bars() -> pd.DataFrame:
    """200 hourly bars with a clean uptrend — enough for SMA-200, ADX, BB."""
    n = 200
    np.random.seed(42)
    closes = np.linspace(100, 130, n) + np.random.normal(0, 0.7, n)
    idx = pd.date_range("2026-04-01 09:30", periods=n, freq="60min")
    return pd.DataFrame({
        "Open":   closes - 0.05,
        "High":   closes + 0.20,
        "Low":    closes - 0.20,
        "Close":  closes,
        "Volume": np.random.randint(500_000, 2_000_000, n),
    }, index=idx)


# ----------------------------------------------------------------------
# Layout / row toggles
# ----------------------------------------------------------------------
class TestEnabledRows:
    def test_default_enables_all_rows(self):
        rows = _enabled_rows({})
        assert rows == ["Price", "Volume", "Oscillators", "Trend"]

    def test_can_collapse_oscillators(self):
        rows = _enabled_rows({"row_oscillators": False, "row_trend": True})
        assert rows == ["Price", "Volume", "Trend"]

    def test_can_collapse_trend(self):
        rows = _enabled_rows({"row_oscillators": True, "row_trend": False})
        assert rows == ["Price", "Volume", "Oscillators"]

    def test_only_price_volume_when_both_off(self):
        rows = _enabled_rows({"row_oscillators": False, "row_trend": False})
        assert rows == ["Price", "Volume"]


# ----------------------------------------------------------------------
# Indicator computation
# ----------------------------------------------------------------------
class TestIndicators:
    def test_baseline_indicators_compute(self, sample_bars):
        # All indicators are in-house pure-pandas — no library guard.
        inds = _compute_indicators(sample_bars, toggles={
            "sma": True, "bbands": True, "rsi": True, "adx": True,
            "atr_bands": False, "ichimoku": False,
            "stoch_rsi": False, "macd": False,
        })
        assert "sma_50"  in inds
        assert "sma_200" in inds
        assert "bb_upper" in inds
        assert "rsi" in inds
        assert "adx_series" in inds
        # Extended overlays explicitly disabled — verify they're absent.
        assert "ichi_tenkan" not in inds
        assert "macd_line" not in inds

    def test_extended_indicators_compute(self, sample_bars):
        # All four extended overlays are now first-class (in-house, no
        # external dep) — verify they all produce the expected keys.
        inds = _compute_indicators(sample_bars, toggles={
            "sma": False, "bbands": False, "rsi": False, "adx": False,
            "atr_bands": True, "ichimoku": True,
            "stoch_rsi": True, "macd": True,
        })
        for k in ("atr_upper", "atr_lower",
                  "ichi_tenkan", "ichi_kijun", "ichi_span_a", "ichi_span_b",
                  "stoch_rsi_k", "stoch_rsi_d",
                  "macd_line", "macd_signal", "macd_hist"):
            assert k in inds, k

    def test_toggles_off_skip_computation(self, sample_bars):
        # All toggles off — empty dict.
        inds = _compute_indicators(sample_bars, toggles={
            "sma": False, "bbands": False, "rsi": False, "adx": False,
            "atr_bands": False, "ichimoku": False,
            "stoch_rsi": False, "macd": False,
        })
        assert inds == {}


# ----------------------------------------------------------------------
# ADX series ↔ adx_strength parity
# ----------------------------------------------------------------------
class TestADXSeriesParity:
    def test_last_value_matches_adx_strength(self, sample_bars):
        """The full-series ADX last value MUST equal multi_tf_regime.adx_strength."""
        series = _adx_series(sample_bars, window=14).dropna()
        latest_from_series = float(series.iloc[-1])
        latest_from_helper = adx_strength(sample_bars, window=14)
        assert latest_from_helper == pytest.approx(latest_from_series, rel=1e-9)


# ----------------------------------------------------------------------
# End-to-end figure build
# ----------------------------------------------------------------------
class TestBuildFigure:
    def test_runs_end_to_end(self, sample_bars):
        fig = build_figure("SPY", "1h", sample_bars, toggles={
            "sma": True, "bbands": True, "rsi": True, "adx": True,
            "atr_bands": True, "ichimoku": True,
            "stoch_rsi": True, "macd": True,
            "row_oscillators": True, "row_trend": True,
        })
        # We just want the layout to materialise without an exception.
        assert fig is not None
        # Subplot count == enabled rows.
        rows_axes = sum(1 for k in fig.layout if k.startswith("yaxis"))
        assert rows_axes == 4

    def test_collapses_when_rows_off(self, sample_bars):
        fig = build_figure("SPY", "1h", sample_bars, toggles={
            "sma": True, "row_oscillators": False, "row_trend": False,
        })
        rows_axes = sum(1 for k in fig.layout if k.startswith("yaxis"))
        assert rows_axes == 2  # Price + Volume only


# ----------------------------------------------------------------------
# Public surface — timeframe options match the planned spec
# ----------------------------------------------------------------------
class TestPublicSurface:
    def test_chart_timeframes_includes_all_planned_intervals(self):
        for tf in ("5m", "15m", "30m", "1h", "4h", "1d"):
            assert tf in CHART_TIMEFRAMES
