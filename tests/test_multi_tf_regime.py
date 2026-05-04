"""
Tests for trading_agent/multi_tf_regime.py.

These tests assert the *contract* the watchlist UI relies on:
  * intraday classification reuses the same _determine_regime rule as the
    daily classifier (no shadow scorer)
  * agreement_score collapses regimes correctly across timeframes
  * a single failing interval does not blank the whole MultiTFRegime
  * ADX strength buckets match the documented thresholds
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from trading_agent.market_data import MarketDataProvider
from trading_agent.multi_tf_regime import (
    DEFAULT_TIMEFRAMES,
    MultiTFRegime,
    adx_strength,
    adx_strength_label,
    classify_multi_tf,
)
from trading_agent.regime import Regime, RegimeAnalysis, RegimeClassifier


# ----------------------------------------------------------------------
# Helpers — synthetic intraday frames
# ----------------------------------------------------------------------
def _bullish_intraday(n: int = 200) -> pd.DataFrame:
    """
    Strong uptrend with realistic volatility.

    Noise σ=2.5 keeps the 20-bar Bollinger Band width above the 4%
    SIDEWAYS-classification floor enforced by ``RegimeClassifier`` — too
    quiet a series will be (correctly) called SIDEWAYS by the same rule
    the daily classifier uses, which is what makes this wrapper's
    parity guarantee meaningful.
    """
    np.random.seed(0)
    closes = np.linspace(100, 160, n) + np.random.normal(0, 2.5, n)
    idx = pd.date_range("2026-04-01 09:30", periods=n, freq="60min")
    return pd.DataFrame({
        "Open":   closes - 0.5,
        "High":   closes + 2.0,
        "Low":    closes - 2.0,
        "Close":  closes,
        "Volume": np.full(n, 1_000_000, dtype="int64"),
    }, index=idx)


def _bearish_intraday(n: int = 200) -> pd.DataFrame:
    np.random.seed(1)
    closes = np.linspace(160, 100, n) + np.random.normal(0, 2.5, n)
    idx = pd.date_range("2026-04-01 09:30", periods=n, freq="60min")
    return pd.DataFrame({
        "Open":   closes - 0.5,
        "High":   closes + 2.0,
        "Low":    closes - 2.0,
        "Close":  closes,
        "Volume": np.full(n, 1_000_000, dtype="int64"),
    }, index=idx)


# ----------------------------------------------------------------------
# Intraday classification — parity with the daily _determine_regime rule.
# ----------------------------------------------------------------------
class TestClassifyIntraday:
    def test_bullish_intraday_returns_bullish(self):
        provider = MarketDataProvider("k", "s")
        # Patch fetch_intraday_bars to return our synthetic frame.
        # Also stub the macro-signal methods — _classify_intraday now
        # populates leadership_zscore / vix_zscore / iv_rank on every
        # intraday cell, and we don't want unit tests hitting the
        # network just because the wrapper got smarter.
        with patch.object(provider, "fetch_intraday_bars",
                          return_value=_bullish_intraday(200)), \
             patch.object(provider, "get_leadership_zscore",
                          return_value=None), \
             patch.object(provider, "get_vix_zscore",
                          return_value=None):
            # Patch the daily delegate so 1d doesn't try to hit yfinance.
            daily = MagicMock(spec=RegimeClassifier)
            daily.classify.return_value = RegimeAnalysis(
                regime=Regime.BULLISH, current_price=130, sma_50=120,
                sma_200=110, sma_50_slope=0.5, rsi_14=60, bollinger_width=0.05,
                reasoning="daily mock",
            )
            out = classify_multi_tf(
                "FAKE", provider,
                intervals=("1h", "1d"),
                daily_classifier=daily,
            )
        assert out.by_interval["1h"].regime == Regime.BULLISH
        assert out.by_interval["1d"].regime == Regime.BULLISH
        assert out.errors == {}

    def test_intraday_with_too_few_bars_records_error(self):
        provider = MarketDataProvider("k", "s")
        # Only 30 bars but the long SMA window is 50 → should fail loud.
        with patch.object(provider, "fetch_intraday_bars",
                          return_value=_bullish_intraday(30)), \
             patch.object(provider, "get_leadership_zscore",
                          return_value=None), \
             patch.object(provider, "get_vix_zscore",
                          return_value=None):
            daily = MagicMock(spec=RegimeClassifier)
            daily.classify.return_value = RegimeAnalysis(
                regime=Regime.BULLISH, current_price=130, sma_50=120,
                sma_200=110, sma_50_slope=0.5, rsi_14=60, bollinger_width=0.05,
                reasoning="daily mock",
            )
            out = classify_multi_tf(
                "FAKE", provider,
                intervals=("1h", "1d"),
                daily_classifier=daily,
            )
        assert "1h" in out.errors
        assert "1d" in out.by_interval  # unaffected


# ----------------------------------------------------------------------
# Agreement scoring
# ----------------------------------------------------------------------
class TestAgreementScore:
    @staticmethod
    def _ra(regime: Regime) -> RegimeAnalysis:
        return RegimeAnalysis(
            regime=regime, current_price=100, sma_50=100, sma_200=100,
            sma_50_slope=0.0, rsi_14=50, bollinger_width=0.05, reasoning="",
        )

    def test_full_alignment_scores_one(self):
        out = MultiTFRegime(ticker="X", by_interval={
            "1d":  self._ra(Regime.BULLISH),
            "4h":  self._ra(Regime.BULLISH),
            "1h":  self._ra(Regime.BULLISH),
            "15m": self._ra(Regime.BULLISH),
        })
        assert out.agreement_score == pytest.approx(1.0)

    def test_split_three_two_scores_correctly(self):
        out = MultiTFRegime(ticker="X", by_interval={
            "1d":  self._ra(Regime.BULLISH),  # anchor
            "4h":  self._ra(Regime.BULLISH),
            "1h":  self._ra(Regime.BULLISH),
            "15m": self._ra(Regime.BEARISH),
            "5m":  self._ra(Regime.BEARISH),
        })
        # 3/5 match the 1d (BULLISH) anchor.
        assert out.agreement_score == pytest.approx(0.6)

    def test_mean_reversion_collapses_to_neutral(self):
        out = MultiTFRegime(ticker="X", by_interval={
            "1d":  self._ra(Regime.BULLISH),
            "4h":  self._ra(Regime.MEAN_REVERSION),
        })
        # MR is bucketed neutral → does NOT match the bullish anchor.
        assert out.agreement_score == pytest.approx(0.5)

    def test_empty_returns_zero(self):
        assert MultiTFRegime(ticker="X").agreement_score == 0.0


# ----------------------------------------------------------------------
# ADX strength helper
# ----------------------------------------------------------------------
class TestADXStrength:
    def test_strong_trend_yields_high_adx(self):
        # ADX needs close-to-close drift to dominate intra-bar range, so
        # we construct a cleaner trend specifically for this assertion.
        # _bullish_intraday() is tuned for the regime classifier (which
        # needs BB-width above 4%), not for ADX.
        n = 200
        closes = np.linspace(100, 200, n)  # 0.50 / bar drift
        idx = pd.date_range("2026-04-01 09:30", periods=n, freq="60min")
        bars = pd.DataFrame({
            "Open":   closes - 0.05,
            "High":   closes + 0.10,   # tight intra-bar range
            "Low":    closes - 0.10,
            "Close":  closes,
            "Volume": np.full(n, 1_000_000, dtype="int64"),
        }, index=idx)
        adx = adx_strength(bars, window=14)
        assert adx is not None
        assert adx > 20, f"expected strong-trend ADX > 20, got {adx}"

    def test_too_few_bars_returns_none(self):
        bars = _bullish_intraday(10)  # < 2*window
        assert adx_strength(bars, window=14) is None

    def test_label_buckets(self):
        assert adx_strength_label(None) == "—"
        assert adx_strength_label(15) == "weak"
        assert adx_strength_label(25) == "developing"
        assert adx_strength_label(45) == "strong"


# ----------------------------------------------------------------------
# Reuse / no-shadow-scorer guard
# ----------------------------------------------------------------------
class TestNoShadowScorer:
    """
    The whole point of this module is to *reuse* RegimeClassifier's
    _determine_regime rule. Verify by patching it and checking that the
    intraday path passes through.
    """

    def test_intraday_calls_determine_regime(self):
        provider = MarketDataProvider("k", "s")
        with patch.object(provider, "fetch_intraday_bars",
                          return_value=_bullish_intraday(200)), \
             patch.object(provider, "get_leadership_zscore",
                          return_value=None), \
             patch.object(provider, "get_vix_zscore",
                          return_value=None):
            with patch.object(
                RegimeClassifier, "_determine_regime",
                return_value=(Regime.BULLISH, "patched")
            ) as patched:
                daily = MagicMock(spec=RegimeClassifier)
                daily.classify.return_value = RegimeAnalysis(
                    regime=Regime.BULLISH, current_price=130, sma_50=120,
                    sma_200=110, sma_50_slope=0.5, rsi_14=60,
                    bollinger_width=0.05, reasoning="daily",
                )
                classify_multi_tf(
                    "FAKE", provider,
                    intervals=("1h",),
                    daily_classifier=daily,
                )
                assert patched.called
