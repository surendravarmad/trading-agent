"""Tests for regime classification logic."""

import pytest
from unittest.mock import MagicMock, patch

from trading_agent.regime import RegimeClassifier, Regime
from trading_agent.market_data import MarketDataProvider


class TestRegimeClassification:
    """Test that regime is correctly identified from price data."""

    def _make_classifier(self, price_data):
        """Helper: build a classifier with mocked data provider."""
        provider = MagicMock(spec=MarketDataProvider)
        provider.fetch_historical_prices.return_value = price_data
        # get_current_price must return a float — MagicMock causes TypeError
        # in comparisons inside the classifier
        provider.get_current_price.return_value = float(
            price_data["Close"].iloc[-1])
        # Static methods must be assigned directly so they behave as callables
        provider.compute_sma = MarketDataProvider.compute_sma
        provider.compute_rsi = MarketDataProvider.compute_rsi
        provider.compute_bollinger_bands = MarketDataProvider.compute_bollinger_bands
        provider.sma_slope = MarketDataProvider.sma_slope
        # ETF macro patch: classify() unpacks the (raw, z) tuple from these
        # signals.  ``MagicMock(spec=…)`` would otherwise return a bare
        # MagicMock that *is* iterable but yields zero items, causing
        # "not enough values to unpack (expected 2, got 0)".  Stub to None
        # so the classifier takes the no-signal branch and the new fields
        # land at their dataclass defaults (0.0 / "" / False).
        provider.get_leadership_zscore = MagicMock(return_value=None)
        provider.get_vix_zscore = MagicMock(return_value=None)
        return RegimeClassifier(provider)

    def test_bullish_regime(self, bullish_prices):
        classifier = self._make_classifier(bullish_prices)
        result = classifier.classify("SPY")
        assert result.regime == Regime.BULLISH
        assert result.current_price > result.sma_200
        assert result.sma_50_slope > 0

    def test_bearish_regime(self, bearish_prices):
        classifier = self._make_classifier(bearish_prices)
        result = classifier.classify("SPY")
        assert result.regime == Regime.BEARISH
        assert result.current_price < result.sma_200
        assert result.sma_50_slope < 0

    def test_sideways_regime(self, sideways_prices):
        classifier = self._make_classifier(sideways_prices)
        result = classifier.classify("SPY")
        assert result.regime == Regime.SIDEWAYS

    def test_analysis_contains_reasoning(self, bullish_prices):
        classifier = self._make_classifier(bullish_prices)
        result = classifier.classify("SPY")
        assert len(result.reasoning) > 0
        assert isinstance(result.rsi_14, float)
        assert isinstance(result.bollinger_width, float)

    def test_new_signal_fields_default(self, bullish_prices):
        """New signal fields are always present with reasonable defaults."""
        classifier = self._make_classifier(bullish_prices)
        result = classifier.classify("SPY")
        assert isinstance(result.mean_reversion_signal, bool)
        assert result.mean_reversion_direction in ("", "upper", "lower")
        # ETF macro patch: Z-score leadership + VIX inter-market gate
        assert isinstance(result.leadership_anchor, str)
        assert isinstance(result.leadership_zscore, float)
        assert isinstance(result.leadership_raw_diff, float)
        assert isinstance(result.vix_zscore, float)
        assert isinstance(result.inter_market_inhibit_bullish, bool)


class TestMeanReversionDetection:
    """Mean reversion regime fires when price touches the 3-std Bollinger Band."""

    def _make_classifier_at_price(self, prices, current_price):
        """Build a classifier that returns *current_price* as get_current_price."""
        provider = MagicMock(spec=MarketDataProvider)
        provider.fetch_historical_prices.return_value = prices
        provider.get_current_price.return_value = float(current_price)
        provider.compute_sma = MarketDataProvider.compute_sma
        provider.compute_rsi = MarketDataProvider.compute_rsi
        provider.compute_bollinger_bands = MarketDataProvider.compute_bollinger_bands
        provider.sma_slope = MarketDataProvider.sma_slope
        # No leadership / VIX support so the new signals stay at defaults
        del provider.get_leadership_zscore
        del provider.get_vix_zscore
        return RegimeClassifier(provider)

    def test_upper_3std_touch_gives_mean_reversion(self, bullish_prices):
        from trading_agent.regime import Regime
        import numpy as np
        # Push price well above the 3-std upper band
        close = bullish_prices["Close"]
        mean = float(close.rolling(20).mean().iloc[-1])
        std = float(close.rolling(20).std().iloc[-1])
        price_above_3std = mean + 3.5 * std

        classifier = self._make_classifier_at_price(bullish_prices, price_above_3std)
        result = classifier.classify("SPY")
        assert result.regime == Regime.MEAN_REVERSION
        assert result.mean_reversion_signal is True
        assert result.mean_reversion_direction == "upper"

    def test_lower_3std_touch_gives_mean_reversion(self, bullish_prices):
        from trading_agent.regime import Regime
        close = bullish_prices["Close"]
        mean = float(close.rolling(20).mean().iloc[-1])
        std = float(close.rolling(20).std().iloc[-1])
        price_below_3std = mean - 3.5 * std

        classifier = self._make_classifier_at_price(bullish_prices, price_below_3std)
        result = classifier.classify("SPY")
        assert result.regime == Regime.MEAN_REVERSION
        assert result.mean_reversion_signal is True
        assert result.mean_reversion_direction == "lower"

    def test_normal_price_no_mean_reversion(self, bullish_prices):
        from trading_agent.regime import Regime
        close = bullish_prices["Close"]
        mean_price = float(close.iloc[-1])  # last close is within bands
        classifier = self._make_classifier_at_price(bullish_prices, mean_price)
        result = classifier.classify("SPY")
        assert result.regime != Regime.MEAN_REVERSION
        assert result.mean_reversion_signal is False


class TestLeadershipAnchorMap:
    """Item 1: every ticker resolves to a sibling anchor (not itself)."""

    def test_spy_anchor_is_qqq(self):
        from trading_agent.regime import LEADERSHIP_ANCHORS
        assert LEADERSHIP_ANCHORS["SPY"] == "QQQ"

    def test_qqq_anchor_is_spy(self):
        from trading_agent.regime import LEADERSHIP_ANCHORS
        assert LEADERSHIP_ANCHORS["QQQ"] == "SPY"

    def test_iwm_and_sectors_anchor_to_spy(self):
        from trading_agent.regime import LEADERSHIP_ANCHORS
        for sector in ("IWM", "XLK", "XLF", "XLE", "XLV", "XLY",
                       "XLI", "XLP", "XLU", "XLB", "XLC", "XLRE"):
            assert LEADERSHIP_ANCHORS[sector] == "SPY", (
                f"{sector} should anchor to SPY")

    def test_no_self_anchor(self):
        """Self-anchoring is degenerate (raw diff always zero)."""
        from trading_agent.regime import LEADERSHIP_ANCHORS
        for ticker, anchor in LEADERSHIP_ANCHORS.items():
            assert ticker != anchor


class TestZScoreLeadershipIntegration:
    """Item 2: classify() reads leadership_zscore via the data provider."""

    def _make_classifier(self, prices, *, leadership=None, vix=None,
                         current_price=None):
        provider = MagicMock(spec=MarketDataProvider)
        provider.fetch_historical_prices.return_value = prices
        provider.get_current_price.return_value = float(
            current_price if current_price is not None
            else prices["Close"].iloc[-1])
        provider.compute_sma = MarketDataProvider.compute_sma
        provider.compute_rsi = MarketDataProvider.compute_rsi
        provider.compute_bollinger_bands = MarketDataProvider.compute_bollinger_bands
        provider.sma_slope = MarketDataProvider.sma_slope
        provider.get_leadership_zscore = MagicMock(return_value=leadership)
        provider.get_vix_zscore = MagicMock(return_value=vix)
        return RegimeClassifier(provider), provider

    def test_z_score_propagated_to_analysis(self, bullish_prices):
        classifier, provider = self._make_classifier(
            bullish_prices, leadership=(0.0008, 1.9), vix=None)
        result = classifier.classify("SPY")
        # Anchor lookup happens against LEADERSHIP_ANCHORS["SPY"] == "QQQ"
        provider.get_leadership_zscore.assert_called_once_with("SPY", "QQQ")
        assert result.leadership_anchor == "QQQ"
        assert result.leadership_zscore == 1.9
        assert result.leadership_raw_diff == 0.0008

    def test_none_result_keeps_defaults(self, bullish_prices):
        classifier, _ = self._make_classifier(
            bullish_prices, leadership=None, vix=None)
        result = classifier.classify("SPY")
        assert result.leadership_zscore == 0.0
        assert result.leadership_raw_diff == 0.0


class TestVIXInterMarketGate:
    """Item 3: VIX z-score > +2σ flips inter_market_inhibit_bullish to True."""

    def _make_classifier(self, prices, *, vix):
        provider = MagicMock(spec=MarketDataProvider)
        provider.fetch_historical_prices.return_value = prices
        provider.get_current_price.return_value = float(prices["Close"].iloc[-1])
        provider.compute_sma = MarketDataProvider.compute_sma
        provider.compute_rsi = MarketDataProvider.compute_rsi
        provider.compute_bollinger_bands = MarketDataProvider.compute_bollinger_bands
        provider.sma_slope = MarketDataProvider.sma_slope
        provider.get_leadership_zscore = MagicMock(return_value=None)
        provider.get_vix_zscore = MagicMock(return_value=vix)
        return RegimeClassifier(provider)

    def test_high_vix_zscore_inhibits_bullish(self, bullish_prices):
        classifier = self._make_classifier(bullish_prices, vix=(0.4, 2.5))
        result = classifier.classify("SPY")
        assert result.vix_zscore == 2.5
        assert result.inter_market_inhibit_bullish is True

    def test_low_vix_zscore_does_not_inhibit(self, bullish_prices):
        classifier = self._make_classifier(bullish_prices, vix=(0.05, 0.4))
        result = classifier.classify("SPY")
        assert result.vix_zscore == 0.4
        assert result.inter_market_inhibit_bullish is False

    def test_threshold_boundary_strictly_greater(self, bullish_prices):
        """Exactly +2.0 σ is NOT enough — gate uses strict > comparison."""
        classifier = self._make_classifier(bullish_prices, vix=(0.3, 2.0))
        result = classifier.classify("SPY")
        assert result.inter_market_inhibit_bullish is False

    def test_none_vix_keeps_defaults(self, bullish_prices):
        classifier = self._make_classifier(bullish_prices, vix=None)
        result = classifier.classify("SPY")
        assert result.vix_zscore == 0.0
        assert result.inter_market_inhibit_bullish is False
