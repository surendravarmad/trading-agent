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
        provider.compute_sma = MarketDataProvider.compute_sma
        provider.compute_rsi = MarketDataProvider.compute_rsi
        provider.compute_bollinger_bands = MarketDataProvider.compute_bollinger_bands
        provider.sma_slope = MarketDataProvider.sma_slope
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
