"""Tests for market data provider — technical indicator calculations."""

import numpy as np
import pandas as pd
import pytest

from trading_agent.market_data import MarketDataProvider


class TestSMA:
    def test_sma_correct_values(self):
        prices = pd.Series([10, 20, 30, 40, 50])
        sma = MarketDataProvider.compute_sma(prices, window=3)
        assert np.isnan(sma.iloc[0])
        assert np.isnan(sma.iloc[1])
        assert sma.iloc[2] == pytest.approx(20.0)
        assert sma.iloc[3] == pytest.approx(30.0)
        assert sma.iloc[4] == pytest.approx(40.0)

    def test_sma_window_larger_than_data(self):
        prices = pd.Series([1, 2, 3])
        sma = MarketDataProvider.compute_sma(prices, window=10)
        assert sma.isna().all()


class TestRSI:
    def test_rsi_range(self, bullish_prices):
        rsi = MarketDataProvider.compute_rsi(bullish_prices["Close"], 14)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_bullish_trend_is_high(self, bullish_prices):
        """In a strong uptrend, RSI should generally be above 50."""
        rsi = MarketDataProvider.compute_rsi(bullish_prices["Close"], 14)
        recent = rsi.tail(20).mean()
        assert recent > 50


class TestBollingerBands:
    def test_bands_ordering(self, bullish_prices):
        upper, middle, lower = MarketDataProvider.compute_bollinger_bands(
            bullish_prices["Close"], 20, 2.0)
        valid_idx = upper.dropna().index
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()


class TestSMASlope:
    def test_positive_slope_for_uptrend(self):
        sma = pd.Series([100, 101, 102, 103, 104])
        slope = MarketDataProvider.sma_slope(sma, lookback=5)
        assert slope > 0

    def test_negative_slope_for_downtrend(self):
        sma = pd.Series([104, 103, 102, 101, 100])
        slope = MarketDataProvider.sma_slope(sma, lookback=5)
        assert slope < 0


class TestStrikeExtraction:
    def test_standard_occ_symbol(self):
        assert MarketDataProvider._extract_strike("SPY250404P00550000") == 550.0

    def test_decimal_strike(self):
        assert MarketDataProvider._extract_strike("SPY250404C00482500") == 482.5

    def test_invalid_symbol(self):
        assert MarketDataProvider._extract_strike("INVALID") == 0.0
