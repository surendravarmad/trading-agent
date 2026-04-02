"""
Phase II — CLASSIFY
Determines the current Market Regime (Bullish, Bearish, Sideways)
based on SMA positioning and slope analysis.
"""

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from trading_agent.market_data import MarketDataProvider

logger = logging.getLogger(__name__)


class Regime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


@dataclass
class RegimeAnalysis:
    """Structured output of the regime classification."""
    regime: Regime
    current_price: float
    sma_50: float
    sma_200: float
    sma_50_slope: float
    rsi_14: float
    bollinger_width: float
    reasoning: str


class RegimeClassifier:
    """
    Classifies market regime using the following rules:

    BULLISH  — Price > SMA-200  AND  SMA-50 slope > 0
    BEARISH  — Price < SMA-200  AND  SMA-50 slope < 0
    SIDEWAYS — Everything else  (price between SMAs or narrow Bollinger Bands)
    """

    BOLLINGER_NARROW_THRESHOLD = 0.04  # 4 % width considered "narrow"

    def __init__(self, data_provider: MarketDataProvider):
        self.data = data_provider

    def classify(self, ticker: str) -> RegimeAnalysis:
        """Run full regime analysis for *ticker*.

        Data sources:
          - yfinance: 200-day historical OHLCV for SMA, RSI, Bollinger Bands
          - Alpaca snapshot: real-time current price for regime decision
        """
        df = self.data.fetch_historical_prices(ticker, period_days=200)
        close = df["Close"]

        sma_50 = self.data.compute_sma(close, 50)
        sma_200 = self.data.compute_sma(close, 200)
        rsi = self.data.compute_rsi(close, 14)
        upper, middle, lower = self.data.compute_bollinger_bands(close, 20, 2.0)
        sma_50_slope = self.data.sma_slope(sma_50, lookback=5)

        # Use Alpaca real-time price for current price (falls back to
        # historical close if the snapshot API is unavailable)
        current_price = self.data.get_current_price(ticker)
        current_sma_50 = float(sma_50.iloc[-1])
        current_sma_200 = float(sma_200.iloc[-1])
        current_rsi = float(rsi.iloc[-1])

        # Bollinger band width as percentage of price
        bb_width = float((upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1])

        regime, reasoning = self._determine_regime(
            current_price, current_sma_50, current_sma_200,
            sma_50_slope, bb_width
        )

        analysis = RegimeAnalysis(
            regime=regime,
            current_price=current_price,
            sma_50=current_sma_50,
            sma_200=current_sma_200,
            sma_50_slope=sma_50_slope,
            rsi_14=current_rsi,
            bollinger_width=bb_width,
            reasoning=reasoning,
        )
        logger.info("[%s] Regime → %s | Price=%.2f SMA50=%.2f SMA200=%.2f Slope=%.4f BB=%.4f RSI=%.1f",
                     ticker, regime.value, current_price, current_sma_50,
                     current_sma_200, sma_50_slope, bb_width, current_rsi)
        return analysis

    def _determine_regime(self, price: float, sma50: float, sma200: float,
                          slope_50: float, bb_width: float):
        """Core classification logic."""
        # Sideways check first: narrow Bollinger Bands
        if bb_width < self.BOLLINGER_NARROW_THRESHOLD:
            return (Regime.SIDEWAYS,
                    f"Bollinger Band width ({bb_width:.4f}) is below "
                    f"threshold ({self.BOLLINGER_NARROW_THRESHOLD}), "
                    f"indicating low volatility / sideways movement.")

        # Bullish: price above SMA-200 and SMA-50 trending up
        if price > sma200 and slope_50 > 0:
            return (Regime.BULLISH,
                    f"Price ({price:.2f}) > SMA-200 ({sma200:.2f}) and "
                    f"SMA-50 slope is positive ({slope_50:.4f}).")

        # Bearish: price below SMA-200 and SMA-50 trending down
        if price < sma200 and slope_50 < 0:
            return (Regime.BEARISH,
                    f"Price ({price:.2f}) < SMA-200 ({sma200:.2f}) and "
                    f"SMA-50 slope is negative ({slope_50:.4f}).")

        # Default: sideways
        return (Regime.SIDEWAYS,
                f"Price ({price:.2f}) is between SMAs or slope direction "
                f"doesn't confirm a trend. SMA-50={sma50:.2f}, "
                f"SMA-200={sma200:.2f}, slope={slope_50:.4f}.")
