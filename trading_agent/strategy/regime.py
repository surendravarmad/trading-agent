"""
Phase II — CLASSIFY
Determines the current Market Regime (Bullish, Bearish, Sideways)
based on SMA positioning and slope analysis.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from trading_agent.market.market_data import MarketDataProvider
from trading_agent.config.loader import RegimeRules

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# ETF leadership anchor map
# ----------------------------------------------------------------------
# For ETFs, raw "vs-SPY" relative strength is a poor signal because:
#   * SPY-vs-SPY is degenerately zero
#   * Sector ETFs (XLK, XLF, …) co-move with SPY by construction
# Anchoring each ticker to a **sibling** benchmark (e.g. QQQ vs SPY,
# IWM vs SPY, sector ETFs vs SPY) yields a meaningful "leadership"
# differential that doesn't decay to zero on benchmark instruments.
#
# To extend: add a new ticker → anchor_ticker entry here.  Tickers
# missing from this map fall through to the default benchmark ``SPY``;
# callers receive ``None`` from the data layer when ticker == anchor,
# which the classifier treats as "no leadership signal".
# Module-level constants seeded from trading_rules.yaml at import time.
# Importable by name (e.g. `from trading_agent.regime import LEADERSHIP_ANCHORS`)
# for backward compatibility with tests and backtest_ui.py.
_regime_rules = RegimeRules()
LEADERSHIP_ANCHORS: dict = _regime_rules.leadership_anchors
VIX_INHIBIT_ZSCORE: float = _regime_rules.vix_inhibit_zscore


class Regime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    MEAN_REVERSION = "mean_reversion"   # 3-std Bollinger Band touch


@dataclass
class RegimeAnalysis:
    """Structured output of the regime classification."""
    regime: Regime
    current_price: float
    sma_50: float
    sma_200: float
    # Raw **dollars-per-day** slope of the 50-day SMA over the last 5 bars
    # (see ``MarketDataProvider.sma_slope``).  Used by downstream code as a
    # sign check only (slope > 0 → bullish trend).  NOT a percentage —
    # dividing by ``sma_50`` gives the equivalent % rate of change.
    sma_50_slope: float
    rsi_14: float
    bollinger_width: float
    reasoning: str
    # Mean reversion signal — set when price touches a 3-std Bollinger Band
    mean_reversion_signal: bool = False
    mean_reversion_direction: str = ""    # "upper" or "lower"
    # ------------------------------------------------------------------
    # ETF macro signals (Items 1, 2, 3 of the ETF-only patch)
    # ------------------------------------------------------------------
    # Item 1 + 2: per-ticker leadership anchor (the sibling we compare
    # against) plus the Z-scored 5-min return differential vs that
    # anchor.  ``leadership_anchor`` is "" when the ticker is its own
    # benchmark or no anchor is configured; ``leadership_zscore`` is
    # 0.0 in that case (no signal).
    leadership_anchor: str = ""
    leadership_zscore: float = 0.0       # Z-score of (ticker - anchor) 5-min return diff
    leadership_raw_diff: float = 0.0     # last raw differential (informational)
    # Item 3: VIX inter-market gate.  ``vix_zscore`` is the 5-min change
    # in ^VIX z-scored against its own intraday distribution; positive
    # values indicate inter-market fear.  ``inter_market_inhibit_bullish``
    # is the consumed boolean used by the strategy planner — True when
    # the gate fires (vix_zscore above VIX_INHIBIT_ZSCORE).
    vix_zscore: float = 0.0
    inter_market_inhibit_bullish: bool = False
    # ------------------------------------------------------------------
    # Capital retainment guards
    iv_rank: float = 0.0               # realized-vol percentile rank 0-100
    high_iv_warning: bool = False      # True when iv_rank > 95 (extreme instability)


class RegimeClassifier:
    """
    Classifies market regime using the following rules:

    BULLISH  — Price > SMA-200  AND  SMA-50 slope > 0
    BEARISH  — Price < SMA-200  AND  SMA-50 slope < 0
    SIDEWAYS — Everything else  (price between SMAs or narrow Bollinger Bands)
    """

    BOLLINGER_NARROW_THRESHOLD = 0.04  # class-level fallback for backward compat

    def __init__(self, data_provider: MarketDataProvider,
                 rules: "RegimeRules | None" = None):
        self.data = data_provider
        r = rules or RegimeRules()
        self.BOLLINGER_NARROW_THRESHOLD = r.bollinger_narrow_threshold
        self.VIX_INHIBIT_ZSCORE = r.vix_inhibit_zscore
        self.LEADERSHIP_ANCHORS = r.leadership_anchors

    def classify(self, ticker: str) -> RegimeAnalysis:
        """Run full regime analysis for *ticker*.

        Data sources:
          - yfinance: 200-day historical OHLCV for SMA, RSI, Bollinger Bands
          - Alpaca snapshot: real-time current price for regime decision
          - Alpaca 5-min bars: relative strength vs SPY/QQQ
        """
        df = self.data.fetch_historical_prices(ticker, period_days=200)
        close = df["Close"]

        sma_50 = self.data.compute_sma(close, 50)
        sma_200 = self.data.compute_sma(close, 200)
        rsi = self.data.compute_rsi(close, 14)
        upper, middle, lower = self.data.compute_bollinger_bands(close, 20, 2.0)
        sma_50_slope = self.data.sma_slope(sma_50, lookback=5)

        # 3-standard-deviation Bollinger Bands for mean-reversion detection
        upper_3std, _, lower_3std = self.data.compute_bollinger_bands(close, 20, 3.0)

        # Use Alpaca real-time price for current price (falls back to
        # historical close if the snapshot API is unavailable)
        current_price = self.data.get_current_price(ticker)
        current_sma_50 = float(sma_50.iloc[-1])
        current_sma_200 = float(sma_200.iloc[-1])
        current_rsi = float(rsi.iloc[-1])

        # Bollinger band width as percentage of price
        bb_width = float((upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1])

        # Mean reversion: check if price has touched the 3-std band
        u3 = float(upper_3std.iloc[-1])
        l3 = float(lower_3std.iloc[-1])
        mr_upper = current_price >= u3
        mr_lower = current_price <= l3
        mean_reversion_signal = mr_upper or mr_lower
        mean_reversion_direction = ("upper" if mr_upper
                                    else "lower" if mr_lower else "")

        # ------------------------------------------------------------------
        # Item 1+2: Z-scored leadership vs the configured anchor benchmark.
        # ------------------------------------------------------------------
        # The pre-ETF code compared every ticker against SPY *and* QQQ
        # using a flat 0.1% threshold.  That broke for ETFs in two ways:
        # (a) SPY-vs-SPY is degenerately zero, so SPY never tripped the
        # bias; (b) sector ETFs co-move with SPY by construction so the
        # raw differential rarely crossed any flat threshold.  The new
        # approach picks one anchor per ticker (LEADERSHIP_ANCHORS) and
        # Z-scores the differential against its own intraday distribution
        # — a 1.5σ leadership move is meaningful regardless of the
        # ticker's idiosyncratic volatility.
        leadership_anchor = self.LEADERSHIP_ANCHORS.get(ticker, "")
        leadership_zscore = 0.0
        leadership_raw_diff = 0.0
        if leadership_anchor and hasattr(self.data, "get_leadership_zscore"):
            result = self.data.get_leadership_zscore(ticker, leadership_anchor)
            if result is not None:
                leadership_raw_diff, leadership_zscore = result

        # ------------------------------------------------------------------
        # Item 3: VIX inter-market gate.
        # ------------------------------------------------------------------
        # When the volatility index spikes >2σ in 5 minutes vs its rolling
        # mean, suppress new bullish-premium openings (Bull Put, Iron
        # Condor put-wing).  This guards short-DTE positions against the
        # macro-fear regime that historically eats credit spreads alive.
        # The signal is a soft inhibit — the planner can still open
        # bearish strategies (Bear Call) when the gate fires.
        vix_zscore = 0.0
        inter_market_inhibit_bullish = False
        if hasattr(self.data, "get_vix_zscore"):
            vix_result = self.data.get_vix_zscore()
            if vix_result is not None:
                _, vix_zscore = vix_result
                inter_market_inhibit_bullish = vix_zscore > self.VIX_INHIBIT_ZSCORE

        # IV rank from realized-volatility percentile over last 200 days.
        # We use rolling 20-day annualised realized vol as an IV proxy and
        # compute what percentile today's reading sits at.
        iv_rank, high_iv_warning = self._compute_iv_rank(close)

        regime, reasoning = self._determine_regime(
            current_price, current_sma_50, current_sma_200,
            sma_50_slope, bb_width, mean_reversion_signal,
            mean_reversion_direction, u3, l3,
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
            mean_reversion_signal=mean_reversion_signal,
            mean_reversion_direction=mean_reversion_direction,
            leadership_anchor=leadership_anchor,
            leadership_zscore=leadership_zscore,
            leadership_raw_diff=leadership_raw_diff,
            vix_zscore=vix_zscore,
            inter_market_inhibit_bullish=inter_market_inhibit_bullish,
            iv_rank=iv_rank,
            high_iv_warning=high_iv_warning,
        )
        logger.info(
            "[%s] Regime → %s | Price=%.2f SMA50=%.2f SMA200=%.2f "
            "Slope=%.4f BB=%.4f RSI=%.1f MR=%s "
            "Lead=%s(z=%+.2f) VIXz=%+.2f InhibitBull=%s "
            "IVRank=%.1f HighVol=%s",
            ticker, regime.value, current_price, current_sma_50,
            current_sma_200, sma_50_slope, bb_width, current_rsi,
            mean_reversion_direction or "none",
            leadership_anchor or "none", leadership_zscore,
            vix_zscore, inter_market_inhibit_bullish,
            iv_rank, high_iv_warning,
        )
        return analysis

    @staticmethod
    def _compute_iv_rank(close: pd.Series,
                         window: int = 20,
                         high_iv_pct: float = 95.0):
        """
        Compute a realized-volatility percentile rank as an IV proxy.

        Method
        ------
        1. Compute rolling *window*-day annualised realised vol over the full
           price series, stepping every 5 days for speed.
        2. Rank today's reading against that distribution via
           ``scipy.stats.percentileofscore`` (``kind='mean'``) — this is the
           standard percentile definition and handles ties symmetrically
           (half the equal-valued observations count below, half above).
           Previous implementation used ``np.sum(hist < current) / n``,
           which is strictly the "strict" (``kind='strict'``) variant and
           systematically under-reports rank whenever ties exist in the
           vol distribution.
        3. Set high_iv_warning = True when iv_rank > *high_iv_pct* (default 95).

        Returns (iv_rank: float, high_iv_warning: bool).
        """
        returns = close.pct_change().dropna()
        if len(returns) < window + 5:
            return 0.0, False

        current_vol = float(returns.tail(window).std() * np.sqrt(252)) * 100

        # Sample historical vols every 5 bars to build the distribution
        hist_vols = []
        for i in range(0, len(returns) - window, 5):
            v = float(returns.iloc[i:i + window].std() * np.sqrt(252)) * 100
            hist_vols.append(v)

        if not hist_vols:
            return 0.0, False

        iv_rank = float(percentileofscore(hist_vols, current_vol, kind="mean"))
        high_iv_warning = iv_rank > high_iv_pct
        return round(iv_rank, 1), high_iv_warning

    def _determine_regime(self, price: float, sma50: float, sma200: float,
                          slope_50: float, bb_width: float,
                          mean_reversion_signal: bool = False,
                          mean_reversion_direction: str = "",
                          upper_3std: float = 0.0,
                          lower_3std: float = 0.0):
        """Core classification logic.

        Mean reversion takes highest priority — a 3-std Bollinger Band
        touch overrides any trend signal.
        """
        # Highest priority: 3-std Bollinger Band touch → Mean Reversion
        if mean_reversion_signal:
            dir_label = ("above upper" if mean_reversion_direction == "upper"
                         else "below lower")
            return (Regime.MEAN_REVERSION,
                    f"Price ({price:.2f}) is {dir_label} 3-std Bollinger Band "
                    f"(upper={upper_3std:.2f}, lower={lower_3std:.2f}). "
                    f"Expect mean reversion.")

        # Sideways check: narrow Bollinger Bands
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
