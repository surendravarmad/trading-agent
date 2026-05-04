"""
multi_tf_regime.py — Multi-timeframe regime classification for the
Watchlist tab.

This module is a **read-only analyst wrapper** around the existing
``RegimeClassifier`` machinery. It does NOT introduce a second source of
classification logic — every decision flows through the same
``_determine_regime`` rule the agent uses on the daily timeframe, just fed
intraday OHLCV bars at different intervals.

Why a wrapper rather than parameterising RegimeClassifier
---------------------------------------------------------
``RegimeClassifier.classify`` is hardcoded to daily bars (``period_days=200``)
because the live agent's strategy planning depends on stable daily SMA-50
and SMA-200 levels. Generalising that method risks a regression in the
trade-decision loop. By reusing only the *pure* primitives — the static
``compute_sma / compute_rsi / compute_bollinger_bands`` helpers and the
``_determine_regime`` static rule — we get identical math at intraday
horizons without forking the agent's daily code path.

Per-timeframe SMA windows
-------------------------
SMA-50 / SMA-200 are well-known landmarks on the daily chart. To get the
**same trend semantics** at lower timeframes we'd need ~50 days × bars-per-day
of intraday history, which exceeds yfinance's free-tier intraday lookback
limits. Instead we scale the windows so each timeframe sees roughly the
same calendar horizon as a daily SMA-50 / SMA-200 sees on the daily chart:

      Interval     short SMA      long SMA   target lookback
      --------     ---------      --------   ---------------
      5m              20             50         ~6 hours / 1 day
      15m             20             50         ~5 days
      30m             20             50         ~10 days
      1h / 60m        20             50         ~10 trading days
      4h              20             50         ~50 trading days
      1d              50            200         ~10 months  (delegates to RegimeClassifier)

These are intentionally smaller than (50, 200) for sub-daily intervals
because the full 50-day / 200-day daily windows have no meaningful intraday
analogue. The (20, 50) pair is the conventional swing-trader pairing on
intraday charts and matches how most platform indicators (TradingView,
ThinkOrSwim) default their multi-tf views.

Output
------
``classify_multi_tf`` returns a ``MultiTFRegime`` dataclass holding one
``RegimeAnalysis`` per requested interval plus an *agreement score* —
the share of timeframes whose trend direction matches the longest one.
The watchlist UI uses agreement to highlight tickers where short-term
and long-term regimes align (high-conviction setups) vs diverge
(chop / topping / bottoming).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trading_agent.market_data import MarketDataProvider
from trading_agent.regime import (
    Regime,
    RegimeAnalysis,
    RegimeClassifier,
    VIX_INHIBIT_ZSCORE,
    leadership_anchor_for,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Per-timeframe SMA windows.  See module docstring for the rationale.
# ----------------------------------------------------------------------
# Intervals are the same set we expose from MarketDataProvider plus "1d"
# which delegates to the existing daily RegimeClassifier path.
DEFAULT_TIMEFRAMES: Tuple[str, ...] = ("1d", "4h", "1h", "15m", "5m")

_SMA_WINDOWS: Dict[str, Tuple[int, int]] = {
    "5m":  (20, 50),
    "15m": (20, 50),
    "30m": (20, 50),
    "60m": (20, 50),
    "1h":  (20, 50),
    "4h":  (20, 50),
    "1d":  (50, 200),
}


@dataclass
class MultiTFRegime:
    """Container for per-timeframe regime analyses."""
    ticker: str
    by_interval: Dict[str, RegimeAnalysis] = field(default_factory=dict)
    # Subset of intervals that failed (insufficient data, rate-limit, etc.).
    # The UI displays a neutral cell for these so a single intraday outage
    # doesn't blank the whole row.
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def regimes(self) -> Dict[str, Regime]:
        """{interval: Regime} convenience map for the UI."""
        return {tf: a.regime for tf, a in self.by_interval.items()}

    @property
    def agreement_score(self) -> float:
        """
        Share of timeframes whose trend direction matches the longest
        interval (typically 1d). Range: 0.0 – 1.0.

        We collapse Regime to a 3-way trend bucket:
          BULLISH                → +1
          BEARISH                → -1
          SIDEWAYS / MEAN_REV.   →  0

        A score of 1.0 means every timeframe agrees with the daily — the
        canonical "all timeframes aligned" swing setup. A score near 0.5
        means the lower timeframes are diverging from the higher ones, a
        common signal that a trend is exhausting.
        """
        if not self.by_interval:
            return 0.0

        anchor_interval = self._longest_interval()
        if anchor_interval is None:
            return 0.0
        anchor_dir = _trend_bucket(self.by_interval[anchor_interval].regime)
        matches = sum(
            1
            for a in self.by_interval.values()
            if _trend_bucket(a.regime) == anchor_dir
        )
        return matches / len(self.by_interval)

    def _longest_interval(self) -> Optional[str]:
        """Return the longest available interval (1d > 4h > 1h > 30m > …)."""
        priority = ["1d", "4h", "1h", "60m", "30m", "15m", "5m"]
        for tf in priority:
            if tf in self.by_interval:
                return tf
        return None


def _trend_bucket(regime: Regime) -> int:
    if regime == Regime.BULLISH:
        return 1
    if regime == Regime.BEARISH:
        return -1
    return 0  # SIDEWAYS, MEAN_REVERSION → neutral for agreement scoring


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def classify_multi_tf(
    ticker: str,
    data_provider: MarketDataProvider,
    intervals: Tuple[str, ...] = DEFAULT_TIMEFRAMES,
    *,
    daily_classifier: Optional[RegimeClassifier] = None,
) -> MultiTFRegime:
    """
    Classify *ticker*'s regime at each of the requested *intervals*.

    Parameters
    ----------
    ticker
        Equity / ETF symbol.
    data_provider
        Shared MarketDataProvider instance — reuses snapshot / price /
        intraday-bars caches, so calling this from the Watchlist UI does
        not duplicate yfinance calls already paid for by the live agent.
    intervals
        Subset of ``("5m","15m","30m","60m","1h","4h","1d")``.
    daily_classifier
        Optional pre-built RegimeClassifier. When provided, ``"1d"`` rows
        delegate to ``daily_classifier.classify()`` so the watchlist sees
        bit-identical output to the live agent. When omitted, a fresh
        classifier is constructed on the same data_provider.

    Returns
    -------
    MultiTFRegime
        Per-interval results plus an agreement score. Failures (no data,
        yfinance error) are captured in ``.errors`` rather than raised.
    """
    if daily_classifier is None:
        daily_classifier = RegimeClassifier(data_provider)

    out = MultiTFRegime(ticker=ticker)

    for interval in intervals:
        try:
            if interval == "1d":
                # Delegate to the existing live-agent classifier so the
                # daily row in the watchlist matches what the agent saw
                # in its most recent cycle, byte for byte.
                analysis = daily_classifier.classify(ticker)
            else:
                analysis = _classify_intraday(
                    ticker, interval, data_provider
                )
            out.by_interval[interval] = analysis
        except Exception as exc:  # noqa: BLE001 — UI must stay alive
            logger.debug("[%s] %s regime classification failed: %s",
                         ticker, interval, exc)
            out.errors[interval] = str(exc)

    logger.info(
        "[%s] Multi-TF regime: %d/%d ok, agreement=%.2f",
        ticker,
        len(out.by_interval),
        len(intervals),
        out.agreement_score,
    )
    return out


# ----------------------------------------------------------------------
# Intraday classification — reuses the same primitives as the daily path.
# ----------------------------------------------------------------------
def _classify_intraday(
    ticker: str,
    interval: str,
    data: MarketDataProvider,
) -> RegimeAnalysis:
    """Fetch intraday bars, compute SMAs/RSI/BB, run _determine_regime."""
    short_w, long_w = _SMA_WINDOWS[interval]
    bars = data.fetch_intraday_bars(ticker, interval)

    close: pd.Series = bars["Close"]
    if len(close) < long_w + 5:
        # Mirror the daily-path failure mode: refuse rather than silently
        # falling through to a degenerate SIDEWAYS classification when
        # the long SMA is mostly NaN.
        raise ValueError(
            f"{ticker} @ {interval}: only {len(close)} bars, "
            f"need at least {long_w + 5} for stable SMA-{long_w}"
        )

    # SAME static helpers used by RegimeClassifier — guaranteed parity.
    sma_short = data.compute_sma(close, short_w)
    sma_long = data.compute_sma(close, long_w)
    rsi = data.compute_rsi(close, 14)
    upper, middle, lower = data.compute_bollinger_bands(close, 20, 2.0)
    upper_3std, _, lower_3std = data.compute_bollinger_bands(close, 20, 3.0)
    sma_short_slope = data.sma_slope(sma_short, lookback=5)
    sma_long_slope = data.sma_slope(sma_long, lookback=5)

    current_price = float(close.iloc[-1])
    bb_width = float((upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1])
    u3 = float(upper_3std.iloc[-1])
    l3 = float(lower_3std.iloc[-1])

    mr_upper = current_price >= u3
    mr_lower = current_price <= l3
    mean_reversion_signal = mr_upper or mr_lower
    mean_reversion_direction = (
        "upper" if mr_upper else "lower" if mr_lower else ""
    )

    # Reuse the static classifier rule — single source of truth.
    classifier = RegimeClassifier(data)
    regime, reasoning = classifier._determine_regime(
        current_price,
        float(sma_short.iloc[-1]),
        float(sma_long.iloc[-1]),
        sma_short_slope,
        bb_width,
        mean_reversion_signal,
        mean_reversion_direction,
        u3,
        l3,
    )

    # ------------------------------------------------------------------
    # Macro / inter-market signals (previously left at defaults).
    # ------------------------------------------------------------------
    # Even though VIX z-score and leadership z-score are *market-wide*
    # signals (not specific to any timeframe), populating them on every
    # RegimeAnalysis keeps consumers honest — anything that reads the
    # intraday object directly (backtester scaffolding, future ML
    # features) sees real values instead of zeros pretending to be data.
    # All three are wrapped in try/except so a single failed RPC can't
    # blank an otherwise-good intraday row.
    leadership_anchor = leadership_anchor_for(ticker)
    leadership_zscore = 0.0
    leadership_raw_diff = 0.0
    leadership_signal_available = False
    if leadership_anchor and hasattr(data, "get_leadership_zscore"):
        try:
            result = data.get_leadership_zscore(ticker, leadership_anchor)
            if result is not None:
                leadership_raw_diff, leadership_zscore = result
                leadership_signal_available = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("[%s] leadership_zscore @ %s failed: %s",
                         ticker, interval, exc)

    vix_zscore = 0.0
    inter_market_inhibit_bullish = False
    if hasattr(data, "get_vix_zscore"):
        try:
            vix_result = data.get_vix_zscore()
            if vix_result is not None:
                _, vix_zscore = vix_result
                inter_market_inhibit_bullish = vix_zscore > VIX_INHIBIT_ZSCORE
        except Exception as exc:  # noqa: BLE001
            logger.debug("[%s] vix_zscore @ %s failed: %s",
                         ticker, interval, exc)

    # IV rank from realized-vol percentile — same algorithm as the daily
    # path, just fed intraday closes.  Returns 0.0 when the bar history
    # is too short for a stable percentile, which is the same fallback
    # the daily classifier uses.
    try:
        iv_rank, high_iv_warning = RegimeClassifier._compute_iv_rank(close)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[%s] iv_rank @ %s failed: %s", ticker, interval, exc)
        iv_rank, high_iv_warning = 0.0, False

    # Trend conflict — same rule the daily path applies.
    trend_conflict = (
        (sma_short_slope < 0 and sma_long_slope > 0) or
        (sma_short_slope > 0 and sma_long_slope < 0)
    )

    # Last bar timestamp — for the watchlist Stale-data chip.
    last_bar_ts = None
    try:
        ts = close.index[-1]
        last_bar_ts = pd.Timestamp(ts).to_pydatetime()
    except Exception:  # noqa: BLE001
        last_bar_ts = None

    return RegimeAnalysis(
        regime=regime,
        current_price=current_price,
        sma_50=float(sma_short.iloc[-1]),   # short window labelled as sma_50
        sma_200=float(sma_long.iloc[-1]),   # long window labelled as sma_200
        sma_50_slope=sma_short_slope,
        rsi_14=float(rsi.iloc[-1]),
        bollinger_width=bb_width,
        reasoning=reasoning,
        mean_reversion_signal=mean_reversion_signal,
        mean_reversion_direction=mean_reversion_direction,
        leadership_anchor=leadership_anchor,
        leadership_zscore=leadership_zscore,
        leadership_raw_diff=leadership_raw_diff,
        leadership_signal_available=leadership_signal_available,
        vix_zscore=vix_zscore,
        inter_market_inhibit_bullish=inter_market_inhibit_bullish,
        iv_rank=iv_rank,
        high_iv_warning=high_iv_warning,
        sma_200_slope=sma_long_slope,
        trend_conflict=trend_conflict,
        last_bar_ts=last_bar_ts,
    )


# ----------------------------------------------------------------------
# ADX strength helper — used by the UI for the regime-strength badge.
# ----------------------------------------------------------------------
def adx_strength(bars: pd.DataFrame, window: int = 14) -> Optional[float]:
    """
    Compute the latest ADX value from an OHLCV DataFrame.

    ADX (Average Directional Index, Wilder 1978) measures **trend strength**
    irrespective of direction:
      * ADX < 20      → weak / chop
      * 20 ≤ ADX < 40 → developing trend
      * ADX ≥ 40      → strong trend

    Implemented here in pure pandas/numpy so the watchlist tab does not
    require ``pandas-ta`` for the categorical regime-strength badge.
    PR #4 swaps this for the pandas-ta version when the chart panel
    pulls in that dependency anyway, and at that point this helper is
    the single thing to update.
    """
    if bars is None or len(bars) < window * 2:
        return None

    high = bars["High"]
    low = bars["Low"]
    close = bars["Close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move

    # Wilder smoothing == EMA with alpha=1/window.
    atr = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr)

    # IMPORTANT: replace 0 → np.nan (NOT pd.NA). pd.NA is the masked-array
    # NA scalar; replacing into a float Series with it coerces dtype to
    # `object`, after which `.ewm().mean()` raises
    #   pandas.errors.DataError: No numeric types to aggregate
    # Using np.nan keeps the Series in float64.
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    # Force-cast in case any upstream op (e.g. resample on an empty
    # group, downcast(infer_objects)) leaked an object dtype.
    dx = pd.to_numeric(dx, errors="coerce")
    if dx.dropna().empty:
        return None
    adx = dx.ewm(alpha=1 / window, adjust=False).mean()

    last = adx.iloc[-1]
    if pd.isna(last):
        return None
    return float(last)


def adx_strength_label(adx_value: Optional[float]) -> str:
    """Bucket the ADX number into the categorical strength label."""
    if adx_value is None:
        return "—"
    if adx_value < 20:
        return "weak"
    if adx_value < 40:
        return "developing"
    return "strong"
