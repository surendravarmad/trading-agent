"""
watchlist_chart.py — Plotly chart panel for the Watchlist tab.

Layout (4-row Plotly subplot stack)
-----------------------------------
    Row 1  Price        Candles · SMA-50 · SMA-200 · BB(20,2σ)
                        · ATR bands · Ichimoku Tenkan/Kijun/Cloud
    Row 2  Volume       Bar chart, color-coded by close direction
    Row 3  Oscillators  RSI(14) · Stoch RSI(14)
    Row 4  Trend        MACD(12,26,9) histogram + signal · ADX(14)

Indicator-toggle UI (sidebar checkboxes) lets the user collapse rows 3+4
or hide individual overlays so the panel stays readable.

Library choice
--------------
All indicators are implemented in pure pandas/numpy in this module.
We previously planned ``pandas-ta`` but its transitive dependency on
``numba`` does not have wheels for Python 3.13+ (source builds need LLVM,
which fails on most user machines). Rolling our own matches the existing
in-house pattern (``MarketDataProvider.compute_sma/rsi/bollinger`` and
``multi_tf_regime._adx_series``) and eliminates a dependency-version
risk. ``adx_strength`` is reused from ``multi_tf_regime`` so the badge
in the table and the line on the chart cannot drift.

Architectural safety
--------------------
This module imports ONLY:
  - the data layer (MarketDataProvider.fetch_intraday_bars)
  - the multi-tf regime helper (adx_strength — pure-pandas fallback)
  - plotly + streamlit
It does NOT import decision_engine, chain_scanner, executor, or
risk_manager. Read-only by construction.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from trading_agent.market_data import MarketDataProvider
from trading_agent.multi_tf_regime import adx_strength

logger = logging.getLogger(__name__)


# Public chart timeframe options (presented in the selector).
CHART_TIMEFRAMES = ("5m", "15m", "30m", "1h", "4h", "1d")


# ----------------------------------------------------------------------
# Indicator computation — pure pandas, no third-party TA library.
# Each function returns NaN for the warm-up window (matches the existing
# pattern in MarketDataProvider.compute_sma / compute_rsi).
# ----------------------------------------------------------------------
def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         window: int = 14) -> pd.Series:
    """
    Average True Range (Wilder, 1978).

    TR = max(H-L, |H - C_{-1}|, |L - C_{-1}|)
    ATR = Wilder-smoothed TR (EMA with alpha = 1/window)
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def _macd(close: pd.Series,
          fast: int = 12, slow: int = 26, signal: int = 9
          ) -> tuple:
    """
    MACD (Appel, 1979).

    line   = EMA(close, fast) - EMA(close, slow)
    signal = EMA(line, signal)
    hist   = line - signal
    """
    ema_fast = close.ewm(span=fast,   adjust=False).mean()
    ema_slow = close.ewm(span=slow,   adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist


def _stoch_rsi(close: pd.Series,
               rsi_window: int = 14,
               stoch_window: int = 14,
               k: int = 3, d: int = 3
               ) -> tuple:
    """
    Stochastic RSI (Chande & Kroll, 1994).

    1. Compute RSI(close, rsi_window) — reuses MarketDataProvider.compute_rsi
       to guarantee parity with the regime classifier.
    2. Stoch = (RSI - min(RSI, stoch_window)) / (max - min)
    3. %K = SMA(Stoch, k);  %D = SMA(%K, d)
    """
    rsi = MarketDataProvider.compute_rsi(close, rsi_window)
    rsi_min = rsi.rolling(stoch_window).min()
    rsi_max = rsi.rolling(stoch_window).max()
    stoch = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    k_line = stoch.rolling(k).mean()
    d_line = k_line.rolling(d).mean()
    return k_line, d_line


def _ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
              tenkan: int = 9, kijun: int = 26, senkou: int = 52
              ) -> dict:
    """
    Ichimoku Kinkō Hyō (Hosoda, 1969).

    tenkan_sen   = (max(H, tenkan) + min(L, tenkan)) / 2          — conversion
    kijun_sen    = (max(H, kijun)  + min(L, kijun))  / 2          — baseline
    senkou_a     = (tenkan + kijun) / 2,  shifted +kijun bars     — cloud A
    senkou_b     = (max(H, senkou) + min(L, senkou)) / 2, shifted — cloud B

    Returns the four series aligned to the bar index (no forward shift
    is applied — the chart layer is free to render the cloud as forward
    projection or in-place; for a single-pane mini-chart in-place looks
    cleaner and avoids confusing the user with a future-shifted series).
    """
    tenkan_line = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_line  = (high.rolling(kijun).max()  + low.rolling(kijun).min())  / 2
    span_a = (tenkan_line + kijun_line) / 2
    span_b = (high.rolling(senkou).max() + low.rolling(senkou).min()) / 2
    return {
        "tenkan": tenkan_line,
        "kijun":  kijun_line,
        "span_a": span_a,
        "span_b": span_b,
    }


def _compute_indicators(bars: pd.DataFrame,
                        toggles: dict) -> dict:
    """
    Compute all indicators the chart will draw.

    Returns a dict of {name: pd.Series} — only the keys the user has
    toggled on. Each indicator is wrapped in a try/except so a single
    failure (insufficient data, NaN cascade) is logged at DEBUG and the
    chart simply skips that overlay rather than blanking the whole panel.
    """
    out: dict = {}
    close = bars["Close"]
    high = bars["High"]
    low = bars["Low"]

    # Baseline overlays — reuse MarketDataProvider helpers for parity
    # with the regime classifier.
    try:
        if toggles.get("sma", True):
            out["sma_50"]  = MarketDataProvider.compute_sma(close, 50)
            out["sma_200"] = MarketDataProvider.compute_sma(close, 200)
        if toggles.get("bbands", True):
            u, m, l = MarketDataProvider.compute_bollinger_bands(close, 20, 2.0)
            out["bb_upper"], out["bb_middle"], out["bb_lower"] = u, m, l
        if toggles.get("rsi", True):
            out["rsi"] = MarketDataProvider.compute_rsi(close, 14)
        if toggles.get("adx", True):
            # Per-bar ADX for the row-4 line plot. adx_strength returns
            # the last value; we recompute the full series here using
            # the same math so the line and the badge match.
            out["adx_series"] = _adx_series(bars, window=14)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Baseline indicator failed: %s", exc)

    # Extended overlays — all in-house, no external library.
    try:
        if toggles.get("atr_bands", True):
            atr = _atr(high, low, close, window=14)
            out["atr_upper"] = close + 2 * atr
            out["atr_lower"] = close - 2 * atr
        if toggles.get("ichimoku", True):
            ich = _ichimoku(high, low, close)
            out["ichi_tenkan"] = ich["tenkan"]
            out["ichi_kijun"]  = ich["kijun"]
            out["ichi_span_a"] = ich["span_a"]
            out["ichi_span_b"] = ich["span_b"]
        if toggles.get("stoch_rsi", True):
            k_line, d_line = _stoch_rsi(close)
            out["stoch_rsi_k"] = k_line
            out["stoch_rsi_d"] = d_line
        if toggles.get("macd", True):
            line, sig, hist = _macd(close)
            out["macd_line"]   = line
            out["macd_signal"] = sig
            out["macd_hist"]   = hist
    except Exception as exc:  # noqa: BLE001
        logger.debug("Extended indicator failed: %s", exc)

    return out


def _adx_series(bars: pd.DataFrame, window: int = 14) -> pd.Series:
    """Full ADX series — same math as multi_tf_regime.adx_strength."""
    high, low, close = bars["High"], bars["Low"], bars["Close"]
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
    atr = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False).mean()


# ----------------------------------------------------------------------
# Chart construction
# ----------------------------------------------------------------------
def build_figure(
    ticker: str,
    interval: str,
    bars: pd.DataFrame,
    toggles: dict,
) -> go.Figure:
    """
    Build the 4-row chart figure. Caller hands in OHLCV bars and the
    toggle dict from the sidebar checkboxes.
    """
    inds = _compute_indicators(bars, toggles)
    rows_enabled = _enabled_rows(toggles)

    fig = make_subplots(
        rows=len(rows_enabled),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=_row_heights(rows_enabled),
        subplot_titles=tuple(rows_enabled),
    )
    row_index = {name: i + 1 for i, name in enumerate(rows_enabled)}

    # --- Row 1: Price + overlays ---------------------------------------
    if "Price" in row_index:
        r = row_index["Price"]
        fig.add_trace(
            go.Candlestick(
                x=bars.index,
                open=bars["Open"], high=bars["High"],
                low=bars["Low"],   close=bars["Close"],
                name=ticker, showlegend=False,
            ),
            row=r, col=1,
        )
        _add_overlay(fig, inds, "sma_50",  "SMA-50",  "#1976d2", r)
        _add_overlay(fig, inds, "sma_200", "SMA-200", "#7e57c2", r)
        _add_overlay(fig, inds, "bb_upper", "BB upper", "rgba(0,200,83,0.5)", r, dash="dot")
        _add_overlay(fig, inds, "bb_lower", "BB lower", "rgba(0,200,83,0.5)", r, dash="dot")
        _add_overlay(fig, inds, "atr_upper", "ATR upper", "rgba(255,109,0,0.5)", r, dash="dash")
        _add_overlay(fig, inds, "atr_lower", "ATR lower", "rgba(255,109,0,0.5)", r, dash="dash")
        _add_overlay(fig, inds, "ichi_tenkan", "Tenkan", "#ff5252", r)
        _add_overlay(fig, inds, "ichi_kijun",  "Kijun",  "#2196f3", r)
        # Ichimoku cloud (Span A vs Span B) drawn as a fill between.
        if "ichi_span_a" in inds and "ichi_span_b" in inds:
            sa, sb = inds["ichi_span_a"], inds["ichi_span_b"]
            fig.add_trace(go.Scatter(
                x=sa.index, y=sa, line=dict(color="rgba(76,175,80,0)"),
                name="Span A", showlegend=False,
            ), row=r, col=1)
            fig.add_trace(go.Scatter(
                x=sb.index, y=sb,
                line=dict(color="rgba(244,67,54,0)"),
                fill="tonexty", fillcolor="rgba(76,175,80,0.10)",
                name="Cloud", showlegend=False,
            ), row=r, col=1)

    # --- Row 2: Volume --------------------------------------------------
    if "Volume" in row_index:
        r = row_index["Volume"]
        colors = np.where(bars["Close"].diff() >= 0, "#26a69a", "#ef5350")
        fig.add_trace(
            go.Bar(x=bars.index, y=bars["Volume"], name="Volume",
                   marker=dict(color=colors), showlegend=False),
            row=r, col=1,
        )

    # --- Row 3: RSI + Stoch RSI ----------------------------------------
    if "Oscillators" in row_index:
        r = row_index["Oscillators"]
        _add_overlay(fig, inds, "rsi", "RSI(14)", "#1976d2", r)
        _add_overlay(fig, inds, "stoch_rsi_k", "StochRSI %K", "#ff6d00", r)
        _add_overlay(fig, inds, "stoch_rsi_d", "StochRSI %D", "#9e9e9e", r, dash="dot")
        # Reference 30 / 70 bands on RSI scale.
        for y, color in ((70, "rgba(213,0,0,0.3)"), (30, "rgba(0,200,83,0.3)")):
            fig.add_hline(y=y, line=dict(color=color, dash="dot"),
                          row=r, col=1)

    # --- Row 4: MACD + ADX ---------------------------------------------
    if "Trend" in row_index:
        r = row_index["Trend"]
        if "macd_hist" in inds:
            hist = inds["macd_hist"].dropna()
            fig.add_trace(
                go.Bar(x=hist.index, y=hist, name="MACD hist",
                       marker=dict(
                           color=np.where(hist >= 0, "#26a69a", "#ef5350")),
                       showlegend=False),
                row=r, col=1,
            )
        _add_overlay(fig, inds, "macd_line",   "MACD",   "#1976d2", r)
        _add_overlay(fig, inds, "macd_signal", "Signal", "#ff6d00", r, dash="dot")
        _add_overlay(fig, inds, "adx_series", "ADX",   "#7e57c2", r)
        # ADX strength reference line at 20.
        if "adx_series" in inds:
            fig.add_hline(y=20, line=dict(color="rgba(126,87,194,0.3)",
                                          dash="dot"), row=r, col=1)

    fig.update_layout(
        height=820,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    fig.update_xaxes(showspikes=True, spikemode="across",
                     spikesnap="cursor", spikethickness=1)
    return fig


# ----------------------------------------------------------------------
# Streamlit panel — composes the controls + chart
# ----------------------------------------------------------------------
def render_chart_panel(provider: MarketDataProvider,
                       tickers: Iterable[str]) -> None:
    """The interactive chart panel below the watchlist table."""
    tickers = list(tickers)
    if not tickers:
        return

    st.divider()
    st.markdown("### 📈 Chart")

    cols = st.columns([2, 2, 6])
    with cols[0]:
        sel_ticker = st.selectbox("Ticker", options=tickers,
                                  key="wl_chart_ticker")
    with cols[1]:
        sel_interval = st.selectbox("Timeframe", options=CHART_TIMEFRAMES,
                                    index=3,  # default 1h
                                    key="wl_chart_interval")
    with cols[2]:
        toggles = _toggle_row()

    try:
        bars = _fetch_bars(provider, sel_ticker, sel_interval)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load {sel_ticker} @ {sel_interval}: {exc}")
        return

    if bars.empty:
        st.warning(f"No bars returned for {sel_ticker} @ {sel_interval}.")
        return

    fig = build_figure(sel_ticker, sel_interval, bars, toggles)
    st.plotly_chart(fig, use_container_width=True)

    # Below-the-chart context: latest ADX value + bucket.
    latest_adx = adx_strength(bars)
    if latest_adx is not None:
        st.caption(
            f"Latest ADX(14) on {sel_interval}: **{latest_adx:.1f}** — "
            f"{'strong trend' if latest_adx >= 40 else 'developing' if latest_adx >= 20 else 'weak/chop'}."
        )


def _toggle_row() -> dict:
    """Indicator-toggle checkboxes in a single row above the chart."""
    with st.expander("Indicators", expanded=False):
        cols = st.columns(4)
        toggles = {
            "sma":       cols[0].checkbox("SMA 50/200",      value=True,  key="t_sma"),
            "bbands":    cols[0].checkbox("Bollinger Bands", value=True,  key="t_bb"),
            "atr_bands": cols[1].checkbox("ATR bands",       value=False, key="t_atr"),
            "ichimoku":  cols[1].checkbox("Ichimoku",        value=True,  key="t_ich"),
            "rsi":       cols[2].checkbox("RSI",             value=True,  key="t_rsi"),
            "stoch_rsi": cols[2].checkbox("Stoch RSI",       value=True,  key="t_srsi"),
            "macd":      cols[3].checkbox("MACD",            value=True,  key="t_macd"),
            "adx":       cols[3].checkbox("ADX",             value=True,  key="t_adx"),
        }
        # Row-level toggles (collapse whole subplot rows).
        cols2 = st.columns(2)
        toggles["row_oscillators"] = cols2[0].checkbox(
            "Show oscillators row (RSI / StochRSI)",
            value=True, key="t_row_osc",
        )
        toggles["row_trend"] = cols2[1].checkbox(
            "Show trend row (MACD / ADX)",
            value=True, key="t_row_trend",
        )
    return toggles


def _enabled_rows(toggles: dict) -> list:
    rows = ["Price", "Volume"]
    if toggles.get("row_oscillators", True):
        rows.append("Oscillators")
    if toggles.get("row_trend", True):
        rows.append("Trend")
    return rows


def _row_heights(rows_enabled: list) -> list:
    """Price row gets the most real estate; sub-panels are compact."""
    weights = {"Price": 0.55, "Volume": 0.15, "Oscillators": 0.15, "Trend": 0.15}
    raw = [weights[r] for r in rows_enabled]
    total = sum(raw)
    return [w / total for w in raw]


def _add_overlay(fig: go.Figure, inds: dict, key: str, name: str,
                 color: str, row: int, *, dash: Optional[str] = None) -> None:
    series = inds.get(key)
    if series is None:
        return
    series = series.dropna()
    if series.empty:
        return
    line = dict(color=color, width=1.4)
    if dash:
        line["dash"] = dash
    fig.add_trace(
        go.Scatter(x=series.index, y=series, mode="lines",
                   name=name, line=line, showlegend=True),
        row=row, col=1,
    )


def _fetch_bars(provider: MarketDataProvider,
                ticker: str, interval: str) -> pd.DataFrame:
    """Daily uses fetch_historical_prices; intraday uses fetch_intraday_bars."""
    if interval == "1d":
        return provider.fetch_historical_prices(ticker)
    return provider.fetch_intraday_bars(ticker, interval)
