"""
watchlist_ui.py — Streamlit Watchlist tab.

PR #3 scope (this file)
-----------------------
- Add / remove tickers (persisted to knowledge_base/watchlist.json).
- Per-ticker multi-timeframe regime table (1d / 4h / 1h / 15m / 5m).
- Macro context strip (VIX z-score + agreement summary).

PR #4 will add the candlestick chart panel below the table — kept as a
separate render fn (``render_chart_panel``) here as a stub so the UI
file stays cohesive.

Caching strategy
----------------
``@st.cache_data`` keyed on ``(ticker, intervals_tuple, refresh_token)``
so a Streamlit rerun within the same refresh window hits the cache. The
``refresh_token`` is bumped manually by the "Refresh" button or
automatically every ``WATCHLIST_REFRESH_SECS`` (default 60s) by the
session-state ticker.

Architectural safety
--------------------
This module ONLY imports the data layer, the multi-tf wrapper, the
watchlist store, and the regime enum. It deliberately does NOT import
``decision_engine``, ``chain_scanner``, ``executor``, or ``risk_manager``
— so the watchlist cannot affect trade decisions, only display.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from trading_agent.config import load_config
from trading_agent.market_data import MarketDataProvider
from trading_agent.multi_tf_regime import (
    DEFAULT_TIMEFRAMES,
    MultiTFRegime,
    adx_strength,
    adx_strength_label,
    classify_multi_tf,
)
from trading_agent.regime import Regime, RegimeClassifier
from trading_agent.streamlit.components import REGIME_COLORS
from trading_agent.watchlist_store import (
    DEFAULT_WATCHLIST_PATH,
    add_ticker,
    load_watchlist,
    remove_ticker,
)

logger = logging.getLogger(__name__)


# Refresh cadence — short enough for an analyst tab, long enough that
# yfinance doesn't rate-limit us. Same env-var convention as
# LIVE_MONITOR_REFRESH_SECS used by the live tab.
WATCHLIST_REFRESH_SECS = int(os.environ.get("WATCHLIST_REFRESH_SECS", "60"))


# ----------------------------------------------------------------------
# Per-process singletons — built once via @st.cache_resource so the
# data provider's snapshot / intraday caches are shared across reruns.
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_data_provider() -> MarketDataProvider:
    cfg = load_config()
    return MarketDataProvider(
        alpaca_api_key=cfg.alpaca.api_key,
        alpaca_secret_key=cfg.alpaca.secret_key,
        alpaca_data_url=cfg.alpaca.data_url,
        alpaca_base_url=cfg.alpaca.base_url,
    )


@st.cache_resource(show_spinner=False)
def _get_daily_classifier() -> RegimeClassifier:
    return RegimeClassifier(_get_data_provider())


# ----------------------------------------------------------------------
# Cached classification — wraps classify_multi_tf with a Streamlit cache.
# ``refresh_token`` is the manual cache-bust knob (bumped on user click /
# auto-refresh tick).
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=WATCHLIST_REFRESH_SECS)
def _classify_cached(
    ticker: str,
    intervals: Tuple[str, ...],
    refresh_token: int,
) -> Dict:
    """
    Returns a serialisable dict so Streamlit's cache hash is stable.
    The UI rebuilds Regime enum values on read.
    """
    provider = _get_data_provider()
    daily = _get_daily_classifier()
    out = classify_multi_tf(ticker, provider,
                            intervals=intervals,
                            daily_classifier=daily)

    by_interval = {}
    adx_by_interval = {}
    for tf, analysis in out.by_interval.items():
        # ``last_bar_ts`` is serialised as an ISO-8601 string so the
        # @st.cache_data hash stays stable (datetime objects from
        # different tz instances would otherwise miss the cache).
        last_bar_iso = (
            analysis.last_bar_ts.isoformat()
            if analysis.last_bar_ts is not None else None
        )
        by_interval[tf] = {
            "regime":           analysis.regime.value,
            "current_price":    analysis.current_price,
            "rsi_14":           analysis.rsi_14,
            "iv_rank":          analysis.iv_rank,
            "leadership_z":     analysis.leadership_zscore,
            "leadership_anchor": analysis.leadership_anchor,
            "leadership_signal_available": analysis.leadership_signal_available,
            "vix_z":            analysis.vix_zscore,
            "reasoning":        analysis.reasoning,
            "trend_conflict":   analysis.trend_conflict,
            "last_bar_ts":      last_bar_iso,
        }
        # ADX strength — only meaningful on intraday/daily bars; skip if
        # we can't fetch them cheaply (e.g. for the daily delegate).
        try:
            if tf == "1d":
                bars = provider.fetch_historical_prices(ticker)
            else:
                bars = provider.fetch_intraday_bars(ticker, tf)
            adx_by_interval[tf] = adx_strength(bars)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[%s] ADX skipped at %s: %s", ticker, tf, exc)
            adx_by_interval[tf] = None

    return {
        "ticker":     ticker,
        "by_interval": by_interval,
        "adx":         adx_by_interval,
        "errors":      dict(out.errors),
        "agreement":   out.agreement_score,
    }


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def render_watchlist() -> None:
    """Top-level entry point — registered as the 4th tab in app.py."""
    st.subheader("📊 Watchlist")
    st.caption(
        "Multi-timeframe regime view. **Read-only** — does not influence "
        "the live agent's trade decisions. Same regime rule "
        "(`_determine_regime`) as the daily classifier, fed intraday bars."
    )

    if "watchlist_refresh_token" not in st.session_state:
        st.session_state.watchlist_refresh_token = int(time.time())

    _render_controls()
    wl = load_watchlist(DEFAULT_WATCHLIST_PATH)
    if not wl.tickers:
        st.info(
            "No tickers yet — add some above. The macro VIX context strip "
            "and per-ticker regime table will appear once you do."
        )
        return

    rows = _classify_all(wl.symbols())
    _render_macro_strip(rows)
    _render_table(rows)

    # PR #4 chart panel — full Plotly stack with indicator toggles.
    from trading_agent.streamlit.watchlist_chart import render_chart_panel
    render_chart_panel(_get_data_provider(), wl.symbols())


def _render_controls() -> None:
    """Add / remove ticker inputs + manual refresh button."""
    cols = st.columns([3, 1, 1, 1])
    with cols[0]:
        new_ticker = st.text_input(
            "Add ticker (uppercased automatically)",
            key="wl_add_input",
            placeholder="e.g. SPY, QQQ, AAPL",
        ).strip().upper()
    with cols[1]:
        if st.button("Add", use_container_width=True):
            if new_ticker:
                add_ticker(new_ticker, path=DEFAULT_WATCHLIST_PATH)
                _bump_refresh()
                st.rerun()
            else:
                st.warning("Enter a ticker first")
    with cols[2]:
        wl = load_watchlist(DEFAULT_WATCHLIST_PATH)
        if wl.tickers:
            to_remove = st.selectbox(
                "Remove",
                options=["—"] + wl.symbols(),
                key="wl_remove_select",
                label_visibility="collapsed",
            )
            if to_remove != "—" and st.button("✕ Remove",
                                              use_container_width=True):
                remove_ticker(to_remove, path=DEFAULT_WATCHLIST_PATH)
                _bump_refresh()
                st.rerun()
    with cols[3]:
        if st.button("⟳ Refresh", use_container_width=True):
            _bump_refresh()
            st.rerun()


def _bump_refresh() -> None:
    st.session_state.watchlist_refresh_token = int(time.time())
    # Streamlit's cache_data TTL also self-expires every
    # WATCHLIST_REFRESH_SECS, but bumping the token forces immediate
    # invalidation when the user explicitly clicks Refresh.
    _classify_cached.clear()


def _classify_all(tickers: List[str]) -> List[Dict]:
    """Classify every ticker; failures are captured per-ticker."""
    rows = []
    token = st.session_state.watchlist_refresh_token
    for t in tickers:
        try:
            rows.append(_classify_cached(t, DEFAULT_TIMEFRAMES, token))
        except Exception as exc:  # noqa: BLE001 — keep UI alive
            logger.warning("[%s] watchlist classification failed: %s", t, exc)
            rows.append({
                "ticker": t,
                "by_interval": {},
                "adx": {},
                "errors": {"_top": str(exc)},
                "agreement": 0.0,
            })
    logger.info("Watchlist refreshed %d tickers", len(rows))
    return rows


def _render_macro_strip(rows: List[Dict]) -> None:
    """Top-of-page summary: VIX z-score, # tickers, avg agreement."""
    # Pull the VIX z-score from the first row that has a 1d cell — it's
    # a market-wide signal so any successful row exposes the same value.
    vix_z = next(
        (r["by_interval"]["1d"]["vix_z"]
         for r in rows
         if "1d" in r["by_interval"]),
        0.0,
    )
    avg_agreement = (
        sum(r["agreement"] for r in rows) / len(rows) if rows else 0.0
    )

    cols = st.columns(3)
    cols[0].metric("Tickers", len(rows))
    cols[1].metric(
        "VIX z-score (5min Δ)",
        f"{vix_z:+.2f}",
        delta="Bullish-inhibit" if vix_z > 2.0 else "OK",
        delta_color="inverse" if vix_z > 2.0 else "normal",
    )
    cols[2].metric(
        "Avg TF agreement",
        f"{avg_agreement:.0%}",
        help="Share of timeframes whose trend matches each ticker's "
             "longest interval (typically 1d). 100% = all aligned.",
    )


# Anchors with reliably-good Alpaca IEX coverage during RTH.  When
# rows anchored to these still come back signal-unavailable, the cause
# is almost certainly market-closed / off-hours / bad credentials —
# NOT IEX feed thinness.  Used by ``_lead_z_health`` to classify the
# failure mode the user is seeing.
_BROAD_MARKET_ANCHORS = frozenset({"SPY", "QQQ", "IWM", "DIA"})


def _lead_z_health(rows: List[Dict]) -> Dict[str, int]:
    """Bucket Lead-z signal availability across the table.

    Returns counts so the caller can pick the right banner copy:
      * ``broad_ok / broad_total``   — rows anchored to SPY/QQQ/IWM/DIA
      * ``sector_ok / sector_total`` — rows anchored to a sector ETF
                                        (XLF, XLK, …)
      * ``no_anchor``                — rows with leadership_anchor == ""
                                        (special tickers, unknown to the map)
      * ``total``                    — rows with a 1d cell (rows with no
                                        1d cell are excluded; we can't
                                        say anything about them)

    The classification is intentionally based on the **anchor**, not the
    ticker itself, because the failure mode is per-anchor: if XLF's 5-min
    series is empty, every JPM/BAC/WFC row anchored to XLF fails together.
    """
    counts = {
        "broad_ok": 0, "broad_total": 0,
        "sector_ok": 0, "sector_total": 0,
        "no_anchor": 0, "total": 0,
    }
    for r in rows:
        d = r.get("by_interval", {}).get("1d")
        if not d:
            continue
        counts["total"] += 1
        anchor = d.get("leadership_anchor", "")
        ok = bool(d.get("leadership_signal_available", False))
        if not anchor:
            counts["no_anchor"] += 1
            continue
        if anchor in _BROAD_MARKET_ANCHORS:
            counts["broad_total"] += 1
            if ok:
                counts["broad_ok"] += 1
        else:
            # Sector ETFs and any other non-broad anchor.
            counts["sector_total"] += 1
            if ok:
                counts["sector_ok"] += 1
    return counts


def _render_lead_z_health_banner(counts: Dict[str, int]) -> None:
    """Render a single banner explaining the current Lead-z source state.

    Picks one of four messages based on the per-anchor success counts so
    the user can self-diagnose whether the failure is *systemic*
    (markets closed / creds bad → all rows fail) vs *per-anchor*
    (only sector-ETF anchors fail → IEX feed thinness).
    """
    broad_ok = counts["broad_ok"]
    broad_total = counts["broad_total"]
    sector_ok = counts["sector_ok"]
    sector_total = counts["sector_total"]
    total = counts["total"]
    if total == 0:
        return  # Empty watchlist — nothing to report.

    anchored_total = broad_total + sector_total
    anchored_ok = broad_ok + sector_ok

    # Happy path: every anchored row has a real reading.  Stay quiet.
    if anchored_total > 0 and anchored_ok == anchored_total:
        st.success(
            f"✅ Lead-z source healthy — {anchored_ok}/{anchored_total} "
            "tickers have a real reading."
        )
        return

    # Total blackout: zero anchored rows succeeded.
    if anchored_total > 0 and anchored_ok == 0:
        st.error(
            f"❌ Lead-z unavailable for all {anchored_total} anchored "
            "tickers.  Likely cause: markets closed (weekend / off-hours), "
            "Alpaca credentials missing or rejected, or the Alpaca data "
            "URL is unreachable.  Check `APCA_API_KEY_ID` / "
            "`APCA_API_SECRET_KEY` and confirm the time matches a US "
            "equities session."
        )
        return

    # Sector-only failure: broad-market anchors work but every sector
    # anchor fails.  Classic IEX feed thinness on XL* ETFs.
    if (broad_total > 0 and broad_ok == broad_total
            and sector_total > 0 and sector_ok == 0):
        st.warning(
            f"⚠️ Lead-z works for SPY/QQQ-anchored rows ({broad_ok}/"
            f"{broad_total}) but fails for all {sector_total} "
            "sector-ETF-anchored rows.  Cause: Alpaca's free IEX feed "
            "has thin volume on sector ETFs (XLF, XLK, XLY, …) so "
            "after `OPEN_BAR_SKIP` is dropped there are <2 5-min bars "
            "to compute a rolling stdev.  Fix: set "
            "`ALPACA_STOCKS_FEED=sip` for full-market coverage, or "
            "remap the affected tickers' anchors to SPY in "
            "`LEADERSHIP_ANCHORS`."
        )
        return

    # Mixed / partial failure.  Surface the breakdown so the user
    # knows which anchors to investigate.
    st.warning(
        f"⚠️ Lead-z partially available: "
        f"{anchored_ok}/{anchored_total} anchored tickers have a real "
        f"reading "
        f"(broad anchors {broad_ok}/{broad_total}, "
        f"sector anchors {sector_ok}/{sector_total}).  "
        "See the legend below for what '— (no data vs X)' means."
    )


def _render_table(rows: List[Dict]) -> None:
    """One row per ticker; one column per timeframe + ADX + IV rank."""
    intervals = list(DEFAULT_TIMEFRAMES)

    # Detect stale data once per render.  We use the 1d cell's last bar
    # because it's the most reliably populated; if the daily delegate
    # was healthy, intraday rows tend to be too.  When we're unsure
    # (no last_bar_ts), default to "fresh" — better to omit a chip
    # than display a misleading one.
    stale_age_minutes = _stale_minutes(rows)

    # Lead-z source-health banner — rendered above the table so the user
    # can interpret "no data" rows immediately.  Computed once per render
    # from the 1d cells (cheap; just dict access).
    _render_lead_z_health_banner(_lead_z_health(rows))

    # Build the dataframe in display order.
    display_rows = []
    for r in rows:
        record = {"Ticker": r["ticker"]}
        for tf in intervals:
            cell = r["by_interval"].get(tf)
            adx = r["adx"].get(tf)
            label = adx_strength_label(adx)
            if cell:
                regime = cell["regime"]
                # ⚠ when long & medium SMAs disagree on direction —
                # informational only, doesn't change the regime label.
                conflict_mark = " ⚠" if cell.get("trend_conflict") else ""
                record[tf] = (f"{_emoji(regime)} {regime}{conflict_mark}"
                              f" · ADX {label}")
            elif tf in r["errors"]:
                record[tf] = "—"
            else:
                record[tf] = ""
        # IV rank + leadership are daily-context fields — pull from 1d.
        d = r["by_interval"].get("1d", {})
        record["IV Rank"] = (f"{d['iv_rank']:.0f}"
                             if d.get("iv_rank") is not None else "—")
        # Lead-z formatting:
        #   * No anchor configured (leadership_anchor == "")  → "—"
        #     (silent zeros pre-fix were misleading: the system literally
        #      had nothing to compare against, but printed +0.00)
        #   * RPC-failed / degenerate stdev / IEX feed empty
        #     (leadership_signal_available == False) → "—" + "(no data)"
        #     so the user can tell "computed and got 0" from "couldn't
        #     compute".  Common cause: sector ETF anchors (XLF, XLK, …)
        #     trade primarily on NYSE Arca and Alpaca's free IEX feed
        #     returns <2 5-min bars after open-skip, which makes
        #     get_5min_return_series return None.
        #   * Anchor configured AND signal available → 3 decimals so
        #     genuinely-tiny but non-zero deltas (SPY vs QQQ on a quiet
        #     day) become visible instead of rounding to +0.00.
        anchor = d.get("leadership_anchor", "")
        signal_ok = d.get("leadership_signal_available", False)
        if not anchor:
            record["Lead z"] = "—"
        elif d.get("leadership_z") is None:
            record["Lead z"] = "—"
        elif not signal_ok:
            record["Lead z"] = f"— (no data vs {anchor})"
        else:
            record["Lead z"] = f"{d['leadership_z']:+.3f} (vs {anchor})"
        record["TF agree"] = f"{r['agreement']:.0%}"
        display_rows.append(record)

    df = pd.DataFrame(display_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Stale-data badge — fires when the most recent 1d bar is hours
    # old AND the live agent is currently outside market hours.  We
    # don't try to detect mid-session stalls here; the live monitor
    # tab owns that signal.
    if stale_age_minutes is not None and stale_age_minutes > 30:
        hours = stale_age_minutes / 60
        st.warning(
            f"⏰ Data stale by ~{hours:.1f}h — "
            "markets likely closed.  Lead-z / VIX-z reflect the last "
            "completed session, not live conditions.",
        )

    # Color-key footnote so the user can decode emoji/label combos.
    legend_parts = [
        f"<span style='color:{REGIME_COLORS[k]}'>{_emoji(k)} {k}</span>"
        for k in ("bullish", "bearish", "sideways", "mean_reversion")
    ]
    st.caption(
        "Regime legend: " + " · ".join(legend_parts)
        + " · ADX label: weak (<20) · developing (20-40) · strong (40+)"
        + " · ⚠ = SMA50/SMA200 slope conflict"
        + " · Lead-z \"— (no data vs X)\" = anchor X exists but the "
        "intraday return series was unavailable (Alpaca IEX feed limit, "
        "weekend / pre-market, or degenerate stdev)",
        unsafe_allow_html=True,
    )


def _stale_minutes(rows: List[Dict]) -> Optional[float]:
    """Return age (minutes) of the most recent 1d bar across all rows.

    Returns None when no row has a usable timestamp — caller treats that
    as "unknown freshness", which suppresses the stale chip.
    """
    latest: Optional[datetime] = None
    for r in rows:
        ts_iso = r.get("by_interval", {}).get("1d", {}).get("last_bar_ts")
        if not ts_iso:
            continue
        try:
            ts = datetime.fromisoformat(ts_iso)
        except (TypeError, ValueError):
            continue
        if latest is None or ts > latest:
            latest = ts
    if latest is None:
        return None
    # Compare in UTC to avoid tz-naive/aware mismatches — both sides
    # are normalised to aware-UTC before subtraction.
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - latest).total_seconds() / 60


def _emoji(regime_value: str) -> str:
    return {
        Regime.BULLISH.value:        "🟢",
        Regime.BEARISH.value:        "🔴",
        Regime.SIDEWAYS.value:       "🟡",
        Regime.MEAN_REVERSION.value: "🟣",
    }.get(regime_value, "⚪")


# Chart panel lives in trading_agent/streamlit/watchlist_chart.py and is
# imported lazily inside render_watchlist() — keeps cold-start light when
# the user never opens this tab.
