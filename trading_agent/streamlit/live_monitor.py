"""
live_monitor.py — Live Monitoring tab.

Displays real-time account metrics, open spread positions, equity curve,
8-guardrail status cards, market status, and the recent signal journal.
Auto-refreshes every 30 seconds via st.rerun().
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from trading_agent.streamlit.components import (
    equity_curve_chart,
    guardrail_cards,
    metric_row,
    positions_table,
    GUARDRAIL_NAMES,
)

JOURNAL_PATH = Path("trade_journal/signals.jsonl")
PAUSE_FLAG = Path("PAUSED")
REFRESH_INTERVAL = 30  # seconds

# Keyword fragments for mapping journal check strings → guardrail slots
_GUARDRAIL_KEYWORDS: List[List[str]] = [
    ["plan invalid", "plan valid"],       # Plan Validity
    ["credit/width", "credit ratio"],     # Credit/Width Ratio
    ["delta"],                            # Delta ≤ Max Delta
    ["max loss"],                         # Max Loss ≤ 2% Equity
    ["paper"],                            # Paper Account
    ["market", "closed", "open"],         # Market Open
    ["bid", "ask", "spread"],             # Bid/Ask Spread
    ["buying power"],                     # Buying Power ≤ 80%
]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_config():
    """Load AppConfig from .env once per process lifetime."""
    try:
        from trading_agent.config import load_config
        return load_config()
    except Exception:
        return None


def _load_journal_df() -> pd.DataFrame:
    """Parse trade_journal/signals.jsonl → DataFrame. Returns empty on missing file."""
    empty = pd.DataFrame(
        columns=["timestamp", "account_balance", "ticker", "action",
                 "regime", "checks_passed", "checks_failed", "notes",
                 "rsi_14", "sma_50", "sma_200"]
    )
    if not JOURNAL_PATH.exists():
        return empty

    rows = []
    with open(JOURNAL_PATH) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rs = rec.get("raw_signal", {})
                rows.append(
                    {
                        "timestamp": pd.to_datetime(rec.get("timestamp")),
                        "account_balance": rs.get("account_balance", 0) or 0,
                        "ticker": rec.get("ticker", ""),
                        "action": rec.get("action", ""),
                        "regime": rs.get("regime", "unknown"),
                        "checks_passed": rs.get("checks_passed") or [],
                        "checks_failed": rs.get("checks_failed") or [],
                        "notes": rec.get("notes", ""),
                        "rsi_14": rs.get("rsi_14", 0) or 0,
                        "sma_50": rs.get("sma_50", 0) or 0,
                        "sma_200": rs.get("sma_200", 0) or 0,
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue

    if not rows:
        return empty
    df = pd.DataFrame(rows)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _guardrail_status_from_journal(df: pd.DataFrame) -> List[Dict]:
    """
    Derive guardrail pass/fail from the last journal row's checks lists.
    Returns a list of 8 dicts: {name, passed, detail}.
    """
    defaults = [{"name": n, "passed": True, "detail": "No data"} for n in GUARDRAIL_NAMES]
    if df.empty:
        return defaults

    last = df.iloc[-1]
    passed_list: List[str] = last.get("checks_passed") or []
    failed_list: List[str] = last.get("checks_failed") or []

    results = []
    for idx, keywords in enumerate(_GUARDRAIL_KEYWORDS):
        passed = True
        detail = "OK"
        # Check if any keyword appears in the failed list
        for fcheck in failed_list:
            fcheck_lower = fcheck.lower()
            if any(kw in fcheck_lower for kw in keywords):
                passed = False
                detail = fcheck[:70]
                break
        # If still passed, find the matching passed check for detail
        if passed:
            for pcheck in passed_list:
                pcheck_lower = pcheck.lower()
                if any(kw in pcheck_lower for kw in keywords):
                    detail = pcheck[:70]
                    break
        results.append({"name": GUARDRAIL_NAMES[idx], "passed": passed, "detail": detail})
    return results


def _fetch_account(config) -> Dict:
    """Fetch Alpaca account dict; returns {} on any failure."""
    try:
        from trading_agent.market_data import MarketDataProvider
        provider = MarketDataProvider(
            alpaca_api_key=config.alpaca.api_key,
            alpaca_secret_key=config.alpaca.secret_key,
            alpaca_data_url=config.alpaca.data_url,
        )
        return provider.get_account_info(config.alpaca.base_url) or {}
    except Exception as exc:
        st.warning(f"Account fetch failed: {exc}")
        return {}


def _fetch_spreads(config) -> List[Dict]:
    """Fetch open spread positions; returns [] on any failure."""
    try:
        import glob as _glob

        from trading_agent.position_monitor import PositionMonitor

        monitor = PositionMonitor(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
            base_url=config.alpaca.base_url,
        )
        positions = monitor.fetch_open_positions()

        # Load persisted trade plans for spread grouping
        trade_plans: List[Dict] = []
        plan_dir = Path(config.logging.trade_plan_dir)
        if plan_dir.exists():
            for fp in plan_dir.glob("trade_plan_*.json"):
                try:
                    data = json.loads(fp.read_text())
                    for entry in data.get("state_history", []):
                        tp = entry.get("trade_plan")
                        if tp:
                            trade_plans.append(tp)
                except Exception:
                    pass

        spreads = monitor.group_into_spreads(positions, trade_plans)
        return [
            {
                "underlying": s.underlying,
                "strategy_name": s.strategy_name,
                "original_credit": s.original_credit,
                "net_unrealized_pl": s.net_unrealized_pl,
                "expiration": s.expiration,
                "exit_signal": s.exit_signal.value,
            }
            for s in spreads
        ]
    except Exception as exc:
        st.warning(f"Position fetch failed: {exc}")
        return []


def _is_market_open(config) -> Optional[bool]:
    """Return True/False/None (None = unavailable)."""
    try:
        from trading_agent.market_data import MarketDataProvider
        provider = MarketDataProvider(
            alpaca_api_key=config.alpaca.api_key,
            alpaca_secret_key=config.alpaca.secret_key,
            alpaca_data_url=config.alpaca.data_url,
        )
        return provider.is_market_open(config.alpaca.base_url)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Auto-refresh (separate function so tests can mock it)
# ---------------------------------------------------------------------------

def _auto_refresh(interval_secs: int = REFRESH_INTERVAL) -> None:
    """Non-blocking countdown using session state; triggers st.rerun() every second."""
    now = time.time()
    last = st.session_state.get("_live_last_refresh", 0)
    elapsed = now - last
    remaining = max(0, int(interval_secs - elapsed))

    if elapsed >= interval_secs:
        st.session_state["_live_last_refresh"] = now
    else:
        st.caption(f"Auto-refreshing in {remaining}s…")
        time.sleep(1)
        st.rerun()


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

@st.fragment(run_every=REFRESH_INTERVAL)
def render_live_monitor() -> None:
    """Render the Live Monitoring tab."""

    # ── Emergency Pause toggle ─────────────────────────────────────────────
    top_left, top_right = st.columns([7, 1])
    with top_left:
        st.subheader("Live Portfolio Monitor")
    with top_right:
        if PAUSE_FLAG.exists():
            st.error("PAUSED")
            if st.button("▶ Resume"):
                PAUSE_FLAG.unlink(missing_ok=True)
                st.rerun()
        else:
            if st.button("⏸ Emergency Pause", type="primary"):
                PAUSE_FLAG.write_text(datetime.now(timezone.utc).isoformat())
                st.success("Agent paused. Remove the PAUSED file or click Resume to restart.")
                st.rerun()

    if PAUSE_FLAG.exists():
        st.warning(
            f"Trading is PAUSED since {PAUSE_FLAG.read_text()[:19]} UTC. "
            "The agent will skip all order submissions until resumed."
        )

    # ── Load data ──────────────────────────────────────────────────────────
    config = _get_config()
    journal_df = _load_journal_df()

    equity = 0.0
    total_pnl = 0.0
    spreads: List[Dict] = []

    if config:
        account = _fetch_account(config)
        equity = float(account.get("equity") or 0)
        total_pnl = float(account.get("unrealized_pl") or 0)
        spreads = _fetch_spreads(config)

    # Fallback equity from journal when Alpaca is unreachable
    if equity == 0.0 and not journal_df.empty:
        nonzero = journal_df[journal_df["account_balance"] > 0]["account_balance"]
        if not nonzero.empty:
            equity = nonzero.iloc[-1]

    # Dominant regime from last 20 signals
    regime = "unknown"
    if not journal_df.empty:
        recent = journal_df.tail(20)
        valid = recent[(recent["regime"].notna()) & (recent["regime"] != "")]
        if not valid.empty:
            regime = valid["regime"].value_counts().idxmax()

    # Seconds until next 5-minute cycle boundary
    now = datetime.now()
    cycle_secs = max(0, 300 - (now.minute * 60 + now.second) % 300)

    # ── Metrics row ────────────────────────────────────────────────────────
    metric_row(equity, total_pnl, regime, cycle_secs)
    st.divider()

    # ── Open positions ─────────────────────────────────────────────────────
    st.subheader("Open Positions")
    positions_table(spreads)
    st.divider()

    # ── Equity curve ───────────────────────────────────────────────────────
    equity_df = journal_df[journal_df["account_balance"] > 0].copy()
    if not equity_df.empty:
        st.subheader("Equity Curve")
        st.plotly_chart(equity_curve_chart(equity_df), use_container_width=True)
        st.divider()

    # ── Guardrail status ───────────────────────────────────────────────────
    st.subheader("Risk Guardrail Status")
    guardrail_status = _guardrail_status_from_journal(journal_df)
    guardrail_cards(guardrail_status)
    st.divider()

    # ── Market status + SMA drift ──────────────────────────────────────────
    st.subheader("Market Status")
    if config:
        open_flag = _is_market_open(config)
        if open_flag is None:
            st.info("Market status unavailable (Alpaca unreachable).")
        else:
            label = "OPEN" if open_flag else "CLOSED"
            color = "green" if open_flag else "red"
            st.markdown(f"Market is currently :{color}[**{label}**]")
    else:
        st.info("Config unavailable — set ALPACA_API_KEY in .env.")

    if not journal_df.empty:
        last = journal_df.iloc[-1]
        sma50 = last.get("sma_50", 0)
        sma200 = last.get("sma_200", 0)
        rsi = last.get("rsi_14", 0)
        if sma50 and sma200:
            drift = (sma50 - sma200) / sma200 * 100
            st.caption(
                f"Last signal — SMA50/SMA200 drift: {drift:+.2f}%  |  RSI-14: {rsi:.1f}"
            )
    st.divider()

    # ── Journal expander ───────────────────────────────────────────────────
    with st.expander("Recent Journal Entries", expanded=False):
        if journal_df.empty:
            st.info("No journal entries found at trade_journal/signals.jsonl.")
        else:
            cols = ["timestamp", "ticker", "action", "regime", "notes"]
            cols = [c for c in cols if c in journal_df.columns]
            st.dataframe(
                journal_df[cols].tail(100).iloc[::-1],
                use_container_width=True,
                hide_index=True,
            )

    # ── Manual refresh ─────────────────────────────────────────────────────
    st.divider()
    col_r1, col_r2 = st.columns([5, 1])
    with col_r1:
        st.caption(f"Auto-refreshes every {REFRESH_INTERVAL}s")
    with col_r2:
        if st.button("Refresh Now"):
            st.rerun()
