"""
live_monitor.py — Live Monitoring tab.

This tab is the single entry point for the trading agent.
It can Start / Stop the agent loop directly, shows real-time
account metrics, open positions, equity curve, guardrail status,
and the recent signal journal.

Agent loop design
-----------------
The agent runs in a subprocess (not a thread) so that the
270-second os._exit(1) timeout guard in run_cycle() cannot kill
the Streamlit dashboard process. Each cycle is launched as:

    python -m trading_agent.agent --env .env

State is persisted in three files:
    AGENT_RUNNING  — sentinel file; presence = agent should keep looping
    AGENT_PID      — PID of the current cycle subprocess
    AGENT_LOG      — last 200 lines of agent stdout/stderr
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from trading_agent.streamlit.components import (
    GUARDRAIL_NAMES,
    equity_curve_chart,
    guardrail_cards,
    metric_row,
    positions_table,
    ungrouped_legs_table,
)
from trading_agent.strategy_presets import (
    AGGRESSIVE,
    BALANCED,
    CONSERVATIVE,
    PRESET_FILE,
    PRESETS,
    PresetConfig,
    load_active_preset,
    save_active_preset,
)


_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def _parse_occ(symbol: str) -> Dict[str, str]:
    """
    Parse an OCC option symbol → underlying, expiration, type, strike.

    OCC format: ROOT(1-6) + YYMMDD(6) + C/P(1) + STRIKE*1000(8)
    Example:  SPY260516P00450000 → SPY, 2026-05-16, Put, 450.00

    Returns the raw symbol as fallback `underlying` if parsing fails.
    """
    m = _OCC_RE.match(symbol or "")
    if not m:
        return {
            "underlying":  symbol or "",
            "expiration": "",
            "type":       "",
            "strike":     "",
        }
    root, ymd, cp, strike = m.groups()
    return {
        "underlying":  root,
        "expiration": f"20{ymd[0:2]}-{ymd[2:4]}-{ymd[4:6]}",
        "type":        "Call" if cp == "C" else "Put",
        "strike":      f"{int(strike) / 1000:.2f}",
    }

# ---------------------------------------------------------------------------
# File-based state paths (relative to repo root)
# ---------------------------------------------------------------------------
JOURNAL_PATH   = Path("trade_journal/signals.jsonl")
PAUSE_FLAG     = Path("PAUSED")
AGENT_RUNNING  = Path("AGENT_RUNNING")   # sentinel: agent loop is active
AGENT_PID      = Path("AGENT_PID")       # PID of the running cycle process
AGENT_LOG      = Path("AGENT_LOG")       # rolling log tail
DRY_RUN_FLAG   = Path("DRY_RUN_MODE")   # sentinel: inject DRY_RUN + FORCE_MARKET_OPEN

REFRESH_INTERVAL   = 30   # seconds between dashboard auto-refreshes
CYCLE_INTERVAL_SEC = 300  # 5-minute trading cycle

# Keyword fragments for mapping journal check strings → guardrail slots
_GUARDRAIL_KEYWORDS: List[List[str]] = [
    ["plan invalid", "plan valid"],
    ["credit/width", "credit ratio"],
    ["delta"],
    ["max loss"],
    ["paper"],
    ["market", "closed", "open"],
    ["bid", "ask", "spread"],
    ["buying power"],
]


# ---------------------------------------------------------------------------
# Agent loop — runs in a background daemon thread inside the dashboard process
# ---------------------------------------------------------------------------

def _append_log(line: str, max_lines: int = 200) -> None:
    """Append one line to AGENT_LOG, keeping only the last max_lines lines."""
    try:
        existing = AGENT_LOG.read_text().splitlines() if AGENT_LOG.exists() else []
        existing.append(line.rstrip())
        AGENT_LOG.write_text("\n".join(existing[-max_lines:]) + "\n")
    except Exception:
        pass


def _run_one_cycle(dry_run: bool = False) -> int:
    """
    Spawn a single agent cycle as a child process.

    Parameters
    ----------
    dry_run : bool
        When True, injects DRY_RUN=true and FORCE_MARKET_OPEN=true into the
        subprocess environment so the full agent pipeline runs — regime
        classification, option chain fetching, risk checks — but no orders
        are submitted to Alpaca. Useful for after-hours simulation.

    Returns the process exit code (0 = success, non-zero = error/timeout).
    """
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "-m", "trading_agent.agent"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    if dry_run:
        env["DRY_RUN"] = "true"
        env["FORCE_MARKET_OPEN"] = "true"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(repo_root),
        env=env,
    )
    AGENT_PID.write_text(str(proc.pid))

    for line in proc.stdout:
        _append_log(line)

    proc.wait()
    if AGENT_PID.exists():
        AGENT_PID.unlink(missing_ok=True)
    return proc.returncode


def _agent_loop() -> None:
    """
    Background thread: run trading cycles every CYCLE_INTERVAL_SEC seconds
    while AGENT_RUNNING sentinel file exists.

    Sleeps 1 second between ticks so the loop reacts quickly to a Stop request.
    """
    _append_log(f"[{_now()}] Agent loop started (PID {os.getpid()})")

    while AGENT_RUNNING.exists():
        if PAUSE_FLAG.exists():
            _append_log(f"[{_now()}] PAUSED — skipping cycle")
            time.sleep(5)
            continue

        is_dry = DRY_RUN_FLAG.exists()
        mode_label = "DRY-RUN" if is_dry else "LIVE"
        _append_log(f"[{_now()}] --- Cycle start [{mode_label}] ---")
        rc = _run_one_cycle(dry_run=is_dry)
        _append_log(f"[{_now()}] --- Cycle end [{mode_label}] (exit={rc}) ---")

        # Wait CYCLE_INTERVAL_SEC, checking every second for a stop/pause signal
        for _ in range(CYCLE_INTERVAL_SEC):
            if not AGENT_RUNNING.exists():
                break
            time.sleep(1)

    _append_log(f"[{_now()}] Agent loop stopped")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Start / Stop helpers
# ---------------------------------------------------------------------------

def _is_loop_running() -> bool:
    """True if the background loop thread is alive (checks sentinel file)."""
    return AGENT_RUNNING.exists()


def _is_dry_run_mode() -> bool:
    return DRY_RUN_FLAG.exists()


def _start_agent(dry_run: bool = False) -> None:
    """Write sentinels and launch the loop in a daemon thread."""
    AGENT_RUNNING.write_text(_now())
    if dry_run:
        DRY_RUN_FLAG.write_text(_now())
    else:
        DRY_RUN_FLAG.unlink(missing_ok=True)
    t = threading.Thread(target=_agent_loop, daemon=True, name="agent-loop")
    t.start()
    st.session_state["_agent_thread"] = t


def _stop_agent() -> None:
    """Remove sentinel — the loop will exit after the current cycle completes."""
    AGENT_RUNNING.unlink(missing_ok=True)
    _append_log(f"[{_now()}] Stop requested from dashboard")


def _kill_current_cycle() -> None:
    """SIGKILL the running cycle subprocess immediately (emergency only)."""
    if AGENT_PID.exists():
        try:
            pid = int(AGENT_PID.read_text().strip())
            os.kill(pid, 9)
            _append_log(f"[{_now()}] SIGKILL sent to PID {pid}")
        except Exception as exc:
            _append_log(f"[{_now()}] Kill failed: {exc}")
        AGENT_PID.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_config():
    try:
        from trading_agent.config import load_config
        return load_config()
    except Exception:
        return None


def _load_journal_df() -> pd.DataFrame:
    empty = pd.DataFrame(
        columns=["timestamp", "account_balance", "ticker", "action",
                 "regime", "checks_passed", "checks_failed", "notes",
                 "rsi_14", "sma_50", "sma_200", "scan_results"]
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
                rows.append({
                    "timestamp":       pd.to_datetime(rec.get("timestamp")),
                    "account_balance": rs.get("account_balance", 0) or 0,
                    "ticker":          rec.get("ticker", ""),
                    "action":          rec.get("action", ""),
                    "regime":          rs.get("regime", "unknown"),
                    "checks_passed":   rs.get("checks_passed") or [],
                    "checks_failed":   rs.get("checks_failed") or [],
                    "notes":           rec.get("notes", ""),
                    "rsi_14":          rs.get("rsi_14", 0) or 0,
                    "sma_50":          rs.get("sma_50", 0) or 0,
                    "sma_200":         rs.get("sma_200", 0) or 0,
                    # Adaptive-scanner block. Empty dict on legacy records
                    # so downstream rendering can do truthiness checks
                    # without KeyError; populated dict on new records.
                    "scan_results":    rs.get("scan_results") or {},
                })
            except (json.JSONDecodeError, KeyError):
                continue

    if not rows:
        return empty
    df = pd.DataFrame(rows)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _scanner_diagnostics_from_journal(
    df: pd.DataFrame, lookback_rows: int = 200,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    """
    Project scanner output from ``signals.jsonl`` into three render-ready
    pieces: per-ticker latest verdict, aggregate reject-reason histogram,
    and per-side candidate counts.

    The latest-per-ticker frame answers *"what did the scanner find on the
    most recent cycle for SPY/QQQ/...?"* — one row per ticker with the
    near-miss credit/EV gap. The histogram answers *"across the last N
    cycles, which filter is rejecting most candidates?"* — telling you
    whether to lower edge_buffer (cw_below_floor dominates) or widen the
    Δ grid (pop_below_min dominates).
    """
    cols = ["ticker", "side", "candidates", "selected", "edge_buffer",
            "min_pop", "grid_total", "grid_priced", "top_reject",
            "top_reject_count", "near_miss_credit", "near_miss_cw",
            "near_miss_floor", "near_miss_ev", "timestamp"]
    empty = pd.DataFrame(columns=cols)
    if df.empty or "scan_results" not in df.columns:
        return empty, {}, {}

    # Restrict to records that actually carry scan_results — older
    # legacy records will silently fall out.
    has_scan = df[df["scan_results"].apply(lambda x: isinstance(x, dict) and bool(x))]
    if has_scan.empty:
        return empty, {}, {}
    has_scan = has_scan.tail(lookback_rows)

    latest_rows: Dict[str, Dict] = {}        # ticker → latest row dict
    reject_hist: Dict[str, int] = {}         # reason → count across window
    side_counts: Dict[str, int] = {}         # side  → candidates_total sum

    for _, rec in has_scan.iterrows():
        sr = rec["scan_results"] or {}
        diag = sr.get("diagnostics") or {}
        side = sr.get("side") or "—"
        side_counts[side] = side_counts.get(side, 0) + int(sr.get("candidates_total") or 0)

        for reason, count in (diag.get("rejects_by_reason") or {}).items():
            reject_hist[reason] = reject_hist.get(reason, 0) + int(count)

        nm = diag.get("best_near_miss") or {}
        rejects = diag.get("rejects_by_reason") or {}
        if rejects:
            top_r, top_c = max(rejects.items(), key=lambda kv: kv[1])
        else:
            top_r, top_c = "—", 0
        latest_rows[rec["ticker"]] = {
            "ticker":           rec["ticker"],
            "side":             side,
            "candidates":       int(sr.get("candidates_total") or 0),
            "selected":         int(sr.get("selected_index") or -1),
            "edge_buffer":      float(sr.get("edge_buffer") or 0.0),
            "min_pop":          float(sr.get("min_pop") or 0.0),
            "grid_total":       int(diag.get("grid_points_total") or 0),
            "grid_priced":      int(diag.get("grid_points_priced") or 0),
            "top_reject":       top_r,
            "top_reject_count": top_c,
            "near_miss_credit": float(nm.get("credit") or 0.0) if nm else 0.0,
            "near_miss_cw":     float(nm.get("cw_ratio") or 0.0) if nm else 0.0,
            "near_miss_floor":  float(nm.get("cw_floor") or 0.0) if nm else 0.0,
            "near_miss_ev":     float(nm.get("ev") or 0.0) if nm else 0.0,
            "timestamp":        rec["timestamp"],
        }

    latest_df = pd.DataFrame(list(latest_rows.values()), columns=cols)
    if not latest_df.empty:
        latest_df.sort_values("timestamp", ascending=False, inplace=True)
        latest_df.reset_index(drop=True, inplace=True)
    return latest_df, reject_hist, side_counts


def _render_scanner_diagnostics_panel(journal_df: pd.DataFrame) -> None:
    """
    Render the "Adaptive Scanner Diagnostics" expander. No-op-friendly:
    when no records carry ``scan_results`` the panel renders an info
    message instead of disappearing — so users running an older agent
    binary aren't left wondering why the panel is empty.
    """
    with st.expander("🔬 Adaptive Scanner Diagnostics", expanded=False):
        if journal_df.empty:
            st.info("No journal entries yet.")
            return

        latest_df, reject_hist, side_counts = _scanner_diagnostics_from_journal(
            journal_df, lookback_rows=200,
        )
        if latest_df.empty:
            st.info(
                "No `scan_results` in the recent journal. The agent may be "
                "running in static-mode (set `SCAN_MODE=adaptive` in the "
                "Strategy Profile to enable the scanner) or you haven't "
                "completed a cycle since upgrading to the diagnostics build."
            )
            return

        # ── Top: per-ticker latest verdict ─────────────────────────────
        st.caption(
            "Latest scanner verdict per ticker (most recent cycle). "
            "**Near-miss** = the highest-EV candidate that *only* failed "
            "the C/W floor — closing the gap between **near_miss_cw** and "
            "**near_miss_floor** is the lever (lower `EDGE_BUFFER`, or wait)."
        )
        # Format floats compactly without copying the full df.
        display = latest_df.copy()
        for col, fmt in (
            ("edge_buffer",      "{:.2%}"),
            ("min_pop",          "{:.2%}"),
            ("near_miss_cw",     "{:.4f}"),
            ("near_miss_floor",  "{:.4f}"),
            ("near_miss_ev",     "{:+.4f}"),
            ("near_miss_credit", "${:.2f}"),
        ):
            display[col] = display[col].apply(
                lambda v, f=fmt: (f.format(v) if v else "—")
            )
        # Drop the timestamp column from the visible table (keeps the row
        # narrow); we display recency implicitly via sort order.
        visible_cols = [c for c in latest_df.columns if c != "timestamp"]
        st.dataframe(
            display[visible_cols],
            width='stretch',
            hide_index=True,
        )

        # ── Bottom: aggregate reject histogram + side counts ──────────
        col_h, col_s = st.columns([3, 1])
        with col_h:
            st.caption("Reject reasons across the last 200 cycles "
                       "(higher = more candidates rejected by that filter).")
            if reject_hist:
                hist_df = pd.DataFrame(
                    sorted(reject_hist.items(), key=lambda kv: -kv[1]),
                    columns=["reason", "count"],
                )
                st.bar_chart(hist_df.set_index("reason"))
            else:
                st.caption("(No rejects recorded — every cycle passed.)")
        with col_s:
            st.caption("Total candidates by side (last 200 cycles).")
            if side_counts:
                side_df = pd.DataFrame(
                    sorted(side_counts.items(), key=lambda kv: -kv[1]),
                    columns=["side", "candidates"],
                )
                st.dataframe(side_df, width='stretch', hide_index=True)
            else:
                st.caption("—")


def _guardrail_status_from_journal(df: pd.DataFrame) -> List[Dict]:
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
        for fcheck in failed_list:
            if any(kw in fcheck.lower() for kw in keywords):
                passed = False
                detail = fcheck[:70]
                break
        if passed:
            for pcheck in passed_list:
                if any(kw in pcheck.lower() for kw in keywords):
                    detail = pcheck[:70]
                    break
        results.append({"name": GUARDRAIL_NAMES[idx], "passed": passed, "detail": detail})
    return results


def _fetch_account(config) -> Dict:
    try:
        from trading_agent.market_data import MarketDataProvider
        provider = MarketDataProvider(
            alpaca_api_key=config.alpaca.api_key,
            alpaca_secret_key=config.alpaca.secret_key,
            alpaca_data_url=config.alpaca.data_url,
            alpaca_base_url=config.alpaca.base_url,
        )
        return provider.get_account_info() or {}
    except Exception as exc:
        st.warning(f"Account fetch failed: {exc}")
        return {}


def _fetch_spreads(config) -> Tuple[List[Dict], List[Dict]]:
    """
    Return (spreads, ungrouped_legs).

    `spreads` are option positions that matched a local `trade_plan_*.json`
    and were aggregated by `PositionMonitor.group_into_spreads`.

    `ungrouped_legs` are option legs in the broker account that did NOT
    match any local trade plan — typically positions opened outside the
    agent or runs whose plan files were rotated/deleted. They are still
    real money in the account, so we surface them rather than silently
    dropping them.
    """
    try:
        from trading_agent.position_monitor import PositionMonitor
        monitor = PositionMonitor(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
            base_url=config.alpaca.base_url,
        )
        positions = monitor.fetch_open_positions()

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

        spread_objs = monitor.group_into_spreads(positions, trade_plans)

        spreads = [
            {
                "underlying":        s.underlying,
                "strategy_name":     s.strategy_name,
                "original_credit":   s.original_credit,
                "net_unrealized_pl": s.net_unrealized_pl,
                "expiration":        s.expiration,
                "exit_signal":       s.exit_signal.value,
            }
            for s in spread_objs
        ]

        # Any leg whose symbol didn't end up in a matched spread is "ungrouped".
        matched_symbols = {leg.symbol for s in spread_objs for leg in s.legs}
        ungrouped_legs = []
        for p in positions:
            if p.symbol in matched_symbols:
                continue
            occ = _parse_occ(p.symbol)
            ungrouped_legs.append(
                {
                    "symbol":          p.symbol,
                    "underlying":      occ["underlying"],
                    "expiration":      occ["expiration"],
                    "type":            occ["type"],
                    "strike":          occ["strike"],
                    "qty":             p.qty,
                    "side":            p.side,
                    "avg_entry_price": p.avg_entry_price,
                    "current_price":   p.current_price,
                    "unrealized_pl":   p.unrealized_pl,
                }
            )

        return spreads, ungrouped_legs
    except Exception as exc:
        st.warning(f"Position fetch failed: {exc}")
        return [], []


def _is_market_open(config) -> Optional[bool]:
    try:
        from trading_agent.market_data import MarketDataProvider
        provider = MarketDataProvider(
            alpaca_api_key=config.alpaca.api_key,
            alpaca_secret_key=config.alpaca.secret_key,
            alpaca_data_url=config.alpaca.data_url,
            alpaca_base_url=config.alpaca.base_url,
        )
        return provider.is_market_open()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Auto-refresh (non-blocking)
# ---------------------------------------------------------------------------

def _auto_refresh(interval_secs: int = REFRESH_INTERVAL) -> None:
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
# Strategy-Profile panel
# ---------------------------------------------------------------------------
#
# The panel writes ``STRATEGY_PRESET.json`` (next to AGENT_RUNNING) when
# the user clicks Apply.  The agent subprocess re-reads that file at the
# start of every cycle (see ``agent.TradingAgent.__init__`` →
# ``load_active_preset``), so changes take effect on the next 5-min tick
# without restarting the loop.
#
# Layout: two top-level selectboxes (risk profile + directional bias)
# plus an expander that's only meaningful when profile == "custom".

_PROFILE_OPTIONS:    List[str] = ["conservative", "balanced", "aggressive", "custom"]
_PROFILE_LABELS: Dict[str, str] = {
    "conservative": "Conservative — ~85% POP, low risk, fewer trades",
    "balanced":     "Balanced — ~75% POP, recommended baseline",
    "aggressive":   "Aggressive — ~65% POP, fat credits, gamma-sensitive",
    "custom":       "Custom — tune every knob yourself",
}

_BIAS_OPTIONS:    List[str] = ["auto", "bullish_only", "bearish_only", "neutral_only"]
_BIAS_LABELS: Dict[str, str] = {
    "auto":         "Auto — trade whatever regime classifier reports",
    "bullish_only": "Bullish only — Bull Puts + Iron Condors + MR",
    "bearish_only": "Bearish only — Bear Calls + Iron Condors + MR",
    "neutral_only": "Neutral only — Iron Condors + MR (no directional)",
}


def _parse_grid(text: str, kind: str) -> Optional[tuple]:
    """
    Parse a comma-separated grid string into a sorted unique tuple.

    ``kind`` is one of ``"int"`` (DTE values) or ``"float"`` (Δ / width %).
    Returns ``None`` on malformed input — caller should fall back to the
    seed value rather than persisting garbage.
    """
    out = []
    for tok in (text or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok) if kind == "int" else float(tok)
        except ValueError:
            return None
        out.append(v)
    if not out:
        return None
    return tuple(sorted(set(out)))


def _custom_inputs(seed: PresetConfig) -> Dict:
    """Render the Custom-mode override widgets and return a dict of values."""
    st.caption(
        "Custom overrides start from the **Balanced** baseline. Only the "
        "fields you change are persisted; everything else stays on the default."
    )

    c1, c2 = st.columns(2)
    with c1:
        max_delta = st.slider(
            "Max short-leg |Δ|", 0.05, 0.45, float(seed.max_delta), 0.01,
            help="0.15 ≈ 85% POP · 0.25 ≈ 75% POP · 0.35 ≈ 65% POP",
            key="cust_max_delta",
        )
        min_credit_ratio = st.slider(
            "Credit/Width floor", 0.10, 0.50, float(seed.min_credit_ratio), 0.05,
            help="Reject spreads whose credit / width is below this floor.",
            key="cust_min_cw",
        )
        max_risk_pct = st.slider(
            "Max account risk per trade (%)",
            0.5, 5.0, float(seed.max_risk_pct) * 100, 0.5,
            help="Hard cap on max-loss as a fraction of account equity.",
            key="cust_max_risk",
        ) / 100.0
        dte_window_days = st.slider(
            "DTE window ± (days)", 1, 14, int(seed.dte_window_days), 1,
            key="cust_dte_window",
        )
    with c2:
        dte_vertical = st.slider(
            "Vertical (Bull Put / Bear Call) DTE",
            5, 60, int(seed.dte_vertical), 1,
            key="cust_dte_v",
        )
        dte_iron_condor = st.slider(
            "Iron Condor DTE", 7, 60, int(seed.dte_iron_condor), 1,
            key="cust_dte_ic",
        )
        dte_mean_reversion = st.slider(
            "Mean-Reversion DTE", 3, 30, int(seed.dte_mean_reversion), 1,
            key="cust_dte_mr",
        )

    st.markdown("**Spread-width policy**")
    wc1, wc2 = st.columns(2)
    with wc1:
        width_mode = st.radio(
            "Width mode",
            options=["pct_of_spot", "fixed_dollar"],
            index=0 if seed.width_mode == "pct_of_spot" else 1,
            format_func=lambda v: "% of spot" if v == "pct_of_spot" else "Fixed $",
            horizontal=True,
            key="cust_width_mode",
        )
    with wc2:
        if width_mode == "pct_of_spot":
            width_value = st.slider(
                "Width (% of spot)", 0.5, 5.0,
                float(seed.width_value) * 100 if seed.width_mode == "pct_of_spot"
                else 1.5,
                0.1, key="cust_width_pct",
            ) / 100.0
        else:
            width_value = st.slider(
                "Width ($)", 1.0, 25.0,
                float(seed.width_value) if seed.width_mode == "fixed_dollar"
                else 5.0,
                0.5, key="cust_width_usd",
            )

    # ── Adaptive scan grids — only meaningful when scan_mode == "adaptive",
    #    but always shown so a user can pre-stage a Custom payload.
    with st.expander(
        "Adaptive scan grids (only used when Scan Mode = Adaptive)",
        expanded=False,
    ):
        st.caption(
            "The chain scanner sweeps the cross-product of these three grids "
            "and picks the highest-EV candidate. Empty an entry to fall back "
            "to the preset default. Comma-separated."
        )
        min_pop = st.slider(
            "Min POP (annualised score floor)", 0.30, 0.85,
            float(seed.min_pop), 0.05,
            help="Drop candidates whose POP (≈ 1 − |Δshort|) is below this.",
            key="cust_min_pop",
        )
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            dte_grid_text = st.text_input(
                "DTE grid (days)",
                value=", ".join(str(d) for d in seed.dte_grid),
                key="cust_dte_grid",
                help="e.g. 7, 14, 21, 30",
            )
        with gc2:
            delta_grid_text = st.text_input(
                "Δ grid (|short delta|)",
                value=", ".join(f"{d:g}" for d in seed.delta_grid),
                key="cust_delta_grid",
                help="e.g. 0.20, 0.25, 0.30, 0.35",
            )
        with gc3:
            width_grid_text = st.text_input(
                "Width grid (% of spot)",
                value=", ".join(f"{w:g}" for w in seed.width_grid_pct),
                key="cust_width_grid",
                help="e.g. 0.010, 0.015, 0.020, 0.025",
            )

    payload = {
        "max_delta":          max_delta,
        "dte_vertical":       dte_vertical,
        "dte_iron_condor":    dte_iron_condor,
        "dte_mean_reversion": dte_mean_reversion,
        "dte_window_days":    dte_window_days,
        "width_mode":         width_mode,
        "width_value":        width_value,
        "min_credit_ratio":   min_credit_ratio,
        "max_risk_pct":       max_risk_pct,
        "min_pop":            min_pop,
    }
    # Only persist grids when they parse cleanly — silently fall back to
    # seed value otherwise so a malformed text box doesn't poison the file.
    parsed_dte    = _parse_grid(dte_grid_text,    "int")
    parsed_delta  = _parse_grid(delta_grid_text,  "float")
    parsed_width  = _parse_grid(width_grid_text,  "float")
    if parsed_dte:    payload["dte_grid"]      = parsed_dte
    if parsed_delta:  payload["delta_grid"]    = parsed_delta
    if parsed_width:  payload["width_grid_pct"] = parsed_width
    return payload


def render_strategy_profile_panel() -> None:
    """
    Two-row Strategy-Profile selector + Apply button.

    Reads the current preset from ``STRATEGY_PRESET.json`` (or the
    BALANCED default when the file is missing) and writes the next
    selection back atomically. Hot-applied on the next cycle.
    """
    current = load_active_preset()
    is_loop_running = AGENT_RUNNING.exists()

    with st.expander(
        f"Strategy Profile — {current.to_summary_line()}",
        expanded=not is_loop_running,
    ):
        st.markdown(
            "Pick a risk profile + directional bias. Changes are written to "
            "`STRATEGY_PRESET.json` and applied **on the next cycle** — no "
            "restart needed. The active preset drives Δ-short, DTE per "
            "strategy, spread width, C/W floor, and the % of equity at risk."
        )

        col_p, col_b = st.columns(2)

        # Decide the index to show as currently-selected.  If the file says
        # "custom", that's preserved; otherwise lookup the current preset's
        # name in the canonical option list.
        try:
            saved_payload = (
                json.loads(PRESET_FILE.read_text())
                if PRESET_FILE.exists() else {}
            )
        except (json.JSONDecodeError, OSError):
            saved_payload = {}

        profile_default = (
            saved_payload.get("profile", current.name)
            if saved_payload.get("profile") in _PROFILE_OPTIONS
            else current.name
        )
        bias_default = current.directional_bias

        with col_p:
            profile = st.selectbox(
                "Risk profile",
                options=_PROFILE_OPTIONS,
                index=_PROFILE_OPTIONS.index(profile_default),
                format_func=lambda v: _PROFILE_LABELS[v],
                key="strat_profile",
            )
        with col_b:
            bias = st.selectbox(
                "Directional bias",
                options=_BIAS_OPTIONS,
                index=_BIAS_OPTIONS.index(bias_default),
                format_func=lambda v: _BIAS_LABELS[v],
                key="strat_bias",
            )

        # ── Scan-mode overlay row ────────────────────────────────────────
        # Static  → planner uses fixed (Δ, DTE, width) preset values.
        # Adaptive → ChainScanner sweeps (DTE × Δ × width) grid and picks
        # the highest-EV candidate that clears |Δshort|×(1+edge_buffer).
        # Both RiskManager and the executor's live-credit recheck switch
        # to the same Δ-aware floor when adaptive is selected, so the
        # planner / risk / exec floors never drift.
        col_s, col_e = st.columns(2)
        scan_default = saved_payload.get("scan_mode") or current.scan_mode
        if scan_default not in ("static", "adaptive"):
            scan_default = "static"
        edge_default = saved_payload.get("edge_buffer", current.edge_buffer)
        with col_s:
            scan_mode = st.radio(
                "Scan mode",
                options=["static", "adaptive"],
                index=0 if scan_default == "static" else 1,
                format_func=lambda v: (
                    "Static — fixed Δ/DTE/width" if v == "static"
                    else "Adaptive — chain scanner picks best EV"
                ),
                horizontal=True,
                key="strat_scan_mode",
                help="Adaptive replaces the static preset triple with a "
                     "(DTE × Δ × width) grid sweep, scoring each candidate "
                     "by per-dollar-risked EV. Floor becomes "
                     "|Δshort|×(1+edge_buffer) so it stays above breakeven "
                     "for whatever Δ the scanner picks.",
            )
        with col_e:
            edge_buffer = st.slider(
                "Edge buffer (over breakeven C/W)",
                0.0, 0.50, float(edge_default), 0.01,
                disabled=(scan_mode != "adaptive"),
                help="Required C/W = |Δshort| × (1 + edge_buffer). "
                     "10% is a reasonable default — drops to 5% in tight "
                     "tape, raise to 20%+ if you want to demand more cushion "
                     "before taking the trade.",
                key="strat_edge_buffer",
            )

        # Preview line for the chosen built-in.
        if profile in PRESETS:
            preview = PRESETS[profile]
            mode_tag = ("ADAPTIVE scan" if scan_mode == "adaptive"
                        else "STATIC scan")
            st.caption(
                f"Selected preset → {preview.description}  · {mode_tag} "
                f"(edge buffer {edge_buffer:.2f})"
            )

        # Custom overrides — only meaningful when profile == "custom".
        custom_payload: Optional[Dict] = None
        if profile == "custom":
            seed = current if current.name == "custom" else BALANCED
            custom_payload = _custom_inputs(seed)

        # Apply / status row
        ac1, ac2 = st.columns([1, 3])
        with ac1:
            apply_clicked = st.button(
                "💾 Apply", type="primary", width='stretch',
                key="strat_apply",
            )
        with ac2:
            if is_loop_running:
                st.caption(
                    "Agent is running — changes apply on the **next cycle** "
                    "(no restart, no current-cycle interruption)."
                )
            else:
                st.caption(
                    "Agent is stopped — the new preset will load on the "
                    "first cycle when you click ▶ Start Agent."
                )

        if apply_clicked:
            try:
                save_active_preset(
                    profile=profile,
                    directional_bias=bias,
                    custom=custom_payload,
                    scan_mode=scan_mode,
                    edge_buffer=edge_buffer,
                )
                st.toast(
                    f"Saved profile=**{profile}**, bias=**{bias}**, "
                    f"scan=**{scan_mode}** (edge {edge_buffer:.2f}) — "
                    "active on next cycle.",
                    icon="✅",
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save preset: {exc}")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

@st.fragment(run_every=REFRESH_INTERVAL)
def render_live_monitor() -> None:
    """Render the Live Monitoring tab — single entry point for the trading agent."""

    loop_running  = _is_loop_running()
    cycle_running = AGENT_PID.exists()
    dry_mode      = _is_dry_run_mode()

    # ── Header row ────────────────────────────────────────────────────────
    st.subheader("Live Portfolio Monitor")

    # ── Strategy profile panel (first — pick risk + bias BEFORE starting) ─
    render_strategy_profile_panel()

    # ── Dry Run toggle (most prominent choice — pick BEFORE starting) ─────
    with st.expander(
        "Dry Run Mode — simulate after hours without real orders",
        expanded=not loop_running,
    ):
        dry_col, info_col = st.columns([1, 2])
        with dry_col:
            new_dry = st.toggle(
                "Enable Dry Run",
                value=dry_mode,
                disabled=loop_running,
                help="Can only be changed while the agent is stopped.",
                key="dry_run_toggle",
            )
            if not loop_running and new_dry != dry_mode:
                if new_dry:
                    DRY_RUN_FLAG.write_text(_now())
                else:
                    DRY_RUN_FLAG.unlink(missing_ok=True)
                st.rerun()

        with info_col:
            if new_dry or dry_mode:
                st.info(
                    "**Dry Run is ON.** The agent will:\n"
                    "- Run the full pipeline: real market data, option chains, "
                    "regime classification, all 8 risk guardrails\n"
                    "- Bypass the market-hours check (`FORCE_MARKET_OPEN=true`) "
                    "so you can simulate after hours\n"
                    "- Log every decision to `signals.jsonl` as `action: dry_run` — "
                    "no orders submitted to Alpaca\n\n"
                    "**Perfect for after-hours review:** see exactly what the agent "
                    "would trade tonight using today's closing prices and live Greeks."
                )
            else:
                st.info(
                    "**Dry Run is OFF — LIVE mode.** Orders are submitted to Alpaca "
                    "when all risk guardrails pass. The market-hours check is enforced "
                    "(`9:25 AM – 4:05 PM ET` only).\n\n"
                    "Enable Dry Run to simulate the full pipeline after hours "
                    "without risking capital."
                )

    # ── Agent control buttons ──────────────────────────────────────────────
    btn_cols = st.columns([2, 2, 2, 1])

    with btn_cols[0]:
        if loop_running:
            if st.button("⏹ Stop Agent", type="secondary", width='stretch'):
                _stop_agent()
                st.toast("Stop requested — current cycle will finish then halt.", icon="⏹")
                st.rerun()
        else:
            label = "▶ Start (Dry Run)" if dry_mode else "▶ Start Agent"
            btn_type = "secondary" if dry_mode else "primary"
            if st.button(label, type=btn_type, width='stretch'):
                _start_agent(dry_run=dry_mode)
                mode = "DRY-RUN" if dry_mode else "LIVE"
                st.toast(f"Agent loop started [{mode}]!", icon="▶")
                st.rerun()

    with btn_cols[1]:
        if PAUSE_FLAG.exists():
            if st.button("▶ Resume", width='stretch'):
                PAUSE_FLAG.unlink(missing_ok=True)
                st.toast("Agent resumed.", icon="▶")
                st.rerun()
        else:
            if st.button("⏸ Pause", width='stretch', disabled=not loop_running):
                PAUSE_FLAG.write_text(_now())
                st.toast("Agent paused — no new orders until resumed.", icon="⏸")
                st.rerun()

    with btn_cols[2]:
        run_once_label = "⚡ Run Once (Dry)" if dry_mode else "⚡ Run Once"
        if st.button(run_once_label, width='stretch', disabled=cycle_running):
            if not loop_running:
                _is_dry = dry_mode

                def _one_shot(is_dry=_is_dry):
                    mode = "DRY-RUN" if is_dry else "LIVE"
                    _append_log(f"[{_now()}] --- One-shot cycle start [{mode}] ---")
                    rc = _run_one_cycle(dry_run=is_dry)
                    _append_log(f"[{_now()}] --- One-shot cycle end [{mode}] (exit={rc}) ---")

                t = threading.Thread(target=_one_shot, daemon=True)
                t.start()
                mode = "dry-run" if dry_mode else "live"
                st.toast(f"Running one {mode} cycle now…", icon="⚡")
                st.rerun()

    with btn_cols[3]:
        if st.button("🔴", width='stretch',
                     help="SIGKILL the running cycle immediately (emergency)",
                     disabled=not cycle_running):
            _kill_current_cycle()
            st.toast("Cycle killed.", icon="🔴")
            st.rerun()

    # ── Status banner ──────────────────────────────────────────────────────
    mode_badge = " · DRY RUN" if dry_mode else " · LIVE"
    if loop_running and cycle_running:
        pid = AGENT_PID.read_text().strip() if AGENT_PID.exists() else "?"
        if dry_mode:
            st.warning(f"Agent is **RUNNING [DRY RUN]** — cycle in progress (PID {pid}) — no orders will be placed")
        else:
            st.success(f"Agent is **RUNNING [LIVE]** — cycle in progress (PID {pid})")
    elif loop_running:
        if dry_mode:
            st.warning(f"Agent is **ACTIVE [DRY RUN]{mode_badge}** — waiting for next cycle — no orders will be placed")
        else:
            st.success(f"Agent is **ACTIVE [LIVE]** — waiting for next cycle")
    elif PAUSE_FLAG.exists():
        st.warning(
            f"Agent is **PAUSED** since {PAUSE_FLAG.read_text()[:19]} UTC. "
            "Click ▶ Resume to continue."
        )
    else:
        if dry_mode:
            st.info("Agent is **STOPPED** · Dry Run mode is armed — click ▶ Start (Dry Run) to simulate.")
        else:
            st.error("Agent is **STOPPED** — click ▶ Start Agent to begin live trading.")

    st.divider()

    # ── Load live data ─────────────────────────────────────────────────────
    config = _get_config()
    journal_df = _load_journal_df()

    equity = 0.0
    total_pnl = 0.0
    spreads: List[Dict] = []
    ungrouped_legs: List[Dict] = []

    if config:
        account = _fetch_account(config)
        equity = float(account.get("equity") or 0)
        total_pnl = float(account.get("unrealized_pl") or 0)
        spreads, ungrouped_legs = _fetch_spreads(config)

    if equity == 0.0 and not journal_df.empty:
        nonzero = journal_df[journal_df["account_balance"] > 0]["account_balance"]
        if not nonzero.empty:
            equity = nonzero.iloc[-1]

    regime = "unknown"
    if not journal_df.empty:
        recent = journal_df.tail(20)
        valid = recent[(recent["regime"].notna()) & (recent["regime"] != "")]
        if not valid.empty:
            regime = valid["regime"].value_counts().idxmax()

    now = datetime.now()
    cycle_secs = max(0, CYCLE_INTERVAL_SEC - (now.minute * 60 + now.second) % CYCLE_INTERVAL_SEC)

    # ── Metrics row ────────────────────────────────────────────────────────
    metric_row(equity, total_pnl, regime, cycle_secs)
    st.divider()

    # ── Open positions ─────────────────────────────────────────────────────
    st.subheader("Open Positions")
    positions_table(spreads)

    if ungrouped_legs:
        st.caption(
            f"Ungrouped Alpaca legs ({len(ungrouped_legs)}) — open option "
            "positions in your broker account that don't match any local "
            "`trade_plan_*.json`. These were likely opened outside the agent "
            "or have missing/rotated plan files. Exit signals do not apply."
        )
        ungrouped_legs_table(ungrouped_legs)
    st.divider()

    # ── Equity curve ───────────────────────────────────────────────────────
    equity_df = journal_df[journal_df["account_balance"] > 0].copy()
    if not equity_df.empty:
        st.subheader("Equity Curve")
        st.plotly_chart(equity_curve_chart(equity_df), width='stretch')
        st.divider()

    # ── Guardrail status ───────────────────────────────────────────────────
    st.subheader("Risk Guardrail Status")
    guardrail_cards(_guardrail_status_from_journal(journal_df))
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
        sma50, sma200, rsi = last.get("sma_50", 0), last.get("sma_200", 0), last.get("rsi_14", 0)
        if sma50 and sma200:
            drift = (sma50 - sma200) / sma200 * 100
            st.caption(f"Last signal — SMA50/SMA200 drift: {drift:+.2f}%  |  RSI-14: {rsi:.1f}")
    st.divider()

    # ── Agent log expander ─────────────────────────────────────────────────
    with st.expander("Agent Log (last 50 lines)", expanded=loop_running):
        if AGENT_LOG.exists():
            lines = AGENT_LOG.read_text().splitlines()
            log_text = "\n".join(lines[-50:])
            st.code(log_text, language="text")
        else:
            st.caption("No log yet — start the agent to see output here.")

    # ── Adaptive scanner diagnostics ───────────────────────────────────────
    # Surfaces the per-ticker scan_results block written by the agent so
    # users can see why candidates are being rejected even when zero
    # trades fire (e.g. cw_below_floor with edge_buffer too tight).
    _render_scanner_diagnostics_panel(journal_df)

    # ── Journal expander ───────────────────────────────────────────────────
    with st.expander("Recent Journal Entries", expanded=False):
        if journal_df.empty:
            st.info("No journal entries found at trade_journal/signals.jsonl.")
        else:
            cols = ["timestamp", "ticker", "action", "regime", "notes"]
            cols = [c for c in cols if c in journal_df.columns]
            st.dataframe(
                journal_df[cols].tail(100).iloc[::-1],
                width='stretch',
                hide_index=True,
            )

    # ── Manual refresh ─────────────────────────────────────────────────────
    # Note: auto-refresh is handled by @st.fragment(run_every=REFRESH_INTERVAL)
    # above. Do NOT call _auto_refresh() here — it would add an extra
    # time.sleep+rerun loop that triggers full-page reruns every second,
    # breaking other tabs (especially the backtesting sidebar).
    st.divider()
    col_r1, col_r2 = st.columns([5, 1])
    with col_r1:
        st.caption(f"Auto-refreshes every {REFRESH_INTERVAL}s via fragment")
    with col_r2:
        if st.button("Refresh Now"):
            st.rerun()
