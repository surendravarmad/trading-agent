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
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from trading_agent.streamlit.components import (
    GUARDRAIL_NAMES,
    equity_curve_chart,
    guardrail_cards,
    metric_row,
    positions_table,
)

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
                })
            except (json.JSONDecodeError, KeyError):
                continue

    if not rows:
        return empty
    df = pd.DataFrame(rows)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


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
        from trading_agent.market.market_data import MarketDataProvider
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


def _fetch_spreads(config) -> List[Dict]:
    try:
        from trading_agent.execution.position_monitor import PositionMonitor
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

        spreads = monitor.group_into_spreads(positions, trade_plans)
        return [{
            "underlying":      s.underlying,
            "strategy_name":   s.strategy_name,
            "original_credit": s.original_credit,
            "net_unrealized_pl": s.net_unrealized_pl,
            "expiration":      s.expiration,
            "exit_signal":     s.exit_signal.value,
        } for s in spreads]
    except Exception as exc:
        st.warning(f"Position fetch failed: {exc}")
        return []


def _is_market_open(config) -> Optional[bool]:
    try:
        from trading_agent.market.market_data import MarketDataProvider
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
            if st.button("⏹ Stop Agent", type="secondary", use_container_width=True):
                _stop_agent()
                st.toast("Stop requested — current cycle will finish then halt.", icon="⏹")
                st.rerun()
        else:
            label = "▶ Start (Dry Run)" if dry_mode else "▶ Start Agent"
            btn_type = "secondary" if dry_mode else "primary"
            if st.button(label, type=btn_type, use_container_width=True):
                _start_agent(dry_run=dry_mode)
                mode = "DRY-RUN" if dry_mode else "LIVE"
                st.toast(f"Agent loop started [{mode}]!", icon="▶")
                st.rerun()

    with btn_cols[1]:
        if PAUSE_FLAG.exists():
            if st.button("▶ Resume", use_container_width=True):
                PAUSE_FLAG.unlink(missing_ok=True)
                st.toast("Agent resumed.", icon="▶")
                st.rerun()
        else:
            if st.button("⏸ Pause", use_container_width=True, disabled=not loop_running):
                PAUSE_FLAG.write_text(_now())
                st.toast("Agent paused — no new orders until resumed.", icon="⏸")
                st.rerun()

    with btn_cols[2]:
        run_once_label = "⚡ Run Once (Dry)" if dry_mode else "⚡ Run Once"
        if st.button(run_once_label, use_container_width=True, disabled=cycle_running):
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
        if st.button("🔴", use_container_width=True,
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

    if config:
        account = _fetch_account(config)
        equity = float(account.get("equity") or 0)
        total_pnl = float(account.get("unrealized_pl") or 0)
        spreads = _fetch_spreads(config)

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
    st.divider()

    # ── Equity curve ───────────────────────────────────────────────────────
    equity_df = journal_df[journal_df["account_balance"] > 0].copy()
    if not equity_df.empty:
        st.subheader("Equity Curve")
        st.plotly_chart(equity_curve_chart(equity_df), use_container_width=True)
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
