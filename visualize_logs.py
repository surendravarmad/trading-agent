#!/usr/bin/env python3
"""
visualize_logs.py
=================
Parses ``trade_journal/signals_live.jsonl`` (or ``signals_backtest.jsonl``,
or the pre-May-2026 ``signals.jsonl`` — auto-detected as a fallback) and
``trade_plans/trade_plan_{TICKER}.json`` files to generate a daily "Agent
Performance Dashboard" as a single interactive HTML file.

Journal resolution
------------------
The default journal is ``trade_journal/signals_live.jsonl`` (post-May-2026
live-mode output). If the default doesn't exist, the script transparently
falls back to the legacy ``trade_journal/signals.jsonl`` so existing
archives still render. Pass ``--journal`` explicitly to render a backtest
run (``trade_journal/signals_backtest.jsonl``) or any other source.

Usage
-----
    python visualize_logs.py                                 # today's live data
    python visualize_logs.py --date 2026-04-02               # specific date
    python visualize_logs.py --tickers SPY QQQ               # specific tickers
    python visualize_logs.py --all-dates                     # no date filter
    python visualize_logs.py --output report.html            # custom output
    python visualize_logs.py --journal trade_journal/signals_backtest.jsonl
    python visualize_logs.py --help                          # full option list
"""

import argparse
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Status / colour constants
# ---------------------------------------------------------------------------

# Maps every action string → a display-friendly status bucket
ACTION_STATUS_MAP: Dict[str, str] = {
    "dry_run":                         "Trade Executed",
    "submitted":                       "Trade Executed",
    "skip":                            "Monitoring/No Action",
    "skipped_existing":                "Monitoring/No Action",
    "skipped_by_llm":                  "Monitoring/No Action",
    "rejected":                        "Risk Rejected",
    "skipped_defense_first":           "Defense First",
    "skipped_liquidation_mode":        "Liquidation Mode",
    "daily_drawdown_circuit_breaker":  "Circuit Breaker",
    "error":                           "Error/Timeout",
    "cycle_timeout":                   "Error/Timeout",
}

# Catppuccin-Mocha–inspired palette
STATUS_COLOR_MAP: Dict[str, str] = {
    "Trade Executed":      "#a6e3a1",  # green
    "Monitoring/No Action":"#89b4fa",  # blue
    "Risk Rejected":       "#fab387",  # peach / orange
    "Defense First":       "#cba6f7",  # mauve / purple
    "Liquidation Mode":    "#f38ba8",  # red
    "Circuit Breaker":     "#eba0ac",  # maroon-ish
    "Error/Timeout":       "#f38ba8",  # red
    "Unknown":             "#6c7086",  # grey
}

# Logic-distribution buckets — evaluated in order; first match wins
_LOGIC_BUCKETS: List[tuple] = [
    (
        "Active Trade",
        lambda r: r.get("status") == "Trade Executed",
    ),
    (
        "Waiting for Trend\n(SMA Filter)",
        lambda r: (
            "sma" in str(r.get("notes", "")).lower()
            or "macro" in str(r.get("notes", "")).lower()
        ),
    ),
    (
        "Overbought/Oversold\n(RSI Filter)",
        lambda r: (
            "rsi" in str(r.get("notes", "")).lower()
            or "overbought" in str(r.get("notes", "")).lower()
            or "oversold" in str(r.get("notes", "")).lower()
        ),
    ),
    (
        "High-IV Block",
        lambda r: (
            r.get("action") == "skipped_defense_first"
            and (
                "iv" in str(r.get("notes", "")).lower()
                or "highiv" in str(r.get("notes", "")).lower()
            )
        ),
    ),
    (
        "Risk Rejected",
        lambda r: r.get("status") == "Risk Rejected",
    ),
    (
        "Defense First\n(Macro Guard)",
        lambda r: r.get("status") == "Defense First",
    ),
    (
        "Circuit Breaker /\nLiquidation",
        lambda r: r.get("status") in ("Circuit Breaker", "Liquidation Mode"),
    ),
    (
        "Error/Timeout",
        lambda r: r.get("status") == "Error/Timeout",
    ),
    (
        "Monitoring/No Action",
        lambda r: r.get("status") == "Monitoring/No Action",
    ),
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Default journal — the May-2026 split moved live-mode output to
# ``signals_live.jsonl``. ``LEGACY_JOURNAL`` is the pre-split filename;
# ``_resolve_journal_path`` falls back to it transparently so old archives
# (and any third-party tooling that still drops files at the old path)
# keep rendering without code changes.
DEFAULT_JOURNAL = "trade_journal/signals_live.jsonl"
LEGACY_JOURNAL  = "trade_journal/signals.jsonl"


def _resolve_journal_path(journal_path: str) -> Path:
    """
    Resolve *journal_path* with one back-compat fallback.

    If the requested file exists, return it as-is. If it doesn't and the
    caller asked for the post-split default ``signals_live.jsonl``, try
    the legacy ``signals.jsonl`` next to it before giving up. Explicit
    user-supplied paths (e.g. ``signals_backtest.jsonl``) are never
    second-guessed — caller's "missing file → empty DataFrame" semantics
    still apply.
    """
    p = Path(journal_path)
    if p.exists():
        return p
    if p.name == "signals_live.jsonl":
        legacy = p.with_name("signals.jsonl")
        if legacy.exists():
            return legacy
    return p


def load_signals(
    journal_path: str,
    filter_date: Optional[str] = None,
    tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load a signal-journal JSONL file into a tidy DataFrame.

    Accepts ``signals_live.jsonl``, ``signals_backtest.jsonl``, or the
    pre-May-2026 ``signals.jsonl`` interchangeably — the schema is
    identical across all three.

    Parameters
    ----------
    journal_path :
        Path to the signal-journal file.
    filter_date :
        ISO date string ``"YYYY-MM-DD"`` — keep only rows from that day.
        ``None`` keeps every row.
    tickers :
        List of ticker symbols to keep.  ``None`` keeps all.

    Returns
    -------
    DataFrame with columns including ``timestamp``, ``ticker``, ``action``,
    ``price``, ``notes``, ``status``, ``color``, ``reason``, and every
    ``raw_signal.*`` field flattened with a ``raw_signal_`` prefix.
    Returns an **empty** DataFrame when the file is missing or empty.
    """
    path = _resolve_journal_path(journal_path)
    if not path.exists():
        return pd.DataFrame()

    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Drop cycle-level events that have no ticker/action
    records = [r for r in records if "ticker" in r and "action" in r]
    if not records:
        return pd.DataFrame()

    df = pd.json_normalize(records, sep="_")

    # Ensure core columns are present
    for col in ("timestamp", "ticker", "action", "price", "exec_status", "notes"):
        if col not in df.columns:
            df[col] = None

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if filter_date:
        target = pd.Timestamp(filter_date).date()
        df = df[df["timestamp"].dt.date == target]

    if tickers:
        df = df[df["ticker"].isin(tickers)]

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Derived display columns
    df["status"] = df["action"].map(ACTION_STATUS_MAP).fillna("Unknown")
    df["color"] = df["status"].map(STATUS_COLOR_MAP).fillna(STATUS_COLOR_MAP["Unknown"])

    # Best available hover reason
    rejection_col = "raw_signal_rejection_reason"
    df["reason"] = df["notes"].fillna("").astype(str)
    if rejection_col in df.columns:
        mask = df["reason"].str.strip() == ""
        df.loc[mask, "reason"] = df.loc[mask, rejection_col].fillna("").astype(str)

    return df


def load_trade_plans(
    plans_dir: str,
    tickers: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load ``trade_plans/trade_plan_{TICKER}.json`` files.

    Each row in the returned DataFrame represents one ``state_history`` entry.
    Columns: ``run_id``, ``timestamp``, ``mode``, ``approved``,
    ``account_balance``, ``short_strike``, ``strategy``,
    ``net_credit``, ``spread_width``, ``expiration``, ``valid``.

    Parameters
    ----------
    plans_dir :
        Directory containing the ``trade_plan_*.json`` files.
    tickers :
        List of tickers to load.  ``None`` loads all found files.

    Returns
    -------
    ``dict`` mapping ticker → DataFrame.
    Returns an **empty** dict when the directory is missing.
    """
    plans_path = Path(plans_dir)
    if not plans_path.exists():
        return {}

    result: Dict[str, pd.DataFrame] = {}

    for filepath in plans_path.glob("trade_plan_*.json"):
        ticker = filepath.stem.replace("trade_plan_", "")
        if tickers and ticker not in tickers:
            continue

        try:
            with filepath.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        history = data.get("state_history", [])
        if not history:
            continue

        rows = []
        for entry in history:
            trade_plan = entry.get("trade_plan") or {}
            risk_verdict = entry.get("risk_verdict") or {}
            legs = trade_plan.get("legs") or []

            # Short strike = first sold leg's strike (the "danger zone")
            sold_legs = [lg for lg in legs if lg.get("action") == "sell"]
            short_strike = sold_legs[0].get("strike") if sold_legs else None

            rows.append({
                "run_id":          entry.get("run_id"),
                "timestamp":       entry.get("timestamp"),
                "mode":            entry.get("mode"),
                "approved":        risk_verdict.get("approved", False),
                "account_balance": risk_verdict.get("account_balance"),
                "short_strike":    short_strike,
                "strategy":        (
                    trade_plan.get("strategy")
                    or trade_plan.get("strategy_name")
                ),
                "net_credit":      trade_plan.get("net_credit"),
                "spread_width":    trade_plan.get("spread_width"),
                "expiration":      trade_plan.get("expiration"),
                "valid":           trade_plan.get("valid", False),
            })

        if rows:
            tdf = pd.DataFrame(rows)
            tdf["timestamp"] = pd.to_datetime(
                tdf["timestamp"], utc=True, errors="coerce"
            )
            tdf = tdf.sort_values("timestamp").reset_index(drop=True)
            result[ticker] = tdf

    return result


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def build_heartbeat_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Horizontal scatter timeline — one dot per agent cycle.

    Colour codes
    ------------
    - Green  → Trade Executed
    - Blue   → Monitoring / No Action
    - Red    → Error / Timeout
    - Orange → Risk Rejected
    - Purple → Defense First / Circuit Breaker

    Hover shows: ticker, price, action, skip reason, timestamp.
    """
    fig = go.Figure()

    if df.empty:
        fig.add_annotation(
            text="No signals found for this period.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#6c7086"),
        )
        _apply_dark_theme(fig, "Heartbeat Timeline — No Data", height=300)
        return fig

    # One trace per status bucket keeps the legend clean
    for status, color in STATUS_COLOR_MAP.items():
        sub = df[df["status"] == status]
        if sub.empty:
            continue

        fig.add_trace(go.Scatter(
            x=sub["timestamp"],
            y=sub["ticker"],
            mode="markers",
            marker=dict(
                color=color,
                size=11,
                line=dict(width=1, color="#1e1e2e"),
                symbol="circle",
            ),
            name=status,
            customdata=sub[["price", "reason", "action"]].values,
            hovertemplate=(
                f"<b>{status}</b><br>"
                "Price: $%{customdata[0]:.2f}<br>"
                "Action: %{customdata[2]}<br>"
                "Reason: %{customdata[1]}<br>"
                "Time: %{x|%Y-%m-%d %H:%M:%S UTC}"
                "<extra></extra>"
            ),
        ))

    _apply_dark_theme(
        fig,
        "<b>Heartbeat Timeline</b> — Agent Cycle Status",
        height=max(300, 80 * df["ticker"].nunique() + 120),
    )
    fig.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="Ticker",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
    )
    return fig


def build_safety_buffer_chart(
    signals_df: pd.DataFrame,
    plans: Dict[str, pd.DataFrame],
    tickers: Optional[List[str]] = None,
) -> go.Figure:
    """
    Line chart — underlying price over the session with short-strike
    "danger zone" annotation.

    - Solid coloured line = underlying price (from the signal journal)
    - Red dashed horizontal line = short strike of the most-recent
      approved trade for that ticker
    """
    if tickers is None:
        t_signals = (
            list(signals_df["ticker"].dropna().unique())
            if not signals_df.empty else []
        )
        tickers = sorted(set(t_signals) | set(plans.keys()))

    fig = go.Figure()

    if signals_df.empty and not plans:
        fig.add_annotation(
            text="No price data available.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#6c7086"),
        )
        _apply_dark_theme(fig, "Safety Buffer Chart — No Data", height=350)
        return fig

    # Catppuccin palette rotated per ticker
    palette = [
        "#89dceb", "#a6e3a1", "#fab387",
        "#f38ba8", "#cba6f7", "#f9e2af",
    ]

    global_x_min = global_x_max = None
    if not signals_df.empty and "timestamp" in signals_df.columns:
        valid_ts = signals_df["timestamp"].dropna()
        if not valid_ts.empty:
            global_x_min = valid_ts.min()
            global_x_max = valid_ts.max()

    for idx, ticker in enumerate(tickers):
        color = palette[idx % len(palette)]

        # --- Price line ---
        if not signals_df.empty:
            sub = signals_df[
                (signals_df["ticker"] == ticker)
                & signals_df["price"].notna()
            ].copy()

            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["timestamp"],
                    y=sub["price"],
                    mode="lines+markers",
                    name=f"{ticker} price",
                    line=dict(color=color, width=2),
                    marker=dict(size=4, color=color),
                    hovertemplate=(
                        f"<b>{ticker}</b> $%{{y:.2f}}<br>"
                        "%{x|%H:%M:%S UTC}<extra></extra>"
                    ),
                ))

        # --- Short strike danger line ---
        if ticker in plans:
            plan_df = plans[ticker]
            active = plan_df[
                (plan_df["approved"] == True) & plan_df["short_strike"].notna()
            ]
            if not active.empty:
                short_strike = float(active.iloc[-1]["short_strike"])

                # x-axis extents for the horizontal rule
                x0 = global_x_min if global_x_min is not None else active["timestamp"].min()
                x1 = global_x_max if global_x_max is not None else active["timestamp"].max()

                fig.add_shape(
                    type="line",
                    x0=x0, x1=x1,
                    y0=short_strike, y1=short_strike,
                    line=dict(color="#f38ba8", width=2, dash="dash"),
                )
                fig.add_annotation(
                    x=x1,
                    y=short_strike,
                    text=f"  {ticker} short strike ${short_strike:.0f}",
                    showarrow=False,
                    font=dict(color="#f38ba8", size=11),
                    xanchor="left",
                )

    _apply_dark_theme(
        fig,
        "<b>Safety Buffer Chart</b> — Underlying Price vs Short Strike",
        height=420,
    )
    fig.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="Price ($)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
    )
    return fig


def build_logic_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Pie chart — what fraction of cycles the agent spent in each state bucket:

    - Active Trade
    - Waiting for Trend (SMA Filter)
    - Overbought / Oversold (RSI Filter)
    - High-IV Block
    - Risk Rejected
    - Defense First
    - Monitoring / No Action
    - Error / Timeout
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for distribution.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#6c7086"),
        )
        _apply_dark_theme(fig, "Logic Distribution — No Data", height=400)
        return fig

    buckets: List[str] = []
    for _, row in df.iterrows():
        assigned = "Other"
        row_dict = row.to_dict()
        for bucket_name, predicate in _LOGIC_BUCKETS:
            try:
                if predicate(row_dict):
                    assigned = bucket_name
                    break
            except Exception:
                pass
        buckets.append(assigned)

    counts = (
        pd.Series(buckets, name="bucket")
        .value_counts()
        .reset_index()
    )
    counts.columns = ["bucket", "count"]

    fig = px.pie(
        counts,
        names="bucket",
        values="count",
        color_discrete_sequence=[
            "#a6e3a1", "#89b4fa", "#fab387", "#cba6f7",
            "#f38ba8", "#eba0ac", "#6c7086", "#f9e2af",
        ],
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{value} cycles (%{percent})<extra></extra>",
        insidetextfont=dict(color="#1e1e2e"),
    )
    _apply_dark_theme(
        fig,
        "<b>Logic Distribution</b> — Agent State Breakdown",
        height=420,
    )
    fig.update_layout(
        legend=dict(font=dict(color="#cdd6f4")),
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------------
# HTML assembly helpers
# ---------------------------------------------------------------------------


def _apply_dark_theme(fig: go.Figure, title: str, height: int = 400) -> None:
    """Apply a consistent dark (Catppuccin Mocha) theme to any figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color="#cdd6f4", size=15)),
        height=height,
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4", family="'Segoe UI', sans-serif"),
        xaxis=dict(gridcolor="#313244", zerolinecolor="#313244"),
        yaxis=dict(gridcolor="#313244", zerolinecolor="#313244"),
        margin=dict(l=60, r=60, t=60, b=50),
    )


def _summary_stats_html(
    df: pd.DataFrame,
    plans: Dict[str, pd.DataFrame],
) -> str:
    """Return an HTML snippet with a top-of-page stats bar."""
    if df.empty:
        return (
            "<p style='color:#6c7086;padding:16px'>"
            "No signal data loaded for this period."
            "</p>"
        )

    total = len(df)
    executed = int((df["status"] == "Trade Executed").sum())
    errors = int((df["status"] == "Error/Timeout").sum())
    defense = int(df["action"].str.startswith("skipped_defense", na=False).sum())
    tickers_seen = int(df["ticker"].nunique())
    active_trades = sum(
        int((p["approved"] == True).sum()) for p in plans.values()
    )
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    date_range = (
        f"{ts_min.strftime('%Y-%m-%d %H:%M')} → {ts_max.strftime('%H:%M')} UTC"
        if pd.notna(ts_min) else "—"
    )

    return f"""
<div class="stats-bar">
  <div class="stat">
    <span class="val">{total}</span>
    <span class="lbl">Total Cycles</span>
  </div>
  <div class="stat">
    <span class="val green">{executed}</span>
    <span class="lbl">Trades Executed</span>
  </div>
  <div class="stat">
    <span class="val blue">{tickers_seen}</span>
    <span class="lbl">Tickers Monitored</span>
  </div>
  <div class="stat">
    <span class="val orange">{active_trades}</span>
    <span class="lbl">Active Positions</span>
  </div>
  <div class="stat">
    <span class="val purple">{defense}</span>
    <span class="lbl">Defense-First Skips</span>
  </div>
  <div class="stat">
    <span class="val red">{errors}</span>
    <span class="lbl">Errors</span>
  </div>
  <div class="stat full-width">
    <span class="lbl">{date_range}</span>
  </div>
</div>
"""


_HTML_STYLE = """
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #11111b; color: #cdd6f4;
  font-family: 'Segoe UI', Roboto, sans-serif;
  padding: 28px 32px;
}
h1 { font-size: 1.65rem; color: #cba6f7; margin-bottom: 4px; }
.subtitle { color: #6c7086; font-size: 0.88rem; margin-bottom: 24px; }
.stats-bar {
  display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 32px;
}
.stat {
  background: #1e1e2e; border: 1px solid #313244; border-radius: 8px;
  padding: 12px 18px; display: flex; flex-direction: column;
  align-items: center; min-width: 118px;
}
.stat.full-width { flex: 1; align-items: flex-start; justify-content: center; }
.val   { font-size: 1.85rem; font-weight: 700; }
.lbl   { font-size: 0.72rem; color: #6c7086; margin-top: 3px; }
.green  { color: #a6e3a1; }
.blue   { color: #89b4fa; }
.orange { color: #fab387; }
.red    { color: #f38ba8; }
.purple { color: #cba6f7; }
.chart-section { margin-bottom: 36px; }
.chart-label {
  font-size: 0.8rem; color: #45475a; text-transform: uppercase;
  letter-spacing: .08em; margin-bottom: 6px;
}
footer {
  margin-top: 48px; font-size: 0.73rem;
  color: #45475a; text-align: center;
}
</style>
"""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def generate_dashboard(
    journal_path: str = DEFAULT_JOURNAL,
    plans_dir: str = "trade_plans",
    tickers: Optional[List[str]] = None,
    filter_date: Optional[str] = None,
    output_path: str = "daily_report.html",
) -> str:
    """
    Full pipeline: load → build charts → write HTML.

    Parameters
    ----------
    journal_path :
        Path to the signal-journal JSONL. Defaults to the live-mode
        journal (``trade_journal/signals_live.jsonl``); pass the
        backtest path explicitly to render a backtest run. The legacy
        ``signals.jsonl`` is auto-detected as a fallback when the
        default doesn't exist (see ``_resolve_journal_path``).
    plans_dir :
        Directory containing ``trade_plan_*.json`` files.
    tickers :
        Tickers to include.  ``None`` uses all available.
    filter_date :
        ISO date string ``"YYYY-MM-DD"`` to filter signals.
        ``None`` defaults to today.
    output_path :
        Destination HTML file path.

    Returns
    -------
    Absolute path to the generated HTML file (str).
    """
    if filter_date is None:
        filter_date = date.today().isoformat()

    signals_df = load_signals(
        journal_path, filter_date=filter_date, tickers=tickers
    )
    plans = load_trade_plans(plans_dir, tickers=tickers)

    # Resolve display ticker list
    display_tickers: List[str] = list(tickers) if tickers else []
    if not display_tickers:
        t_sig = (
            list(signals_df["ticker"].dropna().unique())
            if not signals_df.empty else []
        )
        display_tickers = sorted(set(t_sig) | set(plans.keys()))

    # Build the three visual components
    fig_heartbeat = build_heartbeat_timeline(signals_df)
    fig_safety = build_safety_buffer_chart(
        signals_df, plans, tickers=display_tickers or None
    )
    fig_pie = build_logic_distribution(signals_df)

    # Serialise to HTML div snippets (Plotly JS injected once via CDN)
    heartbeat_div = fig_heartbeat.to_html(full_html=False, include_plotlyjs=False)
    safety_div = fig_safety.to_html(full_html=False, include_plotlyjs=False)
    pie_div = fig_pie.to_html(full_html=False, include_plotlyjs=False)

    stats_html = _summary_stats_html(signals_df, plans)
    report_date = filter_date
    ticker_label = ", ".join(display_tickers) if display_tickers else "All Tickers"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Agent Performance Dashboard — {report_date}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  {_HTML_STYLE}
</head>
<body>
  <h1>Agent Performance Dashboard</h1>
  <p class="subtitle">
    {report_date} &nbsp;|&nbsp; {ticker_label}
    &nbsp;|&nbsp; Generated {generated_at}
  </p>

  {stats_html}

  <div class="chart-section">
    <div class="chart-label">1 — Heartbeat Timeline</div>
    {heartbeat_div}
  </div>

  <div class="chart-section">
    <div class="chart-label">2 — Safety Buffer Chart</div>
    {safety_div}
  </div>

  <div class="chart-section">
    <div class="chart-label">3 — Logic Distribution</div>
    {pie_div}
  </div>

  <footer>
    Generated by <code>visualize_logs.py</code>
    &nbsp;|&nbsp; Trading Agent
  </footer>
</body>
</html>
"""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard written to: {out.resolve()}")
    return str(out.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a daily Agent Performance Dashboard (HTML).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--journal",
        default=DEFAULT_JOURNAL,
        metavar="PATH",
        help=(
            "Path to the signal-journal JSONL "
            f"(default: {DEFAULT_JOURNAL}; "
            "falls back to legacy signals.jsonl if the default is missing)"
        ),
    )
    parser.add_argument(
        "--plans-dir",
        default="trade_plans",
        metavar="DIR",
        help="Directory with trade_plan_*.json files  (default: trade_plans)",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        metavar="TICKER",
        help="Tickers to include (default: all found)",
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Filter to a single date (default: today)",
    )
    parser.add_argument(
        "--all-dates",
        action="store_true",
        help="Include all dates in the journal (overrides --date)",
    )
    parser.add_argument(
        "--output",
        default="daily_report.html",
        metavar="PATH",
        help="Output HTML file  (default: daily_report.html)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    _filter_date = None if args.all_dates else (args.date or date.today().isoformat())
    generate_dashboard(
        journal_path=args.journal,
        plans_dir=args.plans_dir,
        tickers=args.tickers,
        filter_date=_filter_date,
        output_path=args.output,
    )
