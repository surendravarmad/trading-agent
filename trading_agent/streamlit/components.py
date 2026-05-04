"""
components.py — Reusable Plotly charts and Streamlit UI primitives.

All chart-building logic lives here so live_monitor, backtest_ui, and
llm_extension never import plotly directly.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REGIME_COLORS: Dict[str, str] = {
    "bullish": "#00c853",
    "bearish": "#d50000",
    "sideways": "#ff6d00",
    "mean_reversion": "#6200ea",
    "unknown": "#9e9e9e",
}

GUARDRAIL_NAMES: List[str] = [
    "Plan Validity",
    "Credit/Width Ratio",
    "Delta ≤ Max Delta",
    "Max Loss ≤ 2% Equity",
    "Paper Account",
    "Market Open",
    "Bid/Ask Spread",
    "Buying Power ≤ 80%",
]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def equity_curve_chart(df: pd.DataFrame) -> go.Figure:
    """
    Line chart of portfolio equity over time.

    Expected columns: timestamp (datetime-like), account_balance (float).
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["account_balance"],
            mode="lines",
            name="Equity",
            line=dict(color="#1976d2", width=2),
            fill="tozeroy",
            fillcolor="rgba(25,118,210,0.08)",
            hovertemplate="$%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Time",
        yaxis_title="Account Balance ($)",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
        plot_bgcolor="white",
        yaxis=dict(tickformat="$,.0f"),
    )
    return fig


def drawdown_chart(df: pd.DataFrame) -> go.Figure:
    """
    Area chart showing rolling drawdown as a percentage.

    Expected columns: timestamp, account_balance.
    """
    equity = df["account_balance"]
    running_max = equity.cummax()
    # Use np.nan (not pd.NA) to keep dtype float64 — pd.NA coerces to
    # object, which breaks downstream numeric ops like .ewm().mean().
    drawdown_pct = (equity - running_max) / running_max.replace(0, np.nan) * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=drawdown_pct,
            mode="lines",
            name="Drawdown %",
            line=dict(color="#d32f2f", width=2),
            fill="tozeroy",
            fillcolor="rgba(211,47,47,0.12)",
            hovertemplate="%{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Drawdown (%)",
        xaxis_title="Time",
        yaxis_title="Drawdown %",
        height=250,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
        plot_bgcolor="white",
    )
    return fig


def regime_bar_chart(regime_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of total P&L grouped by regime.

    Expected columns: regime (str), pnl (float), trade_count (int).
    """
    colors = [REGIME_COLORS.get(str(r).lower(), "#9e9e9e") for r in regime_df["regime"]]
    fig = go.Figure(
        go.Bar(
            x=regime_df["pnl"],
            y=regime_df["regime"],
            orientation="h",
            marker_color=colors,
            text=[f"{n} trades" for n in regime_df["trade_count"]],
            textposition="outside",
            hovertemplate="%{y}: $%{x:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="P&L by Regime",
        xaxis_title="Total P&L ($)",
        yaxis_title="",
        height=280,
        margin=dict(l=0, r=60, t=40, b=0),
        plot_bgcolor="white",
        xaxis=dict(tickformat="$,.0f"),
    )
    return fig


# ---------------------------------------------------------------------------
# UI Primitives
# ---------------------------------------------------------------------------

def metric_row(equity: float, pnl: float, regime: str, cycle_secs: int) -> None:
    """Four-column metrics bar: equity · P&L · regime badge · cycle countdown."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Equity", f"${equity:,.2f}")

    delta_color = "normal" if pnl >= 0 else "inverse"
    c2.metric("Unrealized P&L", f"${pnl:+,.2f}", delta_color=delta_color)

    with c3:
        color = REGIME_COLORS.get(regime.lower(), "#9e9e9e")
        st.markdown(
            f"""<div style="text-align:center;padding-top:6px">
            <p style="margin:0;font-size:0.85em;color:#888">Dominant Regime</p>
            <span style="background:{color};color:#fff;padding:3px 14px;
            border-radius:12px;font-weight:600;font-size:1em">{regime.upper()}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    c4.metric("Next Cycle In", f"{cycle_secs}s")


def guardrail_cards(guardrail_status: List[Dict]) -> None:
    """
    Two rows of 4 guardrail status cards.

    Each dict must have keys: name (str), passed (bool), detail (str).
    """
    for row_start in (0, 4):
        cols = st.columns(4)
        for offset, col in enumerate(cols):
            idx = row_start + offset
            if idx >= len(guardrail_status):
                break
            g = guardrail_status[idx]
            icon = "✅" if g["passed"] else "❌"
            bg = "#e8f5e9" if g["passed"] else "#ffebee"
            border = "#4caf50" if g["passed"] else "#ef5350"
            col.markdown(
                f"""<div style="background:{bg};border-left:4px solid {border};
                padding:8px 10px;border-radius:4px;margin-bottom:6px;min-height:60px">
                <b>{icon} {g['name']}</b><br>
                <small style="color:#555">{g.get('detail', '')[:70]}</small>
                </div>""",
                unsafe_allow_html=True,
            )


def positions_table(spreads: List[Dict]) -> None:
    """Styled positions table. Each dict is a serialised SpreadPosition."""
    if not spreads:
        st.info("No open positions.")
        return
    rows = []
    for s in spreads:
        credit = round(s.get("original_credit", 0) * 100, 2)
        pnl = round(s.get("net_unrealized_pl", 0), 2)
        pct = f"{pnl / credit * 100:.1f}%" if credit else "—"
        rows.append(
            {
                "Symbol": s.get("underlying", ""),
                "Strategy": s.get("strategy_name", ""),
                "Credit ($)": credit,
                "Unreal. P&L ($)": pnl,
                "% Profit": pct,
                "Expiry": s.get("expiration", ""),
                "Exit Signal": s.get("exit_signal", "hold"),
            }
        )
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


def ungrouped_legs_table(legs: List[Dict]) -> None:
    """
    Render Alpaca option legs that are NOT matched to any local
    trade_plan_*.json file.

    These legs are typically positions opened outside the agent
    (manual entry via the Alpaca web UI, a different machine, or
    runs whose plan files were rotated/deleted). Showing them here
    keeps the dashboard a faithful mirror of the broker's view.
    """
    if not legs:
        return  # silent — nothing to show
    rows = []
    for L in legs:
        rows.append(
            {
                "Symbol":          L.get("symbol", ""),
                "Underlying":      L.get("underlying", ""),
                "Type":            L.get("type", ""),
                "Strike":          L.get("strike", ""),
                "Expiry":          L.get("expiration", ""),
                "Side":            L.get("side", ""),
                "Qty":             L.get("qty", 0),
                "Avg Entry ($)":   round(float(L.get("avg_entry_price", 0)), 2),
                "Current ($)":     round(float(L.get("current_price", 0)), 2),
                "Unreal. P&L ($)": round(float(L.get("unrealized_pl", 0)), 2),
            }
        )
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


def alert_box(message: str, level: str = "warning") -> None:
    """Render an alert at the given severity level (info/warning/error/success)."""
    getattr(st, level, st.warning)(message)
