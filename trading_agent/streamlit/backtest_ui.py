"""
backtest_ui.py — Backtesting tab.

Provides a Backtester class that simulates credit-spread P&L on
historical prices, then renders the results with metrics, charts,
a sortable trade log, and CSV/JSON/Journal export.
"""

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from trading_agent.streamlit.components import (
    drawdown_chart,
    equity_curve_chart,
    regime_bar_chart,
)

STARTING_EQUITY = 100_000.0
SPREAD_WIDTH = 5.0
COMMISSION_ROUND_TRIP = 2.60  # 4 legs × $0.65

DEFAULT_TICKERS = ["SPY", "QQQ", "IWM"]
DEFAULT_START = date.today() - timedelta(days=365)   # needs 200+ bars for SMA warmup
DEFAULT_END = date.today() - timedelta(days=1)

ALL_TICKERS = [
    "SPY", "QQQ", "IWM", "GOOG", "AAPL",
    "MSFT", "AMZN", "META", "SOFI", "TSLA", "JPM",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SimTrade:
    ticker: str
    strategy: str
    regime: str
    entry_date: date
    expiry_date: date
    credit: float
    max_loss: float
    outcome: str = ""   # "win" | "loss"
    pnl: float = 0.0
    hold_days: int = 0


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: Dict
    regime_stats: pd.DataFrame
    skipped: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Simulate credit-spread strategy on historical OHLCV data.

    Regime classification mirrors the real RegimeClassifier rules:
      price > SMA-200 AND slope > 0  → bullish  → Bull Put Spread
      price < SMA-200 AND slope < 0  → bearish  → Bear Call Spread
      otherwise                      → sideways → Iron Condor

    P&L simulation:
      - Credit = spread_width × credit_pct  (default 30 % of width)
      - Win if short strike is never breached within hold period
      - Short strike placed 3 % OTM for puts/calls, ±3 % band for condors
      - Win: capture profit_target_pct of credit minus commission
      - Loss: pay max_loss (width − credit) minus commission
    """

    def __init__(
        self,
        starting_equity: float = STARTING_EQUITY,
        spread_width: float = SPREAD_WIDTH,
        credit_pct: float = 0.30,
        target_dte: int = 45,
        profit_target_pct: float = 0.50,
        commission: float = COMMISSION_ROUND_TRIP,
    ) -> None:
        self.starting_equity = starting_equity
        self.spread_width = spread_width
        self.credit_pct = credit_pct
        self.target_dte = target_dte
        self.profit_target_pct = profit_target_pct
        self.commission = commission

    # ── Regime helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _classify(prices: pd.Series, idx: int) -> str:
        if idx < 200:
            return "sideways"
        window = prices.iloc[max(0, idx - 200): idx + 1]
        sma50 = window.iloc[-min(50, len(window)):].mean()
        sma200 = window.mean()
        slope = (
            window.iloc[-5:].mean() - window.iloc[-10:-5].mean()
            if len(window) >= 10
            else 0.0
        )
        price = window.iloc[-1]
        if price > sma200 and slope > 0:
            return "bullish"
        if price < sma200 and slope < 0:
            return "bearish"
        return "sideways"

    @staticmethod
    def _strategy(regime: str) -> str:
        return {
            "bullish": "Bull Put Spread",
            "bearish": "Bear Call Spread",
            "sideways": "Iron Condor",
        }.get(regime, "Iron Condor")

    # ── Outcome simulation ──────────────────────────────────────────────────

    def _simulate(
        self, prices: pd.Series, entry_idx: int, regime: str, credit: float
    ) -> tuple:
        end_idx = min(entry_idx + self.target_dte, len(prices) - 1)
        fwd = prices.iloc[entry_idx: end_idx + 1]
        entry_p = prices.iloc[entry_idx]

        if regime == "bullish":
            breached = (fwd < entry_p * 0.97).any()
        elif regime == "bearish":
            breached = (fwd > entry_p * 1.03).any()
        else:  # sideways / iron condor
            breached = ((fwd < entry_p * 0.97) | (fwd > entry_p * 1.03)).any()

        hold_days = end_idx - entry_idx
        if breached:
            pnl = -(self.spread_width * 100 - credit * 100) - self.commission
            return "loss", round(pnl, 2), hold_days
        pnl = credit * 100 * self.profit_target_pct - self.commission
        return "win", round(pnl, 2), hold_days

    # ── Main run ────────────────────────────────────────────────────────────

    def run(
        self,
        tickers: List[str],
        start: date,
        end: date,
        timeframe: str = "1Day",
        use_alpaca: bool = False,
    ) -> BacktestResult:
        """
        Execute the backtest and return a BacktestResult.

        Parameters
        ----------
        tickers     : list of ticker symbols
        start / end : date range
        timeframe   : "1Day" or "5Min"
        use_alpaca  : reserved for future Alpaca data source (currently no-op)
        """
        all_trades: List[SimTrade] = []
        equity = self.starting_equity
        equity_curve: List[Dict] = [{"timestamp": pd.Timestamp(start), "account_balance": equity}]

        yf_interval = "1d" if timeframe == "1Day" else "5m"

        skipped: List[str] = []

        for ticker in tickers:
            try:
                raw = yf.download(
                    ticker,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    interval=yf_interval,
                    progress=False,
                    auto_adjust=True,
                )
                if raw.empty:
                    skipped.append(f"{ticker} (no data returned)")
                    continue
                # Handle multi-level columns returned by recent yfinance versions
                if isinstance(raw.columns, pd.MultiIndex):
                    raw = raw.xs(ticker, axis=1, level=1) if ticker in raw.columns.get_level_values(1) else raw
                    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
                prices: pd.Series = raw["Close"].dropna()
            except Exception as exc:
                skipped.append(f"{ticker} (download error: {exc})")
                continue

            if len(prices) < 201:
                skipped.append(
                    f"{ticker} (only {len(prices)} bars — need 201+ for SMA-200 warmup; "
                    f"extend your date range to at least 1 year)"
                )
                continue

            last_entry_idx = -self.target_dte
            for i in range(200, len(prices)):
                if i - last_entry_idx < self.target_dte:
                    continue
                if equity <= 0:
                    break

                regime = self._classify(prices, i)
                strategy = self._strategy(regime)
                credit = self.spread_width * self.credit_pct
                max_loss = self.spread_width - credit

                outcome, pnl, hold_days = self._simulate(prices, i, regime, credit)
                equity = round(equity + pnl, 2)
                last_entry_idx = i

                raw_date = prices.index[i]
                entry_date = raw_date.date() if hasattr(raw_date, "date") else start

                all_trades.append(
                    SimTrade(
                        ticker=ticker,
                        strategy=strategy,
                        regime=regime,
                        entry_date=entry_date,
                        expiry_date=entry_date + timedelta(days=hold_days),
                        credit=credit,
                        max_loss=max_loss,
                        outcome=outcome,
                        pnl=pnl,
                        hold_days=hold_days,
                    )
                )
                equity_curve.append(
                    {"timestamp": pd.Timestamp(entry_date), "account_balance": equity}
                )

        if not all_trades:
            empty_trades = pd.DataFrame(
                columns=["ticker", "strategy", "regime", "entry_date",
                         "expiry_date", "credit", "max_loss", "outcome", "pnl", "hold_days"]
            )
            empty_eq = pd.DataFrame(
                [{"timestamp": pd.Timestamp(start), "account_balance": self.starting_equity}]
            )
            return BacktestResult(
                trades=empty_trades,
                equity_curve=empty_eq,
                metrics=self._metrics([], self.starting_equity),
                regime_stats=pd.DataFrame(columns=["regime", "pnl", "trade_count"]),
                skipped=skipped,
            )

        trades_df = pd.DataFrame([vars(t) for t in all_trades])
        equity_df = pd.DataFrame(equity_curve)

        return BacktestResult(
            trades=trades_df,
            equity_curve=equity_df,
            metrics=self._metrics(all_trades, self.starting_equity),
            regime_stats=self._regime_stats(trades_df),
            skipped=skipped,
        )

    # ── Stats helpers ───────────────────────────────────────────────────────

    def _metrics(self, trades: List[SimTrade], starting: float) -> Dict:
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe": 0.0,
                "avg_hold_days": 0.0,
            }
        wins = [t for t in trades if t.outcome == "win"]
        losses = [t for t in trades if t.outcome == "loss"]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

        pnl_arr = np.array([t.pnl for t in trades])
        sharpe = (
            float(pnl_arr.mean() / pnl_arr.std() * np.sqrt(252))
            if pnl_arr.std() > 0
            else 0.0
        )

        # Max drawdown from trade-by-trade equity
        eq = starting
        peak = starting
        max_dd = 0.0
        for t in sorted(trades, key=lambda x: x.entry_date):
            eq += t.pnl
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        return {
            "total_trades": len(trades),
            "win_rate": round(len(wins) / len(trades) * 100, 1),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe": round(sharpe, 2),
            "avg_hold_days": round(sum(t.hold_days for t in trades) / len(trades), 1),
        }

    @staticmethod
    def _regime_stats(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["regime", "pnl", "trade_count"])
        return (
            df.groupby("regime")
            .agg(pnl=("pnl", "sum"), trade_count=("pnl", "count"))
            .reset_index()
        )


# ---------------------------------------------------------------------------
# Cached runner
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running backtest…")
def _run_cached(
    tickers: tuple,
    start: date,
    end: date,
    timeframe: str,
    use_alpaca: bool,
) -> BacktestResult:
    return Backtester().run(list(tickers), start, end, timeframe, use_alpaca)


# ---------------------------------------------------------------------------
# Journal export helper
# ---------------------------------------------------------------------------

def _export_to_journal(result: BacktestResult) -> None:
    try:
        from trading_agent.journal_kb import JournalKB
        journal = JournalKB(journal_dir="trade_journal")
        m = result.metrics
        journal.log_signal(
            ticker="BACKTEST",
            action="dry_run",
            price=0.0,
            raw_signal={
                "regime": "backtest",
                "strategy": "Backtest Export",
                "plan_valid": True,
                "risk_approved": False,
                "account_balance": STARTING_EQUITY,
                "checks_passed": [],
                "checks_failed": [],
                "backtest_metrics": m,
                "backtest_trade_count": len(result.trades),
            },
            notes=(
                f"Backtest: {m['total_trades']} trades, "
                f"WR={m['win_rate']}%, Sharpe={m['sharpe']:.2f}"
            ),
        )
        st.success("Backtest summary exported to trade journal.")
    except Exception as exc:
        st.error(f"Journal export failed: {exc}")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_backtest_ui() -> None:
    """Render the Backtesting tab."""
    st.subheader("Strategy Backtester")

    # ── Sidebar controls ───────────────────────────────────────────────────
    with st.sidebar:
        st.header("Backtest Settings")
        start_date = st.date_input("Start Date", value=DEFAULT_START)
        end_date = st.date_input("End Date", value=DEFAULT_END)
        tickers = st.multiselect("Tickers", options=ALL_TICKERS, default=DEFAULT_TICKERS)
        timeframe = st.selectbox("Timeframe", options=["1Day", "5Min"], index=0)
        use_alpaca = st.toggle(
            "Use Alpaca Data",
            value=False,
            help="Prefer Alpaca historical bars over yfinance (requires live API key).",
        )
        run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

    if not run_btn and "backtest_result" not in st.session_state:
        st.info("Configure parameters in the sidebar and click **Run Backtest**.")
        return

    if run_btn:
        if not tickers:
            st.error("Select at least one ticker.")
            return
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            return
        result = _run_cached(
            tuple(sorted(tickers)), start_date, end_date, timeframe, use_alpaca
        )
        st.session_state["backtest_result"] = result

    result: Optional[BacktestResult] = st.session_state.get("backtest_result")
    if result is None:
        return

    m = result.metrics

    # ── Skipped ticker warnings ────────────────────────────────────────────
    if result.skipped:
        for msg in result.skipped:
            st.warning(f"Skipped: {msg}")

    if result.trades.empty:
        st.error(
            "No trades were simulated. The most common cause is a date range that is "
            "too short — the backtester needs at least **201 daily bars** (≈ 1 year) "
            "to compute the SMA-200 warmup before it can place the first trade. "
            "Try setting Start Date to at least 1 year ago."
        )
        return

    # ── Summary metric cards ───────────────────────────────────────────────
    cols = st.columns(6)
    for col, (label, value) in zip(
        cols,
        [
            ("Trades", m["total_trades"]),
            ("Win Rate", f"{m['win_rate']:.1f}%"),
            ("Profit Factor", f"{m['profit_factor']:.2f}"),
            ("Max DD", f"{m['max_drawdown_pct']:.1f}%"),
            ("Sharpe", f"{m['sharpe']:.2f}"),
            ("Avg Hold (d)", m["avg_hold_days"]),
        ],
    ):
        col.metric(label, value)

    st.divider()

    # ── Per-regime table + bar chart ───────────────────────────────────────
    if not result.regime_stats.empty:
        left, right = st.columns([2, 3])
        with left:
            st.subheader("Results by Regime")
            st.dataframe(
                result.regime_stats.assign(pnl=result.regime_stats["pnl"].map("${:,.2f}".format)),
                use_container_width=True,
                hide_index=True,
            )
        with right:
            st.plotly_chart(regime_bar_chart(result.regime_stats), use_container_width=True)
        st.divider()

    # ── Equity + drawdown charts ───────────────────────────────────────────
    if not result.equity_curve.empty:
        st.plotly_chart(equity_curve_chart(result.equity_curve), use_container_width=True)
        st.plotly_chart(drawdown_chart(result.equity_curve), use_container_width=True)
        st.divider()

    # ── Sortable trade log ─────────────────────────────────────────────────
    st.subheader("Trade Log")
    if not result.trades.empty:
        sort_col = st.selectbox(
            "Sort by", options=["entry_date", "pnl", "hold_days", "ticker"], index=0
        )
        ascending = st.checkbox("Ascending", value=False)
        st.dataframe(
            result.trades.sort_values(sort_col, ascending=ascending),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # ── Export row ─────────────────────────────────────────────────────────
    col_csv, col_json, col_journal = st.columns(3)

    with col_csv:
        st.download_button(
            "Export CSV",
            data=result.trades.to_csv(index=False).encode(),
            file_name="backtest_trades.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_json:
        st.download_button(
            "Export JSON",
            data=result.trades.to_json(orient="records", date_format="iso", indent=2).encode(),
            file_name="backtest_trades.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_journal:
        if st.button("Export to Journal", use_container_width=True):
            _export_to_journal(result)
