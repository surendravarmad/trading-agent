"""
Parity check: ChainScanner.scan() and Backtester._build_alpaca_plan_via_decide()
must pick the same strikes / credit when fed the same chain.

Both call ``decision_engine.decide()`` under the hood, so the parity is
guaranteed by construction. This driver proves it end-to-end against a
hand-rolled chain so a future regression — someone replacing one path
with a homegrown scorer — gets caught at CI time.

Run:

    python3 run_live_vs_backtest_parity_check.py
"""

from __future__ import annotations

import os
import sys
import types
from datetime import date, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# ── Sandbox stubs (sandbox lacks pip-installed deps) ────────────────────
class _StubCal:
    def valid_days(self, start_date, end_date):
        days = []
        d = start_date
        while d <= end_date:
            if d.weekday() < 5:
                class _D:
                    def __init__(self, dd): self._d = dd
                    def date(self): return self._d
                days.append(_D(d))
            d += timedelta(days=1)
        return days


_mcal = types.ModuleType("pandas_market_calendars")
_mcal.get_calendar = lambda name: _StubCal()
sys.modules["pandas_market_calendars"] = _mcal

_yf = types.ModuleType("yfinance")
_yf.download = lambda *a, **kw: None
_yf.Ticker = lambda *a, **kw: None
sys.modules["yfinance"] = _yf

_scipy = types.ModuleType("scipy")
_scipy_stats = types.ModuleType("scipy.stats")
_scipy_stats.percentileofscore = lambda *a, **kw: 50.0
sys.modules["scipy"] = _scipy
sys.modules["scipy.stats"] = _scipy_stats


class _AnyMod(types.ModuleType):
    def __getattr__(self, name):
        return type(name, (), {})

_plotly = _AnyMod("plotly")
_go = _AnyMod("plotly.graph_objects")
_px = _AnyMod("plotly.express")
_sp = _AnyMod("plotly.subplots")
_plotly.graph_objects = _go
_plotly.express = _px
_plotly.subplots = _sp
sys.modules["plotly"] = _plotly
sys.modules["plotly.graph_objects"] = _go
sys.modules["plotly.express"] = _px
sys.modules["plotly.subplots"] = _sp


class _StubStreamlit:
    def __getattr__(self, name):
        def _noop(*a, **kw): return self
        return _noop
    def __call__(self, *a, **kw): return self


_st = _StubStreamlit()
_st_mod = types.ModuleType("streamlit")
for _attr in ("title", "header", "subheader", "write", "markdown",
              "metric", "table", "dataframe", "expander", "columns",
              "container", "sidebar", "slider", "selectbox", "checkbox",
              "button", "radio", "text_input", "number_input",
              "session_state", "cache_data", "cache_resource",
              "spinner", "progress", "warning", "error", "info", "success",
              "tabs", "json", "code", "set_page_config", "stop"):
    setattr(_st_mod, _attr, _st)
sys.modules["streamlit"] = _st_mod


# ── Imports after stubbing ──────────────────────────────────────────────
from trading_agent.chain_scanner import ChainScanner            # noqa: E402
from trading_agent.streamlit.backtest_ui import Backtester      # noqa: E402


class _Preset:
    edge_buffer    = 0.10
    min_pop        = 0.55
    dte_grid       = (14,)
    delta_grid     = (-0.20, -0.30)
    width_grid_pct = (0.02, 0.05)


def _build_chain(strikes, deltas, bids, asks, dte_days):
    return [
        {"symbol": f"SPY{dte_days:02d}DP{int(s*1000):08d}",
         "strike": s, "delta": d, "bid": b, "ask": a, "type": "put"}
        for s, d, b, a in zip(strikes, deltas, bids, asks)
    ]


class _FakeData:
    def __init__(self, chain): self._chain = chain
    def fetch_option_chain(self, ticker, expiration, opt_type):
        return self._chain


def main() -> int:
    today = date(2026, 5, 1)
    expiration = "2026-05-15"
    spot = 100.0

    # The two paths consume different *chain shapes* — live gets a real
    # snapshot with NBBO bid/ask + Alpaca Greeks; backtest gets synth-Δ
    # from σ_hold and bar-close-as-mid. To prove the *engine* doesn't
    # drift we drive both with the same synth-Δ chain. If decide() is
    # the single source of truth, the two paths must pick the same row.
    from math import erf, log, sqrt
    sigma_hold = 0.20 * sqrt(14 / 252.0)

    def _synth_delta(strike: float) -> float:
        d1 = (log(spot / strike) + 0.5 * sigma_hold * sigma_hold) / sigma_hold
        cdf = 0.5 * (1.0 + erf(d1 / sqrt(2.0)))
        return cdf - 1.0   # put

    # Use a wider strike grid so Δ=-0.20 and Δ=-0.30 land cleanly.
    strikes = [88.0, 92.0, 95.0, 96.0, 97.0, 98.0, 99.0]
    deltas  = [_synth_delta(s) for s in strikes]
    # Bid/ask such that mid roughly tracks intrinsic + a bit of time value.
    bids    = [0.40, 0.85, 1.30, 1.55, 1.85, 2.20, 2.65]
    asks    = [0.45, 0.92, 1.40, 1.65, 1.95, 2.30, 2.75]
    chain_live = _build_chain(strikes, deltas, bids, asks, dte_days=14)

    scanner = ChainScanner(_FakeData(chain_live), _Preset(), dte_window_days=5)
    live_candidates = scanner.scan("SPY", "bull_put", today=today)

    print("=== Live path ===")
    if live_candidates:
        c = live_candidates[0]
        print(f"  pick: short={c.short_strike} long={c.long_strike} "
              f"credit={c.credit:.2f} cw={c.cw_ratio:.4f} Δ={c.short_delta:.3f}")
    else:
        print("  no candidate")

    # ── Backtest path: synthesize chain from contracts + bars ────────────
    contracts = [
        {"symbol": f"SPY14DP{int(s*1000):08d}",
         "strike": s, "type": "put", "expiration": expiration}
        for s in strikes
    ]
    # Bar close = NBBO midpoint, so the synth-chain credit matches the
    # live ``_quote_credit(short_mid, long_mid)`` minus fill_haircut.
    closes = {s: (b + a) / 2.0 for s, b, a in zip(strikes, bids, asks)}
    bars   = {c["symbol"]: [{"c": closes[c["strike"]],
                              "t": today.isoformat()}]
              for c in contracts}

    bt = Backtester(spread_width=5.0, use_alpaca_historical=True,
                    use_unified_engine=True, preset=_Preset())
    bt._fetch_alpaca_option_contracts = lambda ticker, exp, opt_type: (
        contracts if opt_type == "put" else []
    )
    bt._fetch_alpaca_option_bars = lambda symbols, start, end: {
        s: bars[s] for s in symbols if s in bars
    }
    plan, reason, _retry = bt._build_alpaca_plan_for_expiration(
        ticker="SPY", entry_date=today, regime="bullish",
        expiration=expiration, underlying_price=100.0,
        sigma_annual=0.20, effective_sigma_mult=1.0,
        hold_bars=14, bars_per_year=252,
    )
    print("\n=== Backtest path ===")
    if plan:
        print(f"  pick: short={plan['short_strike']} long={plan['long_strike']} "
              f"credit={plan['credit']:.2f}")
    else:
        print(f"  no plan (reason={reason!r})")

    # ── Parity assertions ────────────────────────────────────────────────
    failures = []
    if not live_candidates:
        failures.append("live path returned no candidate")
    if not plan:
        failures.append(f"backtest path returned no plan (reason={reason!r})")
    if live_candidates and plan:
        c = live_candidates[0]
        # Strikes must match exactly — same engine + same chain.
        if c.short_strike != plan["short_strike"]:
            failures.append(
                f"short_strike drift: live={c.short_strike} backtest={plan['short_strike']}"
            )
        if c.long_strike != plan["long_strike"]:
            failures.append(
                f"long_strike drift: live={c.long_strike} backtest={plan['long_strike']}"
            )
        # Credit may differ by < $0.01 due to rounding/synth-delta variance.
        if abs(c.credit - plan["credit"]) > 0.01:
            failures.append(
                f"credit drift: live={c.credit:.4f} backtest={plan['credit']:.4f}"
            )

    if failures:
        print("\nFAIL — live↔backtest drift detected:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nLive ↔ backtest parity OK: same strikes + credit (Δ ≤ $0.01).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
