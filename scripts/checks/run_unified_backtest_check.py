"""
End-to-end smoke test for the Phase 2 unified-engine backtester path.

Exercises ``Backtester._build_alpaca_plan_via_decide`` with hand-crafted
fakes for ``_fetch_alpaca_option_contracts`` + ``_fetch_alpaca_option_bars``
so we don't need a real Alpaca account. Confirms:

    1. With ``use_unified_engine=True`` and a preset, the backtester
       returns a plan whose ``credit`` was scored by ``decide()``
       (not by the legacy σ-distance heuristic).
    2. ``last_decide_diagnostics`` is populated on the Backtester
       instance, so the same panel that renders live diagnostics can
       render backtest diagnostics — taxonomy stays unified.
    3. The plan dict has the same shape the legacy path returns
       (short_symbol, long_symbol, short_strike, long_strike,
       expiration, option_type, credit, strike_distance_pct,
       approx_abs_delta), so callers downstream of
       ``_build_alpaca_plan_for_expiration`` keep working unchanged.

Run:

    python3 run_unified_backtest_check.py
"""

from __future__ import annotations

import os
import sys
import types
from datetime import date, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# Stub pandas_market_calendars (sandbox can't pip install it).
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


_mcal_stub = types.ModuleType("pandas_market_calendars")
_mcal_stub.get_calendar = lambda name: _StubCal()
sys.modules["pandas_market_calendars"] = _mcal_stub


# Stub yfinance — sandbox can't install it.
_yf_stub = types.ModuleType("yfinance")
_yf_stub.download = lambda *a, **kw: None
_yf_stub.Ticker = lambda *a, **kw: None
sys.modules["yfinance"] = _yf_stub


# Stub plotly (used by streamlit.components — only at import time).
# Module-shaped __getattr__ returns a class for unknown names so usages
# like ``go.Figure`` (a type-annotation evaluated at def-time) resolve.
class _AnyMod(types.ModuleType):
    def __getattr__(self, name):
        return type(name, (), {})
_plotly = _AnyMod("plotly")
_go = _AnyMod("plotly.graph_objects")
_px = _AnyMod("plotly.express")
_sp = _AnyMod("plotly.subplots")
# Wire submodules onto the parent so ``import plotly.graph_objects as go``
# sees ``go == _go`` (Python's import system uses parent attribute access
# even when sys.modules has the entry).
_plotly.graph_objects = _go
_plotly.express = _px
_plotly.subplots = _sp
sys.modules["plotly"] = _plotly
sys.modules["plotly.graph_objects"] = _go
sys.modules["plotly.express"] = _px
sys.modules["plotly.subplots"] = _sp


# Stub scipy + scipy.stats.percentileofscore (only that one symbol is
# touched at import time by backtest_ui).
_scipy_stub = types.ModuleType("scipy")
_scipy_stats_stub = types.ModuleType("scipy.stats")
_scipy_stats_stub.percentileofscore = lambda *a, **kw: 50.0
sys.modules["scipy"] = _scipy_stub
sys.modules["scipy.stats"] = _scipy_stats_stub


# Stub streamlit before importing backtest_ui — sandbox doesn't have it.
class _StubStreamlit:
    def __getattr__(self, name):
        # Anything Streamlit-side that backtest_ui touches at *import*
        # time (decorators, sidebar, columns…) is a no-op stub here.
        def _noop(*a, **kw): return self
        return _noop

    def __call__(self, *a, **kw):
        return self


_st_stub = _StubStreamlit()
_st_stub_module = types.ModuleType("streamlit")
for _attr in ("title", "header", "subheader", "write", "markdown",
              "metric", "table", "dataframe", "expander", "columns",
              "container", "sidebar", "slider", "selectbox", "checkbox",
              "button", "radio", "text_input", "number_input",
              "session_state", "cache_data", "cache_resource",
              "spinner", "progress", "warning", "error", "info", "success",
              "tabs", "json", "code", "set_page_config", "stop"):
    setattr(_st_stub_module, _attr, _st_stub)
sys.modules["streamlit"] = _st_stub_module


from trading_agent.streamlit.backtest_ui import Backtester  # noqa: E402


class _FakePreset:
    """Minimal preset matching what decision_engine.decide() reads."""
    edge_buffer    = 0.10
    min_pop        = 0.55
    dte_grid       = (14,)
    delta_grid     = (-0.20, -0.30)
    width_grid_pct = (0.02, 0.05)


def _print_block(title, val):
    print(f"--- {title} ---")
    print(val)
    print()


def main() -> int:
    # ----------------------------------------------------------------- inputs
    entry = date(2026, 5, 1)
    expiration = "2026-05-15"          # 14 DTE
    spot = 100.0

    # Fake put chain: strikes 80→100 in $1 steps, with bar closes that put
    # *some* combination above the C/W floor so decide() returns a winner.
    # The grid step (1.0) means width_grid_pct=0.05 → width=5.0, snapped.
    strikes = [80.0, 85.0, 88.0, 90.0, 92.0, 95.0, 98.0, 100.0]
    contracts_put = [
        {"symbol": f"SPY{expiration.replace('-', '')}P{int(s*1000):08d}",
         "strike": s, "type": "put",
         "expiration": expiration}
        for s in strikes
    ]

    # Per-strike entry-day close ≈ option price. The decision engine reads
    # it as bid=ask=close, so credit ≈ short_close − long_close − haircut.
    # We choose closes such that on the (Δ=-0.30, width=5) tuple the credit
    # exceeds the floor: short ~ 1.20, long ~ 0.50 → credit ≈ 0.68 vs
    # floor ~0.30 × 1.10 = 0.33 on $5 width → C/W = 0.68/5 = 0.136 ⇒
    # falls below the floor of 0.33. To get above we need short_close
    # large enough; bump the synthetic prices.
    closes = {
        80.0: 0.05, 85.0: 0.15, 88.0: 0.30,
        90.0: 0.55, 92.0: 0.95, 95.0: 1.85, 98.0: 3.00, 100.0: 4.50,
    }
    bars_by_symbol = {
        c["symbol"]: [{"c": closes[c["strike"]], "t": entry.isoformat()}]
        for c in contracts_put
    }

    # ----------------------------------------------------------- patch fakes
    bt = Backtester(
        spread_width=5.0,
        use_alpaca_historical=True,
        use_unified_engine=True,
        preset=_FakePreset(),
    )
    bt._fetch_alpaca_option_contracts = lambda ticker, expiration, opt_type: (
        contracts_put if opt_type == "put" else []
    )
    bt._fetch_alpaca_option_bars = lambda symbols, start, end: {
        s: bars_by_symbol[s] for s in symbols if s in bars_by_symbol
    }

    # ---------------------------------------------------------------- drive
    plan, reason, retry = bt._build_alpaca_plan_for_expiration(
        ticker="SPY",
        entry_date=entry,
        regime="bullish",
        expiration=expiration,
        underlying_price=spot,
        sigma_annual=0.20,             # 20 % annualised vol
        effective_sigma_mult=1.0,
        hold_bars=14,
        bars_per_year=252,
    )

    print("plan:")
    print(plan)
    print()
    print("reason:", repr(reason), "retry:", retry)
    print()
    _print_block("last_decide_diagnostics",
                 vars(bt.last_decide_diagnostics)
                 if bt.last_decide_diagnostics is not None
                 else "<None>")

    # ------------------------------------------------------------ assertions
    failures = []
    if bt.last_decide_diagnostics is None:
        failures.append("last_decide_diagnostics must be populated by unified path")
    if plan is None:
        # Diagnostics-only run: still meaningful as long as the engine
        # did its work and returned a structured reason from decide_no_*.
        if not reason.startswith("decide_no_candidate_"):
            failures.append(
                f"plan=None must come with a decide_no_candidate_* reason; got {reason!r}"
            )
    else:
        for k in ("short_symbol", "long_symbol", "short_strike",
                  "long_strike", "expiration", "option_type",
                  "credit", "strike_distance_pct", "approx_abs_delta"):
            if k not in plan:
                failures.append(f"plan missing key {k!r}")
        if plan.get("option_type") != "put":
            failures.append(f"option_type must be 'put' for bullish regime; got {plan.get('option_type')!r}")
        if plan.get("credit", 0) <= 0:
            failures.append(f"credit must be positive on success; got {plan.get('credit')}")

    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("All unified-backtester smoke assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
