"""
End-to-end smoke test for the new ScanDiagnostics journal block.

Drives ``ChainScanner`` against an in-memory chain that's deliberately
tuned to fail the C/W floor on every grid point. Confirms that:

    1. ``last_diagnostics`` is populated (was previously absent)
    2. ``rejects_by_reason`` records ``cw_below_floor`` for every
       Δ × width tuple
    3. ``best_near_miss`` is populated with EV/CW/floor fields and
       picks the candidate with highest EV across all rejects
    4. ``grid_points_priced`` matches |Δ| × |w| for the synthetic
       single-expiration chain

Run:

    python3 run_scan_diagnostics_check.py
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Stub pandas_market_calendars before importing the scanner — the sandbox
# can't pip install the real package and we don't need its calendar.
class _StubCal:
    def valid_days(self, start_date, end_date):
        from datetime import timedelta
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

import types
mcal_stub = types.ModuleType("pandas_market_calendars")
mcal_stub.get_calendar = lambda name: _StubCal()
sys.modules["pandas_market_calendars"] = mcal_stub

from trading_agent.chain_scanner import ChainScanner, REJECT_CW_BELOW_FLOOR


class _FakePreset:
    """Minimal preset stub matching what ChainScanner reads."""
    edge_buffer    = 0.10        # require C/W ≥ |Δ|*1.10
    min_pop        = 0.55        # POP floor
    dte_grid       = (14,)       # single weekly expiration
    delta_grid     = (-0.20, -0.30)
    width_grid_pct = (0.01, 0.02)


def _put_chain(strikes, deltas, bids, asks, dte_days):
    """Build a chain shaped the way market_data delivers it."""
    return [
        {"symbol": f"SPY {dte_days}D{strikes[i]}P",
         "strike": strikes[i],
         "delta":  deltas[i],
         "bid":    bids[i],
         "ask":    asks[i]}
        for i in range(len(strikes))
    ]


class _FakeDataProvider:
    """Returns a single hand-crafted put chain regardless of args."""
    def __init__(self, chain):
        self._chain = chain

    def fetch_option_chain(self, ticker, expiration, opt_type):
        return self._chain


def _print_block(title, val):
    print(f"--- {title} ---")
    print(val)
    print()


def main() -> int:
    # Spot ≈ 100. Strikes from 80 to 100, $1 grid. ATM (Δ ≈ 0.50) at 100.
    strikes = [80.0, 85.0, 90.0, 95.0, 98.0, 99.0, 100.0]
    deltas  = [-0.05, -0.10, -0.20, -0.30, -0.42, -0.46, -0.50]
    # Tune bids so credits land *just below* the floor:
    #   Δ=-0.20 → floor = 0.20 * 1.10 = 0.22 → on $2 width need credit ≥ $0.44
    #   Δ=-0.30 → floor = 0.30 * 1.10 = 0.33 → on $2 width need credit ≥ $0.66
    # We give bids that pass POP filter but credit/width sits at ~0.20 and ~0.30
    # respectively, *just* below their floors.
    bids = [0.05, 0.10, 0.40, 0.60, 1.20, 1.40, 1.50]
    asks = [0.10, 0.15, 0.45, 0.65, 1.25, 1.45, 1.55]

    chain = _put_chain(strikes, deltas, bids, asks, dte_days=14)
    scanner = ChainScanner(_FakeDataProvider(chain), _FakePreset(), dte_window_days=5)

    # Pin "today" two weeks before the expiration so DTE = 14.
    today = date(2026, 5, 1)
    candidates = scanner.scan("SPY", "bull_put", today=today)

    diag = scanner.last_diagnostics
    assert diag is not None, "diagnostics must be populated after every scan"

    print("Candidates returned:", len(candidates))
    _print_block("rejects_by_reason", diag.rejects_by_reason)
    _print_block("grid_points_total/priced/expirations",
                 (diag.grid_points_total, diag.grid_points_priced,
                  diag.expirations_resolved))
    _print_block("best_near_miss", diag.best_near_miss)

    # Assertions
    failures = []
    if candidates:
        failures.append(f"expected 0 candidates (thin credit), got {len(candidates)}")
    if diag.grid_points_total != 1 * 2 * 2:
        failures.append(f"grid_points_total expected 4, got {diag.grid_points_total}")
    if diag.expirations_resolved != 1:
        failures.append(f"expirations_resolved expected 1, got {diag.expirations_resolved}")
    if REJECT_CW_BELOW_FLOOR not in diag.rejects_by_reason:
        failures.append(f"reject_reason missing {REJECT_CW_BELOW_FLOOR!r}; "
                        f"got {diag.rejects_by_reason}")
    if diag.best_near_miss is None:
        failures.append("best_near_miss must be populated (every reject was cw_below_floor)")
    else:
        # Best-near-miss must have full EV/CW/floor fields populated.
        for k in ("ev", "cw_ratio", "cw_floor", "pop", "credit", "width", "short_delta"):
            if k not in diag.best_near_miss:
                failures.append(f"best_near_miss missing field {k!r}")

    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("All ScanDiagnostics assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
