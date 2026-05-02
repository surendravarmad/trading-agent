"""
End-to-end smoke test for the Phase 3 journal split.

Drives ``JournalKB`` with both ``run_mode="live"`` and
``run_mode="backtest"`` against a temp directory and confirms:

    1. Each mode writes to its own filename (signals_live.jsonl /
       signals_backtest.jsonl) — never the legacy signals.jsonl.
    2. Both modes' files have one record per log_signal call.
    3. The legacy filename is *not* created (so we can't accidentally
       fall back to it once the split is in place).
    4. Validation rejects unknown run_modes (typo guard).

Run:

    python3 run_journal_split_check.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from trading_agent.journal_kb import JournalKB


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        # ---- Live mode ---------------------------------------------------
        live = JournalKB(journal_dir=tmp, run_mode="live")
        live.log_signal(
            ticker="SPY", action="dry_run", price=500.0,
            raw_signal={"strategy": "BullPut", "regime": "bullish"},
        )
        live_path = os.path.join(tmp, "signals_live.jsonl")
        if not os.path.exists(live_path):
            failures.append(f"live: missing {live_path}")
        else:
            n = sum(1 for _ in open(live_path))
            if n != 1:
                failures.append(f"live: expected 1 record, got {n}")

        # ---- Backtest mode ----------------------------------------------
        bt = JournalKB(journal_dir=tmp, run_mode="backtest")
        bt.log_signal(
            ticker="BACKTEST", action="dry_run", price=0.0,
            raw_signal={"backtest_metrics": {"total_trades": 42}},
        )
        bt_path = os.path.join(tmp, "signals_backtest.jsonl")
        if not os.path.exists(bt_path):
            failures.append(f"backtest: missing {bt_path}")
        else:
            n = sum(1 for _ in open(bt_path))
            if n != 1:
                failures.append(f"backtest: expected 1 record, got {n}")

        # ---- Legacy file must NOT be created ----------------------------
        legacy = os.path.join(tmp, "signals.jsonl")
        if os.path.exists(legacy):
            failures.append(
                f"legacy: signals.jsonl was created — split is incomplete"
            )

        # ---- Cross-contamination check ----------------------------------
        # Live file must contain only the live record; backtest file must
        # contain only the backtest record.
        if os.path.exists(live_path):
            with open(live_path) as fh:
                contents = fh.read()
            if "BACKTEST" in contents:
                failures.append("live file contains backtest payload")
        if os.path.exists(bt_path):
            with open(bt_path) as fh:
                contents = fh.read()
            if '"ticker": "SPY"' in contents:
                failures.append("backtest file contains live payload")

    # ---- Validation guard ------------------------------------------------
    try:
        JournalKB(journal_dir="/tmp/x", run_mode="liev")
        failures.append("typo run_mode='liev' was accepted; expected ValueError")
    except ValueError:
        pass

    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("All journal-split assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
