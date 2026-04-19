"""
daily_state — per-day persistent state for drawdown + exit debounce
===================================================================

Extracted from agent.py during week 3-4 modularization.  This module
owns the ``daily_state.json`` sidecar that persists three things
across cycles of the same calendar day:

  • ``date``            — current calendar day (ISO string)
  • ``start_equity``    — account equity at the first cycle of the day,
                          used as the baseline for the drawdown circuit
                          breaker
  • ``exit_vote_counts``— per-ticker consecutive-cycle vote counts used
                          by the 3-cycle exit-signal debounce

File concurrency
----------------
All writes go through ``file_locks.update_json_locked`` which uses an
advisory lock on a sidecar ``.lock`` file plus ``os.replace()`` for the
atomic swap.  That guarantees neither an overlapping cron cycle nor a
manual ``python -m trading_agent.agent`` run can corrupt the state.

Rationale for separation from the main agent class
---------------------------------------------------
The prior inline implementation on TradingAgent mixed three concerns:
  1. File I/O (load / save / locking)
  2. Policy (how much drawdown triggers the breaker)
  3. Orchestration (when to call what)

Split here:
  • ``DailyState`` dataclass — the persisted record
  • ``DailyStateStore``       — locked load/save
  • ``check_daily_drawdown``  — pure policy over a store + inputs
  • ``should_exit_spread``    — debounce policy over a store + signal

The main agent just wires these together.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from trading_agent.file_locks import update_json_locked

logger = logging.getLogger(__name__)

DAILY_STATE_FILENAME = "daily_state.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class DailyState:
    """In-memory view of the persisted daily state."""
    date: str = ""
    start_equity: float = 0.0
    exit_vote_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "DailyState":
        return cls(
            date=str(data.get("date", "")),
            start_equity=float(data.get("start_equity", 0.0) or 0.0),
            exit_vote_counts=dict(data.get("exit_vote_counts", {}) or {}),
        )

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "start_equity": self.start_equity,
            "exit_vote_counts": self.exit_vote_counts,
        }


class DailyStateStore:
    """
    Locked, atomic persistent store for DailyState.

    Instances are cheap — one per trade_plan_dir.  All methods are
    thread-safe and cross-process-safe via file_locks.update_json_locked.
    """

    def __init__(self, trade_plan_dir: str) -> None:
        self.path = os.path.join(trade_plan_dir, DAILY_STATE_FILENAME)

    def load(self) -> DailyState:
        """
        Load current state.  Returns a fresh empty DailyState when the
        file is missing or corrupt.
        """
        # update_json_locked with an identity mutator == "read under lock"
        try:
            raw = update_json_locked(self.path, mutator=lambda s: s, default={})
        except Exception as exc:
            logger.warning("daily_state read failed: %s", exc)
            raw = {}
        return DailyState.from_dict(raw)

    def save(self, state: DailyState) -> None:
        """Persist *state* atomically under lock."""
        payload = state.to_dict()
        try:
            update_json_locked(self.path, mutator=lambda _: payload, default={})
        except Exception as exc:
            logger.warning("daily_state write failed: %s", exc)

    def update(self, mutator) -> DailyState:
        """
        Read-modify-write under a single lock.

        ``mutator`` receives a DailyState and must return a DailyState.
        This is the preferred API for debounce updates — it avoids the
        TOCTOU window of separate load()/save() calls.
        """
        def _json_mutator(raw: dict) -> dict:
            current = DailyState.from_dict(raw)
            updated = mutator(current)
            return updated.to_dict()

        try:
            new_raw = update_json_locked(self.path, mutator=_json_mutator, default={})
        except Exception as exc:
            logger.warning("daily_state update failed: %s", exc)
            return DailyState()
        return DailyState.from_dict(new_raw)


# ---------------------------------------------------------------------------
# Policy functions — pure, easy to unit test
# ---------------------------------------------------------------------------
def check_daily_drawdown(
    store: DailyStateStore,
    current_equity: float,
    drawdown_limit: float,
    journal_kb=None,
) -> bool:
    """
    Return True iff today's equity has fallen beyond ``drawdown_limit``.

    Side effect: on the first call of a new calendar day, initialises
    the state with today's starting equity (and resets vote counts).

    Parameters
    ----------
    store           : DailyStateStore instance
    current_equity  : latest account equity
    drawdown_limit  : fraction (e.g. 0.05 for 5%)
    journal_kb      : optional JournalKB for audit trail
    """
    today = datetime.now().date().isoformat()

    def _mutator(state: DailyState) -> DailyState:
        if state.date != today:
            # New day — reset baseline, preserve nothing
            state = DailyState(
                date=today,
                start_equity=current_equity,
                exit_vote_counts={},
            )
        return state

    state = store.update(_mutator)

    start = state.start_equity
    if start <= 0:
        return False

    drawdown = (current_equity - start) / start
    if drawdown >= -drawdown_limit:
        logger.info(
            "Daily drawdown check OK: %.2f%% (limit=%.0f%%)",
            drawdown * 100, drawdown_limit * 100,
        )
        return False

    logger.critical(
        "DAILY DRAWDOWN CIRCUIT BREAKER TRIGGERED: "
        "%.2f%% loss today (limit=%.0f%%, start=$%.2f, now=$%.2f)",
        abs(drawdown) * 100, drawdown_limit * 100, start, current_equity,
    )
    if journal_kb is not None:
        journal_kb.log_cycle_error(
            "daily_drawdown_circuit_breaker",
            {
                "drawdown_pct": round(drawdown * 100, 3),
                "start_equity": start,
                "current_equity": current_equity,
                "limit_pct": drawdown_limit * 100,
                "date": today,
            },
        )
    return True


def tally_exit_vote(
    store: DailyStateStore,
    ticker: str,
    signal_val: str,
    required: int = 3,
) -> int:
    """
    Increment (or reset) the consecutive-cycle vote count for *ticker*.

    Returns the post-update vote count.  Caller compares against
    ``required`` to decide whether to act.

    Resetting: if the current stored signal for this ticker differs
    from *signal_val*, the count restarts at 1.
    """

    def _mutator(state: DailyState) -> DailyState:
        votes = state.exit_vote_counts
        existing = votes.get(ticker, {}) or {}
        if existing.get("signal") == signal_val:
            new_count = int(existing.get("count", 0)) + 1
        else:
            new_count = 1
        votes[ticker] = {"signal": signal_val, "count": new_count}
        state.exit_vote_counts = votes
        return state

    state = store.update(_mutator)
    entry = state.exit_vote_counts.get(ticker, {})
    return int(entry.get("count", 0))
