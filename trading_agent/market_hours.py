"""
market_hours — NYSE trading-hours guard
=======================================

Extracted from agent.py during the week 3-4 modularization so that
market-hours logic can be unit-tested in isolation and reused by
components outside the core agent (e.g. the Streamlit UI, the backtester
validation hooks).

Scope
-----
US equity / option markets only — per the current strategy's charter.
When the vendor-agnostic refactor (week 5-6) lands, these constants will
move into a ``MarketProfile`` dataclass injected via configuration and
this module will become the US-specific implementation of that port.

Behavior
--------
``is_within_market_hours()`` returns True iff:
  • today is an NYSE regular-session trading day, AND
  • the current Eastern time is within
    [09:25, 16:05] ET — a 5-minute buffer on each side.

The buffer exists so a cron job scheduled at :30 past the hour (or
exactly on 9:30 / 16:00) can complete the cycle it already started
without being rejected.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from trading_agent.calendar_utils import is_trading_day

# ---------------------------------------------------------------------------
# US (NYSE) market calendar constants
# ---------------------------------------------------------------------------
EASTERN = ZoneInfo("America/New_York")

# NYSE core session: 9:30 AM – 4:00 PM ET.
# 5-minute buffers let a cron job scheduled exactly on the boundary
# finish its current cycle before the next invocation would be blocked.
MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE   = 9, 25    # 5 min before 9:30
MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE = 16, 5    # 5 min after  16:00


def market_window_str() -> str:
    """Human-readable window string for logs / journal context."""
    return (
        f"{MARKET_OPEN_HOUR:02d}:{MARKET_OPEN_MINUTE:02d}–"
        f"{MARKET_CLOSE_HOUR:02d}:{MARKET_CLOSE_MINUTE:02d} ET "
        f"Mon–Fri"
    )


def is_within_market_hours(now: Optional[datetime] = None) -> bool:
    """
    Return True if *now* (default: current moment) is within the NYSE
    regular session with the configured buffers.

    Uses ``pandas_market_calendars`` (via calendar_utils) to correctly
    skip weekends AND market holidays — the prior weekday-based check
    incorrectly woke the agent on July 4th, MLK Day, Good Friday,
    Thanksgiving, etc.
    """
    now = now or datetime.now(EASTERN)
    # Allow tests to pass a naive / non-Eastern datetime.
    if now.tzinfo is None:
        now = now.replace(tzinfo=EASTERN)
    else:
        now = now.astimezone(EASTERN)

    if not is_trading_day(now.date()):
        return False

    open_boundary = now.replace(
        hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE,
        second=0, microsecond=0,
    )
    close_boundary = now.replace(
        hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE,
        second=0, microsecond=0,
    )
    return open_boundary <= now <= close_boundary
