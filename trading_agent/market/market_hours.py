"""
market_hours — trading-hours guard, parameterised by a MarketProfile
====================================================================

Previously this module hardcoded NYSE constants at module scope.  As
part of the week 5-6 vendor-agnostic refactor the constants moved into
:class:`trading_agent.market_profile.MarketProfile`; this module now
reads them from a profile passed in (or from ``US_MARKET_PROFILE`` by
default) so the same guard can be reused for future venues without
touching the core.

Behavior
--------
``is_within_market_hours(now, profile)`` returns True iff:
  • ``profile.is_trading_day(now.date())`` — regular-session day, AND
  • current time in the profile's timezone is within
    [open_time, close_time] inclusive.

The 5-minute open/close buffers baked into :data:`US_MARKET_PROFILE`
let a cron job scheduled exactly on the NYSE bell (09:30 / 16:00)
complete the cycle it already started without being rejected by the
next invocation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from trading_agent.market.market_profile import MarketProfile, US_MARKET_PROFILE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def market_window_str(profile: MarketProfile = US_MARKET_PROFILE) -> str:
    """Human-readable window string for logs / journal context."""
    return f"{profile.session_window_str} Mon–Fri"


def is_within_market_hours(now: Optional[datetime] = None,
                           profile: MarketProfile = US_MARKET_PROFILE) -> bool:
    """
    Return True if *now* (default: current moment in the profile's TZ)
    is within the profile's regular session with its configured buffers.

    Uses ``profile.is_trading_day`` (NYSE via pandas_market_calendars by
    default) to correctly skip weekends AND market holidays — the prior
    weekday-based check incorrectly woke the agent on July 4th, MLK Day,
    Good Friday, Thanksgiving, etc.
    """
    tz = profile.timezone
    now = now or datetime.now(tz)
    # Allow tests to pass a naive datetime in the profile's timezone.
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    else:
        now = now.astimezone(tz)

    if not profile.is_trading_day(now.date()):
        return False

    open_boundary = now.replace(
        hour=profile.open_hour, minute=profile.open_minute,
        second=0, microsecond=0,
    )
    close_boundary = now.replace(
        hour=profile.close_hour, minute=profile.close_minute,
        second=0, microsecond=0,
    )
    return open_boundary <= now <= close_boundary


# ---------------------------------------------------------------------------
# Back-compat module-level aliases
# ---------------------------------------------------------------------------
# Some internal callers and tests import the bare constants.  We re-export
# the US values here so those imports keep working.  New code should read
# them from :data:`US_MARKET_PROFILE` (or the injected profile) instead.
EASTERN = US_MARKET_PROFILE.timezone
MARKET_OPEN_HOUR = US_MARKET_PROFILE.open_hour
MARKET_OPEN_MINUTE = US_MARKET_PROFILE.open_minute
MARKET_CLOSE_HOUR = US_MARKET_PROFILE.close_hour
MARKET_CLOSE_MINUTE = US_MARKET_PROFILE.close_minute
