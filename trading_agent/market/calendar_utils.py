"""NYSE trading-calendar helpers — wraps pandas_market_calendars.

A single module-level NYSE calendar instance is shared across the agent
to amortise the ~50-100 ms pandas_market_calendars setup cost.

Rationale
---------
Naive `datetime.weekday()` math treats every Monday-Friday as a trading
day, which silently breaks three places in the agent:

  1. ``strategy._pick_expiration`` — when Friday is a market holiday
     (Good Friday, Day-after-Thanksgiving half-day, etc.), weekly
     options actually expire the preceding Thursday.
  2. ``agent._is_within_market_hours`` — naive weekday check runs the
     cycle on MLK, July 4th, etc. when the market is closed.
  3. ``position_monitor._check_dte_safety`` — the "close on Thursday
     before Friday expiration" rule is really "close on the last
     trading day strictly before expiration".
"""

from datetime import date, timedelta
from typing import List, Optional

import pandas_market_calendars as mcal

_NYSE = mcal.get_calendar("NYSE")


def _valid_days(start: date, end: date) -> List[date]:
    """Return sorted list of NYSE regular-session trading days in [start, end]."""
    return [d.date() for d in _NYSE.valid_days(start_date=start, end_date=end)]


def is_trading_day(d: date) -> bool:
    """True if *d* is a full NYSE regular-session trading day.

    False for weekends and NYSE holidays (NYE, Good Friday, Thanksgiving,
    Independence Day, MLK, Presidents' Day, Memorial Day, Juneteenth,
    Labor Day, Christmas).
    """
    return bool(_valid_days(d, d))


def next_weekly_expiration(today: date, target_dte: int,
                           dte_min: int, dte_max: int) -> date:
    """Pick the weekly options expiration closest to *target_dte*.

    Weekly options nominally expire Friday, but on holiday-Fridays they
    expire the preceding Thursday. We therefore return the *last trading
    day of the candidate week* (Mon-Fri).

    Candidate selection — evaluate three adjacent weekly expirations
    (previous, target, next) and pick the one whose DTE falls in
    [dte_min, dte_max] with the **highest** DTE (more theta runway, less
    gamma risk near expiry). If none fit the range, clamp to the
    candidate whose DTE is closest to *target_dte*.
    """
    target = today + timedelta(days=target_dte)

    def last_trading_day_in_week_of(ref: date) -> Optional[date]:
        # Week = Monday-Friday containing *ref*
        monday = ref - timedelta(days=ref.weekday())
        friday = monday + timedelta(days=4)
        days = _valid_days(monday, friday)
        return days[-1] if days else None

    this_fri = last_trading_day_in_week_of(target)
    next_fri = last_trading_day_in_week_of(target + timedelta(days=7))
    prev_fri = last_trading_day_in_week_of(target - timedelta(days=7))

    def dte_ok(d: Optional[date]) -> bool:
        return d is not None and dte_min <= (d - today).days <= dte_max

    in_range = [d for d in (prev_fri, this_fri, next_fri) if dte_ok(d)]
    if in_range:
        # Highest DTE inside range = safest (less gamma near expiration)
        return max(in_range)

    # Nothing fits — fall back to whichever candidate is closest to target
    all_candidates = [d for d in (prev_fri, this_fri, next_fri) if d is not None]
    if not all_candidates:
        # Pathological edge case (e.g. two consecutive weeks fully closed) —
        # return the nominal Friday of target week even if it isn't a trading day
        return target + timedelta(days=(4 - target.weekday()) % 7)
    return min(all_candidates, key=lambda d: abs((d - target).days))


def is_last_trading_day_before(today: date, expiration: date) -> bool:
    """True if *today* is the last NYSE trading day strictly before *expiration*.

    Replaces the naive "today is Thursday AND expires tomorrow" DTE-safety
    check so the rule still fires when Good Friday pushes expiration to
    Thursday (and today is therefore Wednesday).
    """
    if expiration <= today:
        return False
    days = _valid_days(today, expiration - timedelta(days=1))
    return bool(days) and days[-1] == today
