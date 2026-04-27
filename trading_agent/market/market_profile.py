"""
market_profile — session constants for a specific exchange/market
================================================================

Carries the small bundle of facts the core needs to know about *which*
market it is trading:

  * timezone                 — session clock is in this TZ
  * session open/close times — NYSE regular session + 5-min cron buffers
  * is_trading_day oracle    — pluggable calendar (NYSE via
                              pandas_market_calendars by default)
  * trading_days_per_year    — annualisation constant for Sharpe /
                              drawdown reporting
  * contract_multiplier      — 100 for US equity options
  * currency                 — label for journal entries

Design intent
-------------
Prior to this module, ``market_hours.py`` hardcoded US NYSE constants
at module scope.  Moving them into a dataclass with a US default
creates the seam a future multi-venue build would need (e.g. an
``LSE_MARKET_PROFILE`` for London cash equity), while staying
deliberately US-only per the week 5-6 charter.

Consumers receive the profile through :class:`trading_agent.config.AppConfig`
so there is exactly one source of truth at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time as dtime
from typing import Callable

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from trading_agent.market.calendar_utils import is_trading_day


@dataclass(frozen=True)
class MarketProfile:
    """Immutable constants for a single market / exchange."""

    name: str                                 # human label, e.g. "US/NYSE"
    timezone: ZoneInfo                        # session wall-clock TZ

    # Session boundaries with our standard 5-min cron buffer.  These are
    # *inclusive* open/close bounds; a time equal to open_time or
    # close_time is considered inside the session.
    open_hour: int
    open_minute: int
    close_hour: int
    close_minute: int

    # Trading-day oracle.  ``is_trading_day(date) -> bool`` returns True
    # iff the date is a regular-session trading day (excludes weekends
    # and holidays).  Defaults to NYSE via pandas_market_calendars.
    is_trading_day: Callable[[date], bool] = field(default=is_trading_day)

    # Annualisation constant for Sharpe and drawdown-rate reports.
    trading_days_per_year: int = 252

    # Equity-option contract multiplier (US: 100 shares / contract).
    contract_multiplier: int = 100

    # Currency tag used in journal entries and reporting.
    currency: str = "USD"

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------
    @property
    def open_time(self) -> dtime:
        return dtime(self.open_hour, self.open_minute)

    @property
    def close_time(self) -> dtime:
        return dtime(self.close_hour, self.close_minute)

    @property
    def session_window_str(self) -> str:
        tz_short = self.timezone.key.split("/")[-1].replace("_", " ")
        return (
            f"{self.open_hour:02d}:{self.open_minute:02d}–"
            f"{self.close_hour:02d}:{self.close_minute:02d} {tz_short}"
        )


# ---------------------------------------------------------------------------
# US / NYSE — the only profile we support in week 5-6.
# ---------------------------------------------------------------------------
#
# Open buffer: 09:25 ET  (5 min before the 09:30 NYSE bell)
# Close buffer: 16:05 ET (5 min after the 16:00 NYSE bell)
#
# Buffers exist so a cron job scheduled exactly at the boundary can
# finish the cycle it already started without the next invocation
# being rejected by the market-hours guard.
US_MARKET_PROFILE = MarketProfile(
    name="US/NYSE",
    timezone=ZoneInfo("America/New_York"),
    open_hour=9,
    open_minute=25,
    close_hour=16,
    close_minute=5,
    is_trading_day=is_trading_day,
    trading_days_per_year=252,
    contract_multiplier=100,
    currency="USD",
)


__all__ = ["MarketProfile", "US_MARKET_PROFILE"]
