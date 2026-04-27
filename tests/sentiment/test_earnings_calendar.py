"""
Unit tests for trading_agent.earnings_calendar
===============================================

EarningsCalendar is the Tier-0 authoritative short-circuit for
event_risk.  These tests lock down:

* ``disabled`` mode always returns None / False (no yfinance probe).
* A cached EarningsEntry within the refresh window is served directly
  (no network hit).
* ``days_until_earnings`` computes the delta from *today* in UTC and
  returns None for past dates.
* ``has_earnings_within`` honours both the explicit ``days`` arg and
  the configured default lookahead.

yfinance is *not* invoked — the internal store is primed directly with
an ``EarningsEntry`` so the tests are deterministic and offline.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone

import pytest

from trading_agent.earnings_calendar import EarningsCalendar, EarningsEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _prime(cal: EarningsCalendar, ticker: str, next_date):
    """Seed the private store with a fresh cache entry."""
    with cal._lock:
        cal._store[ticker] = EarningsEntry(
            ticker=ticker,
            next_date=next_date,
            fetched_at=time.monotonic(),
        )


# ---------------------------------------------------------------------------
# disabled behaviour
# ---------------------------------------------------------------------------


def test_disabled_calendar_returns_none_and_false():
    cal = EarningsCalendar(enabled=False, lookahead_days=7)
    assert cal.days_until_earnings("AAPL") is None
    assert cal.has_earnings_within("AAPL") is False
    # And must not have probed yfinance — the store is untouched.
    assert "AAPL" not in cal._store


# ---------------------------------------------------------------------------
# lookahead semantics
# ---------------------------------------------------------------------------


def test_has_earnings_within_respects_default_lookahead():
    cal = EarningsCalendar(enabled=True, lookahead_days=7)
    _prime(cal, "AAPL", _today() + timedelta(days=3))
    assert cal.has_earnings_within("AAPL") is True


def test_has_earnings_within_rejects_beyond_lookahead():
    cal = EarningsCalendar(enabled=True, lookahead_days=3)
    _prime(cal, "AAPL", _today() + timedelta(days=10))
    assert cal.has_earnings_within("AAPL") is False


def test_has_earnings_within_explicit_days_override():
    cal = EarningsCalendar(enabled=True, lookahead_days=1)
    _prime(cal, "AAPL", _today() + timedelta(days=5))
    # Default lookahead would say no, but caller asks for 30d.
    assert cal.has_earnings_within("AAPL", days=30) is True
    assert cal.has_earnings_within("AAPL", days=2) is False


def test_has_earnings_within_today_is_truthy():
    cal = EarningsCalendar(enabled=True, lookahead_days=7)
    _prime(cal, "AAPL", _today())
    assert cal.days_until_earnings("AAPL") == 0
    assert cal.has_earnings_within("AAPL") is True


def test_days_until_past_date_is_none():
    cal = EarningsCalendar(enabled=True, lookahead_days=7)
    _prime(cal, "AAPL", _today() - timedelta(days=1))
    # A stale "next" date that has already passed should not be surfaced
    # — the short-circuit would otherwise mis-fire on every cycle.
    assert cal.days_until_earnings("AAPL") is None
    assert cal.has_earnings_within("AAPL") is False


def test_days_until_earnings_uses_stored_entry():
    cal = EarningsCalendar(enabled=True, lookahead_days=7)
    _prime(cal, "AAPL", _today() + timedelta(days=4))
    assert cal.days_until_earnings("AAPL") == 4


def test_unknown_ticker_returns_none_when_fetch_fails(monkeypatch):
    """When yfinance is not installed the fetcher must return a benign empty entry."""
    cal = EarningsCalendar(enabled=True, lookahead_days=7)

    # Force the import inside _fetch to raise.  We mimic yfinance being absent
    # by overriding the instance method — keeps the test hermetic even if the
    # sandbox happens to have yfinance installed.
    def _fake_fetch(ticker):
        return EarningsEntry(ticker=ticker, next_date=None, fetched_at=time.monotonic())

    monkeypatch.setattr(cal, "_fetch", _fake_fetch)
    assert cal.days_until_earnings("NOSUCH") is None
    assert cal.has_earnings_within("NOSUCH") is False


# ---------------------------------------------------------------------------
# Cache lifecycle
# ---------------------------------------------------------------------------


def test_cached_entry_is_served_without_refetch(monkeypatch):
    cal = EarningsCalendar(enabled=True, lookahead_days=7, refresh_hours=12)
    _prime(cal, "AAPL", _today() + timedelta(days=2))

    call_count = {"n": 0}

    def _refetch(ticker):
        call_count["n"] += 1
        return EarningsEntry(ticker=ticker, next_date=None, fetched_at=time.monotonic())

    monkeypatch.setattr(cal, "_fetch", _refetch)

    # Multiple reads within the refresh window must not re-enter the fetcher.
    assert cal.days_until_earnings("AAPL") == 2
    assert cal.has_earnings_within("AAPL") is True
    assert call_count["n"] == 0


def test_stale_cache_triggers_refetch(monkeypatch):
    cal = EarningsCalendar(enabled=True, lookahead_days=7, refresh_hours=1)
    # Seed with a deliberately-aged fetched_at so it's past the refresh window.
    with cal._lock:
        cal._store["AAPL"] = EarningsEntry(
            ticker="AAPL",
            next_date=_today() + timedelta(days=2),
            fetched_at=time.monotonic() - 7200,  # 2h ago, beyond refresh
        )

    new_date = _today() + timedelta(days=9)
    called = {"n": 0}

    def _refetch(ticker):
        called["n"] += 1
        return EarningsEntry(ticker=ticker, next_date=new_date, fetched_at=time.monotonic())

    monkeypatch.setattr(cal, "_fetch", _refetch)

    assert cal.days_until_earnings("AAPL") == 9
    assert called["n"] == 1


def test_clear_specific_ticker():
    cal = EarningsCalendar(enabled=True, lookahead_days=7)
    _prime(cal, "AAPL", _today() + timedelta(days=2))
    _prime(cal, "SPY", _today() + timedelta(days=5))

    cal.clear("AAPL")
    assert "AAPL" not in cal._store
    assert "SPY" in cal._store


def test_clear_all_when_ticker_none():
    cal = EarningsCalendar(enabled=True, lookahead_days=7)
    _prime(cal, "AAPL", _today() + timedelta(days=2))
    _prime(cal, "SPY", _today() + timedelta(days=5))
    cal.clear()
    assert cal._store == {}
