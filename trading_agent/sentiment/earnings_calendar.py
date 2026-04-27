"""
Earnings Calendar — authoritative event_risk short-circuit
===========================================================

Why it exists
-------------
The sentiment pipeline's verifier is the final guard against
hallucinated event_risk claims (e.g. FinGPT inventing an earnings
date that isn't in the evidence).  But there is a more authoritative
source for scheduled earnings than either FinGPT or the reasoning
verifier: the issuer's own published calendar.  If tomorrow is an
earnings day for ``ticker``, no amount of news-layer reasoning should
override that — it's a binary, known-with-certainty catalyst.

This module queries Yahoo Finance's ``get_earnings_dates`` (already a
transitive dependency via yfinance) once per day and caches the
ticker → next_earnings_date mapping.  Consumers ask:

    has_earnings_within(ticker, days) → bool
    days_until_earnings(ticker)       → int | None

When ``has_earnings_within`` returns True within the pipeline's
lookahead, the pipeline short-circuits the FinGPT + verifier stages
and returns a deterministic high-event_risk verdict.  No LLM call,
no hallucination surface, and the answer is identical across cycles.

Design notes
------------
• Cache is in-process, keyed by ticker, refreshed every N hours
  (default 12h — twice per trading day).  Misses gracefully degrade
  to "unknown" (pipeline runs full LLM chain, as before).
• yfinance failures are logged and cached with an empty result so we
  don't hammer the upstream on every cycle when it's rate-limiting.
• The calendar is intentionally *advisory* for recommendation logic:
  we only use it to gate event_risk.  The LLM analyst still sees the
  full news set and can reason about the catalyst qualitatively.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known non-issuer tickers (ETFs / indices / treasuries) have no scheduled
# earnings by construction.  Calling yfinance.get_earnings_dates on them
# returns loud "No fundamentals data found" 404s that pollute the console
# and waste an HTTP round-trip per refresh cycle.  Pre-filter cheaply.
#
# Not exhaustive — unknown ETFs still hit yfinance, fail gracefully, and
# cache an empty result.  This set covers the symbols the agent and
# backtester touch most (broad-market, sectors, vol, treasuries).
# ---------------------------------------------------------------------------
_NON_ISSUER_TICKERS: Set[str] = {
    # Broad-market ETFs
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "IVV", "VEA", "VWO", "EFA",
    "EEM", "AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "EMB",
    # Sector ETFs (XL*)
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE",
    "XLC", "SMH", "XBI", "XOP", "XME", "XRT",
    # Levered / inverse / vol
    "TQQQ", "SQQQ", "SPXL", "SPXS", "UPRO", "SPXU", "VXX", "UVXY", "SVXY",
    # Commodity / currency
    "GLD", "SLV", "USO", "UNG", "UUP", "FXE", "FXY",
    # Common indices that may end up in symbol lists
    "SPX", "NDX", "DJX", "RUT", "VIX",
}


@dataclass
class EarningsEntry:
    ticker: str
    next_date: Optional[date]           # None when no upcoming earnings found
    fetched_at: float                   # monotonic timestamp of last fetch


class EarningsCalendar:
    """
    Lightweight earnings-date oracle backed by yfinance.

    Thread-safe.  Designed to be called from the sentiment pipeline
    worker pool — one cached entry per ticker, refreshed hourly/daily
    per config.
    """

    def __init__(
        self,
        refresh_hours: int = 12,
        lookahead_days: int = 7,
        enabled: bool = True,
    ):
        self._refresh_seconds = max(60, int(refresh_hours) * 3600)
        self._lookahead_days = max(1, int(lookahead_days))
        self._enabled = bool(enabled)
        self._store: Dict[str, EarningsEntry] = {}
        self._lock = threading.Lock()

        if self._enabled:
            logger.info(
                "EarningsCalendar ready (lookahead=%dd, refresh=%dh)",
                self._lookahead_days, refresh_hours,
            )
        else:
            logger.info("EarningsCalendar DISABLED")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def lookahead_days(self) -> int:
        return self._lookahead_days

    def days_until_earnings(self, ticker: str) -> Optional[int]:
        """
        Return the number of days until the next scheduled earnings
        date, or ``None`` if unknown / no upcoming date in the
        catalogue.
        """
        if not self._enabled:
            return None

        entry = self._get_entry(ticker)
        if entry is None or entry.next_date is None:
            return None
        today = datetime.now(timezone.utc).date()
        delta = (entry.next_date - today).days
        return delta if delta >= 0 else None

    def has_earnings_within(self, ticker: str, days: Optional[int] = None) -> bool:
        """True when ticker has scheduled earnings within ``days``."""
        window = days if days is not None else self._lookahead_days
        d = self.days_until_earnings(ticker)
        return d is not None and d <= window

    def clear(self, ticker: Optional[str] = None) -> None:
        with self._lock:
            if ticker is None:
                self._store.clear()
            else:
                self._store.pop(ticker, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_entry(self, ticker: str) -> Optional[EarningsEntry]:
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(ticker)
            if entry and now - entry.fetched_at < self._refresh_seconds:
                return entry

        # Refresh outside the lock — yfinance I/O can take seconds
        fetched = self._fetch(ticker)
        with self._lock:
            self._store[ticker] = fetched
        return fetched

    def _fetch(self, ticker: str) -> EarningsEntry:
        """Call yfinance for next earnings date.  Never raises."""
        next_date: Optional[date] = None

        # Short-circuit ETFs/indices/treasuries — yfinance 404s loudly
        # for these and the answer is deterministically "no earnings".
        if ticker.upper() in _NON_ISSUER_TICKERS:
            logger.debug(
                "[%s] skipping earnings lookup — known non-issuer (ETF/index)",
                ticker,
            )
            return EarningsEntry(
                ticker=ticker,
                next_date=None,
                fetched_at=time.monotonic(),
            )

        try:
            import yfinance as yf
            tk = yf.Ticker(ticker)
            # yfinance exposes `get_earnings_dates(limit=N)` returning a
            # DataFrame indexed by timestamp.  Pull a few ahead in case
            # the nearest is stale.  Silence yfinance's own HTTP-error
            # logger for the duration of the call so transient 404s on
            # unknown tickers don't pollute the console.
            yf_logger = logging.getLogger("yfinance")
            prev_level = yf_logger.level
            yf_logger.setLevel(logging.ERROR + 1)  # effectively silent
            try:
                df = tk.get_earnings_dates(limit=6)
            except Exception:
                df = None
            finally:
                yf_logger.setLevel(prev_level)
            today = datetime.now(timezone.utc).date()
            if df is not None and not df.empty:
                for idx in df.index:
                    # Index entries are pandas Timestamps; normalise to date
                    try:
                        d = idx.date()
                    except AttributeError:
                        continue
                    if d >= today:
                        next_date = d
                        break

            # Fallback: check the lighter-weight `calendar` property
            # (also yfinance — silence its logger again across the call).
            if next_date is None:
                yf_logger.setLevel(logging.ERROR + 1)
                try:
                    cal = getattr(tk, "calendar", None)
                except Exception:
                    cal = None
                finally:
                    yf_logger.setLevel(prev_level)
                if cal is not None:
                    # yfinance returns either a dict {"Earnings Date": [...] }
                    # or a small DataFrame depending on version.
                    dates = []
                    if isinstance(cal, dict):
                        raw = cal.get("Earnings Date") or cal.get("EarningsDate")
                        if raw:
                            if isinstance(raw, list):
                                dates.extend(raw)
                            else:
                                dates.append(raw)
                    else:
                        try:
                            row = cal.loc["Earnings Date"].tolist()
                            dates.extend(row)
                        except Exception:
                            pass
                    for raw in dates:
                        try:
                            d = (
                                raw.date() if hasattr(raw, "date")
                                else datetime.fromisoformat(str(raw)).date()
                            )
                        except Exception:
                            continue
                        if d >= today:
                            next_date = d
                            break

        except ImportError:
            logger.debug("yfinance not installed — earnings calendar inert")
        except Exception as exc:
            logger.debug(
                "[%s] earnings calendar lookup failed: %s", ticker, exc,
            )

        return EarningsEntry(
            ticker=ticker,
            next_date=next_date,
            fetched_at=time.monotonic(),
        )


__all__ = ["EarningsCalendar", "EarningsEntry"]
