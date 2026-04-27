"""
Unit tests for trading_agent.sentiment_cache
=============================================

These tests cover the Tier-0 content-hash gate that sits in front of
the full sentiment pipeline.  The behaviour we want to lock in:

* ``compute_news_hash`` is deterministic, order-invariant, and ignores
  sub-minute timestamp jitter.
* A genuinely new ``NewsItem`` busts the hash.
* ``SentimentHashCache`` honours TTL expiry, LRU eviction, and keeps
  accurate hit/miss counters for stats().

No LLMs, no I/O — all fixtures are stub dataclasses carrying only the
attributes ``compute_news_hash`` reads, so these tests run in a fresh
sandbox without yfinance / Ollama / Anthropic dependencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from trading_agent.sentiment_cache import SentimentHashCache, compute_news_hash


# ---------------------------------------------------------------------------
# Minimal stand-ins for the types compute_news_hash introspects.
# Keeping them local avoids pulling NewsAggregator / yfinance into the test
# path just to construct a plain value object.
# ---------------------------------------------------------------------------


@dataclass
class _StubNewsItem:
    source: str
    ticker: str
    title: str
    published_at: Optional[datetime] = None
    form_type: str = ""
    _slug: str = ""

    def __post_init__(self) -> None:
        if not self._slug:
            # Mimic news_aggregator._make_slug closely enough for hashing.
            self._slug = self.title.lower().replace(" ", "-")[:60].strip("-")


@dataclass
class _StubReport:
    """Placeholder stand-in for VerifiedSentimentReport — cache only stores it."""
    ticker: str
    note: str = "verified"


def _item(
    source: str = "yahoo",
    title: str = "Fed holds rates steady",
    minute: int = 0,
    form_type: str = "",
    ticker: str = "SPY",
) -> _StubNewsItem:
    ts = datetime(2026, 4, 19, 14, minute, 0, tzinfo=timezone.utc)
    return _StubNewsItem(
        source=source,
        ticker=ticker,
        title=title,
        published_at=ts,
        form_type=form_type,
    )


# ---------------------------------------------------------------------------
# compute_news_hash
# ---------------------------------------------------------------------------


def test_hash_is_deterministic_for_same_inputs():
    items = [_item(title="Alpha report"), _item(source="sec_edgar", title="Beta 8-K")]
    assert compute_news_hash("SPY", items) == compute_news_hash("SPY", items)


def test_hash_is_order_invariant():
    a = _item(title="Alpha report")
    b = _item(source="sec_edgar", title="Beta 8-K", form_type="8-K")
    assert compute_news_hash("SPY", [a, b]) == compute_news_hash("SPY", [b, a])


def test_hash_ignores_sub_minute_jitter():
    ts1 = datetime(2026, 4, 19, 14, 30, 1, tzinfo=timezone.utc)
    ts2 = datetime(2026, 4, 19, 14, 30, 58, tzinfo=timezone.utc)
    i1 = _StubNewsItem(source="yahoo", ticker="SPY", title="x", published_at=ts1)
    i2 = _StubNewsItem(source="yahoo", ticker="SPY", title="x", published_at=ts2)
    assert compute_news_hash("SPY", [i1]) == compute_news_hash("SPY", [i2])


def test_hash_changes_when_minute_changes():
    ts1 = datetime(2026, 4, 19, 14, 30, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 4, 19, 14, 31, 0, tzinfo=timezone.utc)
    i1 = _StubNewsItem(source="yahoo", ticker="SPY", title="x", published_at=ts1)
    i2 = _StubNewsItem(source="yahoo", ticker="SPY", title="x", published_at=ts2)
    assert compute_news_hash("SPY", [i1]) != compute_news_hash("SPY", [i2])


def test_hash_changes_when_new_item_added():
    base = [_item(title="Alpha")]
    plus = base + [_item(title="Beta")]
    assert compute_news_hash("SPY", base) != compute_news_hash("SPY", plus)


def test_hash_changes_when_source_changes():
    i1 = _item(source="yahoo", title="same")
    i2 = _item(source="reddit_wsb", title="same")
    assert compute_news_hash("SPY", [i1]) != compute_news_hash("SPY", [i2])


def test_hash_changes_when_form_type_changes():
    i1 = _item(source="sec_edgar", title="filing", form_type="8-K")
    i2 = _item(source="sec_edgar", title="filing", form_type="10-Q")
    assert compute_news_hash("SPY", [i1]) != compute_news_hash("SPY", [i2])


def test_hash_includes_ticker_prefix():
    items = [_item(title="Generic macro headline")]
    # Same evidence, different ticker → different hash so cross-ticker
    # collisions can't silently replay.
    assert compute_news_hash("SPY", items) != compute_news_hash("QQQ", items)


def test_hash_tolerates_none_published_at():
    i = _StubNewsItem(source="yahoo", ticker="SPY", title="no timestamp")
    i2 = _StubNewsItem(source="yahoo", ticker="SPY", title="no timestamp")
    # Both have published_at=None → same slug/source → stable hash.
    assert compute_news_hash("SPY", [i]) == compute_news_hash("SPY", [i2])


def test_empty_list_is_hashable_and_stable():
    assert compute_news_hash("SPY", []) == compute_news_hash("SPY", [])


# ---------------------------------------------------------------------------
# SentimentHashCache
# ---------------------------------------------------------------------------


def test_miss_then_hit():
    cache = SentimentHashCache(max_size=4, ttl_seconds=60)
    report = _StubReport(ticker="SPY")

    assert cache.get("SPY", "h1") is None
    cache.put("SPY", "h1", report)
    assert cache.get("SPY", "h1") is report

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1
    assert stats["capacity"] == 4
    assert stats["hit_rate"] == 0.5


def test_ticker_isolates_cache_entries():
    cache = SentimentHashCache(max_size=4, ttl_seconds=60)
    spy_report = _StubReport(ticker="SPY")
    qqq_report = _StubReport(ticker="QQQ")

    cache.put("SPY", "h1", spy_report)
    cache.put("QQQ", "h1", qqq_report)

    assert cache.get("SPY", "h1") is spy_report
    assert cache.get("QQQ", "h1") is qqq_report


def test_ttl_expiry_discards_stale_entry(monkeypatch):
    cache = SentimentHashCache(max_size=4, ttl_seconds=10)
    report = _StubReport(ticker="SPY")

    fake_time = [1000.0]
    monkeypatch.setattr(
        "trading_agent.sentiment.sentiment_cache.time.monotonic",
        lambda: fake_time[0],
    )

    cache.put("SPY", "h1", report)
    fake_time[0] += 11  # age past TTL
    assert cache.get("SPY", "h1") is None

    stats = cache.stats()
    assert stats["misses"] == 1
    # Expired entry must have been removed to keep memory bounded.
    assert stats["size"] == 0


def test_lru_eviction_drops_oldest_entry():
    cache = SentimentHashCache(max_size=2, ttl_seconds=60)
    r1 = _StubReport(ticker="A")
    r2 = _StubReport(ticker="B")
    r3 = _StubReport(ticker="C")

    cache.put("A", "h", r1)
    cache.put("B", "h", r2)
    # Touch A so B becomes LRU.
    assert cache.get("A", "h") is r1
    cache.put("C", "h", r3)

    assert cache.get("A", "h") is r1  # still present
    assert cache.get("B", "h") is None  # evicted
    assert cache.get("C", "h") is r3  # freshest


def test_clear_resets_store_but_not_counters():
    cache = SentimentHashCache(max_size=4, ttl_seconds=60)
    cache.put("SPY", "h1", _StubReport(ticker="SPY"))
    cache.get("SPY", "h1")  # bump hits
    cache.clear()
    assert cache.get("SPY", "h1") is None
    stats = cache.stats()
    assert stats["size"] == 0
    # Counters persist — stats() is an observability surface, not cleared on purge.
    assert stats["hits"] >= 1


def test_put_same_key_overwrites():
    cache = SentimentHashCache(max_size=4, ttl_seconds=60)
    old = _StubReport(ticker="SPY", note="old")
    new = _StubReport(ticker="SPY", note="new")
    cache.put("SPY", "h1", old)
    cache.put("SPY", "h1", new)
    assert cache.get("SPY", "h1") is new
    assert cache.stats()["size"] == 1
