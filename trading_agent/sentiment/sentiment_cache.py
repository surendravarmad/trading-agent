"""
Sentiment Cache — Tier-0 content-hash gate for the sentiment pipeline
=====================================================================

Why it exists
-------------
Running the full three-stage sentiment pipeline (NewsAggregator →
FinGPT specialist → reasoning-model verifier) on every 5-minute cycle
is both expensive and wasteful when the news didn't actually change.
Yahoo / SEC / Fed refresh cadences are minutes-to-hours, and the Tier-0
cache asks one simple question before spending an LLM call:

    "Have I already produced a verified sentiment report for exactly
     this set of evidence?  If yes, return it.  If no, fall through
     to the LLMs."

Keying
------
The cache key is a deterministic fingerprint of the normalised news
items for a ticker:

    SHA-1 over:  source | form_type | slug | iso-minute-timestamp

so trivial re-orderings or second-level timestamp jitter don't bust
the hash, but a genuine new item does.  We also include the ticker so
two tickers with identical dedupe-slugs (rare, but possible for macro
Fed items) don't collide.

The cache is strictly in-process memory (dict + lock, LRU-capped).  It
does not survive restarts — which is fine, because the first cycle of
a fresh process always wants to produce a fresh verified report.

Safety guarantees
-----------------
  • Anti-hallucination constraint preserved: the *only* thing the cache
    replays is a report that was *already* produced by the full
    pipeline, including the reasoning-model verifier.  It never
    promotes raw FinGPT output to "verified" without the verifier
    having actually run.
  • TTL bound: entries older than ``ttl_seconds`` are discarded on
    access so a stale cache hit can't silently mask a stalled news
    feed on a day the ticker is actually moving.
  • LRU bound: entries evict in insertion order once capacity is hit,
    so long runs can't bloat memory arbitrarily.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Iterable, Optional, Tuple

if TYPE_CHECKING:
    from trading_agent.sentiment.news_aggregator import NewsItem
    from trading_agent.sentiment.sentiment_verifier import VerifiedSentimentReport

logger = logging.getLogger(__name__)


def compute_news_hash(ticker: str, items: Iterable["NewsItem"]) -> str:
    """
    Deterministic SHA-1 fingerprint of the *content* of a news list.

    The function tolerates re-orderings (items are sorted by slug before
    hashing) and sub-minute timestamp jitter (timestamps are truncated
    to the minute).  Bytes outside the slug / source / form_type /
    minute-timestamp are ignored so that surface formatting changes
    (e.g. the aggregator reformatting a title) do not bust the hash.
    """
    h = hashlib.sha1()
    h.update(ticker.encode("utf-8"))

    parts = []
    for it in items:
        ts = ""
        if it.published_at is not None:
            # Minute-level granularity — ignore sub-minute jitter
            ts = it.published_at.strftime("%Y-%m-%dT%H:%M")
        parts.append(
            f"{it.source}|{getattr(it, 'form_type', '')}|{it._slug}|{ts}"
        )
    parts.sort()
    for p in parts:
        h.update(b"\x1e")
        h.update(p.encode("utf-8"))
    return h.hexdigest()


class SentimentHashCache:
    """
    Bounded, TTL + LRU cache of :class:`VerifiedSentimentReport` keyed
    by ``(ticker, news_hash)``.

    Thread-safe: guarded by a single lock.  The pipeline may call
    ``get``/``put`` from a ThreadPoolExecutor worker without races.
    """

    def __init__(self, max_size: int = 32, ttl_seconds: int = 600):
        self._max_size = max(1, int(max_size))
        self._ttl = max(1, int(ttl_seconds))
        self._store: OrderedDict[Tuple[str, str], Tuple["VerifiedSentimentReport", float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        logger.info(
            "SentimentHashCache ready (max=%d entries, ttl=%ds)",
            self._max_size, self._ttl,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get(
        self, ticker: str, news_hash: str,
    ) -> Optional["VerifiedSentimentReport"]:
        """Return the cached report or ``None`` on miss/expiry."""
        key = (ticker, news_hash)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            report, ts = entry
            if time.monotonic() - ts >= self._ttl:
                del self._store[key]
                self._misses += 1
                return None
            # Refresh LRU position
            self._store.move_to_end(key)
            self._hits += 1
            logger.debug(
                "[%s] sentiment cache HIT (hash=%s…)",
                ticker, news_hash[:8],
            )
            return report

    def put(
        self,
        ticker: str,
        news_hash: str,
        report: "VerifiedSentimentReport",
    ) -> None:
        """Insert ``report`` at the freshest LRU slot, evicting oldest."""
        key = (ticker, news_hash)
        with self._lock:
            self._store[key] = (report, time.monotonic())
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug(
                    "sentiment cache EVICT %s (at capacity %d)",
                    evicted_key, self._max_size,
                )

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total else 0.0
            return {
                "size": len(self._store),
                "capacity": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 3),
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


__all__ = ["SentimentHashCache", "compute_news_hash"]
