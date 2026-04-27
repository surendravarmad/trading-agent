"""
Multi-Source News Aggregator
=============================
Pulls financial news from every layer of the information stack and
normalises it into a unified List[NewsItem] for FinGPT to analyse.

Source hierarchy (by authoritative weight):
  1.0  SEC EDGAR 8-K / material-event filings   — no auth, free REST API
  0.95 Federal Reserve press releases / FOMC     — no auth, public RSS
  0.70 Yahoo Finance news                        — no auth, via yfinance
  0.50 Twitter / X cashtag stream               — requires Bearer token (tweepy)
  0.45 Reddit r/options, r/stocks               — requires PRAW credentials
  0.35 Reddit r/wallstreetbets, r/investing      — high noise, strong price signal

Each fetcher is independently fault-tolerant — a failed source is logged
and skipped without aborting the pipeline.

Deduplication: normalised title slugs (first 60 chars, lowercased, punctuation
stripped) collapse cross-source duplicates before the list reaches FinGPT.

TTL cache: per-(ticker, source) with configurable TTL (default 5 min) so
repeated calls within the same trading cycle hit memory, not the network.
"""

import hashlib
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

import requests

logger = logging.getLogger(__name__)

# --- Source weights — seeded from trading_rules.yaml at import time --------
# Importable by name for backward compatibility (fingpt_analyser, tests).
from trading_agent.config.loader import SentimentRules as _SentimentRules
DEFAULT_SOURCE_WEIGHTS: Dict[str, float] = _SentimentRules().source_weights
# Alias for existing imports in fingpt_analyser / tests
SOURCE_WEIGHTS = DEFAULT_SOURCE_WEIGHTS

# --- Cache ----------------------------------------------------------------
# Default TTL when no IntelligenceConfig is supplied.  Real deployments
# flow NEWS_CACHE_TTL through IntelligenceConfig.news_cache_ttl (240 s,
# tuned to hit cache despite 5-minute cycle jitter).
_NEWS_CACHE_TTL = 240


@dataclass
class NewsItem:
    """A single normalised news item from any source."""
    source: str                 # key matching SOURCE_WEIGHTS
    ticker: str
    title: str
    body: str = ""              # snippet / selftext when available
    url: str = ""
    published_at: Optional[datetime] = None
    source_weight: float = 0.5
    author: str = ""
    upvotes: int = 0            # Reddit score; 0 for non-social sources
    form_type: str = ""         # SEC: "8-K", "10-Q", etc.
    # Internal dedupe key — set by aggregator, not callers
    _slug: str = field(default="", repr=False, compare=False)

    def __post_init__(self):
        self._slug = _make_slug(self.title)

    def as_evidence_line(self) -> str:
        """One-line rendering for verifier prompts."""
        ts = (
            self.published_at.strftime("%Y-%m-%d %H:%M UTC")
            if self.published_at else "unknown date"
        )
        weight_tag = f"[w={self.source_weight:.2f}]"
        upvote_tag = f" ({self.upvotes:,} upvotes)" if self.upvotes else ""
        form_tag = f" [{self.form_type}]" if self.form_type else ""
        return (
            f"{weight_tag} [{self.source}{form_tag}] {ts}{upvote_tag}: "
            f"{self.title}"
        )


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _make_slug(title: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]", "", title.lower())[:60].strip()
    return hashlib.md5(cleaned.encode()).hexdigest()


def _parse_rfc822(date_str: str) -> Optional[datetime]:
    """Parse RFC-822 / RSS date strings robustly."""
    try:
        import email.utils
        parsed = email.utils.parsedate_to_datetime(date_str)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _since(hours: int) -> datetime:
    return _utc_now() - timedelta(hours=hours)


# --------------------------------------------------------------------------
# NewsAggregator
# --------------------------------------------------------------------------

class NewsAggregator:
    """
    Parallel multi-source news fetcher with TTL caching and deduplication.

    Enabled sources are controlled via the `sources` parameter (a set of
    source keys). Sources requiring credentials silently skip when the
    corresponding env vars / secrets are absent.
    """

    def __init__(
        self,
        sources: Optional[Set[str]] = None,
        lookback_hours: int = 24,
        max_items_per_source: int = 20,
        cache_ttl: int = _NEWS_CACHE_TTL,
        # Reddit credentials (PRAW)
        reddit_client_id: str = "",
        reddit_client_secret: str = "",
        reddit_user_agent: str = "TradingAgent/1.0",
        # Twitter / X credentials
        twitter_bearer_token: str = "",
        # Optional override of authority weights per source
        source_weights: Optional[Dict[str, float]] = None,
    ):
        self.sources = sources or {
            "yahoo", "sec_edgar", "fed_rss",
            "reddit_wsb", "reddit_stocks", "reddit_options",
        }
        self.lookback_hours = lookback_hours
        self.max_items = max_items_per_source
        self.cache_ttl = cache_ttl
        self.reddit_id = reddit_client_id
        self.reddit_secret = reddit_client_secret
        self.reddit_agent = reddit_user_agent
        self.twitter_token = twitter_bearer_token
        # Per-instance weight map — start from the module default and
        # overlay any caller-supplied overrides so unknown source keys
        # keep their default weight.
        self._weights: Dict[str, float] = dict(DEFAULT_SOURCE_WEIGHTS)
        if source_weights:
            self._weights.update(source_weights)

        # cache: (ticker, source) → (List[NewsItem], epoch)
        self._cache: Dict[Tuple[str, str], Tuple[List[NewsItem], float]] = {}
        self._lock = threading.Lock()

        logger.info(
            "NewsAggregator: sources=%s, lookback=%dh, ttl=%ds",
            sorted(self.sources), lookback_hours, cache_ttl,
        )

    def _weight(self, source: str) -> float:
        """Return the configured authority weight for ``source``."""
        return self._weights.get(source, 0.5)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fetch_all(self, ticker: str) -> List[NewsItem]:
        """
        Fetch from all enabled sources concurrently.
        Returns deduplicated, recency-sorted NewsItem list.
        """
        fetchers = {
            "yahoo":             lambda: self._fetch_yahoo(ticker),
            "sec_edgar":         lambda: self._fetch_sec_edgar(ticker),
            "fed_rss":           lambda: self._fetch_fed_rss(),
            "reddit_wsb":        lambda: self._fetch_reddit(ticker, "wallstreetbets", self._weight("reddit_wsb")),
            "reddit_stocks":     lambda: self._fetch_reddit(ticker, "stocks",          self._weight("reddit_stocks")),
            "reddit_options":    lambda: self._fetch_reddit(ticker, "options",         self._weight("reddit_options")),
            "reddit_investing":  lambda: self._fetch_reddit(ticker, "investing",       self._weight("reddit_investing")),
            "twitter":           lambda: self._fetch_twitter(ticker),
        }

        active = {k: v for k, v in fetchers.items() if k in self.sources}
        all_items: List[NewsItem] = []

        with ThreadPoolExecutor(max_workers=len(active), thread_name_prefix="news") as pool:
            futures = {pool.submit(fn): src for src, fn in active.items()}
            for fut in as_completed(futures):
                src = futures[fut]
                try:
                    items = fut.result(timeout=15)
                    all_items.extend(items)
                    logger.debug("[%s] %s: %d items", ticker, src, len(items))
                except Exception as exc:
                    logger.warning("[%s] %s fetch failed: %s", ticker, src, exc)

        deduped = self._deduplicate(all_items)
        deduped.sort(
            key=lambda i: (
                i.published_at or datetime.min.replace(tzinfo=timezone.utc),
                i.source_weight,
            ),
            reverse=True,
        )
        logger.info(
            "[%s] NewsAggregator: %d items from %d sources (after dedup from %d)",
            ticker, len(deduped), len(active), len(all_items),
        )
        return deduped

    # ------------------------------------------------------------------
    # Per-source fetchers
    # ------------------------------------------------------------------

    def _fetch_yahoo(self, ticker: str) -> List[NewsItem]:
        cached = self._get_cache(ticker, "yahoo")
        if cached is not None:
            return cached

        try:
            import yfinance as yf
            raw = yf.Ticker(ticker).news or []
            items: List[NewsItem] = []
            for entry in raw[: self.max_items]:
                if isinstance(entry, dict):
                    title = (
                        entry.get("title")
                        or entry.get("headline")
                        or entry.get("content", {}).get("title", "")
                    )
                    pub = entry.get("providerPublishTime") or entry.get("published", 0)
                    if pub:
                        pub_dt = datetime.fromtimestamp(int(pub), tz=timezone.utc)
                    else:
                        pub_dt = None
                    url = entry.get("link", "")
                elif hasattr(entry, "title"):
                    title = entry.title
                    pub_dt = None
                    url = getattr(entry, "url", "")
                else:
                    continue
                if not title:
                    continue
                items.append(NewsItem(
                    source="yahoo", ticker=ticker,
                    title=str(title).strip(), url=url,
                    published_at=pub_dt,
                    source_weight=self._weight("yahoo"),
                ))
            self._set_cache(ticker, "yahoo", items)
            return items
        except Exception as exc:
            logger.warning("[%s] Yahoo news fetch failed: %s", ticker, exc)
            return []

    def _fetch_sec_edgar(self, ticker: str) -> List[NewsItem]:
        cached = self._get_cache(ticker, "sec_edgar")
        if cached is not None:
            return cached

        try:
            since = _since(self.lookback_hours).strftime("%Y-%m-%d")
            today = _utc_now().strftime("%Y-%m-%d")
            url = (
                "https://efts.sec.gov/LATEST/search-index"
                f"?q=%22{ticker}%22"
                f"&forms=8-K,10-Q,SC+13G,SC+13D"
                f"&dateRange=custom&startdt={since}&enddt={today}"
            )
            resp = requests.get(
                url,
                headers={"User-Agent": "TradingAgent/1.0 contact@example.com"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            items: List[NewsItem] = []
            for hit in hits[: self.max_items]:
                src = hit.get("_source", {})
                form = src.get("form_type", "")
                entity = src.get("entity_name", ticker)
                file_date = src.get("file_date", "")
                description = src.get("file_description", "")
                title = f"{entity} filed {form}" + (
                    f" — {description}" if description else ""
                )
                pub_dt = None
                if file_date:
                    try:
                        pub_dt = datetime.strptime(file_date, "%Y-%m-%d").replace(
                            tzinfo=timezone.utc
                        )
                    except ValueError:
                        pass
                items.append(NewsItem(
                    source="sec_edgar", ticker=ticker,
                    title=title, form_type=form,
                    published_at=pub_dt,
                    source_weight=self._weight("sec_edgar"),
                    url=src.get("period_of_report", ""),
                ))
            self._set_cache(ticker, "sec_edgar", items)
            return items
        except Exception as exc:
            logger.warning("[%s] SEC EDGAR fetch failed: %s", ticker, exc)
            return []

    def _fetch_fed_rss(self) -> List[NewsItem]:
        """Fetch Federal Reserve press releases (macro context, all tickers)."""
        cached = self._get_cache("__fed__", "fed_rss")
        if cached is not None:
            return cached

        feeds = [
            "https://www.federalreserve.gov/feeds/press_all.xml",
            "https://www.federalreserve.gov/feeds/speeches.xml",
        ]
        since_dt = _since(self.lookback_hours)
        items: List[NewsItem] = []

        for feed_url in feeds:
            try:
                resp = requests.get(feed_url, timeout=10)
                resp.raise_for_status()
                # Parse RSS with stdlib ElementTree
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                root = ET.fromstring(resp.text)

                # Try RSS 2.0 format first
                channel = root.find("channel")
                if channel is not None:
                    for item_el in channel.findall("item")[: self.max_items]:
                        title_el = item_el.find("title")
                        pub_el = item_el.find("pubDate")
                        link_el = item_el.find("link")
                        if title_el is None or not title_el.text:
                            continue
                        pub_dt = _parse_rfc822(pub_el.text) if pub_el is not None and pub_el.text else None
                        if pub_dt and pub_dt < since_dt:
                            continue
                        items.append(NewsItem(
                            source="fed_rss", ticker="MACRO",
                            title=title_el.text.strip(),
                            url=link_el.text.strip() if link_el is not None and link_el.text else "",
                            published_at=pub_dt,
                            source_weight=self._weight("fed_rss"),
                        ))

                # Also try Atom format
                for entry in root.findall("atom:entry", ns)[: self.max_items]:
                    title_el = entry.find("atom:title", ns)
                    pub_el = entry.find("atom:updated", ns) or entry.find("atom:published", ns)
                    link_el = entry.find("atom:link", ns)
                    if title_el is None or not title_el.text:
                        continue
                    pub_dt = None
                    if pub_el is not None and pub_el.text:
                        try:
                            pub_dt = datetime.fromisoformat(
                                pub_el.text.replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass
                    if pub_dt and pub_dt < since_dt:
                        continue
                    url = (link_el.get("href", "") if link_el is not None else "")
                    items.append(NewsItem(
                        source="fed_rss", ticker="MACRO",
                        title=title_el.text.strip(), url=url,
                        published_at=pub_dt,
                        source_weight=self._weight("fed_rss"),
                    ))

            except Exception as exc:
                logger.warning("Fed RSS fetch failed (%s): %s", feed_url, exc)

        self._set_cache("__fed__", "fed_rss", items)
        return items

    def _fetch_reddit(
        self, ticker: str, subreddit: str, weight: float
    ) -> List[NewsItem]:
        if not self.reddit_id or not self.reddit_secret:
            return []

        src_key = f"reddit_{subreddit[:3]}"
        cached = self._get_cache(ticker, src_key)
        if cached is not None:
            return cached

        try:
            import praw
            reddit = praw.Reddit(
                client_id=self.reddit_id,
                client_secret=self.reddit_secret,
                user_agent=self.reddit_agent,
            )
            sub = reddit.subreddit(subreddit)
            since_ts = _since(self.lookback_hours).timestamp()
            items: List[NewsItem] = []
            # Search by ticker symbol first, then cashtag
            for query in [ticker, f"${ticker}"]:
                for post in sub.search(
                    query, sort="new", time_filter="day", limit=self.max_items
                ):
                    if post.created_utc < since_ts:
                        continue
                    pub_dt = datetime.fromtimestamp(
                        post.created_utc, tz=timezone.utc
                    )
                    body = (post.selftext or "")[:300].strip()
                    items.append(NewsItem(
                        source=f"reddit_{subreddit}",
                        ticker=ticker,
                        title=post.title,
                        body=body,
                        url=f"https://reddit.com{post.permalink}",
                        published_at=pub_dt,
                        source_weight=weight,
                        author=str(post.author) if post.author else "",
                        upvotes=post.score,
                    ))
            items = self._deduplicate(items)
            items.sort(key=lambda i: i.upvotes, reverse=True)
            result = items[: self.max_items]
            self._set_cache(ticker, src_key, result)
            return result
        except ImportError:
            logger.debug("praw not installed — Reddit source skipped")
            return []
        except Exception as exc:
            logger.warning("[%s] Reddit r/%s fetch failed: %s", ticker, subreddit, exc)
            return []

    def _fetch_twitter(self, ticker: str) -> List[NewsItem]:
        if not self.twitter_token:
            return []

        cached = self._get_cache(ticker, "twitter")
        if cached is not None:
            return cached

        try:
            import tweepy
            client = tweepy.Client(bearer_token=self.twitter_token, wait_on_rate_limit=False)
            since_id = None
            since_dt = _since(self.lookback_hours)
            query = f"${ticker} lang:en -is:retweet -is:reply"
            resp = client.search_recent_tweets(
                query=query,
                max_results=min(self.max_items, 100),
                tweet_fields=["created_at", "author_id", "public_metrics", "text"],
                start_time=since_dt,
            )
            items: List[NewsItem] = []
            if resp.data:
                for tweet in resp.data:
                    metrics = tweet.public_metrics or {}
                    items.append(NewsItem(
                        source="twitter", ticker=ticker,
                        title=tweet.text[:200],
                        published_at=tweet.created_at,
                        source_weight=self._weight("twitter"),
                        upvotes=metrics.get("like_count", 0),
                    ))
            self._set_cache(ticker, "twitter", items)
            return items
        except ImportError:
            logger.debug("tweepy not installed — Twitter source skipped")
            return []
        except Exception as exc:
            logger.warning("[%s] Twitter fetch failed: %s", ticker, exc)
            return []

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, items: List[NewsItem]) -> List[NewsItem]:
        seen: Set[str] = set()
        unique: List[NewsItem] = []
        for item in items:
            if item._slug not in seen:
                seen.add(item._slug)
                unique.append(item)
        return unique

    # ------------------------------------------------------------------
    # TTL cache
    # ------------------------------------------------------------------

    def _get_cache(
        self, ticker: str, source: str
    ) -> Optional[List[NewsItem]]:
        with self._lock:
            key = (ticker, source)
            entry = self._cache.get(key)
            if not entry:
                return None
            items, ts = entry
            if time.monotonic() - ts < self.cache_ttl:
                return items
            del self._cache[key]
        return None

    def _set_cache(
        self, ticker: str, source: str, items: List[NewsItem]
    ) -> None:
        with self._lock:
            self._cache[(ticker, source)] = (items, time.monotonic())
