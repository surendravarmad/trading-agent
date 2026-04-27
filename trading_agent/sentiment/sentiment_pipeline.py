"""
Sentiment Pipeline — tiered orchestration facade
==================================================

The agent's sentiment chain is expensive (news aggregation + FinGPT
specialist + reasoning verifier) and most 5-minute cycles do not
introduce new evidence.  This module wraps the full chain in a single
object that applies three tiers of gating before spending an LLM call,
while still honouring the "no hallucinations" guarantee:

    Tier 0 (authoritative short-circuit)
      If the earnings calendar reports a scheduled catalyst within the
      configured lookahead, emit a deterministic high-event_risk
      passthrough — no LLM call, no way to invent a conflicting
      opinion.  This is *stricter* than the verifier, not looser.

    Tier 1 (content-hash cache)
      Compute a fingerprint over the aggregated news evidence.  If we
      already produced a VerifiedSentimentReport for exactly this
      evidence set, replay it.  Cache entries are TTL-bounded and only
      contain reports that went through the verifier.

    Tier 2 (full chain)
      NewsAggregator → FinGPT specialist → SentimentVerifier.  Runs
      only when Tier-0/1 missed.  The cache is populated with the
      final *verified* report on the way out.

Lifecycle
---------
The pipeline owns an optional ThreadPoolExecutor for running the
Tier-2 work concurrently with the Phase III/IV path in agent.py.  It
is intended to be used as a context manager so the pool is shut down
deterministically at the end of every cycle — this replaces the
instance-lifetime pool the agent previously held, which never got a
chance to drain before SIGTERM forced a process exit.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, List, Optional

from trading_agent.sentiment.fingpt_analyser import SentimentReport
from trading_agent.sentiment.news_aggregator import NewsItem
from trading_agent.sentiment.sentiment_cache import SentimentHashCache, compute_news_hash
from trading_agent.sentiment.sentiment_verifier import (
    SentimentVerifier,
    VerifiedSentimentReport,
)

if TYPE_CHECKING:
    from trading_agent.config import IntelligenceConfig
    from trading_agent.sentiment.earnings_calendar import EarningsCalendar
    from trading_agent.sentiment.fingpt_analyser import FinGPTAnalyser
    from trading_agent.sentiment.news_aggregator import NewsAggregator

logger = logging.getLogger(__name__)


class SentimentPipeline:
    """
    Facade over (NewsAggregator, FinGPTAnalyser, SentimentVerifier).

    The facade is responsible for:
      • Running the three optional components in order
      • Short-circuiting on an authoritative earnings catalyst
      • Reusing cached verified reports when the evidence is unchanged
      • Exposing a single ``analyse()`` call for the agent
      • Scoping its background pool to a cycle via ``__enter__/__exit__``
    """

    def __init__(
        self,
        cfg: "IntelligenceConfig",
        news_aggregator: Optional["NewsAggregator"] = None,
        fingpt: Optional["FinGPTAnalyser"] = None,
        verifier: Optional[SentimentVerifier] = None,
        earnings_calendar: Optional["EarningsCalendar"] = None,
        hash_cache: Optional[SentimentHashCache] = None,
    ):
        self.cfg = cfg
        self.news_aggregator = news_aggregator
        self.fingpt = fingpt
        self.verifier = verifier
        self.earnings_calendar = earnings_calendar

        # Hash-cache TTL is the longer of news cache TTL and fingpt
        # cache TTL — we want evidence-level and sentiment-level
        # reuse windows aligned.
        cache_ttl = max(cfg.news_cache_ttl, cfg.fingpt_cache_ttl)
        self.cache = hash_cache or SentimentHashCache(
            max_size=cfg.sentiment_hash_cache_size,
            ttl_seconds=cache_ttl,
        )

        # Cycle-scoped pool — created in __enter__, shut down in __exit__.
        self._pool: Optional[ThreadPoolExecutor] = None

        self._enabled = bool(fingpt or news_aggregator or earnings_calendar)
        if self._enabled:
            parts = []
            if news_aggregator: parts.append("news")
            if fingpt:          parts.append("fingpt")
            if verifier:        parts.append("verifier")
            if earnings_calendar: parts.append("earnings")
            logger.info("SentimentPipeline wired: %s", ",".join(parts))
        else:
            logger.info("SentimentPipeline inert (no components enabled)")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "SentimentPipeline":
        if self._enabled and self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="sentiment",
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._pool is not None:
            # wait=True ensures in-flight tasks finish cleanly so their
            # cache inserts complete before the pool dies.  This is the
            # fix for the week 3-4 regression where ``_fingpt_pool`` was
            # never drained.
            self._pool.shutdown(wait=True)
            self._pool = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self, ticker: str, regime: str, current_price: float,
        rsi: float, iv_rank: float, strategy_name: str,
    ) -> Optional[Future]:
        """
        Schedule ``analyse()`` on the background pool; returns a Future
        whose result is the VerifiedSentimentReport (or None).

        When the pipeline is inert, returns None and the caller should
        proceed without sentiment.
        """
        if not self._enabled or self._pool is None:
            return None
        return self._pool.submit(
            self.analyse, ticker, regime, current_price,
            rsi, iv_rank, strategy_name,
        )

    def analyse(
        self, ticker: str, regime: str, current_price: float,
        rsi: float, iv_rank: float, strategy_name: str,
    ) -> Optional[VerifiedSentimentReport]:
        """Synchronous execution of the three-tier pipeline."""
        # Tier 0 — authoritative earnings short-circuit
        ec_report = self._earnings_short_circuit(ticker)
        if ec_report is not None:
            return ec_report

        # Evidence ingestion (needed by Tier 1 hash and Tier 2 chain)
        news_items: List[NewsItem] = []
        if self.news_aggregator is not None:
            try:
                news_items = self.news_aggregator.fetch_all(ticker)
            except Exception as exc:
                logger.warning("[%s] NewsAggregator failed: %s", ticker, exc)

        # Tier 1 — content-hash cache
        news_hash = compute_news_hash(ticker, news_items) if news_items else ""
        if news_hash:
            cached = self.cache.get(ticker, news_hash)
            if cached is not None:
                return cached

        # Tier 2 — FinGPT specialist
        sentiment = self._run_fingpt(
            ticker, news_items, regime, current_price, rsi, iv_rank,
            strategy_name,
        )
        if sentiment is None:
            return None

        # Tier 2 — reasoning verifier (always runs if FinGPT produced output)
        verified = self._run_verifier(sentiment, news_items)

        # Populate the hash cache with the verified (not raw) result
        if verified is not None and news_hash:
            self.cache.put(ticker, news_hash, verified)
        return verified

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _earnings_short_circuit(
        self, ticker: str,
    ) -> Optional[VerifiedSentimentReport]:
        """Return a deterministic high-risk passthrough if earnings loom."""
        if self.earnings_calendar is None:
            return None
        try:
            days = self.earnings_calendar.days_until_earnings(ticker)
        except Exception as exc:
            logger.debug(
                "[%s] earnings calendar lookup errored: %s", ticker, exc,
            )
            return None
        if days is None or days > self.earnings_calendar.lookahead_days:
            return None

        logger.info(
            "[%s] earnings in %d day(s) — pipeline short-circuits to avoid",
            ticker, days,
        )
        synthetic = SentimentReport(
            ticker=ticker,
            sentiment_score=0.0,
            event_risk=1.0,
            confidence=1.0,
            headlines=[f"Scheduled earnings in {days} day(s)"],
            key_themes=["scheduled_earnings"],
            recommendation="avoid",
            reasoning=(
                "Authoritative earnings calendar reports a scheduled "
                "announcement within the lookahead window. Credit-spread "
                "premium sellers must avoid binary catalysts."
            ),
        )
        return VerifiedSentimentReport(
            original=synthetic,
            verified_sentiment_score=0.0,
            verified_event_risk=1.0,
            verified_confidence=1.0,
            verified_recommendation="avoid",
            verified_reasoning=synthetic.reasoning,
            evidence_mapping=[],
            hallucination_flags=[],
            agreement_score=1.0,
            confidence_delta=0.0,
            verifier_warnings=[],
            verifier_model="earnings_calendar",
            passthrough=True,
        )

    def _run_fingpt(
        self, ticker: str, news_items: List[NewsItem], regime: str,
        current_price: float, rsi: float, iv_rank: float,
        strategy_name: str,
    ) -> Optional[SentimentReport]:
        if self.fingpt is None:
            return None
        try:
            if news_items:
                return self.fingpt.analyse_items(
                    ticker, news_items, regime, current_price,
                    rsi, iv_rank, strategy_name,
                )
            return self.fingpt.analyse(
                ticker, regime, current_price, rsi, iv_rank, strategy_name,
            )
        except Exception as exc:
            logger.warning("[%s] FinGPT analysis failed: %s", ticker, exc)
            return None

    def _run_verifier(
        self, sentiment: SentimentReport, news_items: List[NewsItem],
    ) -> VerifiedSentimentReport:
        if self.verifier is None:
            return SentimentVerifier._passthrough(sentiment)
        try:
            return self.verifier.verify(sentiment, news_items)
        except Exception as exc:
            logger.warning(
                "[%s] Sentiment verifier failed: %s — passing through",
                sentiment.ticker, exc,
            )
            return SentimentVerifier._passthrough(sentiment)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: "IntelligenceConfig") -> "SentimentPipeline":
        """
        Build a fully wired pipeline from one :class:`IntelligenceConfig`.

        Any component whose prerequisites are missing is silently
        skipped — the facade degrades to a no-op without blocking the
        agent.  Never raises from construction.
        """
        from trading_agent.sentiment.news_aggregator import NewsAggregator
        from trading_agent.sentiment.fingpt_analyser import FinGPTAnalyser
        from trading_agent.sentiment.earnings_calendar import EarningsCalendar
        import json as _json

        # ---- NewsAggregator ---------------------------------------------
        aggregator: Optional[NewsAggregator] = None
        if cfg.fingpt_enabled:
            try:
                sources = {
                    s.strip() for s in cfg.news_sources.split(",") if s.strip()
                }
                if cfg.reddit_client_id and cfg.reddit_client_secret:
                    sources.update({
                        "reddit_wsb", "reddit_stocks",
                        "reddit_options", "reddit_investing",
                    })
                if cfg.twitter_bearer_token:
                    sources.add("twitter")

                overrides = None
                if cfg.news_source_weights_json:
                    try:
                        overrides = _json.loads(cfg.news_source_weights_json)
                    except Exception as exc:
                        logger.warning(
                            "NEWS_SOURCE_WEIGHTS_JSON invalid: %s", exc,
                        )

                aggregator = NewsAggregator(
                    sources=sources,
                    lookback_hours=cfg.news_lookback_hours,
                    max_items_per_source=cfg.news_max_items_per_source,
                    cache_ttl=cfg.news_cache_ttl,
                    reddit_client_id=cfg.reddit_client_id,
                    reddit_client_secret=cfg.reddit_client_secret,
                    reddit_user_agent=cfg.reddit_user_agent,
                    twitter_bearer_token=cfg.twitter_bearer_token,
                    source_weights=overrides,
                )
            except Exception as exc:
                logger.warning("NewsAggregator init failed: %s", exc)

        # ---- FinGPT specialist ------------------------------------------
        fingpt: Optional[FinGPTAnalyser] = None
        if cfg.fingpt_enabled:
            try:
                fingpt = FinGPTAnalyser(cfg=cfg, enabled=True)
            except Exception as exc:
                logger.warning("FinGPTAnalyser init failed: %s", exc)

        # ---- Verifier ----------------------------------------------------
        verifier: Optional[SentimentVerifier] = None
        if cfg.verifier_enabled:
            try:
                verifier = SentimentVerifier(cfg=cfg, enabled=True)
            except Exception as exc:
                logger.warning("SentimentVerifier init failed: %s", exc)

        # ---- Earnings calendar ------------------------------------------
        earnings: Optional[EarningsCalendar] = None
        if cfg.earnings_calendar_enabled:
            try:
                earnings = EarningsCalendar(
                    refresh_hours=cfg.earnings_calendar_refresh_hours,
                    lookahead_days=cfg.earnings_calendar_lookahead_days,
                    enabled=True,
                )
            except Exception as exc:
                logger.warning("EarningsCalendar init failed: %s", exc)

        return cls(
            cfg=cfg,
            news_aggregator=aggregator,
            fingpt=fingpt,
            verifier=verifier,
            earnings_calendar=earnings,
        )


__all__ = ["SentimentPipeline"]
