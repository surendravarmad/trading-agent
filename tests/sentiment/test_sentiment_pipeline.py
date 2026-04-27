"""
Unit tests for trading_agent.sentiment_pipeline
=================================================

These tests exercise the three-tier gating contract that was the whole
point of the Gate 3/4 overhaul:

    Tier 0 — authoritative earnings short-circuit
    Tier 1 — verified-report content-hash cache replay
    Tier 2 — full NewsAggregator → FinGPT → Verifier chain

We mock every external component so the tests are hermetic — no
yfinance, no Ollama, no network.  The assertions focus on control
flow: which tier fired, which components were called, and how many
times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from trading_agent.fingpt_analyser import SentimentReport
from trading_agent.sentiment_cache import SentimentHashCache
from trading_agent.sentiment_pipeline import SentimentPipeline
from trading_agent.sentiment_verifier import VerifiedSentimentReport


# ---------------------------------------------------------------------------
# Minimal IntelligenceConfig fake.  We avoid pulling in the real one so the
# test doesn't drag ``pandas_market_calendars`` through config.py's
# ``market_profile`` import in a fresh sandbox.
# ---------------------------------------------------------------------------


@dataclass
class _FakeCfg:
    news_cache_ttl: int = 240
    fingpt_cache_ttl: int = 300
    sentiment_hash_cache_size: int = 8
    # Unused by the paths exercised below, but the dataclass must be
    # truthy and attribute-complete for the pipeline's __init__ checks.
    fingpt_enabled: bool = True
    verifier_enabled: bool = True
    earnings_calendar_enabled: bool = True


# ---------------------------------------------------------------------------
# Stub NewsItem — carries only the attributes compute_news_hash reads.
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
            self._slug = self.title.lower().replace(" ", "-")[:60].strip("-")


def _news(title: str, source: str = "yahoo", minute: int = 0) -> _StubNewsItem:
    return _StubNewsItem(
        source=source,
        ticker="SPY",
        title=title,
        published_at=datetime(2026, 4, 19, 14, minute, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Factory helpers — build mocks with the exact surface the pipeline needs.
# ---------------------------------------------------------------------------


def _make_fingpt(sentiment: Optional[SentimentReport] = None) -> MagicMock:
    fingpt = MagicMock()
    fingpt.analyse_items.return_value = sentiment or SentimentReport(
        ticker="SPY",
        sentiment_score=0.2,
        event_risk=0.3,
        confidence=0.7,
        headlines=["headline A"],
        key_themes=["macro"],
        recommendation="proceed",
        reasoning="positive tone",
    )
    fingpt.analyse.return_value = fingpt.analyse_items.return_value
    return fingpt


def _make_verifier(
    verified: Optional[VerifiedSentimentReport] = None,
) -> MagicMock:
    v = MagicMock()
    v.verify.return_value = verified or VerifiedSentimentReport(
        original=SentimentReport(
            ticker="SPY",
            sentiment_score=0.2,
            event_risk=0.3,
            confidence=0.7,
            headlines=["headline A"],
            key_themes=["macro"],
            recommendation="proceed",
            reasoning="positive tone",
        ),
        verified_sentiment_score=0.15,
        verified_event_risk=0.35,
        verified_confidence=0.65,
        verified_recommendation="proceed",
        verified_reasoning="verifier confirmed",
        evidence_mapping=[],
        hallucination_flags=[],
        agreement_score=0.9,
        confidence_delta=-0.05,
        verifier_warnings=[],
        verifier_model="mock-reasoner",
        passthrough=False,
    )
    return v


def _make_aggregator(items: List[_StubNewsItem]) -> MagicMock:
    agg = MagicMock()
    agg.fetch_all.return_value = list(items)
    return agg


def _make_earnings(days_until: Optional[int], lookahead: int = 7) -> MagicMock:
    ec = MagicMock()
    ec.days_until_earnings.return_value = days_until
    ec.lookahead_days = lookahead
    return ec


# ---------------------------------------------------------------------------
# Tier 0 — authoritative earnings short-circuit
# ---------------------------------------------------------------------------


def test_earnings_short_circuit_skips_llm_chain():
    fingpt = _make_fingpt()
    verifier = _make_verifier()
    aggregator = _make_aggregator([_news("any")])
    earnings = _make_earnings(days_until=2, lookahead=7)

    pipeline = SentimentPipeline(
        cfg=_FakeCfg(),
        news_aggregator=aggregator,
        fingpt=fingpt,
        verifier=verifier,
        earnings_calendar=earnings,
    )

    result = pipeline.analyse("SPY", "bullish", 500.0, 55.0, 40.0, "bull_put")

    assert result is not None
    assert result.verifier_model == "earnings_calendar"
    assert result.verified_event_risk == 1.0
    assert result.verified_recommendation == "avoid"
    assert result.passthrough is True

    # No LLM call — no aggregation, no FinGPT, no verifier.
    aggregator.fetch_all.assert_not_called()
    fingpt.analyse_items.assert_not_called()
    fingpt.analyse.assert_not_called()
    verifier.verify.assert_not_called()


def test_earnings_outside_lookahead_falls_through_to_llm_chain():
    fingpt = _make_fingpt()
    verifier = _make_verifier()
    aggregator = _make_aggregator([_news("alpha")])
    earnings = _make_earnings(days_until=30, lookahead=7)

    pipeline = SentimentPipeline(
        cfg=_FakeCfg(),
        news_aggregator=aggregator,
        fingpt=fingpt,
        verifier=verifier,
        earnings_calendar=earnings,
    )

    result = pipeline.analyse("SPY", "bullish", 500.0, 55.0, 40.0, "bull_put")

    assert result is not None
    assert result.verifier_model != "earnings_calendar"
    fingpt.analyse_items.assert_called_once()
    verifier.verify.assert_called_once()


def test_earnings_calendar_exception_does_not_abort_pipeline():
    fingpt = _make_fingpt()
    verifier = _make_verifier()
    aggregator = _make_aggregator([_news("alpha")])
    earnings = MagicMock()
    earnings.days_until_earnings.side_effect = RuntimeError("yfinance down")
    earnings.lookahead_days = 7

    pipeline = SentimentPipeline(
        cfg=_FakeCfg(),
        news_aggregator=aggregator,
        fingpt=fingpt,
        verifier=verifier,
        earnings_calendar=earnings,
    )

    result = pipeline.analyse("SPY", "bullish", 500.0, 55.0, 40.0, "bull_put")
    assert result is not None
    # Still ran the full chain — a flaky calendar must not gag the agent.
    fingpt.analyse_items.assert_called_once()
    verifier.verify.assert_called_once()


# ---------------------------------------------------------------------------
# Tier 1 — content-hash cache
# ---------------------------------------------------------------------------


def test_cache_replay_avoids_rerunning_llm_chain():
    fingpt = _make_fingpt()
    verifier = _make_verifier()
    items = [_news("alpha"), _news("beta", source="sec_edgar")]
    aggregator = _make_aggregator(items)

    pipeline = SentimentPipeline(
        cfg=_FakeCfg(),
        news_aggregator=aggregator,
        fingpt=fingpt,
        verifier=verifier,
        earnings_calendar=None,
    )

    first = pipeline.analyse("SPY", "bullish", 500.0, 55.0, 40.0, "bull_put")
    second = pipeline.analyse("SPY", "bullish", 500.0, 55.0, 40.0, "bull_put")

    # Same VerifiedSentimentReport instance replayed from cache.
    assert first is second
    # Aggregator still runs (cheap, cached internally) so the hash can
    # be recomputed, but the expensive LLM stages ran exactly once.
    assert aggregator.fetch_all.call_count == 2
    assert fingpt.analyse_items.call_count == 1
    assert verifier.verify.call_count == 1

    stats = pipeline.cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_new_evidence_busts_cache_and_reruns_chain():
    fingpt = _make_fingpt()
    verifier = _make_verifier()

    items_run_1 = [_news("alpha")]
    items_run_2 = [_news("alpha"), _news("beta", source="reddit_wsb")]
    aggregator = MagicMock()
    aggregator.fetch_all.side_effect = [items_run_1, items_run_2]

    pipeline = SentimentPipeline(
        cfg=_FakeCfg(),
        news_aggregator=aggregator,
        fingpt=fingpt,
        verifier=verifier,
        earnings_calendar=None,
    )

    pipeline.analyse("SPY", "bullish", 500.0, 55.0, 40.0, "bull_put")
    pipeline.analyse("SPY", "bullish", 500.0, 55.0, 40.0, "bull_put")

    assert fingpt.analyse_items.call_count == 2
    assert verifier.verify.call_count == 2
    stats = pipeline.cache.stats()
    # First call was a miss; second call was also a miss because the hash
    # changed.  Cache now holds both reports.
    assert stats["misses"] == 2
    assert stats["hits"] == 0
    assert stats["size"] == 2


# ---------------------------------------------------------------------------
# Degraded-component behaviour
# ---------------------------------------------------------------------------


def test_no_fingpt_returns_none():
    aggregator = _make_aggregator([_news("alpha")])
    pipeline = SentimentPipeline(
        cfg=_FakeCfg(),
        news_aggregator=aggregator,
        fingpt=None,
        verifier=_make_verifier(),
        earnings_calendar=None,
    )
    assert pipeline.analyse("SPY", "bullish", 500.0, 55.0, 40.0, "bull_put") is None


def test_no_verifier_falls_back_to_passthrough():
    fingpt = _make_fingpt()
    aggregator = _make_aggregator([_news("alpha")])
    pipeline = SentimentPipeline(
        cfg=_FakeCfg(),
        news_aggregator=aggregator,
        fingpt=fingpt,
        verifier=None,
        earnings_calendar=None,
    )
    result = pipeline.analyse("SPY", "bullish", 500.0, 55.0, 40.0, "bull_put")
    assert result is not None
    # Passthrough echoes FinGPT's scores unchanged and marks itself as such.
    assert result.passthrough is True
    assert result.verified_sentiment_score == pytest.approx(0.2)
    assert result.verified_event_risk == pytest.approx(0.3)


def test_aggregator_failure_still_runs_fingpt_analyse_fallback():
    """When aggregator raises, pipeline falls back to FinGPT's self-fetch path."""
    fingpt = _make_fingpt()
    aggregator = MagicMock()
    aggregator.fetch_all.side_effect = RuntimeError("feed down")

    pipeline = SentimentPipeline(
        cfg=_FakeCfg(),
        news_aggregator=aggregator,
        fingpt=fingpt,
        verifier=_make_verifier(),
        earnings_calendar=None,
    )
    result = pipeline.analyse("SPY", "bullish", 500.0, 55.0, 40.0, "bull_put")
    assert result is not None
    # With no evidence, analyse() is the legacy self-fetch entry point.
    fingpt.analyse.assert_called_once()
    fingpt.analyse_items.assert_not_called()


def test_verifier_exception_falls_back_to_passthrough():
    fingpt = _make_fingpt()
    aggregator = _make_aggregator([_news("alpha")])
    verifier = MagicMock()
    verifier.verify.side_effect = RuntimeError("verifier died")

    pipeline = SentimentPipeline(
        cfg=_FakeCfg(),
        news_aggregator=aggregator,
        fingpt=fingpt,
        verifier=verifier,
        earnings_calendar=None,
    )
    result = pipeline.analyse("SPY", "bullish", 500.0, 55.0, 40.0, "bull_put")
    assert result is not None
    assert result.passthrough is True


# ---------------------------------------------------------------------------
# Cycle-scoped executor lifecycle
# ---------------------------------------------------------------------------


def test_enter_creates_pool_and_exit_shuts_it_down():
    pipeline = SentimentPipeline(
        cfg=_FakeCfg(),
        news_aggregator=_make_aggregator([_news("alpha")]),
        fingpt=_make_fingpt(),
        verifier=_make_verifier(),
        earnings_calendar=None,
    )
    assert pipeline._pool is None
    with pipeline as p:
        assert p is pipeline
        assert pipeline._pool is not None
    assert pipeline._pool is None


def test_submit_returns_none_when_inert():
    # No components wired → pipeline is inert → submit returns None.
    pipeline = SentimentPipeline(
        cfg=_FakeCfg(),
        news_aggregator=None,
        fingpt=None,
        verifier=None,
        earnings_calendar=None,
    )
    with pipeline:
        fut = pipeline.submit("SPY", "bullish", 500.0, 55.0, 40.0, "bull_put")
        assert fut is None


def test_submit_schedules_future_and_returns_report():
    pipeline = SentimentPipeline(
        cfg=_FakeCfg(),
        news_aggregator=_make_aggregator([_news("alpha")]),
        fingpt=_make_fingpt(),
        verifier=_make_verifier(),
        earnings_calendar=None,
    )
    with pipeline:
        fut = pipeline.submit("SPY", "bullish", 500.0, 55.0, 40.0, "bull_put")
        assert fut is not None
        # Must complete without raising and yield a verified report.
        result = fut.result(timeout=5)
        assert result is not None
        assert result.ticker == "SPY"


def test_submit_without_context_manager_returns_none():
    """The pool only exists inside a ``with`` block — callers that forget it degrade safely."""
    pipeline = SentimentPipeline(
        cfg=_FakeCfg(),
        news_aggregator=_make_aggregator([_news("alpha")]),
        fingpt=_make_fingpt(),
        verifier=_make_verifier(),
        earnings_calendar=None,
    )
    assert pipeline.submit("SPY", "bullish", 500.0, 55.0, 40.0, "bull_put") is None
