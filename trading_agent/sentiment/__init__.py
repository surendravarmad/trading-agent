"""Sentiment analysis package (3-tier pipeline)."""
from trading_agent.sentiment.sentiment_pipeline import SentimentPipeline
from trading_agent.sentiment.earnings_calendar import EarningsCalendar
from trading_agent.sentiment.sentiment_cache import SentimentHashCache, compute_news_hash
from trading_agent.sentiment.news_aggregator import (
    NewsAggregator, NewsItem, DEFAULT_SOURCE_WEIGHTS, SOURCE_WEIGHTS,
)
from trading_agent.sentiment.fingpt_analyser import FinGPTAnalyser, SentimentReport
from trading_agent.sentiment.sentiment_verifier import (
    SentimentVerifier, VerifiedSentimentReport,
)

__all__ = [
    "SentimentPipeline",
    "EarningsCalendar",
    "SentimentHashCache", "compute_news_hash",
    "NewsAggregator", "NewsItem", "DEFAULT_SOURCE_WEIGHTS", "SOURCE_WEIGHTS",
    "FinGPTAnalyser", "SentimentReport",
    "SentimentVerifier", "VerifiedSentimentReport",
]
