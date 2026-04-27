# Backward-compatibility shim — real implementation has moved to the subpackage.
# Delete this file after all callers have been updated to use the new import path.
from trading_agent.sentiment.news_aggregator import *  # noqa: F401,F403
from trading_agent.sentiment.news_aggregator import NewsAggregator, NewsItem, DEFAULT_SOURCE_WEIGHTS, SOURCE_WEIGHTS
