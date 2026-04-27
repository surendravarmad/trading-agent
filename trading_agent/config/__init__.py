"""
Configuration package — loads environment variables and provides
typed, validated settings for the entire trading agent.

Re-exports from submodules so all existing imports continue to work:
    from trading_agent.config import AppConfig, load_config
    from trading_agent.config import TradingRulesConfig, load_trading_rules
    from trading_agent.config.loader import StrategyRules, RegimeRules, ...
"""

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

from trading_agent.market.market_profile import MarketProfile, US_MARKET_PROFILE
from trading_agent.config.loader import (
    TradingRulesConfig,
    StrategyRules,
    RegimeRules,
    PositionMonitorRules,
    AgentRules,
    ExecutionRules,
    CacheRules,
    SentimentRules,
    BacktestRules,
    load_trading_rules,
)


@dataclass(frozen=True)
class AlpacaConfig:
    api_key: str
    secret_key: str
    base_url: str
    data_url: str


@dataclass(frozen=True)
class TradingConfig:
    tickers: List[str]
    mode: str                   # "live" or "dry_run"
    max_risk_pct: float         # max loss per trade as fraction of account
    min_credit_ratio: float     # minimum credit / spread-width
    max_delta: float            # sold-strike delta ceiling
    dry_run: bool
    force_market_open: bool = False    # force market open for testing after hours
    daily_drawdown_limit: float = 0.05  # kill process if account drops >X% in one day
    max_buying_power_pct: float = 0.80  # enter liquidation mode if BP >X% used
    margin_multiplier: float = 2.0      # broker's buying power multiplier (2.0 = standard margin, 1.0 = cash)
    liquidity_max_spread: float = 0.05  # absolute floor of the bid/ask gate ($)
    liquidity_bps_of_mid: float = 0.0005  # slope of the bid/ask gate (5 bps × mid)
    stale_spread_pct: float = 0.01       # rel-spread above which the quote is treated as stale (soft-pass)
    schedule_interval: str = "5m"       # cycle interval (for startup log / docs)


@dataclass(frozen=True)
class LoggingConfig:
    log_level: str
    log_dir: str
    trade_plan_dir: str


@dataclass(frozen=True)
class IntelligenceConfig:
    enabled: bool                   # Enable LLM analyst layer
    llm_provider: str               # "ollama", "lmstudio", "openai"
    llm_base_url: str
    llm_model: str
    llm_embedding_model: str
    llm_api_key: str
    llm_temperature: float
    journal_dir: str
    knowledge_base_dir: str
    # ------------------------------------------------------------------
    # Per-role LLM tuning (analyst / fingpt / verifier)
    # ------------------------------------------------------------------
    # The three LLM callers have different temperature / context / timeout
    # profiles and historically each hard-coded its own LLMConfig inside
    # the module.  Centralising here means one ``.env`` controls all of
    # them and a future adapter swap touches a single place.
    analyst_max_tokens: int = 2048
    analyst_timeout: int = 60
    # FinGPT sentiment analyser (second-eye news layer)
    fingpt_enabled: bool = False
    fingpt_model: str = "qwen2.5-trading"   # swap to any Ollama-pulled FinGPT GGUF
    fingpt_news_limit: int = 10
    fingpt_cache_ttl: int = 300              # seconds; matches intraday TTL
    fingpt_temperature: float = 0.1          # deterministic scoring
    fingpt_max_tokens: int = 512             # short JSON response
    fingpt_timeout: int = 45
    # Multi-source news aggregator
    news_sources: str = "yahoo,sec_edgar,fed_rss"   # comma-separated source keys
    news_lookback_hours: int = 24
    news_max_items_per_source: int = 20
    news_cache_ttl: int = 240                # 4 min — stretched from 300s so
                                              # that back-to-back 5-min cycles
                                              # hit cache despite timing jitter
    # Reddit credentials (PRAW) — optional; skip Reddit if absent
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "TradingAgent/1.0"
    # Twitter / X credentials — optional; skip Twitter if absent
    twitter_bearer_token: str = ""
    # Sentiment verifier — reasoning model that cross-checks FinGPT output
    verifier_enabled: bool = False
    verifier_provider: str = "ollama"       # "ollama" | "anthropic"
    verifier_model: str = "qwq:32b"        # local: qwq:32b, deepseek-r1:32b; cloud: claude-sonnet-4-6
    verifier_api_key: str = ""             # Anthropic API key when verifier_provider=anthropic
    verifier_temperature: float = 0.15      # low but non-zero — reasoning models benefit
    verifier_max_tokens: int = 2048
    verifier_timeout: int = 90              # reasoning is slower
    # Tier-0 content-hash cache (skip LLM entirely when news unchanged)
    sentiment_hash_cache_size: int = 32     # per-ticker LRU cap
    # Earnings calendar — authoritative short-circuit for event_risk
    earnings_calendar_enabled: bool = True
    earnings_calendar_lookahead_days: int = 7
    earnings_calendar_refresh_hours: int = 12   # refresh twice per trading day
    # Source authority overrides (JSON object mapping source → weight)
    # Leave blank to use the defaults in news_aggregator.SOURCE_WEIGHTS.
    news_source_weights_json: str = ""


@dataclass
class AppConfig:
    alpaca: AlpacaConfig
    trading: TradingConfig
    logging: LoggingConfig
    intelligence: IntelligenceConfig = None
    # Market-level constants (timezone, session boundaries, trading-day
    # oracle, annualisation, contract multiplier).  Defaults to the US
    # NYSE profile.  Moving these out of module scope is the week 5-6
    # vendor-agnostic seam — a future venue swap replaces this field
    # without touching core strategy / risk code.
    market_profile: MarketProfile = field(default_factory=lambda: US_MARKET_PROFILE)
    # Trader-tunable algorithm parameters loaded from trading_rules.yaml
    rules: TradingRulesConfig = field(default_factory=TradingRulesConfig)


def load_config(env_path: str = None) -> AppConfig:
    """Load configuration from .env file and environment variables."""
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

    alpaca = AlpacaConfig(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2"),
        data_url=os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets/v2"),
    )

    tickers_raw = os.getenv("TICKERS", "SPY,QQQ")
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    trading = TradingConfig(
        tickers=tickers,
        mode=os.getenv("MODE", "dry_run"),
        max_risk_pct=float(os.getenv("MAX_RISK_PCT", "0.02")),
        # Defaults reflect the README design intent (premium-rich, ~80% POP).
        # Override per-deployment via .env if you want a looser fill policy.
        min_credit_ratio=float(os.getenv("MIN_CREDIT_RATIO", "0.33")),
        max_delta=float(os.getenv("MAX_DELTA", "0.20")),
        dry_run=os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes"),
        force_market_open=os.getenv("FORCE_MARKET_OPEN", "false").lower() in ("true", "1", "yes"),
        daily_drawdown_limit=float(os.getenv("DAILY_DRAWDOWN_LIMIT", "0.05")),
        max_buying_power_pct=float(os.getenv("MAX_BUYING_POWER_PCT", "0.80")),
        margin_multiplier=float(os.getenv("MARGIN_MULTIPLIER", "2.0")),
        liquidity_max_spread=float(os.getenv("LIQUIDITY_MAX_SPREAD", "0.05")),
        liquidity_bps_of_mid=float(os.getenv("LIQUIDITY_BPS_OF_MID", "0.0005")),
        stale_spread_pct=float(os.getenv("STALE_SPREAD_PCT", "0.01")),
        schedule_interval=os.getenv("SCHEDULE_INTERVAL", "5m"),
    )

    logging_cfg = LoggingConfig(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_dir=os.getenv("LOG_DIR", "logs"),
        trade_plan_dir=os.getenv("TRADE_PLAN_DIR", "trade_plans"),
    )

    intelligence = IntelligenceConfig(
        enabled=os.getenv("LLM_ENABLED", "false").lower() in ("true", "1", "yes"),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434"),
        llm_model=os.getenv("LLM_MODEL", "mistral"),
        llm_embedding_model=os.getenv("LLM_EMBEDDING_MODEL", "nomic-embed-text"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        journal_dir=os.getenv("TRADE_JOURNAL_DIR", "trade_journal"),
        knowledge_base_dir=os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base"),
        analyst_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        analyst_timeout=int(os.getenv("LLM_TIMEOUT", "60")),
        fingpt_enabled=os.getenv("FINGPT_ENABLED", "false").lower() in ("true", "1", "yes"),
        fingpt_model=os.getenv("FINGPT_MODEL", "qwen2.5-trading"),
        fingpt_news_limit=int(os.getenv("FINGPT_NEWS_LIMIT", "10")),
        fingpt_cache_ttl=int(os.getenv("FINGPT_CACHE_TTL", "300")),
        fingpt_temperature=float(os.getenv("FINGPT_TEMPERATURE", "0.1")),
        fingpt_max_tokens=int(os.getenv("FINGPT_MAX_TOKENS", "512")),
        fingpt_timeout=int(os.getenv("FINGPT_TIMEOUT", "45")),
        news_sources=os.getenv("NEWS_SOURCES", "yahoo,sec_edgar,fed_rss"),
        news_lookback_hours=int(os.getenv("NEWS_LOOKBACK_HOURS", "24")),
        news_max_items_per_source=int(os.getenv("NEWS_MAX_ITEMS_PER_SOURCE", "20")),
        news_cache_ttl=int(os.getenv("NEWS_CACHE_TTL", "240")),
        reddit_client_id=os.getenv("REDDIT_CLIENT_ID", ""),
        reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT", "TradingAgent/1.0"),
        twitter_bearer_token=os.getenv("TWITTER_BEARER_TOKEN", ""),
        verifier_enabled=os.getenv("VERIFIER_ENABLED", "false").lower() in ("true", "1", "yes"),
        verifier_provider=os.getenv("VERIFIER_PROVIDER", "ollama"),
        verifier_model=os.getenv("VERIFIER_MODEL", "qwq:32b"),
        verifier_api_key=os.getenv("VERIFIER_API_KEY", ""),
        verifier_temperature=float(os.getenv("VERIFIER_TEMPERATURE", "0.15")),
        verifier_max_tokens=int(os.getenv("VERIFIER_MAX_TOKENS", "2048")),
        verifier_timeout=int(os.getenv("VERIFIER_TIMEOUT", "90")),
        sentiment_hash_cache_size=int(os.getenv("SENTIMENT_HASH_CACHE_SIZE", "32")),
        earnings_calendar_enabled=os.getenv("EARNINGS_CALENDAR_ENABLED", "true").lower() in ("true", "1", "yes"),
        earnings_calendar_lookahead_days=int(os.getenv("EARNINGS_CALENDAR_LOOKAHEAD_DAYS", "7")),
        earnings_calendar_refresh_hours=int(os.getenv("EARNINGS_CALENDAR_REFRESH_HOURS", "12")),
        news_source_weights_json=os.getenv("NEWS_SOURCE_WEIGHTS_JSON", ""),
    )

    return AppConfig(alpaca=alpaca, trading=trading, logging=logging_cfg,
                     intelligence=intelligence, rules=load_trading_rules())


__all__ = [
    "AppConfig", "AlpacaConfig", "TradingConfig", "LoggingConfig",
    "IntelligenceConfig", "load_config",
    "TradingRulesConfig", "StrategyRules", "RegimeRules",
    "PositionMonitorRules", "AgentRules", "ExecutionRules",
    "CacheRules", "SentimentRules", "BacktestRules", "load_trading_rules",
]
