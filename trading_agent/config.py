"""
Configuration module — loads environment variables and provides
typed, validated settings for the entire trading agent.
"""

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv


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
    liquidity_max_spread: float = 0.05  # reject underlying if bid/ask spread >= X
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


@dataclass
class AppConfig:
    alpaca: AlpacaConfig
    trading: TradingConfig
    logging: LoggingConfig
    intelligence: IntelligenceConfig = None


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
        min_credit_ratio=float(os.getenv("MIN_CREDIT_RATIO", "0.25")),  # lowered for more trades
        max_delta=float(os.getenv("MAX_DELTA", "0.25")),  # increased for more options
        dry_run=os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes"),
        force_market_open=os.getenv("FORCE_MARKET_OPEN", "false").lower() in ("true", "1", "yes"),
        daily_drawdown_limit=float(os.getenv("DAILY_DRAWDOWN_LIMIT", "0.05")),
        max_buying_power_pct=float(os.getenv("MAX_BUYING_POWER_PCT", "0.80")),
        margin_multiplier=float(os.getenv("MARGIN_MULTIPLIER", "2.0")),
        liquidity_max_spread=float(os.getenv("LIQUIDITY_MAX_SPREAD", "0.05")),
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
    )

    return AppConfig(alpaca=alpaca, trading=trading, logging=logging_cfg,
                     intelligence=intelligence)
