"""Tests for configuration loading."""

import os
import pytest
from trading_agent.config import load_config


def test_load_config_from_env(tmp_path):
    """Config loads all fields from a .env file correctly."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ALPACA_API_KEY=test_key\n"
        "ALPACA_SECRET_KEY=test_secret\n"
        "ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2\n"
        "TICKERS=SPY,QQQ,IWM\n"
        "MODE=dry_run\n"
        "MAX_RISK_PCT=0.03\n"
        "MIN_CREDIT_RATIO=0.30\n"
        "MAX_DELTA=0.25\n"
        "DRY_RUN=true\n"
        "LOG_LEVEL=DEBUG\n"
    )
    cfg = load_config(str(env_file))

    assert cfg.alpaca.api_key == "test_key"
    assert cfg.alpaca.secret_key == "test_secret"
    assert cfg.trading.tickers == ["SPY", "QQQ", "IWM"]
    assert cfg.trading.max_risk_pct == 0.03
    assert cfg.trading.min_credit_ratio == 0.30
    assert cfg.trading.max_delta == 0.25
    assert cfg.trading.dry_run is True
    assert cfg.logging.log_level == "DEBUG"


def test_load_config_defaults(tmp_path, monkeypatch):
    """Config falls back to sane defaults when env vars are missing.

    monkeypatch clears any TICKERS / MAX_RISK_PCT that may be set in the
    project's real .env so the test only sees the code-level defaults.
    """
    for var in ("TICKERS", "MAX_RISK_PCT", "DRY_RUN", "MIN_CREDIT_RATIO",
                "MAX_DELTA", "MODE", "FORCE_MARKET_OPEN"):
        monkeypatch.delenv(var, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text("")  # empty
    cfg = load_config(str(env_file))

    assert cfg.trading.tickers == ["SPY", "QQQ"]
    assert cfg.trading.max_risk_pct == 0.02
    assert cfg.trading.dry_run is True
