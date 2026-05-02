"""
Shared fixtures for the trading agent test suite.
All tests use synthetic data — no real API calls.
"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from trading_agent.config import AppConfig, AlpacaConfig, TradingConfig, LoggingConfig
from trading_agent.market.market_data import MarketDataProvider
from trading_agent.strategy.strategy import SpreadLeg, SpreadPlan


# ------------------------------------------------------------------
# Config fixtures
# ------------------------------------------------------------------

@pytest.fixture
def test_config(tmp_path):
    return AppConfig(
        alpaca=AlpacaConfig(
            api_key="TEST_KEY",
            secret_key="TEST_SECRET",
            base_url="https://paper-api.alpaca.markets/v2",
            data_url="https://data.alpaca.markets/v2",
        ),
        trading=TradingConfig(
            tickers=["SPY", "QQQ"],
            mode="dry_run",
            max_risk_pct=0.02,
            min_credit_ratio=0.33,
            max_delta=0.20,
            dry_run=True,
        ),
        logging=LoggingConfig(
            log_level="DEBUG",
            log_dir=str(tmp_path / "logs"),
            trade_plan_dir=str(tmp_path / "plans"),
        ),
    )


# ------------------------------------------------------------------
# Synthetic price data
# ------------------------------------------------------------------

@pytest.fixture
def bullish_prices():
    """Generate 200 days of prices in a clear uptrend above SMA-200."""
    np.random.seed(42)
    _end = pd.offsets.BDay().rollback(pd.Timestamp.today().normalize())
    dates = pd.bdate_range(end=_end, periods=200)
    base = np.linspace(400, 500, 200)
    noise = np.random.normal(0, 8.0, 200)
    prices = base + noise
    return pd.DataFrame({
        "Open": prices - 0.5,
        "High": prices + 4,
        "Low": prices - 4,
        "Close": prices,
        "Volume": np.random.randint(50_000_000, 150_000_000, 200),
    }, index=dates)


@pytest.fixture
def bearish_prices():
    """Generate 200 days of prices in a clear downtrend below SMA-200."""
    np.random.seed(42)
    _end = pd.offsets.BDay().rollback(pd.Timestamp.today().normalize())
    dates = pd.bdate_range(end=_end, periods=200)
    base = np.linspace(500, 400, 200)
    noise = np.random.normal(0, 8.0, 200)
    prices = base + noise
    return pd.DataFrame({
        "Open": prices - 0.5,
        "High": prices + 4,
        "Low": prices - 4,
        "Close": prices,
        "Volume": np.random.randint(50_000_000, 150_000_000, 200),
    }, index=dates)


@pytest.fixture
def sideways_prices():
    """Generate 200 days of prices oscillating in a narrow range."""
    np.random.seed(42)
    _end = pd.offsets.BDay().rollback(pd.Timestamp.today().normalize())
    dates = pd.bdate_range(end=_end, periods=200)
    base = 450 + 3 * np.sin(np.linspace(0, 8 * np.pi, 200))
    noise = np.random.normal(0, 0.5, 200)
    prices = base + noise
    return pd.DataFrame({
        "Open": prices - 0.3,
        "High": prices + 1,
        "Low": prices - 1,
        "Close": prices,
        "Volume": np.random.randint(50_000_000, 150_000_000, 200),
    }, index=dates)


# ------------------------------------------------------------------
# Mock option contracts
# ------------------------------------------------------------------

@pytest.fixture
def sample_put_contracts():
    """Synthetic put option chain for testing."""
    return [
        {"symbol": "SPY250425P00480000", "strike": 480.0, "bid": 1.20, "ask": 1.40,
         "mid": 1.30, "delta": -0.15, "theta": -0.05, "vega": 0.10, "gamma": 0.01,
         "iv": 0.18, "expiration": "2025-04-25", "type": "put"},
        {"symbol": "SPY250425P00475000", "strike": 475.0, "bid": 0.80, "ask": 1.00,
         "mid": 0.90, "delta": -0.10, "theta": -0.04, "vega": 0.08, "gamma": 0.008,
         "iv": 0.17, "expiration": "2025-04-25", "type": "put"},
        {"symbol": "SPY250425P00470000", "strike": 470.0, "bid": 0.50, "ask": 0.65,
         "mid": 0.575, "delta": -0.07, "theta": -0.03, "vega": 0.06, "gamma": 0.005,
         "iv": 0.16, "expiration": "2025-04-25", "type": "put"},
    ]


@pytest.fixture
def sample_call_contracts():
    """Synthetic call option chain for testing."""
    return [
        {"symbol": "SPY250425C00520000", "strike": 520.0, "bid": 1.10, "ask": 1.30,
         "mid": 1.20, "delta": 0.18, "theta": -0.05, "vega": 0.10, "gamma": 0.01,
         "iv": 0.18, "expiration": "2025-04-25", "type": "call"},
        {"symbol": "SPY250425C00525000", "strike": 525.0, "bid": 0.70, "ask": 0.90,
         "mid": 0.80, "delta": 0.12, "theta": -0.04, "vega": 0.08, "gamma": 0.008,
         "iv": 0.17, "expiration": "2025-04-25", "type": "call"},
        {"symbol": "SPY250425C00530000", "strike": 530.0, "bid": 0.40, "ask": 0.55,
         "mid": 0.475, "delta": 0.08, "theta": -0.03, "vega": 0.06, "gamma": 0.005,
         "iv": 0.16, "expiration": "2025-04-25", "type": "call"},
    ]


# ------------------------------------------------------------------
# Sample spread plan
# ------------------------------------------------------------------

@pytest.fixture
def valid_spread_plan():
    """A spread plan that should pass all risk checks."""
    return SpreadPlan(
        ticker="SPY",
        strategy_name="Bull Put Spread",
        regime="bullish",
        legs=[
            SpreadLeg(symbol="SPY250425P00480000", strike=480.0,
                      action="sell", option_type="put",
                      delta=-0.15, theta=-0.05, bid=1.20, ask=1.40, mid=1.30),
            SpreadLeg(symbol="SPY250425P00475000", strike=475.0,
                      action="buy", option_type="put",
                      delta=-0.10, theta=-0.04, bid=0.80, ask=1.00, mid=0.90),
        ],
        spread_width=5.0,
        net_credit=0.40,
        max_loss=460.0,  # (5.0 - 0.40) * 100
        credit_to_width_ratio=0.08,  # Below threshold — will be adjusted in tests
        expiration="2025-04-25",
        reasoning="Test plan",
    )
