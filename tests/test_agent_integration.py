"""
Integration test — runs the full agent pipeline with mocked external APIs.
Verifies that the four phases (Perceive → Classify → Plan → Act) wire
together correctly end-to-end.
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from trading_agent.agent import TradingAgent
from trading_agent.config import AppConfig, AlpacaConfig, TradingConfig, LoggingConfig


@pytest.fixture
def integration_config(tmp_path):
    return AppConfig(
        alpaca=AlpacaConfig(
            api_key="INT_TEST_KEY",
            secret_key="INT_TEST_SECRET",
            base_url="https://paper-api.alpaca.markets/v2",
            data_url="https://data.alpaca.markets/v2",
        ),
        trading=TradingConfig(
            tickers=["SPY"],
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


class TestFullPipeline:

    def test_full_cycle_bullish(self, integration_config, bullish_prices,
                                 sample_put_contracts):
        """End-to-end: bullish regime → bull put spread → dry-run log."""
        agent = TradingAgent(integration_config)

        # Mock external calls
        agent.data_provider.fetch_historical_prices = MagicMock(
            return_value=bullish_prices)
        agent.data_provider.fetch_option_chain = MagicMock(
            return_value=sample_put_contracts)
        agent.data_provider.get_account_info = MagicMock(return_value={
            "equity": "100000",
        })
        agent.data_provider.is_market_open = MagicMock(return_value=True)

        results = agent.run_cycle()

        assert len(results) == 1
        r = results[0]
        assert r["ticker"] == "SPY"
        assert r["regime"] == "bullish"
        assert r["strategy"] == "Bull Put Spread"
        assert r["execution"]["status"] in ("dry_run", "rejected")

        # Verify a plan file was created
        plan_dir = integration_config.logging.trade_plan_dir
        plans = os.listdir(plan_dir)
        assert len(plans) >= 1

    def test_full_cycle_no_account(self, integration_config):
        """Agent aborts gracefully when account info is unavailable."""
        agent = TradingAgent(integration_config)
        agent.data_provider.get_account_info = MagicMock(return_value=None)

        results = agent.run_cycle()
        assert results[0]["status"] == "error"
        assert "account" in results[0]["reason"].lower()

    def test_full_cycle_handles_exception(self, integration_config):
        """Agent catches unhandled errors per-ticker without crashing."""
        agent = TradingAgent(integration_config)
        agent.data_provider.get_account_info = MagicMock(return_value={
            "equity": "100000",
        })
        agent.data_provider.is_market_open = MagicMock(return_value=True)
        # Make classify throw an exception
        agent.regime_classifier.classify = MagicMock(
            side_effect=Exception("API timeout"))

        results = agent.run_cycle()
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "timeout" in results[0]["reason"].lower()
