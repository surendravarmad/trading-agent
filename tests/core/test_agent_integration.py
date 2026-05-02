"""
Integration test — runs the full agent pipeline with mocked external APIs.
Verifies that the two-stage cycle (Monitor → Open) wires together correctly.
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from trading_agent.core.agent import TradingAgent
from trading_agent.config import (AppConfig, AlpacaConfig, TradingConfig,
                                   LoggingConfig, IntelligenceConfig)


def _make_config(tmp_path, tickers=None):
    return AppConfig(
        alpaca=AlpacaConfig(
            api_key="INT_TEST_KEY",
            secret_key="INT_TEST_SECRET",
            base_url="https://paper-api.alpaca.markets/v2",
            data_url="https://data.alpaca.markets/v2",
        ),
        trading=TradingConfig(
            tickers=tickers or ["SPY"],
            mode="dry_run",
            max_risk_pct=0.02,
            min_credit_ratio=0.33,
            max_delta=0.20,
            dry_run=True,
            force_market_open=True,   # bypass market-hours check in tests
        ),
        logging=LoggingConfig(
            log_level="DEBUG",
            log_dir=str(tmp_path / "logs"),
            trade_plan_dir=str(tmp_path / "plans"),
        ),
        intelligence=IntelligenceConfig(
            enabled=False,
            llm_provider="ollama",
            llm_base_url="",
            llm_model="",
            llm_embedding_model="",
            llm_api_key="",
            llm_temperature=0.0,
            journal_dir=str(tmp_path / "journal"),  # isolated per test
            knowledge_base_dir=str(tmp_path / "kb"),
        ),
    )


def _mock_agent(agent, prices, option_chain):
    """Apply standard API mocks to an agent instance."""
    dp = agent.data_provider
    dp.fetch_historical_prices = MagicMock(return_value=prices)
    dp.fetch_option_chain = MagicMock(return_value=option_chain)
    dp.get_account_info = MagicMock(
        return_value={"equity": "100000", "buying_power": "50000"})
    dp.is_market_open = MagicMock(return_value=True)
    # get_current_price is separate from fetch_historical_prices after the
    # TTL cache split — mock it explicitly so classify() gets a real float
    dp.get_current_price = MagicMock(
        return_value=float(prices["Close"].iloc[-1]))
    # New methods added for 5-min optimisation
    dp.fetch_batch_snapshots = MagicMock(return_value={"SPY": 500.0})
    dp.prefetch_historical_parallel = MagicMock()
    # Liquidity check — return a liquid spread (< $0.05) so tests pass by default
    dp.get_underlying_bid_ask = MagicMock(return_value=(499.98, 500.01))
    # Stage 1: no open positions so we skip straight to Stage 2
    agent.position_monitor.fetch_open_positions = MagicMock(return_value=[])


class TestFullPipeline:

    def test_full_cycle_bullish(self, tmp_path, bullish_prices, sample_put_contracts):
        """End-to-end: bullish regime → bull put spread → dry-run plan file."""
        agent = TradingAgent(_make_config(tmp_path))
        _mock_agent(agent, bullish_prices, sample_put_contracts)

        results = agent.run_cycle()

        trades = results["new_trades"]
        assert len(trades) == 1
        r = trades[0]
        assert r["ticker"] == "SPY"
        assert r["regime"] == "bullish"
        assert r["strategy"] == "Bull Put Spread"
        assert r["execution"]["status"] in ("dry_run", "rejected")

        # Plan file persisted using new single-file format
        plan_dir = tmp_path / "plans"
        assert os.path.exists(plan_dir / "trade_plan_SPY.json")
        with open(plan_dir / "trade_plan_SPY.json") as f:
            data = json.load(f)
        assert "state_history" in data
        assert data["ticker"] == "SPY"

    def test_full_cycle_no_account(self, tmp_path):
        """Agent aborts gracefully when account info is unavailable."""
        agent = TradingAgent(_make_config(tmp_path))
        agent.data_provider.get_account_info = MagicMock(return_value=None)

        results = agent.run_cycle()
        # Returns a top-level error dict, not a list
        assert results["status"] == "error"
        assert "account" in results["reason"].lower()

    def test_full_cycle_handles_per_ticker_exception(self, tmp_path, bullish_prices):
        """Agent catches unhandled errors per-ticker without crashing."""
        agent = TradingAgent(_make_config(tmp_path))
        agent.data_provider.get_account_info = MagicMock(
            return_value={"equity": "100000", "buying_power": "50000"})
        agent.data_provider.is_market_open = MagicMock(return_value=True)
        agent.data_provider.fetch_batch_snapshots = MagicMock(return_value={})
        agent.data_provider.prefetch_historical_parallel = MagicMock()
        agent.data_provider.get_underlying_bid_ask = MagicMock(return_value=(499.98, 500.01))
        agent.position_monitor.fetch_open_positions = MagicMock(return_value=[])
        # Make classify throw per ticker
        agent.regime_classifier.classify = MagicMock(
            side_effect=Exception("API timeout"))

        results = agent.run_cycle()
        trades = results["new_trades"]
        assert len(trades) == 1
        assert trades[0]["status"] == "error"
        assert "timeout" in trades[0]["reason"].lower()

    def test_prefetch_called_for_all_tickers(self, tmp_path, bullish_prices,
                                              sample_put_contracts):
        """prefetch_historical_parallel is called with all configured tickers."""
        cfg = _make_config(tmp_path, tickers=["SPY", "QQQ"])
        agent = TradingAgent(cfg)
        _mock_agent(agent, bullish_prices, sample_put_contracts)
        agent.data_provider.fetch_batch_snapshots = MagicMock(return_value={
            "SPY": 500.0, "QQQ": 450.0})

        agent.run_cycle()

        agent.data_provider.prefetch_historical_parallel.assert_called_once_with(
            ["SPY", "QQQ"])

    def test_batch_snapshot_called_for_all_tickers(self, tmp_path, bullish_prices,
                                                    sample_put_contracts):
        """fetch_batch_snapshots is called with all configured tickers."""
        cfg = _make_config(tmp_path, tickers=["SPY", "QQQ"])
        agent = TradingAgent(cfg)
        _mock_agent(agent, bullish_prices, sample_put_contracts)
        agent.data_provider.fetch_batch_snapshots = MagicMock(return_value={
            "SPY": 500.0, "QQQ": 450.0})

        agent.run_cycle()

        agent.data_provider.fetch_batch_snapshots.assert_called_once_with(
            ["SPY", "QQQ"])

    def test_journal_kb_logs_signal_on_dry_run(self, tmp_path, bullish_prices,
                                                sample_put_contracts):
        """JournalKB signals.jsonl is written after each cycle."""
        agent = TradingAgent(_make_config(tmp_path))
        _mock_agent(agent, bullish_prices, sample_put_contracts)

        agent.run_cycle()

        journal_dir = agent.journal_kb.journal_dir
        jsonl_path = os.path.join(journal_dir, "signals.jsonl")
        assert os.path.exists(jsonl_path)
        lines = open(jsonl_path).readlines()
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert record["ticker"] == "SPY"
        assert "action" in record
        assert "raw_signal" in record

    def test_signal_contains_thesis(self, tmp_path, bullish_prices,
                                     sample_put_contracts):
        """JournalKB record includes the trade thesis (why/why_now/exit_plan)."""
        agent = TradingAgent(_make_config(tmp_path))
        _mock_agent(agent, bullish_prices, sample_put_contracts)

        agent.run_cycle()

        jsonl_path = os.path.join(agent.journal_kb.journal_dir, "signals.jsonl")
        record = json.loads(open(jsonl_path).readline())
        thesis = record["raw_signal"].get("thesis", {})
        assert "why" in thesis
        assert "why_now" in thesis
        assert "exit_plan" in thesis


class TestDailyDrawdown:

    def test_daily_state_file_created_on_first_run(self, tmp_path, bullish_prices,
                                                    sample_put_contracts):
        """First cycle of the day writes daily_state.json."""
        agent = TradingAgent(_make_config(tmp_path))
        _mock_agent(agent, bullish_prices, sample_put_contracts)

        agent.run_cycle()

        state_path = tmp_path / "plans" / "daily_state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert "start_equity" in state
        assert "date" in state

    def test_drawdown_circuit_breaker_fires(self, tmp_path, bullish_prices,
                                             sample_put_contracts):
        """When equity drops >5% from day start, _check_daily_drawdown returns True."""
        import datetime as dt
        agent = TradingAgent(_make_config(tmp_path))

        today = dt.date.today().isoformat()
        state_path = tmp_path / "plans" / "daily_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(
            {"date": today, "start_equity": 100_000.0}))

        # Equity has dropped 6% — exceeds 5% limit
        assert agent._check_daily_drawdown(94_000.0) is True

    def test_drawdown_within_limit_passes(self, tmp_path):
        import datetime as dt
        agent = TradingAgent(_make_config(tmp_path))

        today = dt.date.today().isoformat()
        state_path = tmp_path / "plans" / "daily_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(
            {"date": today, "start_equity": 100_000.0}))

        # Equity dropped only 2% — under the 5% limit
        assert agent._check_daily_drawdown(98_000.0) is False

    def test_new_day_resets_baseline(self, tmp_path):
        """State from a previous day is replaced with today's equity."""
        import datetime as dt
        agent = TradingAgent(_make_config(tmp_path))

        yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
        state_path = tmp_path / "plans" / "daily_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(
            {"date": yesterday, "start_equity": 50_000.0}))

        # Even though equity would be -90% vs yesterday, a new day resets
        assert agent._check_daily_drawdown(100_000.0) is False
        new_state = json.loads(state_path.read_text())
        assert new_state["date"] == dt.date.today().isoformat()
        assert new_state["start_equity"] == pytest.approx(100_000.0)


class TestLiquidationMode:

    def test_liquidation_mode_skips_new_trades(self, tmp_path, bullish_prices,
                                                sample_put_contracts):
        """When buying power is exhausted, new trades are skipped."""
        agent = TradingAgent(_make_config(tmp_path))
        _mock_agent(agent, bullish_prices, sample_put_contracts)
        # Only 5% BP remaining (95% used > 80% limit)
        agent.data_provider.get_account_info = MagicMock(
            return_value={"equity": "100000", "buying_power": "5000"})

        results = agent.run_cycle()
        trades = results["new_trades"]
        assert len(trades) == 1
        assert trades[0]["status"] == "skipped"
        assert "liquidation" in trades[0]["reason"].lower()

    def test_sufficient_bp_allows_trades(self, tmp_path, bullish_prices,
                                          sample_put_contracts):
        """With plenty of buying power, trades proceed normally."""
        agent = TradingAgent(_make_config(tmp_path))
        _mock_agent(agent, bullish_prices, sample_put_contracts)
        # 50% BP remaining (75% of initial 2x used < 80% limit)
        agent.data_provider.get_account_info = MagicMock(
            return_value={"equity": "100000", "buying_power": "50000"})

        results = agent.run_cycle()
        trades = results["new_trades"]
        assert len(trades) == 1
        assert trades[0].get("status") != "skipped"

    def test_zero_equity_triggers_liquidation(self, tmp_path):
        """Equity at or below zero triggers emergency liquidation mode."""
        agent = TradingAgent(_make_config(tmp_path))
        assert agent._check_liquidation_mode(equity=0.0, buying_power=0.0) is True
        assert agent._check_liquidation_mode(equity=-1000.0, buying_power=0.0) is True

    def test_fresh_margin_account_no_liquidation(self, tmp_path):
        """A fresh margin account (buying_power = 2x equity, nothing deployed) should NOT trigger."""
        agent = TradingAgent(_make_config(tmp_path))
        # buying_power == 2x equity means 0% deployed
        assert agent._check_liquidation_mode(equity=100_000.0, buying_power=200_000.0) is False
