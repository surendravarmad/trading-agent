"""
Tests for the after-hours market shutdown guard.

Coverage
--------
- _is_within_market_hours() boundary conditions
  - Before market open (08:59 ET, weekday)
  - Exactly at open boundary (09:25 ET)
  - During market hours (11:00 ET)
  - Exactly at close boundary (16:05 ET)
  - After market close (16:06 ET)
  - Weekend (Saturday / Sunday)
- run_cycle() calls os._exit(0) outside hours
- run_cycle() does NOT call os._exit(0) during hours
- force_market_open=True bypasses the shutdown
"""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

import trading_agent.agent as agent_module
from trading_agent.agent import _is_within_market_hours, TradingAgent
from trading_agent.config import (
    AppConfig, AlpacaConfig, TradingConfig, LoggingConfig, IntelligenceConfig,
)

_ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _et(year, month, day, hour, minute, second=0):
    """Return a timezone-aware datetime in US/Eastern."""
    return datetime(year, month, day, hour, minute, second, tzinfo=_ET)


def _make_config(tmp_path, force_market_open=False):
    return AppConfig(
        alpaca=AlpacaConfig(
            api_key="TEST_KEY",
            secret_key="TEST_SECRET",
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
            force_market_open=force_market_open,
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
            journal_dir=str(tmp_path / "journal"),
            knowledge_base_dir=str(tmp_path / "kb"),
        ),
    )


# ---------------------------------------------------------------------------
# _is_within_market_hours() — time-boundary unit tests
# All dates chosen to be regular weekdays (Wednesday 2026-04-01 is a Wednesday)
# ---------------------------------------------------------------------------

class TestIsWithinMarketHours:

    # Wednesday 2026-04-01 used throughout (not a holiday in the test scope)
    _WED = (2026, 4, 1)

    def _patch(self, dt):
        """Context manager: freeze _EASTERN datetime.now() to `dt`."""
        return patch(
            "trading_agent.agent.datetime",
            **{"now.return_value": dt},
        )

    def test_before_open_returns_false(self):
        # 08:59 ET — one minute before the 09:25 open boundary
        with self._patch(_et(*self._WED, 8, 59)):
            assert _is_within_market_hours() is False

    def test_exactly_at_open_boundary_returns_true(self):
        # 09:25 ET — the open boundary itself must be included
        with self._patch(_et(*self._WED, 9, 25)):
            assert _is_within_market_hours() is True

    def test_mid_session_returns_true(self):
        # 11:00 ET — well within trading hours
        with self._patch(_et(*self._WED, 11, 0)):
            assert _is_within_market_hours() is True

    def test_just_before_close_boundary_returns_true(self):
        # 16:04 ET — one minute before the 16:05 close boundary
        with self._patch(_et(*self._WED, 16, 4)):
            assert _is_within_market_hours() is True

    def test_exactly_at_close_boundary_returns_true(self):
        # 16:05 ET — the close boundary itself must be included
        with self._patch(_et(*self._WED, 16, 5)):
            assert _is_within_market_hours() is True

    def test_one_minute_after_close_returns_false(self):
        # 16:06 ET — one minute past close boundary
        with self._patch(_et(*self._WED, 16, 6)):
            assert _is_within_market_hours() is False

    def test_evening_returns_false(self):
        # 20:00 ET — evening, market long closed
        with self._patch(_et(*self._WED, 20, 0)):
            assert _is_within_market_hours() is False

    def test_midnight_returns_false(self):
        with self._patch(_et(*self._WED, 0, 0)):
            assert _is_within_market_hours() is False

    def test_saturday_returns_false(self):
        # 2026-04-04 is a Saturday, noon ET
        with self._patch(_et(2026, 4, 4, 12, 0)):
            assert _is_within_market_hours() is False

    def test_sunday_returns_false(self):
        # 2026-04-05 is a Sunday, noon ET
        with self._patch(_et(2026, 4, 5, 12, 0)):
            assert _is_within_market_hours() is False

    def test_monday_during_hours_returns_true(self):
        # 2026-04-06 is a Monday
        with self._patch(_et(2026, 4, 6, 10, 30)):
            assert _is_within_market_hours() is True

    def test_friday_during_hours_returns_true(self):
        # 2026-04-03 is a Friday
        with self._patch(_et(2026, 4, 3, 15, 45)):
            assert _is_within_market_hours() is True

    def test_friday_after_close_returns_false(self):
        # 2026-04-03 Friday 16:30 ET — market closed for the week
        with self._patch(_et(2026, 4, 3, 16, 30)):
            assert _is_within_market_hours() is False


# ---------------------------------------------------------------------------
# TradingAgent.run_cycle() — shutdown integration tests
# ---------------------------------------------------------------------------

class TestAfterHoursShutdown:
    """Verify run_cycle() calls os._exit(0) outside hours."""

    def _make_agent(self, tmp_path, force_market_open=False):
        agent = TradingAgent(_make_config(tmp_path, force_market_open=force_market_open))
        # Stub out the JournalKB so we can inspect calls without real file I/O
        agent.journal_kb = MagicMock()
        return agent

    def test_after_close_calls_exit_0(self, tmp_path):
        """After 16:06 ET on a weekday → os._exit(0)."""
        agent = self._make_agent(tmp_path)
        with (
            patch("trading_agent.agent._is_within_market_hours", return_value=False),
            patch("os._exit") as mock_exit,
        ):
            agent.run_cycle()
            mock_exit.assert_called_once_with(0)

    def test_after_close_logs_cycle_error(self, tmp_path):
        """After-hours shutdown must write a journal entry."""
        agent = self._make_agent(tmp_path)
        with (
            patch("trading_agent.agent._is_within_market_hours", return_value=False),
            patch("os._exit"),
        ):
            agent.run_cycle()
            # The after-hours entry must be the first log_cycle_error call.
            # (A second call may follow because the mocked os._exit doesn't
            # actually stop execution, so the cycle continues and may fail
            # on the account-fetch with no API credentials.)
            first_call = agent.journal_kb.log_cycle_error.call_args_list[0]
            assert "after_hours_shutdown" in first_call[0][0]

    def test_during_hours_does_not_call_exit(self, tmp_path, bullish_prices,
                                              sample_put_contracts):
        """During market hours, os._exit should NOT be called for after-hours."""
        agent = self._make_agent(tmp_path)
        # Stub all API calls so the cycle runs without real network
        dp = agent.data_provider
        dp.fetch_historical_prices = MagicMock(return_value=bullish_prices)
        dp.fetch_option_chain = MagicMock(return_value=sample_put_contracts)
        dp.get_account_info = MagicMock(
            return_value={"equity": "100000", "buying_power": "50000"})
        dp.is_market_open = MagicMock(return_value=True)
        dp.get_current_price = MagicMock(
            return_value=float(bullish_prices["Close"].iloc[-1]))
        dp.fetch_batch_snapshots = MagicMock(return_value={"SPY": 500.0})
        dp.prefetch_historical_parallel = MagicMock()
        dp.get_underlying_bid_ask = MagicMock(return_value=(499.98, 500.01))
        agent.position_monitor.fetch_open_positions = MagicMock(return_value=[])

        with (
            patch("trading_agent.agent._is_within_market_hours", return_value=True),
            patch("os._exit") as mock_exit,
        ):
            agent.run_cycle()
            # os._exit should NOT have been called with 0 (after-hours reason)
            for call in mock_exit.call_args_list:
                assert call[0][0] != 0, (
                    "os._exit(0) must not be called during market hours"
                )

    def test_force_market_open_bypasses_shutdown(self, tmp_path, bullish_prices,
                                                  sample_put_contracts):
        """force_market_open=True must skip the after-hours guard entirely."""
        agent = self._make_agent(tmp_path, force_market_open=True)
        dp = agent.data_provider
        dp.fetch_historical_prices = MagicMock(return_value=bullish_prices)
        dp.fetch_option_chain = MagicMock(return_value=sample_put_contracts)
        dp.get_account_info = MagicMock(
            return_value={"equity": "100000", "buying_power": "50000"})
        dp.is_market_open = MagicMock(return_value=True)
        dp.get_current_price = MagicMock(
            return_value=float(bullish_prices["Close"].iloc[-1]))
        dp.fetch_batch_snapshots = MagicMock(return_value={"SPY": 500.0})
        dp.prefetch_historical_parallel = MagicMock()
        dp.get_underlying_bid_ask = MagicMock(return_value=(499.98, 500.01))
        agent.position_monitor.fetch_open_positions = MagicMock(return_value=[])

        with (
            patch("trading_agent.agent._is_within_market_hours", return_value=False),
            patch("os._exit") as mock_exit,
        ):
            agent.run_cycle()
            for call in mock_exit.call_args_list:
                assert call[0][0] != 0, (
                    "os._exit(0) must not be called when force_market_open=True"
                )

    def test_weekend_calls_exit_0(self, tmp_path):
        """Saturday / Sunday must also trigger after-hours shutdown."""
        agent = self._make_agent(tmp_path)
        # _is_within_market_hours already tested for weekends above;
        # here we wire through the full run_cycle() path.
        sat = _et(2026, 4, 4, 12, 0)   # Saturday noon ET
        with (
            patch("trading_agent.agent.datetime", **{"now.return_value": sat}),
            patch("os._exit") as mock_exit,
        ):
            agent.run_cycle()
            mock_exit.assert_called_once_with(0)

    def test_exit_code_is_0_not_1(self, tmp_path):
        """After-hours exit must use code 0 (clean stop), not 1 (error)."""
        agent = self._make_agent(tmp_path)
        with (
            patch("trading_agent.agent._is_within_market_hours", return_value=False),
            patch("os._exit") as mock_exit,
        ):
            agent.run_cycle()
            code = mock_exit.call_args[0][0]
            assert code == 0, f"Expected exit(0), got exit({code})"
