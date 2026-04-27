"""
Tests for the after-hours market shutdown guard.

Coverage
--------
- is_within_market_hours() boundary conditions
  - Before market open (08:59 ET, weekday)
  - Exactly at open boundary (09:25 ET)
  - During market hours (11:00 ET)
  - Exactly at close boundary (16:05 ET)
  - After market close (16:06 ET)
  - Weekend (Saturday / Sunday)
- run_cycle() calls graceful_exit(0) outside hours
- run_cycle() does NOT call graceful_exit(0) during hours
- force_market_open=True bypasses the shutdown

Week 3-4 refactor note
----------------------
The after-hours exit path now goes through
``trading_agent.shutdown.graceful_exit`` instead of ``os._exit``.  The
graceful path flushes logs + writes a journal shutdown marker, then
calls ``sys.exit(code)``.  Tests patch ``graceful_exit`` directly so
the assertion is expressive (we care about the code and the reason)
and so ``sys.exit`` doesn't actually raise.
"""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

import trading_agent.agent as agent_module
from trading_agent.market_hours import is_within_market_hours
from trading_agent.agent import TradingAgent
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
# is_within_market_hours() — time-boundary unit tests
# All dates chosen to be regular weekdays (Wednesday 2026-04-01 is a Wednesday)
# ---------------------------------------------------------------------------

class TestIsWithinMarketHours:

    # Wednesday 2026-04-01 used throughout (not a holiday in the test scope)
    _WED = (2026, 4, 1)

    def _at(self, dt):
        """
        Call is_within_market_hours with an explicit *now* — cleaner than
        patching datetime inside the module and works regardless of where
        the function's datetime import lives.
        """
        return is_within_market_hours(dt)

    def test_before_open_returns_false(self):
        # 08:59 ET — one minute before the 09:25 open boundary
        assert self._at(_et(*self._WED, 8, 59)) is False

    def test_exactly_at_open_boundary_returns_true(self):
        # 09:25 ET — the open boundary itself must be included
        assert self._at(_et(*self._WED, 9, 25)) is True

    def test_mid_session_returns_true(self):
        # 11:00 ET — well within trading hours
        assert self._at(_et(*self._WED, 11, 0)) is True

    def test_just_before_close_boundary_returns_true(self):
        # 16:04 ET — one minute before the 16:05 close boundary
        assert self._at(_et(*self._WED, 16, 4)) is True

    def test_exactly_at_close_boundary_returns_true(self):
        # 16:05 ET — the close boundary itself must be included
        assert self._at(_et(*self._WED, 16, 5)) is True

    def test_one_minute_after_close_returns_false(self):
        # 16:06 ET — one minute past close boundary
        assert self._at(_et(*self._WED, 16, 6)) is False

    def test_evening_returns_false(self):
        # 20:00 ET — evening, market long closed
        assert self._at(_et(*self._WED, 20, 0)) is False

    def test_midnight_returns_false(self):
        assert self._at(_et(*self._WED, 0, 0)) is False

    def test_saturday_returns_false(self):
        # 2026-04-04 is a Saturday, noon ET
        assert self._at(_et(2026, 4, 4, 12, 0)) is False

    def test_sunday_returns_false(self):
        # 2026-04-05 is a Sunday, noon ET
        assert self._at(_et(2026, 4, 5, 12, 0)) is False

    def test_monday_during_hours_returns_true(self):
        # 2026-04-06 is a Monday
        assert self._at(_et(2026, 4, 6, 10, 30)) is True

    def test_friday_during_hours_returns_true(self):
        # 2026-04-10 is a Friday (post-Easter, a regular NYSE trading day).
        # Note: 2026-04-03 was Good Friday — NYSE is closed — which the
        # calendar-aware check now correctly rejects. See the dedicated
        # holiday test below.
        assert self._at(_et(2026, 4, 10, 15, 45)) is True

    def test_friday_after_close_returns_false(self):
        # 2026-04-10 Friday 16:30 ET — market closed for the week
        assert self._at(_et(2026, 4, 10, 16, 30)) is False

    def test_good_friday_returns_false(self):
        """Good Friday 2026 (April 3) — NYSE holiday, market closed even
        though it's technically a weekday. Regression check for the
        pandas_market_calendars swap."""
        assert self._at(_et(2026, 4, 3, 11, 0)) is False   # 11 AM ET, would be session


# ---------------------------------------------------------------------------
# TradingAgent.run_cycle() — shutdown integration tests
# ---------------------------------------------------------------------------

class TestAfterHoursShutdown:
    """Verify run_cycle() calls graceful_exit(0) outside hours."""

    def _make_agent(self, tmp_path, force_market_open=False):
        agent = TradingAgent(_make_config(tmp_path, force_market_open=force_market_open))
        # Stub out the JournalKB so we can inspect calls without real file I/O
        agent.journal_kb = MagicMock()
        return agent

    def test_after_close_calls_graceful_exit_0(self, tmp_path):
        """After 16:06 ET on a weekday → graceful_exit(0)."""
        agent = self._make_agent(tmp_path)
        with (
            patch("trading_agent.core.agent._is_within_market_hours", return_value=False),
            patch("trading_agent.core.agent._shutdown.graceful_exit") as mock_exit,
        ):
            agent.run_cycle()
            assert mock_exit.called, "graceful_exit must be called after hours"
            code = mock_exit.call_args[0][0] if mock_exit.call_args[0] \
                else mock_exit.call_args.kwargs.get("code")
            assert code == 0, f"Expected graceful_exit(0), got {code}"

    def test_after_close_logs_cycle_error(self, tmp_path):
        """After-hours shutdown must write a journal entry."""
        agent = self._make_agent(tmp_path)
        with (
            patch("trading_agent.core.agent._is_within_market_hours", return_value=False),
            patch("trading_agent.core.agent._shutdown.graceful_exit"),
        ):
            agent.run_cycle()
            # The after-hours entry must be the first log_cycle_error call.
            first_call = agent.journal_kb.log_cycle_error.call_args_list[0]
            assert "after_hours_shutdown" in first_call[0][0]

    def test_during_hours_does_not_call_graceful_exit(self, tmp_path, bullish_prices,
                                                       sample_put_contracts):
        """During market hours, graceful_exit should NOT be called for after-hours."""
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
            patch("trading_agent.core.agent._is_within_market_hours", return_value=True),
            patch("trading_agent.core.agent._shutdown.graceful_exit") as mock_exit,
        ):
            agent.run_cycle()
            # graceful_exit(0) must not fire for after-hours.  It MAY fire
            # for drawdown breaker (code 1) on a degenerate account — assert
            # only that no zero-code call happened.
            for call in mock_exit.call_args_list:
                args = call[0]
                kwargs = call.kwargs
                code = args[0] if args else kwargs.get("code")
                assert code != 0, (
                    "graceful_exit(0) must not be called during market hours"
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
            patch("trading_agent.core.agent._is_within_market_hours", return_value=False),
            patch("trading_agent.core.agent._shutdown.graceful_exit") as mock_exit,
        ):
            agent.run_cycle()
            for call in mock_exit.call_args_list:
                args = call[0]
                kwargs = call.kwargs
                code = args[0] if args else kwargs.get("code")
                assert code != 0, (
                    "graceful_exit(0) must not be called when force_market_open=True"
                )

    def test_weekend_calls_graceful_exit_0(self, tmp_path):
        """Saturday / Sunday must also trigger after-hours shutdown."""
        agent = self._make_agent(tmp_path)
        sat = _et(2026, 4, 4, 12, 0)   # Saturday noon ET
        with (
            patch("trading_agent.market_hours.datetime",
                  **{"now.return_value": sat}),
            patch("trading_agent.core.agent.datetime", **{"now.return_value": sat}),
            patch("trading_agent.core.agent._shutdown.graceful_exit") as mock_exit,
        ):
            agent.run_cycle()
            assert mock_exit.called, "graceful_exit must fire on weekends"
            code = mock_exit.call_args[0][0] if mock_exit.call_args[0] \
                else mock_exit.call_args.kwargs.get("code")
            assert code == 0

    def test_exit_code_is_0_not_1(self, tmp_path):
        """After-hours exit must use code 0 (clean stop), not 1 (error)."""
        agent = self._make_agent(tmp_path)
        with (
            patch("trading_agent.core.agent._is_within_market_hours", return_value=False),
            patch("trading_agent.core.agent._shutdown.graceful_exit") as mock_exit,
        ):
            agent.run_cycle()
            code = mock_exit.call_args[0][0] if mock_exit.call_args[0] \
                else mock_exit.call_args.kwargs.get("code")
            assert code == 0, f"Expected graceful_exit(0), got graceful_exit({code})"
