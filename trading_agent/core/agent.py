"""
The Trading Agent Orchestrator
================================
Runs a two-stage cycle:

  STAGE 1 — MONITOR existing positions
    → Fetch open positions from Alpaca
    → Classify current regime for each underlying
    → Evaluate exit signals (stop-loss, profit-target, regime-shift)
    → Close spreads that trigger an exit signal

  STAGE 2 — OPEN new positions (the original four-phase loop)
    I.   PERCEIVE   — fetch market data
    II.  CLASSIFY   — determine regime
    III. PLAN       — select strategy and strikes
    IV.  ACT        — validate risk, execute or log

5-minute cycle design notes
------------------------------
• run_cycle() is wrapped in a 270-second (4.5 min) hard-timeout guard.
  If the cycle has not completed by then, a TIMEOUT event is logged to
  JournalKB and the guard calls shutdown.hard_exit(1) so the cron
  scheduler can cleanly launch the next run without a zombie process.

• All historical price data is pre-fetched in parallel via
  prefetch_historical_parallel() before the per-ticker loop begins.

• All current prices are fetched in a single batch API call via
  fetch_batch_snapshots() before the per-ticker loop.

• JournalKB.log_signal() is called for EVERY ticker on EVERY cycle
  regardless of LLM enablement, execution mode, or failure type.

Week 3-4 modularization
-----------------------
Several concerns previously inlined in this file were extracted:

  • market_hours.py  — NYSE trading-hours guard
  • daily_state.py   — DailyStateStore + drawdown + debounce policy
  • thesis_builder.py — raw_signal thesis dict
  • shutdown.py      — graceful vs hard exit paths, signal handlers
  • file_locks.py    — locked appends + atomic JSON writes
  • logger_setup.py  — now uses RotatingFileHandler

The TradingAgent class is a thin orchestrator over those modules.
"""

import contextlib
import glob
import json
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List, Optional

from trading_agent.config import AppConfig, load_config
from trading_agent.intelligence.journal_kb import JournalKB
from trading_agent.utils.logger_setup import setup_logging
from trading_agent.market.market_data import MarketDataProvider, InsufficientDataError
from trading_agent.core.ports import (
    MarketDataPort,
    ExecutionPort,
    PositionsPort,
    OrdersPort,
)
from trading_agent.strategy.regime import Regime, RegimeClassifier, RegimeAnalysis
from trading_agent.strategy.strategy import StrategyPlanner, SpreadPlan
from trading_agent.strategy.risk_manager import RiskManager, RiskVerdict
from trading_agent.execution.executor import OrderExecutor
from trading_agent.execution.position_monitor import (
    PositionMonitor, ExitSignal, SpreadPosition, IMMEDIATE_EXIT_SIGNALS,
)
from trading_agent.execution.order_tracker import OrderTracker
from trading_agent.intelligence.llm_client import LLMClient, LLMConfig
from trading_agent.intelligence.trade_journal import TradeJournal
from trading_agent.intelligence.knowledge_base import KnowledgeBase
from trading_agent.intelligence.llm_analyst import LLMAnalyst, AnalystDecision
from trading_agent.sentiment.fingpt_analyser import FinGPTAnalyser, SentimentReport
from trading_agent.sentiment.news_aggregator import NewsAggregator
from trading_agent.sentiment.sentiment_verifier import SentimentVerifier, VerifiedSentimentReport
from trading_agent.sentiment.sentiment_pipeline import SentimentPipeline

# --- Week 3-4 extractions ---
from trading_agent.market.market_hours import (
    EASTERN,
    is_within_market_hours as _is_within_market_hours,
    market_window_str,
)
from trading_agent.utils.daily_state import (
    DailyStateStore,
    check_daily_drawdown,
    tally_exit_vote,
)
from trading_agent.utils.thesis_builder import build_thesis
from trading_agent.utils import shutdown as _shutdown

logger = logging.getLogger(__name__)

# Module-level defaults — seeded from trading_rules.yaml.
# Importable by name for backward compatibility; TradingAgent.__init__
# uses self.config.rules.agent for the actual runtime values.
from trading_agent.config.loader import AgentRules as _AgentRules
from trading_agent.core.stage_monitor import _StageMonitorMixin
from trading_agent.core.stage_plan import _StagePlanMixin

_agent_rules = _AgentRules()
CYCLE_TIMEOUT_SECONDS = _agent_rules.cycle_timeout_seconds
EXIT_DEBOUNCE_REQUIRED = _agent_rules.exit_debounce_required


class TradingAgent(_StageMonitorMixin, _StagePlanMixin):
    """
    Autonomous credit-spread trading agent.

    Stage 1 methods (position monitoring) live in stage_monitor.py.
    Stage 2 methods (new trade entry)    live in stage_plan.py.
    This class owns: __init__, run_cycle, _run_cycle_impl, and the CLI.

    Lifecycle::
        agent = TradingAgent.from_env()
        results = agent.run_cycle()
    """

    def __init__(self, config: AppConfig):
        self.config = config

        # MarketDataProvider now satisfies both MarketDataPort and
        # AccountPort (its get_account_info/is_market_open methods no
        # longer accept base_url — the adapter owns its endpoint).
        self.data_provider: MarketDataPort = MarketDataProvider(
            alpaca_api_key=config.alpaca.api_key,
            alpaca_secret_key=config.alpaca.secret_key,
            alpaca_data_url=config.alpaca.data_url,
            alpaca_base_url=config.alpaca.base_url,
        )
        self.regime_classifier = RegimeClassifier(
            self.data_provider,
            rules=config.rules.regime,
        )
        self.strategy_planner = StrategyPlanner(
            data_provider=self.data_provider,
            max_delta=config.trading.max_delta,
            min_credit_ratio=config.trading.min_credit_ratio,
            rules=config.rules.strategy,
        )
        self.risk_manager = RiskManager(
            max_risk_pct=config.trading.max_risk_pct,
            min_credit_ratio=config.trading.min_credit_ratio,
            max_delta=config.trading.max_delta,
            liquidity_max_spread=config.trading.liquidity_max_spread,
            liquidity_bps_of_mid=config.trading.liquidity_bps_of_mid,
            stale_spread_pct=config.trading.stale_spread_pct,
            max_buying_power_pct=config.trading.max_buying_power_pct,
            margin_multiplier=config.trading.margin_multiplier,
        )
        # The three broker-facing adapters are typed as ports so the
        # agent core never reaches into vendor-specific internals.  The
        # concrete classes still satisfy these Protocols via structural
        # typing — no inheritance required.
        self.executor: ExecutionPort = OrderExecutor(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
            base_url=config.alpaca.base_url,
            trade_plan_dir=config.logging.trade_plan_dir,
            dry_run=config.trading.dry_run,
            data_provider=self.data_provider,   # for live quote refresh on execution
            max_risk_pct=config.trading.max_risk_pct,            # shared w/ RiskManager #4
            min_credit_ratio=config.trading.min_credit_ratio,    # shared w/ RiskManager #2
            rules=config.rules.execution,
        )
        self.position_monitor: PositionsPort = PositionMonitor(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
            base_url=config.alpaca.base_url,
            rules=config.rules.position_monitor,
        )
        self.order_tracker: OrdersPort = OrderTracker(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
            base_url=config.alpaca.base_url,
        )

        # JournalKB writes signals.jsonl + signals.md into the same
        # trade_journal/ directory — no extra folder needed.
        journal_dir = (
            config.intelligence.journal_dir
            if config.intelligence and config.intelligence.journal_dir
            else "trade_journal"
        )
        self.journal_kb = JournalKB(journal_dir)

        # Daily state store (drawdown + exit debounce)
        self.daily_state = DailyStateStore(config.logging.trade_plan_dir)

        # Register graceful-shutdown handlers so SIGTERM (docker stop,
        # systemctl stop, cron cancel) flushes the journal + logs
        # instead of losing the in-flight write buffer.
        _shutdown.install_signal_handlers(journal=self.journal_kb)

        # Intelligence layer (LLM + RAG + Journal) — optional
        self.llm_analyst = self._init_intelligence(config)

        # Tiered sentiment pipeline (news → FinGPT → verifier) behind a
        # single facade.  The facade owns a cycle-scoped ThreadPoolExecutor
        # so the worker is always drained on cycle exit — no more
        # instance-lifetime pool surviving a SIGTERM mid-call.  Short
        # circuits (earnings calendar, content-hash cache) live inside
        # the facade, not the agent, so the orchestration call site
        # stays trivial.
        self.sentiment_pipeline: Optional[SentimentPipeline] = (
            SentimentPipeline.from_config(config.intelligence)
            if config.intelligence else None
        )
        # Back-compat handles — a few tests and journal helpers still
        # reach for these instance attributes by name.  Expose the
        # underlying components without reintroducing ownership.
        self.fingpt_analyser: Optional[FinGPTAnalyser] = (
            self.sentiment_pipeline.fingpt if self.sentiment_pipeline else None
        )
        self.news_aggregator: Optional[NewsAggregator] = (
            self.sentiment_pipeline.news_aggregator if self.sentiment_pipeline else None
        )
        self.sentiment_verifier: Optional[SentimentVerifier] = (
            self.sentiment_pipeline.verifier if self.sentiment_pipeline else None
        )

    def _init_intelligence(self, config: AppConfig):
        """Initialize the LLM intelligence layer if enabled."""
        intel_cfg = config.intelligence
        if not intel_cfg or not intel_cfg.enabled:
            logger.info("Intelligence layer DISABLED — rule-based mode only")
            return None

        try:
            llm_config = LLMConfig(
                provider=intel_cfg.llm_provider,
                base_url=intel_cfg.llm_base_url,
                model=intel_cfg.llm_model,
                embedding_model=intel_cfg.llm_embedding_model,
                api_key=intel_cfg.llm_api_key,
                temperature=intel_cfg.llm_temperature,
            )
            llm_client = LLMClient(llm_config)
            journal = TradeJournal(journal_dir=intel_cfg.journal_dir)
            kb = KnowledgeBase(
                kb_dir=intel_cfg.knowledge_base_dir,
                embed_fn=llm_client.embed,
            )
            analyst = LLMAnalyst(
                llm_client=llm_client,
                journal=journal,
                knowledge_base=kb,
                enabled=True,
            )
            logger.info(
                "Intelligence layer ENABLED — model=%s, provider=%s",
                intel_cfg.llm_model, intel_cfg.llm_provider,
            )
            return analyst

        except Exception as exc:
            logger.warning(
                "Failed to initialize intelligence layer: %s — "
                "continuing in rule-based mode",
                exc,
            )
            return None

    # Sentiment pipeline construction lives in
    # ``SentimentPipeline.from_config`` (wired once in __init__).  The
    # per-cycle call site is `_with_sentiment_pipeline()` below, which
    # context-manages the facade's background pool.

    @classmethod
    def from_env(cls, env_path: str = None) -> "TradingAgent":
        """Factory: create agent from environment / .env file."""
        config = load_config(env_path)
        setup_logging(config.logging.log_level, config.logging.log_dir)
        return cls(config)

    # ==================================================================
    # Main cycle — public entry point
    # ==================================================================

    @property
    def _cycle_timeout(self) -> int:
        return self.config.rules.agent.cycle_timeout_seconds

    @property
    def _exit_debounce_required(self) -> int:
        return self.config.rules.agent.exit_debounce_required

    def run_cycle(self) -> Dict:
        """
        Execute one full cycle with a hard timeout guard.

        If the cycle exceeds _cycle_timeout seconds the guard logs a
        TIMEOUT event to JournalKB and terminates the process so the
        scheduler can launch the next run cleanly.
        """
        cycle_timeout = self._cycle_timeout
        # --- Timeout guard -----------------------------------------------
        def _on_timeout():
            reason = (
                f"Cycle TIMEOUT after {cycle_timeout}s "
                "— killing process to unblock scheduler"
            )
            logger.error(reason)
            self.journal_kb.log_cycle_error(
                "cycle_timeout",
                {"timeout_seconds": cycle_timeout},
            )
            # Use hard_exit — process may be hung on a syscall or
            # deadlocked on a mutex; clean teardown is unsafe.
            _shutdown.hard_exit(1, reason=reason)

        timer = threading.Timer(cycle_timeout, _on_timeout)
        timer.daemon = True
        timer.start()
        # -----------------------------------------------------------------

        cycle_start = time.monotonic()
        # Cycle-scope the sentiment pipeline's worker pool so its thread
        # is drained cleanly on every exit path — including SIGTERM and
        # uncaught exceptions.  The nullcontext fallback covers the
        # case where intelligence is disabled entirely.
        pipeline_ctx: contextlib.AbstractContextManager = (
            self.sentiment_pipeline
            if self.sentiment_pipeline is not None
            else contextlib.nullcontext()
        )
        try:
            with pipeline_ctx:
                result = self._run_cycle_impl()
        except Exception as exc:
            logger.exception("CYCLE FAILED with unhandled exception: %s", exc)
            self.journal_kb.log_cycle_error(
                str(exc), {"tickers": self.config.trading.tickers},
            )
            result = {
                "status": "error",
                "reason": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        finally:
            timer.cancel()

        elapsed = time.monotonic() - cycle_start
        logger.info(
            "Cycle completed in %.1fs / %ds budget",
            elapsed, cycle_timeout,
        )
        if elapsed > cycle_timeout * 0.8:
            logger.warning(
                "Cycle used %.0f%% of time budget — consider "
                "reducing ticker count or increasing interval",
                100 * elapsed / cycle_timeout,
            )
        return result

    # ==================================================================
    # Cycle implementation
    # ==================================================================

    def _run_cycle_impl(self) -> Dict:
        """Core cycle logic — called inside run_cycle()'s timeout guard."""
        # --- After-hours shutdown guard -----------------------------------
        # Exit cleanly (code 0) when invoked outside NYSE market hours so
        # that a cron scheduler does not waste cycles on a closed market.
        # Set FORCE_MARKET_OPEN=true (or force_market_open=True in config)
        # to bypass this check in tests or paper-trading outside hours.
        if not self.config.trading.force_market_open and not _is_within_market_hours():
            now_et = datetime.now(EASTERN)
            reason = (
                f"Outside NYSE market hours "
                f"({now_et.strftime('%A %H:%M ET')}) — shutting down cleanly"
            )
            logger.info(reason)
            self.journal_kb.log_cycle_error(
                "after_hours_shutdown",
                {
                    "local_time_et": now_et.isoformat(),
                    "market_window": market_window_str(),
                },
            )
            # graceful_exit: we decided to stop; logs + journal are healthy.
            _shutdown.graceful_exit(0, reason="after_hours_shutdown", context={
                "local_time_et": now_et.isoformat(),
            })
        # ------------------------------------------------------------------

        logger.info("=" * 70)
        logger.info(
            "TRADING CYCLE START — %s",
            datetime.now(timezone.utc).isoformat(),
        )
        logger.info(
            "Tickers: %s | Mode: %s | Dry-run: %s",
            self.config.trading.tickers,
            self.config.trading.mode,
            self.config.trading.dry_run,
        )
        logger.info("=" * 70)

        # Pre-flight: fetch account info
        account = self.data_provider.get_account_info()
        if not account:
            msg = "Cannot fetch account info — aborting cycle."
            logger.error(msg)
            self.journal_kb.log_cycle_error(msg)
            return {
                "status": "error",
                "reason": "Account info unavailable",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        account_balance = float(account.get("equity", 0))
        account_buying_power = float(account.get("buying_power", account_balance))
        account_type = "paper" if "paper" in self.config.alpaca.base_url else "live"
        market_open = self.data_provider.is_market_open()

        logger.info(
            "Account: balance=$%s, buying_power=$%s, type=%s, market_open=%s",
            f"{account_balance:,.2f}",
            f"{account_buying_power:,.2f}",
            account_type, market_open,
        )
        logger.info("Schedule interval: %s", self.config.trading.schedule_interval)

        # --- Daily Drawdown Circuit Breaker ---
        if check_daily_drawdown(
            self.daily_state,
            current_equity=account_balance,
            drawdown_limit=self.config.trading.daily_drawdown_limit,
            journal_kb=self.journal_kb,
        ):
            reason = (
                f"Daily drawdown limit "
                f"({self.config.trading.daily_drawdown_limit * 100:.0f}%) "
                f"exceeded — stopping all trading"
            )
            logger.critical(reason)
            # graceful_exit: drawdown is a decided policy stop, not a hang.
            _shutdown.graceful_exit(1, reason="daily_drawdown_breaker", context={
                "equity": account_balance,
                "limit_pct": self.config.trading.daily_drawdown_limit * 100,
            })

        # --- Liquidation Mode Check ---
        liquidation_mode = self._check_liquidation_mode(
            account_balance, account_buying_power)
        if liquidation_mode:
            logger.warning(
                "LIQUIDATION MODE: buying power >%.0f%% used — "
                "closing positions only, no new trades",
                self.config.trading.max_buying_power_pct * 100,
            )

        # ------------------------------------------------------------------
        # Pre-fetch data for all tickers in parallel (5-min optimisation)
        # ------------------------------------------------------------------
        tickers = self.config.trading.tickers
        logger.info("Pre-fetching market data for %d ticker(s)…", len(tickers))
        self.data_provider.prefetch_historical_parallel(tickers)
        self.data_provider.fetch_batch_snapshots(tickers)

        # ------------------------------------------------------------------
        # Stage 1: MONITOR existing positions
        # ------------------------------------------------------------------
        logger.info("=" * 70)
        logger.info("STAGE 1 — MONITOR EXISTING POSITIONS")
        logger.info("=" * 70)

        monitor_results = self._stage_monitor(account_balance)

        tickers_with_positions = set()
        for sr in monitor_results.get("positions", []):
            if sr.get("signal") == ExitSignal.HOLD.value:
                tickers_with_positions.add(sr.get("underlying", ""))

        # ------------------------------------------------------------------
        # Stage 2: OPEN new positions
        # ------------------------------------------------------------------
        logger.info("=" * 70)
        logger.info("STAGE 2 — OPEN NEW POSITIONS")
        logger.info("=" * 70)

        new_trade_results = []
        for ticker in tickers:
            # Check for shutdown between tickers so a SIGTERM mid-cycle
            # stops the loop cleanly at the next safe point.
            if _shutdown.shutdown_requested():
                logger.warning(
                    "Shutdown requested — aborting ticker loop at %s", ticker)
                break

            if ticker in tickers_with_positions:
                logger.info("[%s] Already has an open spread — skipping", ticker)
                self.journal_kb.log_signal(
                    ticker=ticker,
                    action="skipped_existing",
                    price=self._cached_price(ticker),
                    raw_signal={"reason": "Existing open position"},
                )
                new_trade_results.append({
                    "ticker": ticker,
                    "status": "skipped",
                    "reason": "Existing open position",
                })
                continue

            if liquidation_mode:
                self.journal_kb.log_signal(
                    ticker=ticker,
                    action="skipped_liquidation_mode",
                    price=self._cached_price(ticker),
                    raw_signal={"reason": "Liquidation Mode — buying power exhausted"},
                )
                new_trade_results.append({
                    "ticker": ticker,
                    "status": "skipped",
                    "reason": "Liquidation Mode",
                })
                continue

            try:
                result = self._process_ticker(
                    ticker, account_balance, account_buying_power,
                    account_type, market_open,
                )
                new_trade_results.append(result)
            except InsufficientDataError as exc:
                # Expected condition — ticker has too little history for a
                # reliable SMA-200 classification. Log as a warning and
                # skip cleanly; this is not an error worth paging on.
                logger.warning("[%s] Skipped — %s", ticker, exc)
                new_trade_results.append({
                    "ticker": ticker,
                    "status": "skipped",
                    "reason": f"insufficient_data: {exc}",
                })
            except Exception as exc:
                logger.exception("[%s] Unhandled error: %s", ticker, exc)
                self.journal_kb.log_error(
                    ticker=ticker,
                    error=str(exc),
                    price=self._cached_price(ticker),
                )
                new_trade_results.append({
                    "ticker": ticker,
                    "status": "error",
                    "reason": str(exc),
                })

        # ------------------------------------------------------------------
        # Order status summary
        # ------------------------------------------------------------------
        order_summary = self._check_order_statuses()

        logger.info("=" * 70)
        logger.info(
            "TRADING CYCLE COMPLETE — %d tickers processed",
            len(new_trade_results),
        )
        self._print_summary(new_trade_results)
        logger.info("=" * 70)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "account_balance": account_balance,
            "monitor": monitor_results,
            "new_trades": new_trade_results,
            "order_summary": order_summary,
        }


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    """Run a single trading cycle from the command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Autonomous Options Trading Agent")
    parser.add_argument("--env", type=str, default=None,
                        help="Path to .env file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Override: force dry-run mode")
    args = parser.parse_args()

    agent = TradingAgent.from_env(args.env)
    if args.dry_run:
        agent.executor.dry_run = True

    results = agent.run_cycle()
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
