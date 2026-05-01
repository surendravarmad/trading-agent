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
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

from trading_agent.config import AppConfig, load_config
from trading_agent.journal_kb import JournalKB
from trading_agent.logger_setup import setup_logging
from trading_agent.market_data import MarketDataProvider, InsufficientDataError
from trading_agent.ports import (
    MarketDataPort,
    ExecutionPort,
    PositionsPort,
    OrdersPort,
)
from trading_agent.regime import Regime, RegimeClassifier, RegimeAnalysis
from trading_agent.strategy import StrategyPlanner, SpreadPlan
from trading_agent.strategy_presets import (
    PresetConfig,
    load_active_preset,
    regime_is_allowed,
)
from trading_agent.risk_manager import RiskManager, RiskVerdict
from trading_agent.executor import OrderExecutor
from trading_agent.position_monitor import (
    PositionMonitor, ExitSignal, SpreadPosition, IMMEDIATE_EXIT_SIGNALS,
)
from trading_agent.order_tracker import OrderTracker
from trading_agent.llm_client import LLMClient, LLMConfig
from trading_agent.trade_journal import TradeJournal
from trading_agent.knowledge_base import KnowledgeBase
from trading_agent.llm_analyst import LLMAnalyst, AnalystDecision
from trading_agent.fingpt_analyser import FinGPTAnalyser, SentimentReport
from trading_agent.news_aggregator import NewsAggregator
from trading_agent.sentiment_verifier import SentimentVerifier, VerifiedSentimentReport
from trading_agent.sentiment_pipeline import SentimentPipeline

# --- Week 3-4 extractions ---
from trading_agent.market_hours import (
    EASTERN,
    is_within_market_hours as _is_within_market_hours,
    market_window_str,
)
from trading_agent.daily_state import (
    DailyStateStore,
    check_daily_drawdown,
    tally_exit_vote,
)
from trading_agent.thesis_builder import build_thesis
from trading_agent import shutdown as _shutdown

logger = logging.getLogger(__name__)

# Kill the process if a cycle takes longer than this (seconds).
# The external scheduler (cron / APScheduler) will start the next run cleanly.
CYCLE_TIMEOUT_SECONDS = 270   # 4 min 30 sec

# Number of consecutive cycles an exit signal must repeat before acting.
EXIT_DEBOUNCE_REQUIRED = 3

# Stale-order policy.
#
# An open limit order whose age (since Alpaca's `created_at`) exceeds this
# threshold is cancelled at the start of every cycle — the next planning
# pass will re-price against fresh quotes.  Set high enough to give a
# midday combo a fair shot at filling, low enough to recover before
# theta has chewed through the original limit.
STALE_ORDER_MAX_AGE_MIN = 15

# OCC option symbol → underlying root.  Format ROOT(1-6) + YYMMDD(6) +
# C/P(1) + STRIKE*1000(8).  Used both for stale-order cancel scoping
# and open-order dedup, so identical to the dashboard helper in
# streamlit/live_monitor.py.
_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def _root_from_occ(symbol: str) -> str:
    """Return the OCC root (e.g. ``GOOG260508P00337500`` → ``GOOG``)
    or the empty string if the symbol can't be parsed."""
    if not symbol:
        return ""
    m = _OCC_RE.match(symbol)
    return m.group(1) if m else ""


class TradingAgent:
    """
    Autonomous credit-spread trading agent.

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

        # Load the active Strategy-Profile preset (Conservative / Balanced /
        # Aggressive / Custom) chosen via the Streamlit dashboard. The preset
        # bundles max_delta + per-strategy DTE + width policy + C/W floor +
        # max-risk %, plus the directional-bias filter. Falls back to BALANCED
        # if STRATEGY_PRESET.json is missing or malformed (logged at info-level).
        # Each subprocess re-reads the file on init, so dashboard changes apply
        # on the next 5-min cycle without restarting the loop.
        self.preset: PresetConfig = load_active_preset()
        logger.info("Strategy preset → %s", self.preset.to_summary_line())

        # Risk knobs come from the preset; everything else stays from the
        # AppConfig env-loaded baseline (liquidity floors, margin, etc).
        max_delta        = self.preset.max_delta
        min_credit_ratio = self.preset.min_credit_ratio
        max_risk_pct     = self.preset.max_risk_pct

        self.regime_classifier = RegimeClassifier(self.data_provider)
        self.strategy_planner = StrategyPlanner(
            data_provider=self.data_provider,
            max_delta=max_delta,
            min_credit_ratio=min_credit_ratio,
            dte_vertical=self.preset.dte_vertical,
            dte_iron_condor=self.preset.dte_iron_condor,
            dte_mean_reversion=self.preset.dte_mean_reversion,
            dte_window_days=self.preset.dte_window_days,
            width_mode=self.preset.width_mode,
            width_value=self.preset.width_value,
            preset=self.preset,
        )
        self.risk_manager = RiskManager(
            max_risk_pct=max_risk_pct,
            min_credit_ratio=min_credit_ratio,
            max_delta=max_delta,
            liquidity_max_spread=config.trading.liquidity_max_spread,
            liquidity_bps_of_mid=config.trading.liquidity_bps_of_mid,
            stale_spread_pct=config.trading.stale_spread_pct,
            max_buying_power_pct=config.trading.max_buying_power_pct,
            margin_multiplier=config.trading.margin_multiplier,
            # Adaptive mode: replace the static C/W floor with a Δ-aware one —
            # same formula the scanner uses: |Δshort| × (1 + edge_buffer).
            delta_aware_floor=(self.preset.scan_mode == "adaptive"),
            edge_buffer=self.preset.edge_buffer,
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
            # Adaptive mode: live-credit recheck + 1-tick haircut both use the
            # same Δ-aware floor RiskManager is enforcing, so a scanner-picked
            # plan can never be vetoed at execution time by a stale static
            # floor.  Mirrors the kwargs passed to RiskManager above.
            delta_aware_floor=(self.preset.scan_mode == "adaptive"),
            edge_buffer=self.preset.edge_buffer,
        )
        self.position_monitor: PositionsPort = PositionMonitor(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
            base_url=config.alpaca.base_url,
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
        # Short-circuit when the intelligence layer is fully disabled so we
        # don't pay the SentimentPipeline factory cost (transitively imports
        # NewsAggregator + FinGPTAnalyser + EarningsCalendar) for tests and
        # rule-only deployments. Was a measurable per-test hit because every
        # `TradingAgent(...)` instantiation triggered the factory.
        intel_cfg = config.intelligence
        intel_disabled = (
            intel_cfg is None
            or not getattr(intel_cfg, "enabled", False)
        )
        self.sentiment_pipeline: Optional[SentimentPipeline] = (
            None if intel_disabled
            else SentimentPipeline.from_config(intel_cfg)
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

    def run_cycle(self) -> Dict:
        """
        Execute one full cycle with a hard timeout guard.

        If the cycle exceeds CYCLE_TIMEOUT_SECONDS the guard logs a
        TIMEOUT event to JournalKB and terminates the process so the
        scheduler can launch the next run cleanly.
        """
        # --- Timeout guard -----------------------------------------------
        def _on_timeout():
            reason = (
                f"Cycle TIMEOUT after {CYCLE_TIMEOUT_SECONDS}s "
                "— killing process to unblock scheduler"
            )
            logger.error(reason)
            self.journal_kb.log_cycle_error(
                "cycle_timeout",
                {"timeout_seconds": CYCLE_TIMEOUT_SECONDS},
            )
            # Use hard_exit — process may be hung on a syscall or
            # deadlocked on a mutex; clean teardown is unsafe.
            _shutdown.hard_exit(1, reason=reason)

        timer = threading.Timer(CYCLE_TIMEOUT_SECONDS, _on_timeout)
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
            elapsed, CYCLE_TIMEOUT_SECONDS,
        )
        if elapsed > CYCLE_TIMEOUT_SECONDS * 0.8:
            logger.warning(
                "Cycle used %.0f%% of time budget — consider "
                "reducing ticker count or increasing interval",
                100 * elapsed / CYCLE_TIMEOUT_SECONDS,
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
        # Stage 1.5: Stale-order maintenance
        # ------------------------------------------------------------------
        # Cancel stuck limits that have been on the book longer than
        # STALE_ORDER_MAX_AGE_MIN so the next planning pass can re-price
        # against a fresh mid.  Then collect the tickers of every
        # remaining open order and union them into tickers_with_positions
        # — the original dedup only blocked tickers with FILLED spreads,
        # which let the cycle stack identical limit orders on top of an
        # unfilled one (root cause of the duplicate-GOOG issue).
        try:
            self._cancel_stale_orders(tickers)
        except Exception as exc:
            logger.warning("Stale-order maintenance failed: %s", exc)

        try:
            tickers_with_open_orders = self._tickers_with_open_orders()
            if tickers_with_open_orders:
                logger.info(
                    "Open orders pending fill on: %s — these tickers will "
                    "be skipped in Stage 2 to avoid duplicate submissions",
                    sorted(tickers_with_open_orders),
                )
                tickers_with_positions |= tickers_with_open_orders
        except Exception as exc:
            logger.warning("Open-order dedup failed: %s", exc)

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
                # tickers_with_positions includes BOTH filled spreads
                # (Stage 1) and pending limit orders (Stage 1.5 dedup),
                # so this branch covers both.  The journal reason is
                # kept generic to avoid log churn for a single string.
                logger.info(
                    "[%s] Already has an open spread or pending order — skipping",
                    ticker,
                )
                self.journal_kb.log_signal(
                    ticker=ticker,
                    action="skipped_existing",
                    price=self._cached_price(ticker),
                    raw_signal={"reason": "Existing open position or pending order"},
                )
                new_trade_results.append({
                    "ticker": ticker,
                    "status": "skipped",
                    "reason": "Existing open position or pending order",
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

    # ==================================================================
    # Stage 1: Position monitoring
    # ==================================================================

    def _stage_monitor(self, account_balance: float) -> Dict:
        """
        Fetch positions, classify regimes, evaluate exit signals,
        and close spreads that need closing.
        """
        positions = self.position_monitor.fetch_open_positions()
        if not positions:
            logger.info("No open option positions found.")
            return {"total_spreads": 0, "positions": [], "closed": []}

        trade_plans = self._load_trade_plans()
        spreads = self.position_monitor.group_into_spreads(positions, trade_plans)
        if not spreads:
            logger.info("Could not match positions to any trade plans.")
            return {"total_spreads": 0, "positions": [], "closed": []}

        underlyings = {s.underlying for s in spreads}
        current_regimes: Dict = {}
        underlying_prices: Dict[str, float] = {}
        for ticker in underlyings:
            try:
                analysis = self.regime_classifier.classify(ticker)
                current_regimes[ticker] = analysis.regime
                logger.info(
                    "[%s] Current regime: %s",
                    ticker, analysis.regime.value,
                )
            except Exception as exc:
                logger.warning("[%s] Could not classify regime: %s", ticker, exc)
            price = self._cached_price(ticker)
            if price > 0:
                underlying_prices[ticker] = price

        spreads = self.position_monitor.evaluate(
            spreads, current_regimes, underlying_prices)

        closed = []
        for spread in spreads:
            if spread.exit_signal != ExitSignal.HOLD and self._should_exit_spread(spread):
                if self.config.trading.dry_run:
                    logger.info(
                        "[%s] DRY RUN — would close %s (%s: %s)",
                        spread.underlying, spread.strategy_name,
                        spread.exit_signal.value, spread.exit_reason,
                    )
                    closed.append({
                        "underlying": spread.underlying,
                        "signal": spread.exit_signal.value,
                        "reason": spread.exit_reason,
                        "action": "dry_run_close",
                    })
                else:
                    result = self.executor.close_spread(spread)
                    closed.append(result)

                if self.llm_analyst:
                    self._learn_from_close(spread)

        summary = self.position_monitor.summary(spreads)
        summary["closed"] = closed
        return summary

    def _load_trade_plans(self) -> List[Dict]:
        """
        Load trade plans from the plan directory.

        Handles two formats:
          • New  — trade_plan_{TICKER}.json  (state_history array)
          • Old  — trade_plan_{TICKER}_{TS}.json  (flat dict, legacy)
        """
        plan_dir = self.config.logging.trade_plan_dir
        if not os.path.isdir(plan_dir):
            return []

        plans = []
        for path in sorted(glob.glob(
                os.path.join(plan_dir, "trade_plan_*.json"))):
            try:
                with open(path) as fh:
                    data = json.load(fh)

                if "state_history" in data:
                    # New format: flatten all approved history entries
                    for entry in data["state_history"]:
                        plans.append(entry)
                else:
                    # Old timestamped format
                    plans.append(data)

            except Exception as exc:
                logger.warning("Could not load plan %s: %s", path, exc)

        logger.info("Loaded %d trade plan(s) from %s", len(plans), plan_dir)
        return plans

    # ==================================================================
    # Stage 2: New trade entry
    # ==================================================================

    def _process_ticker(self, ticker: str, balance: float,
                        buying_power: float,
                        acct_type: str, market_open: bool) -> Dict:
        """Full four-phase pipeline for a single ticker (+ LLM Phase V)."""
        logger.info("-" * 50)
        logger.info("[%s] Phase I  — PERCEIVE", ticker)

        # Liquidity check on underlying
        underlying_bid_ask = self.data_provider.get_underlying_bid_ask(ticker)

        logger.info("[%s] Phase II — CLASSIFY", ticker)
        analysis: RegimeAnalysis = self.regime_classifier.classify(ticker)

        # --- Directional-bias filter (Strategy Profile) ------------------
        # The active preset can restrict which regimes are tradeable.  This
        # check runs immediately after classify so we short-circuit before
        # spinning up the sentiment pipeline or option-chain fetch — both
        # of which are expensive and pointless when the regime would be
        # filtered out anyway.  Mean-reversion is always allowed (the 3-σ
        # touch override is a fear-spike signal, not a directional view).
        if not regime_is_allowed(
            analysis.regime.value, self.preset.directional_bias
        ):
            reason = (
                f"DirectionalBias={self.preset.directional_bias} blocks "
                f"regime={analysis.regime.value}"
            )
            logger.info("[%s] %s — skipping ticker", ticker, reason)
            self.journal_kb.log_signal(
                ticker=ticker,
                action="skipped_bias",
                price=analysis.current_price,
                raw_signal={
                    "regime": analysis.regime.value,
                    "directional_bias": self.preset.directional_bias,
                    "preset": self.preset.name,
                    "reason": reason,
                },
            )
            return {
                "ticker": ticker,
                "regime": analysis.regime.value,
                "strategy": "skipped_bias",
                "plan_valid": False,
                "risk_approved": False,
                "status": "skipped",
                "reason": reason,
            }

        # --- High-IV block: IV rank > 95th pct blocks all new entries ---
        if getattr(analysis, "high_iv_warning", False):
            reason = (
                f"HighIV: IV rank {getattr(analysis, 'iv_rank', 0):.1f} > 95th pct "
                f"— extreme volatility, blocking all new entries"
            )
            logger.warning("[%s] %s | strategy_mode=defense_first", ticker, reason)
            self.journal_kb.log_defense_first(
                ticker, reason, analysis.current_price,
                {
                    "regime": analysis.regime.value,
                    "iv_rank": getattr(analysis, "iv_rank", 0.0),
                    "high_iv_warning": True,
                },
            )
            return {
                "ticker": ticker,
                "regime": analysis.regime.value,
                "strategy": "skipped",
                "plan_valid": False,
                "risk_approved": False,
                "status": "skipped",
                "reason": reason,
                "strategy_mode": "defense_first",
            }

        # Launch tiered sentiment pipeline (Tier-0 earnings → Tier-1 cache →
        # Tier-2 FinGPT + verifier) in the background immediately after
        # Phase II so it runs concurrently with Phase III + IV, adding
        # near-zero wall-clock latency when the Tier-0/1 short-circuit
        # applies (the common case once the cache is warm).
        fingpt_future: Optional[Future] = None
        if self.sentiment_pipeline is not None:
            fingpt_future = self.sentiment_pipeline.submit(
                ticker,
                analysis.regime.value,
                analysis.current_price,
                analysis.rsi_14,
                getattr(analysis, "iv_rank", 0.0),
                self._regime_to_strategy(analysis.regime),
            )

        logger.info(
            "[%s] Phase III — PLAN (%s → %s)",
            ticker, analysis.regime.value,
            self._regime_to_strategy(analysis.regime),
        )
        plan: SpreadPlan = self.strategy_planner.plan(ticker, analysis)

        # Snapshot adaptive-scan results immediately so the next ticker's
        # plan() call doesn't overwrite ``last_scan_candidates`` before
        # we get a chance to journal them.  Returns None in static mode.
        scan_results = self._snapshot_scan_results()

        logger.info("[%s] Phase IV — RISK CHECK", ticker)
        verdict: RiskVerdict = self.risk_manager.evaluate(
            plan, balance, acct_type, market_open,
            self.config.trading.force_market_open,
            underlying_bid_ask=underlying_bid_ask,
            account_buying_power=buying_power,
        )

        # Resolve tiered sentiment pipeline result (earnings → cache →
        # FinGPT + verifier).  Timeout is 60s: news fetching adds
        # ~5-15s on top of inference; the Tier-0/1 short-circuits
        # return in <100 ms so the typical case is effectively free.
        #
        # Efficiency gate: if Phase III/IV produced no tradeable
        # candidate (invalid plan or risk rejection), the sentiment
        # readout is not consumed downstream — cancel the future to
        # skip the LLM calls entirely.
        sentiment: Optional[VerifiedSentimentReport] = None
        if fingpt_future is not None:
            if not (plan.valid and verdict.approved):
                # Best-effort cancellation — if the worker has already
                # begun the LLM call, it'll finish; we just drop the
                # result.  Either way we don't block the cycle.
                fingpt_future.cancel()
                logger.debug(
                    "[%s] No tradeable candidate — sentiment future dropped",
                    ticker,
                )
            else:
                try:
                    sentiment = fingpt_future.result(timeout=60)
                except Exception as exc:
                    logger.warning(
                        "[%s] Sentiment pipeline future failed: %s",
                        ticker, exc,
                    )

        # Phase V: LLM Analysis (if enabled)
        llm_decision = None
        if self.llm_analyst and plan.valid and verdict.approved:
            logger.info("[%s] Phase V  — LLM ANALYSIS", ticker)
            llm_decision = self.llm_analyst.analyze_trade(
                ticker, analysis, plan, verdict, sentiment=sentiment)

            if llm_decision.action == "skip":
                logger.warning(
                    "[%s] LLM SKIPPED trade (confidence=%.2f): %s",
                    ticker, llm_decision.confidence,
                    llm_decision.reasoning[:150],
                )

                self._log_signal(
                    ticker, "skipped_by_llm", analysis, plan, verdict,
                    llm_decision, exec_result=None,
                    scan_results=scan_results,
                )

                return {
                    "ticker": ticker,
                    "regime": analysis.regime.value,
                    "strategy": plan.strategy_name,
                    "plan_valid": plan.valid,
                    "risk_approved": verdict.approved,
                    "llm_decision": "skip",
                    "llm_reasoning": llm_decision.reasoning,
                    "llm_confidence": llm_decision.confidence,
                    "execution": {"status": "skipped_by_llm"},
                    "analysis": self._analysis_dict(analysis),
                }

        # Execute trade
        logger.info("[%s] Phase VI — EXECUTE", ticker)
        exec_result = self.executor.execute(verdict)

        # Journal the trade (if LLM enabled)
        if (self.llm_analyst and llm_decision
                and exec_result.get("status") in ("submitted", "dry_run")):
            try:
                entry = self.llm_analyst.create_journal_entry(
                    ticker, analysis, plan, verdict, llm_decision)
                entry.order_status = exec_result.get("status", "")
                entry.order_id = exec_result.get("order_id", "")
                trade_id = self.llm_analyst.journal.open_trade(entry)
                exec_result["trade_journal_id"] = trade_id
            except Exception as exc:
                logger.warning("[%s] Failed to journal trade: %s", ticker, exc)

        # Always log to JournalKB
        self._log_signal(
            ticker, exec_result.get("status", "unknown"),
            analysis, plan, verdict, llm_decision, exec_result,
            scan_results=scan_results,
        )

        result = {
            "ticker": ticker,
            "regime": analysis.regime.value,
            "strategy": plan.strategy_name,
            "plan_valid": plan.valid,
            "risk_approved": verdict.approved,
            "execution": exec_result,
            "analysis": self._analysis_dict(analysis),
        }

        if llm_decision:
            result["llm_decision"] = llm_decision.action
            result["llm_confidence"] = llm_decision.confidence
            result["llm_reasoning"] = llm_decision.reasoning
            result["llm_warnings"] = llm_decision.warnings

        if sentiment:
            # Use the SentimentReadout surface (verified_* fields exposed
            # as plain attribute aliases) so the journal emits
            # identical keys regardless of which pipeline tier produced
            # the result (earnings short-circuit, cache hit, or full
            # FinGPT + verifier chain).
            result["fingpt_sentiment"] = sentiment.sentiment_score
            result["fingpt_event_risk"] = sentiment.event_risk
            result["fingpt_recommendation"] = sentiment.recommendation
            result["fingpt_themes"] = sentiment.key_themes
            result["fingpt_agreement"] = sentiment.agreement_score
            result["fingpt_hallucination_flags"] = sentiment.hallucination_flags
            result["fingpt_verified_by"] = sentiment.verifier_model

        return result

    # ==================================================================
    # JournalKB signal helper
    # ==================================================================

    # ------------------------------------------------------------------
    # Adaptive-scan journal helper
    # ------------------------------------------------------------------

    # Top-K candidates persisted per cycle.  10 is enough to reconstruct
    # the scanner's decision (best + a few near-misses) without bloating
    # signals.jsonl on a 12-ticker, 4-grid sweep.
    _SCAN_JOURNAL_TOPK = 10

    def _snapshot_scan_results(self) -> Optional[Dict]:
        """
        Capture the planner's most recent scanner output as a journal-safe
        dict, or return ``None`` when the planner is in static mode.

        MUST be called immediately after ``strategy_planner.plan(...)`` —
        the next plan() invocation resets ``last_scan_candidates`` and
        the snapshot would otherwise reflect a different ticker.

        The returned shape is::

            {
              "side":           "bull_put" | "bear_call",
              "scan_mode":      "adaptive",
              "edge_buffer":    0.10,
              "min_pop":        0.55,
              "candidates_total": 8,
              "selected_index":  0,             # index into the K below
              "top_k": [ <SpreadCandidate.to_journal_dict()>, ... ],
              "diagnostics": {
                  "grid_points_total":    16,
                  "grid_points_priced":   12,
                  "expirations_resolved": 4,
                  "rejects_by_reason":    {"cw_below_floor": 11, ...},
                  "best_near_miss":       {<SpreadCandidate-like dict>}
              }
            }

        ``selected_index`` is 0 when the scanner picked a candidate
        (top-of-list) and -1 when no candidate cleared the floor.

        The ``diagnostics`` block is the actionable answer to *"why didn't
        the scanner pass?"*. ``best_near_miss`` is the single highest-EV
        candidate that failed only the C/W floor — quoting it lets the
        user see "the closest we came was C/W=0.18, needed 0.22" without
        digging through trading_agent.log.
        """
        planner = self.strategy_planner
        if not getattr(planner, "is_adaptive", False):
            return None
        candidates = list(getattr(planner, "last_scan_candidates", []) or [])
        side = getattr(planner, "last_scan_side", None)
        diagnostics = getattr(planner, "last_scan_diagnostics", None)
        # No scan ran this ticker (e.g. iron condor or mean-reversion path
        # in adaptive preset — those still use the static builders today).
        if not candidates and side is None and diagnostics is None:
            return None
        top_k = candidates[: self._SCAN_JOURNAL_TOPK]
        block: Dict = {
            "scan_mode":        "adaptive",
            "side":             side,
            "edge_buffer":      float(getattr(self.preset, "edge_buffer", 0.10)),
            "min_pop":          float(getattr(self.preset, "min_pop", 0.55)),
            "candidates_total": len(candidates),
            "selected_index":   0 if candidates else -1,
            "top_k":            [c.to_journal_dict() for c in top_k],
        }
        if diagnostics is not None:
            block["diagnostics"] = diagnostics
        return block

    def _log_signal(
        self,
        ticker: str,
        action: str,
        analysis: "RegimeAnalysis",
        plan: "SpreadPlan",
        verdict: "RiskVerdict",
        llm_decision: Optional["AnalystDecision"],
        exec_result: Optional[Dict],
        *,
        scan_results: Optional[Dict] = None,
    ) -> None:
        """Build raw_signal dict and write to JournalKB."""
        thesis = build_thesis(analysis, plan, verdict)

        raw: Dict = {
            "regime": analysis.regime.value,
            "strategy": plan.strategy_name,
            "plan_valid": plan.valid,
            "rejection_reason": plan.rejection_reason if not plan.valid else None,
            "risk_approved": verdict.approved,
            "net_credit": plan.net_credit if plan.valid else None,
            "max_loss": plan.max_loss if plan.valid else None,
            "credit_to_width_ratio": (
                plan.credit_to_width_ratio if plan.valid else None
            ),
            "spread_width": plan.spread_width if plan.valid else None,
            "expiration": plan.expiration if plan.valid else None,
            "sma_50": analysis.sma_50,
            "sma_200": analysis.sma_200,
            "rsi_14": analysis.rsi_14,
            "mean_reversion_signal": getattr(analysis, "mean_reversion_signal", False),
            "mean_reversion_direction": getattr(analysis, "mean_reversion_direction", ""),
            "leadership_anchor": getattr(analysis, "leadership_anchor", ""),
            "leadership_zscore": getattr(analysis, "leadership_zscore", 0.0),
            "leadership_raw_diff": getattr(analysis, "leadership_raw_diff", 0.0),
            "vix_zscore": getattr(analysis, "vix_zscore", 0.0),
            "inter_market_inhibit_bullish": getattr(
                analysis, "inter_market_inhibit_bullish", False),
            "account_balance": verdict.account_balance,
            "checks_passed": verdict.checks_passed,
            "checks_failed": verdict.checks_failed,
            "llm_decision": llm_decision.action if llm_decision else None,
            "llm_confidence": llm_decision.confidence if llm_decision else None,
            "order_id": (
                exec_result.get("order_id") if exec_result else None
            ),
            "run_id": exec_result.get("run_id") if exec_result else None,
            "thesis": thesis,
        }

        # Adaptive-scan diagnostics: top-K candidates + selected pick. Only
        # set when the planner ran the scanner this cycle; static mode emits
        # nothing so the journal stays compact.
        if scan_results is not None:
            raw["scan_results"] = scan_results

        exec_status = exec_result.get("status") if exec_result else action

        try:
            self.journal_kb.log_signal(
                ticker=ticker,
                action=action,
                price=analysis.current_price,
                raw_signal=raw,
                exec_status=exec_status,
            )
        except Exception as exc:
            logger.warning("[%s] JournalKB log failed: %s", ticker, exc)

    # ==================================================================
    # Risk guardrail helpers (delegated to daily_state module)
    # ==================================================================

    def _should_exit_spread(self, spread: SpreadPosition) -> bool:
        """
        3-cycle debounce guard for non-immediate exit signals.

        Immediate signals (HARD_STOP, STRIKE_PROXIMITY, DTE_SAFETY) bypass
        debounce and return True immediately.  All other signals require
        the SAME signal on 3 consecutive cycles before this returns True.
        """
        if spread.exit_signal == ExitSignal.HOLD:
            return False

        if spread.exit_signal in IMMEDIATE_EXIT_SIGNALS:
            logger.warning(
                "[%s] IMMEDIATE exit signal %s — bypassing debounce",
                spread.underlying, spread.exit_signal.value,
            )
            return True

        count = tally_exit_vote(
            self.daily_state,
            ticker=spread.underlying,
            signal_val=spread.exit_signal.value,
            required=EXIT_DEBOUNCE_REQUIRED,
        )

        if count >= EXIT_DEBOUNCE_REQUIRED:
            logger.warning(
                "[%s] Exit signal %s confirmed after %d cycles — acting",
                spread.underlying, spread.exit_signal.value, count,
            )
            return True

        logger.info(
            "[%s] Exit signal %s vote %d/%d — debouncing (next check in ~5 min)",
            spread.underlying, spread.exit_signal.value,
            count, EXIT_DEBOUNCE_REQUIRED,
        )
        return False

    def _check_liquidation_mode(self, equity: float,
                                buying_power: float) -> bool:
        """
        Returns True if buying power usage exceeds the configured threshold,
        signalling the agent should close positions rather than open new ones.
        """
        if equity <= 0:
            logger.warning("Equity <= 0 (%.2f) — Emergency Liquidation Mode", equity)
            return True
        initial_bp = equity * self.config.trading.margin_multiplier
        pct_used = 1.0 - (buying_power / initial_bp)
        limit = self.config.trading.max_buying_power_pct
        if pct_used > limit:
            logger.warning(
                "Buying power %.1f%% used (limit=%.0f%%) — Liquidation Mode",
                pct_used * 100, limit * 100,
            )
            self.journal_kb.log_cycle_error(
                "liquidation_mode_activated",
                {
                    "buying_power_used_pct": round(pct_used * 100, 1),
                    "limit_pct": limit * 100,
                    "equity": equity,
                    "buying_power": buying_power,
                },
            )
            return True
        return False

    # ==================================================================
    # Open-order dedup + stale-order maintenance
    # ==================================================================

    def _tickers_with_open_orders(self) -> Set[str]:
        """
        Return the set of underlying tickers that currently have at least
        one open (pending-fill) order on Alpaca.

        For multi-leg orders the broker's top-level ``symbol`` is empty,
        so we recover the underlying by parsing the OCC root from each
        leg.  Equity / single-leg orders carry the underlying directly.
        """
        out: Set[str] = set()
        try:
            open_orders = self.order_tracker.fetch_open_orders()
        except Exception as exc:
            logger.warning("Could not fetch open orders for dedup: %s", exc)
            return out

        for o in open_orders:
            top = (o.symbol or "").upper().strip()
            if top:
                # Equity / single-leg: top-level symbol is the ticker
                # (e.g. "SPY") — keep it as-is unless it's an OCC string.
                root = _root_from_occ(top) or top
                if root:
                    out.add(root)
            for leg in (o.legs or []):
                root = _root_from_occ((leg.get("symbol") or "").upper())
                if root:
                    out.add(root)
        return out

    def _cancel_stale_orders(self, agent_tickers: List[str]) -> None:
        """
        Cancel limit orders that have been on the broker's book longer
        than ``STALE_ORDER_MAX_AGE_MIN`` minutes.

        Scoping
        -------
        Only orders whose underlying matches one of ``agent_tickers``
        are cancelled — this keeps any manual trade you placed on a
        ticker the agent doesn't manage off the chopping block.  Orders
        younger than the threshold are also untouched, so a manual
        order placed in the last 15 minutes is safe.

        Why this matters
        ----------------
        The executor submits at the planning-time mid with
        ``time_in_force="day"`` and never re-prices.  On a 7-DTE put
        spread the achievable credit erodes minute by minute as theta
        drains, so an unfilled limit becomes structurally un-fillable.
        Cancelling and re-planning is the only way to keep the limit
        anywhere near the live mid.
        """
        try:
            open_orders = self.order_tracker.fetch_open_orders()
        except Exception as exc:
            logger.warning("Could not fetch open orders for stale check: %s", exc)
            return

        if not open_orders:
            return

        ticker_set = {t.upper() for t in (agent_tickers or [])}
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=STALE_ORDER_MAX_AGE_MIN)
        cancelled = 0

        for o in open_orders:
            # Recover the underlying.  Multi-leg orders have empty
            # top-level symbol; fall back to the first leg's OCC root.
            roots = set()
            top = (o.symbol or "").upper().strip()
            if top:
                roots.add(_root_from_occ(top) or top)
            for leg in (o.legs or []):
                r = _root_from_occ((leg.get("symbol") or "").upper())
                if r:
                    roots.add(r)
            if not roots & ticker_set:
                continue   # not on a ticker the agent manages

            # Parse Alpaca's RFC3339 created_at.  It's already UTC.
            created_raw = o.created_at or ""
            if not created_raw:
                continue
            try:
                created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
            except ValueError:
                logger.debug("Order %s: unparseable created_at %r — skipping",
                             o.order_id, created_raw)
                continue
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if created > cutoff:
                continue   # younger than the threshold

            age_min = (datetime.now(timezone.utc) - created).total_seconds() / 60.0
            logger.warning(
                "[%s] Cancelling stale order %s (age %.1f min > %d min) — "
                "next cycle will re-price",
                next(iter(roots & ticker_set)), o.order_id, age_min,
                STALE_ORDER_MAX_AGE_MIN,
            )
            if self.order_tracker.cancel_order(o.order_id):
                cancelled += 1

        if cancelled:
            self.journal_kb.log_cycle_error(
                "stale_orders_cancelled",
                {
                    "count": cancelled,
                    "max_age_minutes": STALE_ORDER_MAX_AGE_MIN,
                },
            )

    # ==================================================================
    # Order status check
    # ==================================================================

    def _check_order_statuses(self) -> Dict:
        """Fetch recent orders and log a summary."""
        try:
            open_orders = self.order_tracker.fetch_open_orders()
            recent_fills = self.order_tracker.fetch_recent_fills(limit=10)

            open_summary = self.order_tracker.summarize_orders(open_orders)
            fill_summary = self.order_tracker.summarize_orders(recent_fills)

            logger.info(
                "Open orders: %d | Recent fills: %d",
                open_summary["total"], fill_summary["total"],
            )

            return {
                "open_orders": open_summary,
                "recent_fills": fill_summary,
            }
        except Exception as exc:
            logger.warning("Could not check order statuses: %s", exc)
            return {"error": str(exc)}

    # ==================================================================
    # Helpers
    # ==================================================================

    def _cached_price(self, ticker: str) -> float:
        """Return cached price for *ticker* or 0.0 if unavailable.

        Delegates to the MarketDataPort.get_cached_price method so the
        agent never reaches into adapter-private caches.  Pre-week-5-6
        this poked ``data_provider._snapshot_cache`` and
        ``_price_cache`` directly, which was the classic "leaky
        abstraction" symptom the port refactor eliminates.
        """
        price = self.data_provider.get_cached_price(ticker)
        return float(price) if price is not None else 0.0

    def _check_daily_drawdown(self, current_equity: float) -> bool:
        """Thin wrapper for tests / legacy callers.

        The canonical implementation is
        :func:`trading_agent.daily_state.check_daily_drawdown`, which
        was extracted during the week 3-4 refactor.  This method keeps
        the pre-refactor instance-method shape so existing integration
        tests (and anyone who held on to the previous surface) don't
        have to rewire.  All real policy lives in daily_state.
        """
        return check_daily_drawdown(
            self.daily_state,
            current_equity=current_equity,
            drawdown_limit=self.config.trading.daily_drawdown_limit,
            journal_kb=self.journal_kb,
        )

    def _learn_from_close(self, spread: SpreadPosition):
        """Run post-trade LLM analysis when a spread is closed."""
        try:
            recent = self.llm_analyst.journal.get_trades_by_ticker(
                spread.underlying, limit=5)
            for trade in recent:
                if (trade.strategy_name == spread.strategy_name
                        and not trade.timestamp_closed):
                    self.llm_analyst.journal.close_trade(
                        trade_id=trade.trade_id,
                        exit_signal=spread.exit_signal.value,
                        exit_reason=spread.exit_reason,
                        realized_pl=spread.net_unrealized_pl,
                    )
                    trade = self.llm_analyst.journal.get_trade(trade.trade_id)
                    if trade:
                        self.llm_analyst.analyze_outcome(trade)
                        # Back-fill the outcome into the KB document so
                        # future RAG searches return outcome-labelled results
                        self.llm_analyst.knowledge_base.update_trade_outcome(
                            trade_id=trade.trade_id,
                            outcome_label=trade.outcome_label,
                            realized_pl=trade.realized_pl,
                            exit_signal=trade.exit_signal,
                            exit_reason=trade.exit_reason,
                            updated_text=trade.to_embedding_text(),
                        )
                    break
        except Exception as exc:
            logger.warning(
                "[%s] Post-trade learning failed: %s",
                spread.underlying, exc,
            )

    @staticmethod
    def _regime_to_strategy(regime) -> str:
        return {
            Regime.BULLISH: "Bull Put Spread",
            Regime.BEARISH: "Bear Call Spread",
            Regime.SIDEWAYS: "Iron Condor",
            Regime.MEAN_REVERSION: "Mean Reversion Spread",
        }.get(regime, "Unknown")

    @staticmethod
    def _analysis_dict(analysis: "RegimeAnalysis") -> Dict:
        return {
            "price": analysis.current_price,
            "sma_50": analysis.sma_50,
            "sma_200": analysis.sma_200,
            "rsi": analysis.rsi_14,
            "reasoning": analysis.reasoning,
        }

    def _print_summary(self, results: List[Dict]):
        """Log a human-readable summary table."""
        logger.info(
            "\n%-6s | %-10s | %-18s | %-8s | %-10s | %s",
            "Ticker", "Regime", "Strategy", "Valid", "Risk OK", "Status",
        )
        logger.info("-" * 80)
        for r in results:
            logger.info(
                "%-6s | %-10s | %-18s | %-8s | %-10s | %s",
                r.get("ticker", "?"),
                r.get("regime", "?"),
                r.get("strategy", "?"),
                r.get("plan_valid", "?"),
                r.get("risk_approved", "?"),
                r.get("execution", {}).get("status", r.get("status", "?")),
            )


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
