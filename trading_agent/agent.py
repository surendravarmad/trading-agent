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
"""

import glob
import json
import logging
import os
from datetime import datetime
from typing import Dict, List

from trading_agent.config import AppConfig, load_config
from trading_agent.logger_setup import setup_logging
from trading_agent.market_data import MarketDataProvider
from trading_agent.regime import Regime, RegimeClassifier, RegimeAnalysis
from trading_agent.strategy import StrategyPlanner, SpreadPlan
from trading_agent.risk_manager import RiskManager, RiskVerdict
from trading_agent.executor import OrderExecutor
from trading_agent.position_monitor import (
    PositionMonitor, ExitSignal, SpreadPosition,
)
from trading_agent.order_tracker import OrderTracker
from trading_agent.llm_client import LLMClient, LLMConfig
from trading_agent.trade_journal import TradeJournal, TradeEntry
from trading_agent.knowledge_base import KnowledgeBase
from trading_agent.llm_analyst import LLMAnalyst, AnalystDecision

logger = logging.getLogger(__name__)


class TradingAgent:
    """
    Autonomous credit-spread trading agent.

    Lifecycle:
        agent = TradingAgent.from_env()
        results = agent.run_cycle()
    """

    def __init__(self, config: AppConfig):
        self.config = config

        # Wire up components
        self.data_provider = MarketDataProvider(
            alpaca_api_key=config.alpaca.api_key,
            alpaca_secret_key=config.alpaca.secret_key,
            alpaca_data_url=config.alpaca.data_url,
        )
        self.regime_classifier = RegimeClassifier(self.data_provider)
        self.strategy_planner = StrategyPlanner(
            data_provider=self.data_provider,
            max_delta=config.trading.max_delta,
            min_credit_ratio=config.trading.min_credit_ratio,
        )
        self.risk_manager = RiskManager(
            max_risk_pct=config.trading.max_risk_pct,
            min_credit_ratio=config.trading.min_credit_ratio,
            max_delta=config.trading.max_delta,
        )
        self.executor = OrderExecutor(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
            base_url=config.alpaca.base_url,
            trade_plan_dir=config.logging.trade_plan_dir,
            dry_run=config.trading.dry_run,
        )
        self.position_monitor = PositionMonitor(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
            base_url=config.alpaca.base_url,
        )
        self.order_tracker = OrderTracker(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
            base_url=config.alpaca.base_url,
        )

        # Intelligence layer (LLM + RAG + Journal)
        self.llm_analyst = self._init_intelligence(config)

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

            logger.info("Intelligence layer ENABLED — model=%s, provider=%s",
                         intel_cfg.llm_model, intel_cfg.llm_provider)
            return analyst

        except Exception as exc:
            logger.warning("Failed to initialize intelligence layer: %s — "
                          "continuing in rule-based mode", exc)
            return None

    @classmethod
    def from_env(cls, env_path: str = None) -> "TradingAgent":
        """Factory: create agent from environment / .env file."""
        config = load_config(env_path)
        setup_logging(config.logging.log_level, config.logging.log_dir)
        return cls(config)

    # ==================================================================
    # Main cycle
    # ==================================================================

    def run_cycle(self) -> List[Dict]:
        """
        Execute one full cycle:
          Stage 1 — Monitor & manage existing positions
          Stage 2 — Open new positions for tickers without open spreads
        """
        logger.info("=" * 70)
        logger.info("TRADING CYCLE START — %s", datetime.utcnow().isoformat())
        logger.info("Tickers: %s | Mode: %s | Dry-run: %s",
                     self.config.trading.tickers,
                     self.config.trading.mode,
                     self.config.trading.dry_run)
        logger.info("=" * 70)

        # Pre-flight: fetch account info
        account = self.data_provider.get_account_info(self.config.alpaca.base_url)
        if not account:
            logger.error("Cannot fetch account info — aborting cycle.")
            return [{"status": "error", "reason": "Account info unavailable"}]

        account_balance = float(account.get("equity", 0))
        account_type = "paper" if "paper" in self.config.alpaca.base_url else "live"
        market_open = self.data_provider.is_market_open(self.config.alpaca.base_url)

        logger.info("Account: balance=$%s, type=%s, market_open=%s",
                     f"{account_balance:,.2f}", account_type, market_open)

        # ------------------------------------------------------------------
        # Stage 1: MONITOR existing positions
        # ------------------------------------------------------------------
        logger.info("=" * 70)
        logger.info("STAGE 1 — MONITOR EXISTING POSITIONS")
        logger.info("=" * 70)

        monitor_results = self._stage_monitor(account_balance)

        # Determine which tickers already have open positions
        tickers_with_positions = set()
        for sr in monitor_results.get("positions", []):
            if sr.get("signal") == ExitSignal.HOLD.value:
                tickers_with_positions.add(sr.get("underlying", ""))

        # ------------------------------------------------------------------
        # Stage 2: OPEN new positions for tickers without spreads
        # ------------------------------------------------------------------
        logger.info("=" * 70)
        logger.info("STAGE 2 — OPEN NEW POSITIONS")
        logger.info("=" * 70)

        new_trade_results = []
        for ticker in self.config.trading.tickers:
            if ticker in tickers_with_positions:
                logger.info("[%s] Already has an open spread — skipping new entry",
                            ticker)
                new_trade_results.append({
                    "ticker": ticker,
                    "status": "skipped",
                    "reason": "Existing open position",
                })
                continue

            try:
                result = self._process_ticker(
                    ticker, account_balance, account_type, market_open)
                new_trade_results.append(result)
            except Exception as exc:
                logger.exception("[%s] Unhandled error in cycle: %s", ticker, exc)
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
        logger.info("TRADING CYCLE COMPLETE — %d tickers processed",
                     len(new_trade_results))
        self._print_summary(new_trade_results)
        logger.info("=" * 70)

        return {
            "timestamp": datetime.utcnow().isoformat(),
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
        # 1. Fetch open option positions
        positions = self.position_monitor.fetch_open_positions()
        if not positions:
            logger.info("No open option positions found.")
            return {"total_spreads": 0, "positions": [], "closed": []}

        # 2. Load trade plans to match positions to their original entry
        trade_plans = self._load_trade_plans()

        # 3. Group positions into spreads
        spreads = self.position_monitor.group_into_spreads(positions, trade_plans)
        if not spreads:
            logger.info("Could not match positions to any trade plans.")
            return {"total_spreads": 0, "positions": [], "closed": []}

        # 4. Classify current regimes for underlyings with positions
        underlyings = {s.underlying for s in spreads}
        current_regimes = {}
        for ticker in underlyings:
            try:
                analysis = self.regime_classifier.classify(ticker)
                current_regimes[ticker] = analysis.regime
                logger.info("[%s] Current regime: %s", ticker, analysis.regime.value)
            except Exception as exc:
                logger.warning("[%s] Could not classify regime: %s", ticker, exc)

        # 5. Evaluate exit signals
        spreads = self.position_monitor.evaluate(spreads, current_regimes)

        # 6. Close spreads with exit signals + post-trade learning
        closed = []
        for spread in spreads:
            if spread.exit_signal != ExitSignal.HOLD:
                if self.config.trading.dry_run:
                    logger.info("[%s] DRY RUN — would close %s (%s: %s)",
                                spread.underlying, spread.strategy_name,
                                spread.exit_signal.value, spread.exit_reason)
                    closed.append({
                        "underlying": spread.underlying,
                        "signal": spread.exit_signal.value,
                        "reason": spread.exit_reason,
                        "action": "dry_run_close",
                    })
                else:
                    result = self.executor.close_spread(spread)
                    closed.append(result)

                # Post-trade LLM analysis (learn from outcome)
                if self.llm_analyst:
                    self._learn_from_close(spread)

        summary = self.position_monitor.summary(spreads)
        summary["closed"] = closed
        return summary

    def _load_trade_plans(self) -> List[Dict]:
        """Load all trade plan JSON files from the plan directory."""
        plan_dir = self.config.logging.trade_plan_dir
        if not os.path.isdir(plan_dir):
            return []
        plans = []
        for path in sorted(glob.glob(os.path.join(plan_dir, "trade_plan_*.json"))):
            try:
                with open(path) as f:
                    plans.append(json.load(f))
            except Exception as exc:
                logger.warning("Could not load plan %s: %s", path, exc)
        logger.info("Loaded %d trade plan(s) from %s", len(plans), plan_dir)
        return plans

    # ==================================================================
    # Stage 2: New trade entry (original four-phase loop)
    # ==================================================================

    def _process_ticker(self, ticker: str, balance: float,
                        acct_type: str, market_open: bool) -> Dict:
        """Full four-phase pipeline for a single ticker (+ LLM Phase V)."""
        logger.info("-" * 50)
        logger.info("[%s] Phase I  — PERCEIVE", ticker)

        # Phase II: Classify
        logger.info("[%s] Phase II — CLASSIFY", ticker)
        analysis: RegimeAnalysis = self.regime_classifier.classify(ticker)

        # Phase III: Plan
        logger.info("[%s] Phase III — PLAN (%s → %s)", ticker,
                     analysis.regime.value,
                     self._regime_to_strategy(analysis.regime))
        plan: SpreadPlan = self.strategy_planner.plan(ticker, analysis)

        # Phase IV: Validate risk
        logger.info("[%s] Phase IV — RISK CHECK", ticker)
        verdict: RiskVerdict = self.risk_manager.evaluate(
            plan, balance, acct_type, market_open,
            self.config.trading.force_market_open)

        # Phase V: LLM Analysis (if enabled)
        llm_decision = None
        if self.llm_analyst and plan.valid and verdict.approved:
            logger.info("[%s] Phase V  — LLM ANALYSIS", ticker)
            llm_decision = self.llm_analyst.analyze_trade(
                ticker, analysis, plan, verdict)

            if llm_decision.action == "skip":
                logger.warning(
                    "[%s] LLM SKIPPED trade (confidence=%.2f): %s",
                    ticker, llm_decision.confidence,
                    llm_decision.reasoning[:150])
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
                    "analysis": {
                        "price": analysis.current_price,
                        "sma_50": analysis.sma_50,
                        "sma_200": analysis.sma_200,
                        "rsi": analysis.rsi_14,
                        "reasoning": analysis.reasoning,
                    },
                }

        # Execute trade
        logger.info("[%s] Phase VI — EXECUTE", ticker)
        exec_result = self.executor.execute(verdict)

        # Journal the trade (if LLM enabled)
        if self.llm_analyst and llm_decision and exec_result.get("status") in ("submitted", "dry_run"):
            try:
                entry = self.llm_analyst.create_journal_entry(
                    ticker, analysis, plan, verdict, llm_decision)
                entry.order_status = exec_result.get("status", "")
                entry.order_id = exec_result.get("order_id", "")
                trade_id = self.llm_analyst.journal.open_trade(entry)
                exec_result["trade_journal_id"] = trade_id
            except Exception as exc:
                logger.warning("[%s] Failed to journal trade: %s", ticker, exc)

        result = {
            "ticker": ticker,
            "regime": analysis.regime.value,
            "strategy": plan.strategy_name,
            "plan_valid": plan.valid,
            "risk_approved": verdict.approved,
            "execution": exec_result,
            "analysis": {
                "price": analysis.current_price,
                "sma_50": analysis.sma_50,
                "sma_200": analysis.sma_200,
                "rsi": analysis.rsi_14,
                "reasoning": analysis.reasoning,
            },
        }

        if llm_decision:
            result["llm_decision"] = llm_decision.action
            result["llm_confidence"] = llm_decision.confidence
            result["llm_reasoning"] = llm_decision.reasoning
            result["llm_warnings"] = llm_decision.warnings

        return result

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

            logger.info("Open orders: %d | Recent fills: %d",
                        open_summary["total"], fill_summary["total"])

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

    def _learn_from_close(self, spread: SpreadPosition):
        """Run post-trade LLM analysis when a spread is closed."""
        try:
            # Find the journal entry for this spread
            recent = self.llm_analyst.journal.get_trades_by_ticker(
                spread.underlying, limit=5)
            for trade in recent:
                if (trade.strategy_name == spread.strategy_name and
                        not trade.timestamp_closed):
                    # Close in journal
                    self.llm_analyst.journal.close_trade(
                        trade_id=trade.trade_id,
                        exit_signal=spread.exit_signal.value,
                        exit_reason=spread.exit_reason,
                        realized_pl=spread.net_unrealized_pl,
                    )
                    # LLM post-trade analysis
                    trade = self.llm_analyst.journal.get_trade(trade.trade_id)
                    if trade:
                        self.llm_analyst.analyze_outcome(trade)
                    break
        except Exception as exc:
            logger.warning("[%s] Post-trade learning failed: %s",
                          spread.underlying, exc)

    @staticmethod
    def _regime_to_strategy(regime) -> str:
        return {
            Regime.BULLISH: "Bull Put Spread",
            Regime.BEARISH: "Bear Call Spread",
            Regime.SIDEWAYS: "Iron Condor",
        }.get(regime, "Unknown")

    def _print_summary(self, results: List[Dict]):
        """Log a human-readable summary table."""
        logger.info("\n%-6s | %-10s | %-18s | %-8s | %-10s | %s",
                     "Ticker", "Regime", "Strategy", "Valid", "Risk OK", "Status")
        logger.info("-" * 80)
        for r in results:
            logger.info("%-6s | %-10s | %-18s | %-8s | %-10s | %s",
                         r.get("ticker", "?"),
                         r.get("regime", "?"),
                         r.get("strategy", "?"),
                         r.get("plan_valid", "?"),
                         r.get("risk_approved", "?"),
                         r.get("execution", {}).get("status",
                                                     r.get("status", "?")))


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    """Run a single trading cycle from the command line."""
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Options Trading Agent")
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
