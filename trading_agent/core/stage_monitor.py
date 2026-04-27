"""
Stage 1 — Monitor Existing Positions (mixin)
=============================================
Provides the Stage 1 methods used by TradingAgent:

  _stage_monitor         — fetch positions, evaluate exit signals, close spreads
  _load_trade_plans      — load trade-plan JSON files from disk
  _should_exit_spread    — 3-cycle debounce guard for exit signals
  _check_liquidation_mode — buying-power liquidation check
  _check_order_statuses  — order tracker summary
  _cached_price          — price cache accessor
  _check_daily_drawdown  — thin wrapper around daily_state
  _learn_from_close      — post-trade LLM learning

These are defined as a mixin (_StageMonitorMixin) so they can be imported
and added to TradingAgent without rewriting class hierarchy.
"""

import glob
import json
import logging
import os
from typing import Dict, List, Optional

from trading_agent.utils.daily_state import (
    check_daily_drawdown,
    tally_exit_vote,
)
from trading_agent.execution.position_monitor import (
    ExitSignal,
    SpreadPosition,
    IMMEDIATE_EXIT_SIGNALS,
)

logger = logging.getLogger(__name__)


class _StageMonitorMixin:
    """
    Mixin providing Stage 1 (position monitoring) methods for TradingAgent.
    Relies on attributes set by TradingAgent.__init__:
      self.config, self.position_monitor, self.regime_classifier,
      self.executor, self.daily_state, self.journal_kb, self.llm_analyst
    """

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
            required=self._exit_debounce_required,
        )

        if count >= self._exit_debounce_required:
            logger.warning(
                "[%s] Exit signal %s confirmed after %d cycles — acting",
                spread.underlying, spread.exit_signal.value, count,
            )
            return True

        logger.info(
            "[%s] Exit signal %s vote %d/%d — debouncing (next check in ~5 min)",
            spread.underlying, spread.exit_signal.value,
            count, self._exit_debounce_required,
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
        """Return cached price for *ticker* or 0.0 if unavailable."""
        price = self.data_provider.get_cached_price(ticker)
        return float(price) if price is not None else 0.0

    def _check_daily_drawdown(self, current_equity: float) -> bool:
        """Thin wrapper for tests / legacy callers.

        The canonical implementation is
        :func:`trading_agent.daily_state.check_daily_drawdown`.
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
