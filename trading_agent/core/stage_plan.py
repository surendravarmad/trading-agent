"""
Stage 2 — Open New Positions (mixin)
=====================================
Provides the Stage 2 methods used by TradingAgent:

  _process_ticker      — per-ticker Phase I–VI pipeline
  _log_signal          — JournalKB signal helper
  _regime_to_strategy  — regime → strategy name mapping
  _analysis_dict       — RegimeAnalysis → summary dict
  _print_summary       — human-readable cycle summary

These are defined as a mixin (_StagePlanMixin) so they can be imported
and composed into TradingAgent without rewriting class hierarchy.
"""

import logging
from concurrent.futures import Future
from typing import Dict, List, Optional, TYPE_CHECKING

from trading_agent.strategy.regime import Regime, RegimeAnalysis
from trading_agent.strategy.strategy import SpreadPlan
from trading_agent.strategy.risk_manager import RiskVerdict
from trading_agent.utils.thesis_builder import build_thesis

if TYPE_CHECKING:
    from trading_agent.intelligence.llm_analyst import AnalystDecision
    from trading_agent.sentiment.sentiment_verifier import VerifiedSentimentReport

logger = logging.getLogger(__name__)


class _StagePlanMixin:
    """
    Mixin providing Stage 2 (new trade entry) methods for TradingAgent.
    Relies on attributes set by TradingAgent.__init__:
      self.config, self.data_provider, self.regime_classifier,
      self.strategy_planner, self.risk_manager, self.executor,
      self.journal_kb, self.llm_analyst, self.sentiment_pipeline
    """

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

        # Launch tiered sentiment pipeline concurrently after Phase II
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

        logger.info("[%s] Phase IV — RISK CHECK", ticker)
        verdict: RiskVerdict = self.risk_manager.evaluate(
            plan, balance, acct_type, market_open,
            self.config.trading.force_market_open,
            underlying_bid_ask=underlying_bid_ask,
            account_buying_power=buying_power,
        )

        # Resolve sentiment pipeline result
        sentiment: Optional["VerifiedSentimentReport"] = None
        if fingpt_future is not None:
            if not (plan.valid and verdict.approved):
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
                    llm_decision, exec_result=None)
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

        # Phase VI: Execute trade
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

    def _log_signal(
        self,
        ticker: str,
        action: str,
        analysis: "RegimeAnalysis",
        plan: "SpreadPlan",
        verdict: "RiskVerdict",
        llm_decision: Optional["AnalystDecision"],
        exec_result: Optional[Dict],
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
            logger.warning("[%s] JournalKB log_signal failed: %s", ticker, exc)

    # ==================================================================
    # Static helpers
    # ==================================================================

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
