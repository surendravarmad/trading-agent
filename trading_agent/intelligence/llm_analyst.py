"""
LLM Analyst — The Intelligence Layer
======================================
The brain of the trading agent. Uses a local LLM (Ollama) augmented
with RAG (past trade history + lessons) to:

  1. ANALYZE  — Richer trade reasoning beyond simple indicators
  2. DECIDE   — Final approval/modification/skip decision on each trade
  3. LEARN    — Post-trade analysis that feeds back into the knowledge base
  4. TUNE     — Recommend parameter adjustments based on accumulated data

The analyst operates as an advisory layer ON TOP of the rule-based system.
Rules are never bypassed — the LLM can only tighten constraints, not loosen.
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from trading_agent.intelligence.llm_client import LLMClient, LLMConfig
from trading_agent.intelligence.trade_journal import TradeJournal, TradeEntry
from trading_agent.intelligence.knowledge_base import KnowledgeBase
from trading_agent.strategy.regime import Regime, RegimeAnalysis
from trading_agent.strategy.strategy import SpreadPlan
from trading_agent.strategy.risk_manager import RiskVerdict
from trading_agent.sentiment.fingpt_analyser import SentimentReport
from trading_agent.core.ports import SentimentReadout

logger = logging.getLogger(__name__)


@dataclass
class AnalystDecision:
    """The LLM analyst's verdict on a proposed trade."""
    action: str                      # "approve", "modify", "skip"
    confidence: float                # 0.0 - 1.0
    reasoning: str                   # Full chain-of-thought reasoning
    risk_assessment: str             # Additional risk factors spotted
    similar_trades_summary: str      # What happened in similar past trades
    modifications: Dict              # Suggested parameter changes (if action="modify")
    warnings: List[str]              # Edge cases or concerns flagged


SYSTEM_PROMPT = """You are an expert options trading analyst specializing in credit spreads.
Your role is to review proposed trades and provide a final decision: approve, modify, or skip.

CORE PRINCIPLES:
1. Capital preservation is the #1 priority — when in doubt, skip the trade.
2. You are an ADVISORY layer on top of rule-based risk checks. You can only TIGHTEN
   constraints, never loosen them. If the risk manager rejected a trade, you cannot
   override that rejection.
3. Theta decay is the edge — we sell time, not direction.
4. Context matters: similar past trades, current volatility environment, and
   regime stability all factor into the decision.

DECISION FRAMEWORK:
- APPROVE: Trade passes all rules AND you see favorable conditions in context
- MODIFY: Trade is sound but you recommend adjusting strikes, DTE, or position size
- SKIP: Red flags in market context, recent similar trades that failed, regime instability

You MUST respond in JSON format with these exact fields:
{
  "action": "approve" | "modify" | "skip",
  "confidence": 0.0-1.0,
  "reasoning": "Your chain-of-thought analysis",
  "risk_assessment": "Additional risk factors you identified",
  "similar_trades_summary": "What you learned from similar past trades",
  "modifications": {"field": "new_value"} or {},
  "warnings": ["list", "of", "concerns"]
}"""


ANALYSIS_PROMPT_TEMPLATE = """## Proposed Trade

**Ticker:** {ticker}
**Strategy:** {strategy_name}
**Regime:** {regime} (SMA50 slope: ${sma_50_slope:.4f}/day — raw price delta, sign indicates trend direction)

**Market Data:**
- Current Price: ${current_price:.2f}
- SMA-50: ${sma_50:.2f}
- SMA-200: ${sma_200:.2f}
- RSI-14: {rsi_14:.1f}
- Bollinger Width: {bollinger_width:.4f}

**Spread Details:**
- Sold strike delta: {sold_delta:.3f}
- Spread width: ${spread_width:.2f}
- Net credit: ${net_credit:.2f}
- Credit/Width ratio: {credit_ratio:.4f}
- Max loss: ${max_loss:.2f}
- Expiration: {expiration} ({dte} DTE)

**Risk Manager Verdict:** {risk_verdict}

{sentiment_section}

{similar_trades_section}

{lessons_section}

{performance_stats_section}

Based on this analysis, should this trade be approved, modified, or skipped?
Respond in the JSON format specified in your system instructions."""


POST_TRADE_PROMPT_TEMPLATE = """## Post-Trade Analysis

Analyze this completed trade and extract lessons learned:

**Trade:** {strategy_name} on {ticker}
**Entry regime:** {entry_regime}
**Entry price:** ${entry_price:.2f}

**Spread:** Credit ${net_credit:.2f}, Width ${spread_width:.2f}, Ratio {credit_ratio:.4f}
**Sold delta:** {sold_delta:.3f}, DTE at entry: {dte_at_entry}

**Outcome:**
- Exit signal: {exit_signal}
- Exit reason: {exit_reason}
- Realized P&L: ${realized_pl:.2f} ({realized_pl_pct:.1f}%)
- Hold duration: {hold_duration_days} days
- Regime at close: {regime_at_close}

**LLM reasoning at entry:** {entry_reasoning}

Please provide:
1. What went right or wrong
2. What signals were present at entry that predicted this outcome
3. Specific, actionable lessons for future trades
4. Whether the entry parameters (delta, DTE, credit ratio) were appropriate

Respond in JSON:
{{
  "analysis": "Your detailed post-trade analysis",
  "lessons": ["lesson1", "lesson2", ...],
  "parameter_feedback": {{
    "delta_assessment": "too aggressive/conservative/appropriate",
    "dte_assessment": "too short/long/appropriate",
    "credit_ratio_assessment": "too low/high/appropriate"
  }},
  "pattern_identified": "Description of any pattern you noticed"
}}"""


TUNING_PROMPT_TEMPLATE = """## Parameter Tuning Review

Based on the trading agent's performance history, recommend parameter adjustments.

**Current Parameters:**
- max_delta: {max_delta}
- min_credit_ratio: {min_credit_ratio}
- max_risk_pct: {max_risk_pct}
- TARGET_DTE: {target_dte}
- stop_loss_pct: {stop_loss_pct}
- profit_target_pct: {profit_target_pct}

**Performance Stats:**
{stats_json}

**Recent Lessons:**
{recent_lessons}

Should any parameters be adjusted? Only recommend changes if there's clear
evidence from the data. Conservative changes only — capital preservation first.

Respond in JSON:
{{
  "recommendations": [
    {{
      "parameter": "param_name",
      "current_value": 0.0,
      "suggested_value": 0.0,
      "reasoning": "Why this change improves outcomes",
      "confidence": 0.0-1.0
    }}
  ],
  "overall_assessment": "Brief assessment of strategy health",
  "no_changes_needed": true/false
}}"""


class LLMAnalyst:
    """
    Intelligence layer that uses a local LLM + RAG to make
    better trading decisions and learn from outcomes.
    """

    def __init__(self, llm_client: LLMClient,
                 journal: TradeJournal,
                 knowledge_base: KnowledgeBase,
                 enabled: bool = True):
        self.llm = llm_client
        self.journal = journal
        self.kb = knowledge_base
        self.enabled = enabled

        # Check if LLM is reachable
        if self.enabled and not self.llm.is_available():
            logger.warning(
                "LLM not available at %s — analyst will operate in "
                "passthrough mode (all trades approved by default)",
                self.llm.config.base_url)
            self.enabled = False

    # ==================================================================
    # Phase 1: Pre-trade analysis and decision
    # ==================================================================

    def analyze_trade(self, ticker: str, analysis: RegimeAnalysis,
                      plan: SpreadPlan, verdict: RiskVerdict,
                      sentiment: Optional[SentimentReadout] = None) -> AnalystDecision:
        """
        Review a proposed trade plan and decide: approve, modify, or skip.

        This runs AFTER the rule-based risk manager — if the risk manager
        rejected the trade, the analyst cannot override that.

        Args:
            sentiment: Optional FinGPT SentimentReport from the news layer.
                       When provided, macro/event context is injected into
                       the analysis prompt. A FinGPT "avoid" recommendation
                       is escalated to a warning but does not override risk
                       manager approval — it informs the LLM's own decision.
        """
        if not self.enabled:
            return self._passthrough_decision(verdict)

        # If risk manager already rejected, don't bother analyzing
        if not verdict.approved:
            return AnalystDecision(
                action="skip",
                confidence=1.0,
                reasoning=f"Risk manager rejected: {verdict.summary}",
                risk_assessment="Pre-flight risk checks failed",
                similar_trades_summary="",
                modifications={},
                warnings=verdict.checks_failed,
            )

        # Gather RAG context
        similar_trades = self._find_similar_trades(ticker, analysis, plan)
        lessons = self._find_relevant_lessons(ticker, analysis)
        stats = self.journal.get_stats()

        # Build the prompt (sentiment section injected when available)
        prompt = self._build_analysis_prompt(
            ticker, analysis, plan, verdict, similar_trades, lessons, stats,
            sentiment=sentiment)

        # Query the LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        decision_data = self.llm.chat_json(messages)

        if not decision_data:
            logger.warning("LLM returned empty response — defaulting to approve")
            return self._passthrough_decision(verdict)

        # Parse decision
        decision = AnalystDecision(
            action=decision_data.get("action", "approve"),
            confidence=float(decision_data.get("confidence", 0.5)),
            reasoning=decision_data.get("reasoning", ""),
            risk_assessment=decision_data.get("risk_assessment", ""),
            similar_trades_summary=decision_data.get("similar_trades_summary", ""),
            modifications=decision_data.get("modifications", {}),
            warnings=decision_data.get("warnings", []),
        )

        # Safety: LLM can never approve what the risk manager rejected
        if decision.action == "approve" and not verdict.approved:
            decision.action = "skip"
            decision.warnings.append("LLM tried to approve a risk-rejected trade — overridden")

        logger.info("[%s] LLM Analyst: %s (confidence=%.2f) — %s",
                     ticker, decision.action.upper(), decision.confidence,
                     decision.reasoning[:100])

        return decision

    # ==================================================================
    # Phase 2: Post-trade learning
    # ==================================================================

    def analyze_outcome(self, trade: TradeEntry) -> Dict:
        """
        Analyze a completed trade and extract lessons for the knowledge base.
        Called when a trade is closed (any exit signal).
        """
        if not self.enabled:
            return {"analysis": "", "lessons": []}

        prompt = POST_TRADE_PROMPT_TEMPLATE.format(
            strategy_name=trade.strategy_name,
            ticker=trade.ticker,
            entry_regime=trade.regime,
            entry_price=trade.current_price,
            net_credit=trade.net_credit,
            spread_width=trade.spread_width,
            credit_ratio=trade.credit_to_width_ratio,
            sold_delta=trade.sold_delta,
            dte_at_entry=trade.dte_at_entry,
            exit_signal=trade.exit_signal,
            exit_reason=trade.exit_reason,
            realized_pl=trade.realized_pl,
            realized_pl_pct=trade.realized_pl_pct,
            hold_duration_days=trade.hold_duration_days,
            regime_at_close=trade.regime_at_close,
            entry_reasoning=trade.llm_reasoning or "No LLM reasoning recorded",
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        result = self.llm.chat_json(messages)
        if not result:
            return {"analysis": "", "lessons": []}

        # Store in knowledge base
        analysis_text = result.get("analysis", "")
        lessons = result.get("lessons", [])

        # Add trade to KB for future similarity search
        self.kb.add_trade(
            trade_id=trade.trade_id,
            text=trade.to_embedding_text(),
            metadata={
                "ticker": trade.ticker,
                "strategy": trade.strategy_name,
                "outcome": trade.outcome_label,
                "pl": trade.realized_pl,
                "regime": trade.regime,
            },
        )

        # Add each lesson as a separate KB document
        for lesson in lessons:
            self.kb.add_lesson(
                lesson_text=lesson,
                trade_id=trade.trade_id,
                metadata={
                    "ticker": trade.ticker,
                    "strategy": trade.strategy_name,
                    "outcome": trade.outcome_label,
                },
            )

        # Add pattern if identified
        pattern = result.get("pattern_identified", "")
        if pattern:
            self.kb.add_strategy_note(
                note=pattern,
                strategy=trade.strategy_name,
                metadata={"source_trade": trade.trade_id},
            )

        # Update trade journal
        self.journal.add_llm_analysis(trade.trade_id, analysis_text, lessons)

        logger.info("[%s] Post-trade analysis complete: %d lessons extracted",
                     trade.ticker, len(lessons))

        return result

    # ==================================================================
    # Phase 3: Parameter tuning recommendations
    # ==================================================================

    def recommend_tuning(self, current_params: Dict) -> Dict:
        """
        Analyze performance history and recommend parameter adjustments.
        Should be called periodically (e.g., weekly) not every cycle.
        """
        if not self.enabled:
            return {"no_changes_needed": True, "recommendations": []}

        stats = self.journal.get_stats()
        if stats.get("total_trades", 0) < 10:
            logger.info("Not enough trade history for tuning (%d trades)",
                        stats.get("total_trades", 0))
            return {"no_changes_needed": True,
                    "reason": "Insufficient trade history (need 10+)"}

        # Get recent lessons
        recent_lessons = self.kb.get_all_lessons()
        lessons_text = "\n".join(
            f"- {l.text}" for l in recent_lessons[-20:])

        prompt = TUNING_PROMPT_TEMPLATE.format(
            max_delta=current_params.get("max_delta", 0.20),
            min_credit_ratio=current_params.get("min_credit_ratio", 0.33),
            max_risk_pct=current_params.get("max_risk_pct", 0.02),
            target_dte=current_params.get("target_dte", 45),
            stop_loss_pct=current_params.get("stop_loss_pct", 0.50),
            profit_target_pct=current_params.get("profit_target_pct", 0.75),
            stats_json=json.dumps(stats, indent=2),
            recent_lessons=lessons_text or "No lessons recorded yet.",
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        result = self.llm.chat_json(messages)
        if not result:
            return {"no_changes_needed": True, "recommendations": []}

        logger.info("Tuning recommendations: %s",
                     json.dumps(result, indent=2)[:500])
        return result

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _passthrough_decision(self, verdict: RiskVerdict) -> AnalystDecision:
        """Default decision when LLM is disabled — defer to rule-based system."""
        return AnalystDecision(
            action="approve" if verdict.approved else "skip",
            confidence=1.0 if verdict.approved else 0.0,
            reasoning="LLM analyst disabled — deferring to rule-based system",
            risk_assessment="",
            similar_trades_summary="",
            modifications={},
            warnings=[],
        )

    def _find_similar_trades(self, ticker: str, analysis: RegimeAnalysis,
                             plan: SpreadPlan) -> List[Tuple]:
        """Search KB for similar historical trades."""
        query = (
            f"{plan.strategy_name} on {ticker}, "
            f"{analysis.regime.value} regime, "
            f"RSI={analysis.rsi_14:.1f}, "
            f"price=${analysis.current_price:.2f}, "
            f"credit ratio={plan.credit_to_width_ratio:.4f}"
        )
        return self.kb.get_similar_trades(query, top_k=3)

    def _find_relevant_lessons(self, ticker: str,
                                analysis: RegimeAnalysis) -> List[Tuple]:
        """Search KB for relevant lessons."""
        query = (
            f"{ticker} {analysis.regime.value} regime, "
            f"RSI={analysis.rsi_14:.1f}, "
            f"BB width={analysis.bollinger_width:.4f}"
        )
        return self.kb.get_relevant_lessons(query, top_k=3)

    def _build_analysis_prompt(self, ticker, analysis, plan, verdict,
                                similar_trades, lessons, stats,
                                sentiment: Optional[SentimentReadout] = None) -> str:
        """Build the full analysis prompt with RAG context and optional FinGPT sentiment."""

        # Similar trades section
        if similar_trades:
            similar_lines = []
            for doc, score in similar_trades:
                similar_lines.append(
                    f"[Similarity: {score:.2f}]\n{doc.text}\n")
            similar_section = (
                "## Similar Past Trades\n" + "\n".join(similar_lines))
        else:
            similar_section = "## Similar Past Trades\nNo similar trades found yet."

        # Lessons section
        if lessons:
            lesson_lines = [f"- {doc.text}" for doc, _ in lessons]
            lessons_section = (
                "## Relevant Lessons\n" + "\n".join(lesson_lines))
        else:
            lessons_section = "## Relevant Lessons\nNo lessons recorded yet."

        # Performance stats
        if stats.get("total_trades", 0) > 0:
            stats_section = (
                f"## Performance History\n"
                f"Win rate: {stats.get('win_rate', 0)*100:.1f}% "
                f"({stats.get('wins', 0)}W / {stats.get('losses', 0)}L)\n"
                f"Avg P&L: ${stats.get('avg_pl_per_trade', 0):.2f}\n"
                f"Total P&L: ${stats.get('total_pl', 0):.2f}")
        else:
            stats_section = "## Performance History\nNo completed trades yet."

        # Get sold delta
        sold_delta = 0.0
        for leg in plan.legs:
            if leg.action == "sell":
                sold_delta = abs(leg.delta)
                break

        # Calculate DTE
        dte = 0
        try:
            from datetime import datetime
            exp = datetime.strptime(plan.expiration, "%Y-%m-%d")
            dte = (exp - datetime.utcnow()).days
        except Exception:
            dte = 30

        # FinGPT sentiment section (empty string when not available)
        if sentiment:
            sentiment_section = sentiment.to_prompt_section()
            if sentiment.recommendation == "avoid":
                sentiment_section += (
                    "\n\n> ⚠️ FinGPT flags HIGH EVENT RISK — strongly consider skipping "
                    "this trade regardless of technical signals."
                )
        else:
            sentiment_section = "## FinGPT Sentiment Analysis\nNot available (disabled or no headlines)."

        return ANALYSIS_PROMPT_TEMPLATE.format(
            ticker=ticker,
            strategy_name=plan.strategy_name,
            regime=analysis.regime.value,
            sma_50_slope=analysis.sma_50_slope,
            current_price=analysis.current_price,
            sma_50=analysis.sma_50,
            sma_200=analysis.sma_200,
            rsi_14=analysis.rsi_14,
            bollinger_width=analysis.bollinger_width,
            sold_delta=sold_delta,
            spread_width=plan.spread_width,
            net_credit=plan.net_credit,
            credit_ratio=plan.credit_to_width_ratio,
            max_loss=plan.max_loss,
            expiration=plan.expiration,
            dte=dte,
            risk_verdict="APPROVED" if verdict.approved else "REJECTED",
            sentiment_section=sentiment_section,
            similar_trades_section=similar_section,
            lessons_section=lessons_section,
            performance_stats_section=stats_section,
        )

    # ==================================================================
    # Journal integration helpers
    # ==================================================================

    def create_journal_entry(self, ticker: str, analysis: RegimeAnalysis,
                             plan: SpreadPlan, verdict: RiskVerdict,
                             decision: AnalystDecision) -> TradeEntry:
        """Create a TradeEntry from the current trade context."""
        sold_delta = 0.0
        for leg in plan.legs:
            if leg.action == "sell":
                sold_delta = abs(leg.delta)
                break

        dte = 0
        try:
            from datetime import datetime
            exp = datetime.strptime(plan.expiration, "%Y-%m-%d")
            dte = (exp - datetime.utcnow()).days
        except Exception:
            pass

        return TradeEntry(
            ticker=ticker,
            strategy_name=plan.strategy_name,
            regime=analysis.regime.value,
            current_price=analysis.current_price,
            sma_50=analysis.sma_50,
            sma_200=analysis.sma_200,
            sma_50_slope=analysis.sma_50_slope,
            rsi_14=analysis.rsi_14,
            bollinger_width=analysis.bollinger_width,
            legs=[l.__dict__ if hasattr(l, '__dict__') else l
                  for l in plan.legs],
            spread_width=plan.spread_width,
            net_credit=plan.net_credit,
            max_loss=plan.max_loss,
            credit_to_width_ratio=plan.credit_to_width_ratio,
            sold_delta=sold_delta,
            expiration=plan.expiration,
            dte_at_entry=dte,
            risk_approved=verdict.approved,
            risk_checks_passed=verdict.checks_passed,
            risk_checks_failed=verdict.checks_failed,
            llm_reasoning=decision.reasoning,
            llm_confidence=decision.confidence,
            llm_decision=decision.action,
            llm_modifications=decision.modifications,
        )
