"""
Fine-Tuning Data Pipeline
===========================
Exports trade journal data into formats suitable for fine-tuning
local LLMs (Ollama/LM Studio) on the agent's own trading history.

Supports two export formats:
  1. JSONL (OpenAI chat format) — for Ollama/LM Studio fine-tuning
  2. Alpaca instruction format — for LoRA/QLoRA fine-tuning

The pipeline generates training examples from:
  - Winning trades → "This is what a good trade looks like"
  - Losing trades → "These are the warning signs to avoid"
  - Lessons → "Rules and patterns to follow"
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List

from trading_agent.trade_journal import TradeJournal, TradeEntry
from trading_agent.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class FineTuningExporter:
    """
    Exports curated training data from the trade journal and
    knowledge base for fine-tuning local LLMs.
    """

    def __init__(self, journal: TradeJournal, knowledge_base: KnowledgeBase,
                 export_dir: str = "fine_tuning_data"):
        self.journal = journal
        self.kb = knowledge_base
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Export: OpenAI JSONL format (for Ollama fine-tuning)
    # ------------------------------------------------------------------

    def export_chat_jsonl(self, min_trades: int = 20) -> str:
        """
        Export training data in OpenAI chat JSONL format.

        Each line is a JSON object with 'messages' array containing
        system, user, and assistant messages that teach the model
        to make good trading decisions.

        Returns the filepath of the exported file.
        """
        closed_trades = self.journal.get_closed_trades(limit=500)
        if len(closed_trades) < min_trades:
            logger.warning(
                "Only %d closed trades (need %d) — skipping fine-tuning export",
                len(closed_trades), min_trades)
            return ""

        examples = []

        # 1. Trade decision examples
        for trade in closed_trades:
            example = self._trade_to_chat_example(trade)
            if example:
                examples.append(example)

        # 2. Lesson examples
        lessons = self.kb.get_all_lessons()
        for lesson in lessons:
            example = self._lesson_to_chat_example(lesson)
            if example:
                examples.append(example)

        # Write JSONL file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.export_dir,
                                f"training_chat_{timestamp}.jsonl")

        with open(filepath, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

        logger.info("Exported %d training examples to %s",
                     len(examples), filepath)
        return filepath

    # ------------------------------------------------------------------
    # Export: Alpaca instruction format (for LoRA/QLoRA)
    # ------------------------------------------------------------------

    def export_alpaca_format(self, min_trades: int = 20) -> str:
        """
        Export training data in Alpaca instruction format:
        {"instruction": "...", "input": "...", "output": "..."}

        Returns the filepath of the exported file.
        """
        closed_trades = self.journal.get_closed_trades(limit=500)
        if len(closed_trades) < min_trades:
            return ""

        examples = []

        for trade in closed_trades:
            example = self._trade_to_alpaca_example(trade)
            if example:
                examples.append(example)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.export_dir,
                                f"training_alpaca_{timestamp}.json")

        with open(filepath, "w") as f:
            json.dump(examples, f, indent=2)

        logger.info("Exported %d Alpaca-format examples to %s",
                     len(examples), filepath)
        return filepath

    # ------------------------------------------------------------------
    # Export: Preference pairs (DPO format for RLHF)
    # ------------------------------------------------------------------

    def export_preference_pairs(self) -> str:
        """
        Export win/loss trade pairs for Direct Preference Optimization.
        Pairs a winning trade (chosen) with a losing trade (rejected)
        under similar market conditions.
        """
        closed = self.journal.get_closed_trades(limit=500)
        wins = [t for t in closed if t.outcome_label == "win"]
        losses = [t for t in closed if t.outcome_label == "loss"]

        if len(wins) < 5 or len(losses) < 5:
            logger.info("Not enough win/loss pairs for DPO export")
            return ""

        pairs = []
        for win in wins:
            # Find a loss with similar conditions
            best_loss = self._find_similar_loss(win, losses)
            if best_loss:
                pair = {
                    "prompt": self._build_trade_context(win),
                    "chosen": self._build_trade_decision(win, "approve"),
                    "rejected": self._build_trade_decision(best_loss, "approve"),
                }
                pairs.append(pair)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.export_dir,
                                f"training_dpo_{timestamp}.jsonl")

        with open(filepath, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        logger.info("Exported %d DPO preference pairs to %s",
                     len(pairs), filepath)
        return filepath

    # ------------------------------------------------------------------
    # Training data summary
    # ------------------------------------------------------------------

    def get_training_summary(self) -> Dict:
        """Summarize available training data."""
        closed = self.journal.get_closed_trades(limit=500)
        lessons = self.kb.get_all_lessons()

        wins = [t for t in closed if t.outcome_label == "win"]
        losses = [t for t in closed if t.outcome_label == "loss"]

        return {
            "total_closed_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "lessons": len(lessons),
            "ready_for_chat_ft": len(closed) >= 20,
            "ready_for_dpo": len(wins) >= 5 and len(losses) >= 5,
            "estimated_examples": len(closed) + len(lessons),
            "strategies_covered": list({t.strategy_name for t in closed}),
            "tickers_covered": list({t.ticker for t in closed}),
        }

    # ------------------------------------------------------------------
    # Internal: Build training examples
    # ------------------------------------------------------------------

    def _trade_to_chat_example(self, trade: TradeEntry) -> Dict:
        """Convert a closed trade to a chat fine-tuning example."""
        context = self._build_trade_context(trade)

        # Build the ideal response based on actual outcome
        if trade.outcome_label == "win":
            decision = {
                "action": "approve",
                "confidence": 0.8,
                "reasoning": (
                    f"The {trade.strategy_name} on {trade.ticker} in a "
                    f"{trade.regime} regime was a good trade. "
                    f"RSI at {trade.rsi_14:.1f} confirmed the regime. "
                    f"Credit ratio of {trade.credit_to_width_ratio:.4f} "
                    f"provided adequate compensation for risk. "
                    f"Result: P&L ${trade.realized_pl:.2f} "
                    f"({trade.realized_pl_pct:.1f}%) over "
                    f"{trade.hold_duration_days} days."
                ),
                "risk_assessment": "Acceptable risk profile",
                "similar_trades_summary": "",
                "modifications": {},
                "warnings": [],
            }
        elif trade.outcome_label == "loss":
            decision = {
                "action": "skip",
                "confidence": 0.7,
                "reasoning": (
                    f"The {trade.strategy_name} on {trade.ticker} should have "
                    f"been avoided. The {trade.regime} regime with RSI "
                    f"{trade.rsi_14:.1f} had warning signs. "
                    f"Exit signal was {trade.exit_signal}: {trade.exit_reason}. "
                    f"Loss: ${abs(trade.realized_pl):.2f}."
                ),
                "risk_assessment": "Risk factors were underestimated",
                "similar_trades_summary": "",
                "modifications": {},
                "warnings": [
                    f"Trade lost ${abs(trade.realized_pl):.2f}",
                    f"Exit: {trade.exit_signal}",
                ],
            }
        else:
            return {}

        return {
            "messages": [
                {"role": "system", "content": "You are an expert options trading analyst. Respond in JSON."},
                {"role": "user", "content": context},
                {"role": "assistant", "content": json.dumps(decision)},
            ]
        }

    def _trade_to_alpaca_example(self, trade: TradeEntry) -> Dict:
        """Convert a trade to Alpaca instruction format."""
        context = self._build_trade_context(trade)

        if trade.outcome_label == "win":
            output = (
                f"APPROVE this trade. The {trade.strategy_name} on {trade.ticker} "
                f"in {trade.regime} regime with credit ratio "
                f"{trade.credit_to_width_ratio:.4f} is well-structured. "
                f"Expected outcome: profit of ~${trade.realized_pl:.2f}."
            )
        elif trade.outcome_label == "loss":
            output = (
                f"SKIP this trade. Warning signs: {trade.exit_reason}. "
                f"The {trade.regime} regime was unstable. "
                f"Expected outcome: loss of ~${abs(trade.realized_pl):.2f}."
            )
        else:
            return {}

        return {
            "instruction": "Analyze this proposed credit spread trade and decide: approve, modify, or skip.",
            "input": context,
            "output": output,
        }

    def _lesson_to_chat_example(self, lesson_doc) -> Dict:
        """Convert a KB lesson to a chat training example."""
        return {
            "messages": [
                {"role": "system", "content": "You are an expert options trading analyst."},
                {"role": "user", "content": (
                    "What's an important lesson you've learned from "
                    "recent credit spread trading?"
                )},
                {"role": "assistant", "content": lesson_doc.text},
            ]
        }

    def _build_trade_context(self, trade: TradeEntry) -> str:
        """Build the market context prompt from a trade entry."""
        return (
            f"Proposed trade: {trade.strategy_name} on {trade.ticker}\n"
            f"Regime: {trade.regime}\n"
            f"Price: ${trade.current_price:.2f} "
            f"(SMA50={trade.sma_50:.2f}, SMA200={trade.sma_200:.2f})\n"
            f"RSI: {trade.rsi_14:.1f}, BB Width: {trade.bollinger_width:.4f}\n"
            f"Credit: ${trade.net_credit:.2f}, Width: ${trade.spread_width:.2f}\n"
            f"Credit/Width ratio: {trade.credit_to_width_ratio:.4f}\n"
            f"Sold delta: {trade.sold_delta:.3f}, DTE: {trade.dte_at_entry}\n"
            f"Should this trade be approved, modified, or skipped?"
        )

    def _build_trade_decision(self, trade: TradeEntry, action: str) -> str:
        """Build a decision string for DPO training."""
        return json.dumps({
            "action": action,
            "reasoning": (
                f"{'Good' if trade.outcome_label == 'win' else 'Bad'} trade: "
                f"{trade.strategy_name} resulted in "
                f"P&L ${trade.realized_pl:.2f} "
                f"({trade.exit_signal}: {trade.exit_reason})"
            ),
        })

    def _find_similar_loss(self, win: TradeEntry,
                            losses: List[TradeEntry]) -> TradeEntry:
        """Find a losing trade with similar market conditions to a winner."""
        best = None
        best_score = float("inf")

        for loss in losses:
            # Simple distance metric on key features
            score = (
                abs(win.rsi_14 - loss.rsi_14) +
                abs(win.credit_to_width_ratio - loss.credit_to_width_ratio) * 100 +
                abs(win.sold_delta - loss.sold_delta) * 100
            )
            if win.strategy_name == loss.strategy_name:
                score *= 0.5  # Prefer same strategy type

            if score < best_score:
                best_score = score
                best = loss

        return best
