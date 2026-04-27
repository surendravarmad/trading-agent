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

from trading_agent.intelligence.trade_journal import TradeJournal, TradeEntry
from trading_agent.intelligence.knowledge_base import KnowledgeBase

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
        """
        Convert a closed trade to a chat fine-tuning example.

        Uses the full outcome-aware context so the model learns which
        feature combinations predicted the result, not just whether to
        approve or skip an abstract prompt.
        """
        # Skip trades with insufficient data for a quality example
        if not self._is_quality_example(trade):
            return {}

        context = self._build_trade_context_with_outcome(trade)

        if trade.outcome_label == "win":
            confidence = min(0.95, 0.65 + trade.realized_pl_pct / 100)
            reasoning = (
                f"The {trade.strategy_name} on {trade.ticker} succeeded in a "
                f"{trade.regime} regime. Key factors: RSI {trade.rsi_14:.1f} "
                f"confirmed regime direction, credit ratio "
                f"{trade.credit_to_width_ratio:.4f} provided adequate "
                f"compensation, sold delta {trade.sold_delta:.3f} stayed OTM. "
                f"Result: +${trade.realized_pl:.2f} "
                f"({trade.realized_pl_pct:.1f}%) in {trade.hold_duration_days}d."
            )
            decision = {
                "action": "approve",
                "confidence": round(confidence, 2),
                "reasoning": reasoning,
                "risk_assessment": "Risk profile appropriate for regime",
                "modifications": {},
                "warnings": [],
            }
        elif trade.outcome_label == "loss":
            reasoning = (
                f"The {trade.strategy_name} on {trade.ticker} lost in a "
                f"{trade.regime} regime. Warning signs: "
                f"RSI {trade.rsi_14:.1f}, BB width {trade.bollinger_width:.4f}. "
                f"Exit triggered by {trade.exit_signal}: {trade.exit_reason}. "
                f"Loss: ${abs(trade.realized_pl):.2f} in {trade.hold_duration_days}d."
            )
            if trade.lessons_learned:
                reasoning += " Lessons: " + "; ".join(trade.lessons_learned[:2])
            decision = {
                "action": "skip",
                "confidence": 0.75,
                "reasoning": reasoning,
                "risk_assessment": "Risk factors were underestimated",
                "modifications": {},
                "warnings": [
                    f"Historical loss: ${abs(trade.realized_pl):.2f}",
                    f"Exit trigger: {trade.exit_signal}",
                ],
            }
        else:
            return {}

        return {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert options trading analyst specialising "
                        "in credit spreads. Given a proposed trade and full "
                        "market context, decide: approve, modify, or skip. "
                        "Respond in JSON."
                    ),
                },
                {"role": "user", "content": context},
                {"role": "assistant", "content": json.dumps(decision)},
            ]
        }

    def _trade_to_alpaca_example(self, trade: TradeEntry) -> Dict:
        """Convert a trade to Alpaca instruction format."""
        if not self._is_quality_example(trade):
            return {}

        context = self._build_trade_context_with_outcome(trade)

        if trade.outcome_label == "win":
            output = (
                f"APPROVE. The {trade.strategy_name} on {trade.ticker} in a "
                f"{trade.regime} regime is well-structured. "
                f"Credit ratio {trade.credit_to_width_ratio:.4f}, delta "
                f"{trade.sold_delta:.3f}, RSI {trade.rsi_14:.1f}. "
                f"This setup historically profits "
                f"${trade.realized_pl:.2f} ({trade.realized_pl_pct:.1f}%)."
            )
        elif trade.outcome_label == "loss":
            output = (
                f"SKIP. Warning: {trade.exit_reason}. "
                f"This {trade.regime} regime setup with RSI {trade.rsi_14:.1f} "
                f"and BB width {trade.bollinger_width:.4f} historically triggers "
                f"{trade.exit_signal} exits resulting in "
                f"-${abs(trade.realized_pl):.2f} losses."
            )
        else:
            return {}

        return {
            "instruction": (
                "Analyze this proposed credit spread trade with full market "
                "context and decide: approve, modify, or skip. "
                "Cite specific indicators in your reasoning."
            ),
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
        """
        Build a rich, structured market context prompt from a trade entry.

        Richer context = better training signal.  The model needs to see
        all the inputs it will have at inference time so it can learn which
        combinations of features predict wins vs losses.
        """
        sma_ratio = (trade.sma_50 / trade.sma_200) if trade.sma_200 else 0
        price_vs_200 = "ABOVE" if trade.current_price > trade.sma_200 else "BELOW"
        slope_dir = "rising" if trade.sma_50_slope > 0 else "falling"
        risk_pct = (trade.max_loss / trade.account_balance * 100
                    if getattr(trade, "account_balance", 0) else 0)

        lines = [
            f"Proposed trade: {trade.strategy_name} on {trade.ticker}",
            "",
            "--- Market State ---",
            f"Regime:        {trade.regime}",
            f"Price:         ${trade.current_price:.2f}  ({price_vs_200} SMA-200)",
            f"SMA-50:        ${trade.sma_50:.2f}  (slope: {slope_dir}, "
            f"SMA50/200={sma_ratio:.3f})",
            f"SMA-200:       ${trade.sma_200:.2f}",
            f"RSI-14:        {trade.rsi_14:.1f}",
            f"BB Width:      {trade.bollinger_width:.4f}",
            f"IV Rank:       {trade.iv_rank:.1f}",
            "",
            "--- Spread Structure ---",
            f"Strategy:      {trade.strategy_name}",
            f"Net credit:    ${trade.net_credit:.2f}",
            f"Spread width:  ${trade.spread_width:.2f}",
            f"Credit ratio:  {trade.credit_to_width_ratio:.4f}  "
            f"(min required: 0.25)",
            f"Sold delta:    {trade.sold_delta:.3f}  (max allowed: 0.20)",
            f"DTE at entry:  {trade.dte_at_entry}  (target: 44)",
            f"Expiration:    {trade.expiration}",
            f"Max loss:      ${trade.max_loss:.2f}  ({risk_pct:.2f}% of account)",
            "",
            "Should this trade be approved, modified, or skipped?",
        ]
        return "\n".join(lines)

    def _build_trade_context_with_outcome(self, trade: TradeEntry) -> str:
        """
        Full context including outcome — used only in training data so the
        model can learn to recognise which features actually predicted the result.
        """
        base = self._build_trade_context(trade)
        if not trade.exit_signal:
            return base

        outcome_lines = [
            "",
            "--- Actual Outcome (training label) ---",
            f"Result:        {trade.outcome_label}",
            f"P&L:           ${trade.realized_pl:.2f} ({trade.realized_pl_pct:.1f}%)",
            f"Hold duration: {trade.hold_duration_days} days",
            f"Exit signal:   {trade.exit_signal}",
            f"Exit reason:   {trade.exit_reason}",
        ]
        if trade.llm_post_analysis:
            outcome_lines.append(
                f"Post-trade analysis: {trade.llm_post_analysis[:200]}"
            )
        return base + "\n".join(outcome_lines)

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
        """
        Find a losing trade whose market conditions are similar to *win*.

        DPO quality depends entirely on pairing quality.  A useful pair
        must isolate the feature that caused the loss — which means the
        pair should look similar on most dimensions but differ on the
        one(s) that matter.

        Rules (in order of importance):
          1. Must be the same strategy type  (hard filter — no cross-strategy pairs)
          2. Must be the same regime         (hard filter — apples to apples)
          3. Closest on: RSI, credit ratio, sold delta, BB width, DTE
        """
        # Hard filters: same strategy and regime only
        candidates = [
            loss for loss in losses
            if (loss.strategy_name == win.strategy_name
                and loss.regime == win.regime)
        ]

        if not candidates:
            # Relax: allow same strategy, any regime (still better than cross-strategy)
            candidates = [
                loss for loss in losses
                if loss.strategy_name == win.strategy_name
            ]

        if not candidates:
            return None

        best = None
        best_score = float("inf")
        for loss in candidates:
            # Normalised distance across 5 features
            score = (
                abs(win.rsi_14 - loss.rsi_14) / 100 +
                abs(win.credit_to_width_ratio - loss.credit_to_width_ratio) +
                abs(win.sold_delta - loss.sold_delta) +
                abs(win.bollinger_width - loss.bollinger_width) * 10 +
                abs(win.dte_at_entry - loss.dte_at_entry) / 45
            )
            if score < best_score:
                best_score = score
                best = loss

        return best

    @staticmethod
    def _is_quality_example(trade: TradeEntry) -> bool:
        """
        Filter out low-quality training examples.

        A quality example requires:
          - A clear win or loss label (not breakeven / missing)
          - Non-zero market indicators (trade was fully observed)
          - An exit signal (trade was actually closed and measured)
        """
        if trade.outcome_label not in ("win", "loss"):
            return False
        if not trade.exit_signal:
            return False
        if trade.rsi_14 <= 0 or trade.current_price <= 0:
            return False
        return True
