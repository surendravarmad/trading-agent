"""
Trade Journal
==============
Structured logging system that captures every trade's full lifecycle:
  - Entry context (regime, Greeks, indicators, LLM reasoning)
  - Execution details (order status, fill prices)
  - Outcome tracking (P&L, hold duration, exit signal)
  - Lessons learned (LLM post-trade analysis)

This journal feeds the RAG knowledge base and fine-tuning pipeline.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeEntry:
    """Complete record of a single trade from plan to close."""

    # Identity
    trade_id: str = ""
    ticker: str = ""
    strategy_name: str = ""
    timestamp_opened: str = ""
    timestamp_closed: str = ""

    # Market context at entry
    regime: str = ""
    current_price: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    sma_50_slope: float = 0.0
    rsi_14: float = 0.0
    bollinger_width: float = 0.0
    iv_rank: float = 0.0            # implied volatility rank (0-100)

    # Trade structure
    legs: List[Dict] = field(default_factory=list)
    spread_width: float = 0.0
    net_credit: float = 0.0
    max_loss: float = 0.0
    credit_to_width_ratio: float = 0.0
    sold_delta: float = 0.0
    expiration: str = ""
    dte_at_entry: int = 0

    # Risk assessment
    risk_approved: bool = False
    risk_checks_passed: List[str] = field(default_factory=list)
    risk_checks_failed: List[str] = field(default_factory=list)

    # LLM analysis at entry
    llm_reasoning: str = ""
    llm_confidence: float = 0.0      # 0.0 - 1.0
    llm_decision: str = ""           # "approve", "modify", "skip"
    llm_modifications: Dict = field(default_factory=dict)

    # Execution
    order_status: str = ""           # "submitted", "filled", "rejected", "dry_run"
    order_id: str = ""
    fill_price: float = 0.0

    # Outcome (filled when trade closes)
    exit_signal: str = ""            # "profit_target", "stop_loss", "regime_shift", "expired"
    exit_reason: str = ""
    realized_pl: float = 0.0
    realized_pl_pct: float = 0.0
    hold_duration_days: int = 0
    max_drawdown: float = 0.0
    regime_at_close: str = ""

    # Post-trade LLM analysis
    llm_post_analysis: str = ""
    lessons_learned: List[str] = field(default_factory=list)
    outcome_label: str = ""          # "win", "loss", "breakeven", "expired_worthless"

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_embedding_text(self) -> str:
        """
        Generate a canonical structured summary for embedding in the RAG store.

        Uses a fixed-format template so that semantically similar trades
        (same regime, similar RSI band, same strategy) produce embeddings
        that are geometrically close in vector space.  Freeform prose causes
        synonyms and word-order variance to scatter similar trades far apart.

        Template sections (always in this order):
          [TRADE]      strategy / ticker / regime — the primary discriminator
          [MARKET]     price-action indicators in fixed numeric buckets
          [STRUCTURE]  spread geometry — credit, width, ratio, delta, DTE
          [OUTCOME]    result section — empty until trade closes
          [LESSONS]    post-trade learnings (LLM-generated)
        """
        sma_ratio = (self.sma_50 / self.sma_200) if self.sma_200 else 0
        price_vs_sma200 = (
            "above" if self.current_price > self.sma_200 else "below"
        )
        slope_dir = (
            "rising" if self.sma_50_slope > 0 else "falling"
        )

        # Bucket RSI into ranges so similar values cluster: oversold/neutral/overbought
        if self.rsi_14 < 30:
            rsi_bucket = "oversold (<30)"
        elif self.rsi_14 < 45:
            rsi_bucket = "low-neutral (30-45)"
        elif self.rsi_14 < 55:
            rsi_bucket = "mid-neutral (45-55)"
        elif self.rsi_14 < 70:
            rsi_bucket = "high-neutral (55-70)"
        else:
            rsi_bucket = "overbought (>70)"

        parts = [
            # --- Primary discriminator ---
            f"[TRADE] {self.strategy_name} | {self.ticker} | regime={self.regime}",

            # --- Market state ---
            f"[MARKET] price=${self.current_price:.2f} ({price_vs_sma200} SMA200) "
            f"| SMA50/200={sma_ratio:.3f} slope={slope_dir} "
            f"| RSI={self.rsi_14:.1f} ({rsi_bucket}) "
            f"| BB_width={self.bollinger_width:.4f} "
            f"| IV_rank={self.iv_rank:.1f}",

            # --- Spread structure ---
            f"[STRUCTURE] credit=${self.net_credit:.2f} "
            f"| width=${self.spread_width:.2f} "
            f"| ratio={self.credit_to_width_ratio:.4f} "
            f"| sold_delta={self.sold_delta:.3f} "
            f"| DTE={self.dte_at_entry} "
            f"| expiration={self.expiration}",
        ]

        # --- Outcome (only present after close) ---
        if self.exit_signal:
            parts.append(
                f"[OUTCOME] {self.outcome_label} "
                f"| P&L=${self.realized_pl:.2f} ({self.realized_pl_pct:.1f}%) "
                f"| held={self.hold_duration_days}d "
                f"| exit={self.exit_signal} "
                f"| reason={self.exit_reason}"
            )
        else:
            parts.append("[OUTCOME] open — result not yet available")

        # --- LLM reasoning (if available) ---
        if self.llm_reasoning:
            parts.append(f"[REASONING] {self.llm_reasoning[:300]}")

        # --- Lessons (if available) ---
        if self.lessons_learned:
            parts.append("[LESSONS] " + " | ".join(self.lessons_learned))

        return "\n".join(parts)


class TradeJournal:
    """
    Persistent trade journal that stores every trade's full lifecycle.

    Storage structure:
      journal_dir/
        trades/
          {trade_id}.json       — individual trade records
        index.json              — lightweight index for quick lookups
        stats.json              — aggregate statistics
    """

    def __init__(self, journal_dir: str = "trade_journal"):
        self.journal_dir = journal_dir
        self.trades_dir = os.path.join(journal_dir, "trades")
        self.index_path = os.path.join(journal_dir, "index.json")
        self.stats_path = os.path.join(journal_dir, "stats.json")
        os.makedirs(self.trades_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def open_trade(self, trade: TradeEntry) -> str:
        """
        Record a new trade entry (at time of order submission).
        Returns the trade_id.
        """
        if not trade.trade_id:
            trade.trade_id = self._generate_id(trade.ticker)
        if not trade.timestamp_opened:
            trade.timestamp_opened = datetime.utcnow().isoformat()

        self._save_trade(trade)
        self._update_index(trade, action="open")
        logger.info("[%s] Trade journal: OPENED trade %s (%s)",
                     trade.ticker, trade.trade_id, trade.strategy_name)
        return trade.trade_id

    def close_trade(self, trade_id: str, exit_signal: str, exit_reason: str,
                    realized_pl: float, regime_at_close: str = "",
                    max_drawdown: float = 0.0) -> Optional[TradeEntry]:
        """
        Update a trade record with its outcome when closing.
        """
        trade = self.get_trade(trade_id)
        if not trade:
            logger.warning("Trade %s not found in journal", trade_id)
            return None

        trade.timestamp_closed = datetime.utcnow().isoformat()
        trade.exit_signal = exit_signal
        trade.exit_reason = exit_reason
        trade.realized_pl = realized_pl
        trade.regime_at_close = regime_at_close
        trade.max_drawdown = max_drawdown

        # Calculate derived fields
        if trade.net_credit > 0:
            credit_value = trade.net_credit * 100
            trade.realized_pl_pct = round((realized_pl / credit_value) * 100, 2)

        if trade.timestamp_opened:
            try:
                opened = datetime.fromisoformat(trade.timestamp_opened)
                closed = datetime.fromisoformat(trade.timestamp_closed)
                trade.hold_duration_days = (closed - opened).days
            except ValueError:
                pass

        # Label outcome
        if realized_pl > 0:
            trade.outcome_label = "win"
        elif realized_pl < -10:
            trade.outcome_label = "loss"
        elif exit_signal == "expired":
            trade.outcome_label = "expired_worthless"
        else:
            trade.outcome_label = "breakeven"

        self._save_trade(trade)
        self._update_index(trade, action="close")
        self._update_stats()

        logger.info("[%s] Trade journal: CLOSED trade %s — %s P&L=$%.2f",
                     trade.ticker, trade_id, trade.outcome_label, realized_pl)
        return trade

    def add_llm_analysis(self, trade_id: str, post_analysis: str,
                         lessons: List[str]) -> Optional[TradeEntry]:
        """Add post-trade LLM analysis and lessons learned."""
        trade = self.get_trade(trade_id)
        if not trade:
            return None

        trade.llm_post_analysis = post_analysis
        trade.lessons_learned = lessons
        self._save_trade(trade)

        logger.info("[%s] Added LLM post-analysis to trade %s",
                     trade.ticker, trade_id)
        return trade

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_trade(self, trade_id: str) -> Optional[TradeEntry]:
        """Load a single trade by ID."""
        path = os.path.join(self.trades_dir, f"{trade_id}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return self._dict_to_entry(data)
        except Exception as exc:
            logger.error("Failed to load trade %s: %s", trade_id, exc)
            return None

    def get_recent_trades(self, limit: int = 20) -> List[TradeEntry]:
        """Get the most recent trades (both open and closed)."""
        index = self._load_index()
        entries = sorted(index.get("trades", []),
                         key=lambda x: x.get("timestamp", ""), reverse=True)
        trades = []
        for entry in entries[:limit]:
            trade = self.get_trade(entry["trade_id"])
            if trade:
                trades.append(trade)
        return trades

    def get_trades_by_ticker(self, ticker: str,
                             limit: int = 50) -> List[TradeEntry]:
        """Get all trades for a specific ticker."""
        index = self._load_index()
        matching = [e for e in index.get("trades", [])
                    if e.get("ticker", "") == ticker]
        matching.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        trades = []
        for entry in matching[:limit]:
            trade = self.get_trade(entry["trade_id"])
            if trade:
                trades.append(trade)
        return trades

    def get_closed_trades(self, limit: int = 100) -> List[TradeEntry]:
        """Get only completed (closed) trades for analysis."""
        index = self._load_index()
        closed = [e for e in index.get("trades", [])
                  if e.get("status") == "closed"]
        closed.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        trades = []
        for entry in closed[:limit]:
            trade = self.get_trade(entry["trade_id"])
            if trade:
                trades.append(trade)
        return trades

    def get_stats(self) -> Dict:
        """Get aggregate performance statistics."""
        if os.path.exists(self.stats_path):
            with open(self.stats_path) as f:
                return json.load(f)
        return self._compute_stats()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _compute_stats(self) -> Dict:
        """Compute aggregate performance metrics from closed trades."""
        closed = self.get_closed_trades(limit=500)
        if not closed:
            return {"total_trades": 0}

        wins = [t for t in closed if t.outcome_label == "win"]
        losses = [t for t in closed if t.outcome_label == "loss"]
        total_pl = sum(t.realized_pl for t in closed)
        avg_pl = total_pl / len(closed) if closed else 0
        avg_hold = (sum(t.hold_duration_days for t in closed) / len(closed)
                    if closed else 0)

        win_rate = len(wins) / len(closed) if closed else 0
        avg_win = (sum(t.realized_pl for t in wins) / len(wins)
                   if wins else 0)
        avg_loss = (sum(t.realized_pl for t in losses) / len(losses)
                    if losses else 0)

        # By strategy
        by_strategy = {}
        for t in closed:
            strat = t.strategy_name
            if strat not in by_strategy:
                by_strategy[strat] = {"count": 0, "pl": 0, "wins": 0}
            by_strategy[strat]["count"] += 1
            by_strategy[strat]["pl"] += t.realized_pl
            if t.outcome_label == "win":
                by_strategy[strat]["wins"] += 1

        # By ticker
        by_ticker = {}
        for t in closed:
            tk = t.ticker
            if tk not in by_ticker:
                by_ticker[tk] = {"count": 0, "pl": 0, "wins": 0}
            by_ticker[tk]["count"] += 1
            by_ticker[tk]["pl"] += t.realized_pl
            if t.outcome_label == "win":
                by_ticker[tk]["wins"] += 1

        stats = {
            "total_trades": len(closed),
            "total_pl": round(total_pl, 2),
            "win_rate": round(win_rate, 4),
            "avg_pl_per_trade": round(avg_pl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_hold_days": round(avg_hold, 1),
            "wins": len(wins),
            "losses": len(losses),
            "by_strategy": by_strategy,
            "by_ticker": by_ticker,
            "updated_at": datetime.utcnow().isoformat(),
        }
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_id(self, ticker: str) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{ticker}_{ts}"

    def _save_trade(self, trade: TradeEntry):
        path = os.path.join(self.trades_dir, f"{trade.trade_id}.json")
        with open(path, "w") as f:
            json.dump(trade.to_dict(), f, indent=2)

    def _load_index(self) -> Dict:
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                return json.load(f)
        return {"trades": []}

    def _update_index(self, trade: TradeEntry, action: str):
        index = self._load_index()
        entries = index.get("trades", [])

        # Update or add
        found = False
        for i, e in enumerate(entries):
            if e["trade_id"] == trade.trade_id:
                entries[i] = {
                    "trade_id": trade.trade_id,
                    "ticker": trade.ticker,
                    "strategy": trade.strategy_name,
                    "status": "closed" if trade.timestamp_closed else "open",
                    "timestamp": trade.timestamp_opened,
                    "outcome": trade.outcome_label,
                    "pl": trade.realized_pl,
                }
                found = True
                break

        if not found:
            entries.append({
                "trade_id": trade.trade_id,
                "ticker": trade.ticker,
                "strategy": trade.strategy_name,
                "status": "open",
                "timestamp": trade.timestamp_opened,
                "outcome": "",
                "pl": 0,
            })

        index["trades"] = entries
        with open(self.index_path, "w") as f:
            json.dump(index, f, indent=2)

    def _update_stats(self):
        stats = self._compute_stats()
        with open(self.stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    @staticmethod
    def _dict_to_entry(data: Dict) -> TradeEntry:
        """Safely convert a dict to TradeEntry, handling missing fields."""
        entry = TradeEntry()
        for key, value in data.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
        return entry
