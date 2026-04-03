"""
Journal Knowledge Base
=======================
Structured signal logger for every trade attempt/signal, written
regardless of whether the LLM intelligence layer is enabled.

Two output files maintained in *journal_dir*:
  signals.jsonl  — one JSON object per line (LLM fine-tuning / RAG-ready)
  signals.md     — append-only Markdown table (human-readable)

JSONL record schema
-------------------
{
  "timestamp":   ISO-8601 UTC string
  "ticker":      str
  "action":      str   ("dry_run" | "submitted" | "rejected" | "skip" |
                         "error" | "skipped_by_llm" | "skipped_existing")
  "price":       float  (current underlying price at signal time)
  "exec_status": str   (mirrors action or final order status)
  "notes":       str   (brief human-readable summary ≤ 120 chars)
  "raw_signal": {
      "regime":                str
      "strategy":              str
      "plan_valid":            bool
      "risk_approved":         bool
      "net_credit":            float
      "max_loss":              float
      "credit_to_width_ratio": float
      "spread_width":          float
      "expiration":            str
      "dte":                   int
      "sma_50":                float
      "sma_200":               float
      "rsi_14":                float
      "account_balance":       float
      "checks_passed":         list[str]
      "checks_failed":         list[str]
      "llm_decision":          str | None
      "llm_confidence":        float | None
      "rejection_reason":      str | None
      "error":                 str | None
  }
}
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_MD_HEADER = (
    "| Timestamp (UTC) | Ticker | Action | Price | Strategy | Regime "
    "| Risk OK | Status | Confidence | Notes |\n"
    "|-----------------|--------|--------|-------|----------|--------"
    "|---------|--------|------------|-------|\n"
)


class JournalKB:
    """
    Append-only signal journal — always active, LLM-independent.

    Usage::
        jkb = JournalKB("journal_kb")
        jkb.log_signal(ticker="AAPL", action="dry_run", price=178.5,
                       raw_signal={...})
    """

    def __init__(self, journal_dir: str = "journal_kb"):
        self.journal_dir = journal_dir
        os.makedirs(journal_dir, exist_ok=True)
        self.jsonl_path = os.path.join(journal_dir, "signals.jsonl")
        self.md_path = os.path.join(journal_dir, "signals.md")
        self._ensure_md_header()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_signal(
        self,
        ticker: str,
        action: str,
        price: float,
        raw_signal: Dict[str, Any],
        exec_status: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        """
        Log one trade signal/attempt.  Called for every ticker every
        cycle regardless of LLM enablement or execution outcome.

        Parameters
        ----------
        ticker      : underlying symbol
        action      : signal disposition (see schema above)
        price       : underlying price at decision time
        raw_signal  : full market + plan + risk context dict
        exec_status : final order status if different from action
        notes       : optional ≤120-char human summary
        """
        ts = datetime.now(timezone.utc).isoformat()
        status = exec_status or action

        if notes is None:
            notes = self._auto_notes(raw_signal, action)

        record: Dict[str, Any] = {
            "timestamp": ts,
            "ticker": ticker,
            "action": action,
            "price": round(float(price), 4),
            "exec_status": status,
            "notes": notes[:120],
            "raw_signal": raw_signal,
        }

        self._write_jsonl(record)
        self._write_md_row(ts, ticker, action, price, raw_signal, status, notes)

    def log_defense_first(
        self,
        ticker: str,
        reason: str,
        price: float,
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Log a capital-retainment skip event (macro guard, high-IV block, etc.).
        Includes strategy_mode: defense_first in the raw_signal for LLM training.
        """
        raw: Dict[str, Any] = {"strategy_mode": "defense_first", "reason": reason}
        if extra:
            raw.update(extra)
        self.log_signal(
            ticker=ticker,
            action="skipped_defense_first",
            price=price,
            raw_signal=raw,
            notes=f"defense_first: {reason[:80]}",
        )

    def log_error(
        self,
        ticker: str,
        error: str,
        price: float = 0.0,
        context: Optional[Dict] = None,
    ) -> None:
        """Log a per-ticker processing failure."""
        self.log_signal(
            ticker=ticker,
            action="error",
            price=price,
            raw_signal={"error": error, **(context or {})},
            exec_status="error",
            notes=error[:120],
        )

    def log_cycle_error(self, error: str, context: Optional[Dict] = None) -> None:
        """Log a full-cycle failure (e.g. account fetch failure, timeout)."""
        ts = datetime.now(timezone.utc).isoformat()
        record: Dict[str, Any] = {
            "timestamp": ts,
            "event": "cycle_error",
            "error": error[:500],
            "context": context or {},
        }
        self._write_jsonl(record)
        logger.error("JournalKB cycle_error: %s", error)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_md_header(self) -> None:
        """Write the Markdown header if the file doesn't exist yet."""
        if not os.path.exists(self.md_path):
            with open(self.md_path, "w") as fh:
                fh.write("# Trade Signal Journal\n\n")
                fh.write(_MD_HEADER)

    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        try:
            with open(self.jsonl_path, "a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.error("JournalKB JSONL write failed: %s", exc)

    def _write_md_row(
        self,
        ts: str,
        ticker: str,
        action: str,
        price: float,
        raw: Dict[str, Any],
        status: str,
        notes: str,
    ) -> None:
        try:
            strategy = raw.get("strategy") or "—"
            regime = raw.get("regime") or "—"
            risk_ok = raw.get("risk_approved", "—")
            conf = raw.get("llm_confidence")
            conf_str = f"{conf:.2f}" if isinstance(conf, float) else "—"
            safe_notes = str(notes).replace("|", "\\|")[:80]
            row = (
                f"| {ts[:19]} | {ticker} | {action} | ${price:.2f} "
                f"| {strategy} | {regime} | {risk_ok} | {status} "
                f"| {conf_str} | {safe_notes} |\n"
            )
            with open(self.md_path, "a") as fh:
                fh.write(row)
        except Exception as exc:
            logger.error("JournalKB Markdown write failed: %s", exc)

    @staticmethod
    def _auto_notes(raw: Dict[str, Any], action: str) -> str:
        """Generate a concise summary string from raw signal data."""
        parts = []
        if raw.get("strategy"):
            parts.append(raw["strategy"])
        if raw.get("net_credit"):
            parts.append(f"cr={raw['net_credit']:.2f}")
        ratio = raw.get("credit_to_width_ratio")
        if ratio:
            parts.append(f"ratio={ratio:.2f}")
        if raw.get("rejection_reason"):
            parts.append(raw["rejection_reason"][:60])
        if raw.get("error"):
            parts.append(raw["error"][:60])
        return (f"{action}: " + ", ".join(parts)) if parts else action
