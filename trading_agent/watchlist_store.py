"""
watchlist_store.py — JSON-backed persistence for the Watchlist tab.

Storage
-------
File: ``knowledge_base/watchlist.json`` (gitignored — ``knowledge_base/``
is already covered by .gitignore alongside other per-checkout state).

Format (schema v1)::

    {
      "schema_version": 1,
      "tickers": [
        {"symbol": "SPY", "added_at": "2026-04-19T12:30:00Z", "note": ""},
        {"symbol": "QQQ", "added_at": "2026-04-19T12:31:00Z", "note": "tech"}
      ]
    }

Design notes
------------
* **Atomic writes** — write to ``watchlist.json.tmp`` then ``os.replace``
  so a crash mid-write can never leave a half-written file. Same pattern
  ``strategy_presets.PRESET_FILE`` uses for the active-preset file.
* **Schema versioning** — ``schema_version`` lets us migrate gracefully
  later (e.g., when v2 adds per-ticker alert thresholds for a regime
  flip). Unknown future versions are loaded as v1 with a warning.
* **Symbol normalisation** — Tickers are uppercased and stripped on
  insert to prevent ``"spy "`` and ``"SPY"`` becoming two rows.
* **De-duplication** — ``add_ticker`` is idempotent; calling twice with
  the same symbol updates the note rather than creating a duplicate.
* **No agent-loop coupling** — this module is import-isolated from
  ``agent.py``, ``decision_engine.py``, and the executor. The watchlist
  cannot affect trade decisions, only display.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_WATCHLIST_PATH = Path("knowledge_base/watchlist.json")
SCHEMA_VERSION = 1

# A single process-wide lock guards atomic read-modify-write sequences
# against concurrent Streamlit reruns (which Streamlit does freely).
# Must be reentrant: the public mutators (add_ticker / remove_ticker /
# update_note) hold the lock across a load → mutate → save_watchlist
# sequence, and save_watchlist itself acquires the lock for the os.replace.
# A plain Lock would self-deadlock on the second acquire.
_WRITE_LOCK = threading.RLock()


@dataclass
class WatchlistEntry:
    """One ticker in the watchlist."""
    symbol: str
    added_at: str = ""
    note: str = ""

    def __post_init__(self):
        self.symbol = (self.symbol or "").strip().upper()
        if not self.added_at:
            self.added_at = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z")


@dataclass
class Watchlist:
    """In-memory representation of the watchlist file."""
    tickers: List[WatchlistEntry] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def symbols(self) -> List[str]:
        return [e.symbol for e in self.tickers]


# ----------------------------------------------------------------------
# Load / save
# ----------------------------------------------------------------------
def load_watchlist(path: Path = DEFAULT_WATCHLIST_PATH) -> Watchlist:
    """Read the watchlist from disk; return an empty list if missing."""
    p = Path(path)
    if not p.exists():
        return Watchlist()

    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Watchlist file unreadable (%s) — returning empty", exc)
        return Watchlist()

    version = int(raw.get("schema_version", 1))
    if version != SCHEMA_VERSION:
        logger.warning(
            "Watchlist schema_version=%d differs from current=%d; "
            "loading best-effort", version, SCHEMA_VERSION,
        )

    tickers = []
    for row in raw.get("tickers", []):
        try:
            entry = WatchlistEntry(
                symbol=row.get("symbol", ""),
                added_at=row.get("added_at", ""),
                note=row.get("note", ""),
            )
            if entry.symbol:
                tickers.append(entry)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed watchlist row %r: %s", row, exc)

    return Watchlist(tickers=tickers, schema_version=SCHEMA_VERSION)


def save_watchlist(wl: Watchlist, path: Path = DEFAULT_WATCHLIST_PATH) -> None:
    """Atomic write to *path*. Creates parent dir if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tickers": [asdict(e) for e in wl.tickers],
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    with _WRITE_LOCK:
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, p)
    logger.debug("Watchlist saved (%d tickers) → %s", len(wl.tickers), p)


# ----------------------------------------------------------------------
# Mutations — used by the UI's add / remove / annotate buttons.
# ----------------------------------------------------------------------
def add_ticker(symbol: str,
               note: str = "",
               path: Path = DEFAULT_WATCHLIST_PATH) -> Watchlist:
    """Add *symbol* (idempotent). Updates note if symbol already present."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise ValueError("symbol must be a non-empty string")

    with _WRITE_LOCK:
        wl = load_watchlist(path)
        for existing in wl.tickers:
            if existing.symbol == symbol:
                if note:
                    existing.note = note
                save_watchlist(wl, path)
                return wl
        wl.tickers.append(WatchlistEntry(symbol=symbol, note=note))
        save_watchlist(wl, path)
    return wl


def remove_ticker(symbol: str,
                  path: Path = DEFAULT_WATCHLIST_PATH) -> Watchlist:
    """Remove *symbol* if present. No-op if not."""
    symbol = (symbol or "").strip().upper()
    with _WRITE_LOCK:
        wl = load_watchlist(path)
        wl.tickers = [e for e in wl.tickers if e.symbol != symbol]
        save_watchlist(wl, path)
    return wl


def update_note(symbol: str,
                note: str,
                path: Path = DEFAULT_WATCHLIST_PATH) -> Optional[Watchlist]:
    """Update the note for *symbol*. Returns None if not present."""
    symbol = (symbol or "").strip().upper()
    with _WRITE_LOCK:
        wl = load_watchlist(path)
        for e in wl.tickers:
            if e.symbol == symbol:
                e.note = note
                save_watchlist(wl, path)
                return wl
    return None
