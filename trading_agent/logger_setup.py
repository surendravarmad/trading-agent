"""
Logging configuration — sets up console + rotating file handlers
with a consistent format across all modules.

Week 3-4 upgrade
----------------
The prior implementation used a plain FileHandler keyed by UTC date.
That had two failure modes for a long-running agent:

  1. Within a single day the log file grew unbounded.  A noisy cycle
     (e.g. LLM analyst dumping full prompts) could fill the disk.
  2. Old files were never purged.  Over weeks of running, thousands
     of ``trading_agent_YYYYMMDD.log`` files would accumulate.

The new implementation uses ``RotatingFileHandler``:
  • 10 MB per file (tunable via env ``LOG_MAX_BYTES``)
  • 7 rollover files kept (tunable via env ``LOG_BACKUP_COUNT``)
  • Old files are automatically renamed ``...log.1``, ``...log.2``, …
    and anything past BACKUP_COUNT is deleted on rollover.

The file path no longer embeds the date — rotation is size-driven.
Callers that rely on per-date files should grep by timestamp inside
the file contents instead.
"""

import logging
import logging.handlers
import os
from typing import Optional

# ---------------------------------------------------------------------------
# Rotation defaults
# ---------------------------------------------------------------------------
# Plain integers so they're trivially overridable from the environment
# without import-time side effects on tests that monkey-patch os.environ.
DEFAULT_MAX_BYTES      = 10 * 1024 * 1024   # 10 MB
DEFAULT_BACKUP_COUNT   = 7
LOG_FILENAME           = "trading_agent.log"


def _int_from_env(name: str, default: int) -> int:
    """Read an int env var with graceful fallback on parse errors."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return default


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
) -> logging.Logger:
    """
    Configure the root logger with console + size-rotated file output.

    Parameters
    ----------
    log_level    : INFO / DEBUG / WARNING / …
    log_dir      : directory for log files (created if missing)
    max_bytes    : rotation threshold per file (None → env/DEFAULT)
    backup_count : number of rollover files kept (None → env/DEFAULT)

    Returns
    -------
    The root logger.
    """
    os.makedirs(log_dir, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)
    log_file = os.path.join(log_dir, LOG_FILENAME)

    if max_bytes is None:
        max_bytes = _int_from_env("LOG_MAX_BYTES", DEFAULT_MAX_BYTES)
    if backup_count is None:
        backup_count = _int_from_env("LOG_BACKUP_COUNT", DEFAULT_BACKUP_COUNT)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — stderr by default
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)

    # Rotating file handler — size-driven, bounded retention
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
        delay=False,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers on re-init (Streamlit hot reload / test reuse)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    # Silence chatty third-party libraries at INFO / DEBUG
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)

    root.info(
        "Logging initialised — level=%s, file=%s, rotate=%dMB × %d backups",
        log_level, log_file, max_bytes // (1024 * 1024), backup_count,
    )
    return root
