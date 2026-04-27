"""
shutdown — graceful termination helpers
=======================================

Why this module exists
----------------------
Prior versions called ``os._exit(0)`` or ``os._exit(1)`` in four places
inside ``agent.py``:

  1. After-hours shutdown (clean, code 0)
  2. Daily drawdown circuit breaker (hard kill, code 1)
  3. Cycle timeout guard (hard kill, code 1)
  4. Via test helpers that patch _exit

``os._exit`` has harsh semantics:
  • No flushing of stdio buffers → log lines in flight are lost
  • No cleanup of threading.Timer / background threads
  • No atexit handlers run → tempfiles linger
  • SIGTERM handlers never fire → deploy tooling can't observe the exit

That's appropriate for a *true* hard-kill (e.g. cycle hung), but for
the after-hours path and drawdown breaker a clean exit is better:
the log file should be flushed, the journal should record the reason,
and any open file descriptors should close normally.

What this module provides
-------------------------
• ``install_signal_handlers()`` — installs SIGTERM / SIGINT handlers
  that set a module-level ``_shutdown_requested`` flag and call
  ``graceful_exit()``.  Callers (e.g. a long-running scheduler) can
  poll ``shutdown_requested()`` between cycles.

• ``graceful_exit(code, reason, journal)`` — the controlled exit path:
  flushes logging handlers, records a shutdown marker in the journal,
  then calls ``sys.exit(code)``.  Use this for "we decided to stop"
  paths (after-hours, drawdown breaker, successful shutdown request).

• ``hard_exit(code, reason)`` — unchanged ``os._exit`` semantics,
  reserved for paths where the process may be in an unrecoverable
  state (cycle timeout, deadlock).  Still calls the journal but does
  not assume logging is healthy.

Design choices
--------------
• No global state except ``_shutdown_requested`` and ``_journal_ref``.
  Signal handlers must be reentrancy-safe and can't take locks.

• ``logging.shutdown()`` is called inside ``graceful_exit`` — this is
  the stdlib-recommended way to flush + close all handlers in order.

• Signal handlers schedule exit via ``graceful_exit`` but don't
  themselves call it directly if the signal fires during sensitive
  work.  We set the flag and exit only when control returns to the
  scheduler.  The simpler immediate-exit variant is also supported
  via ``install_signal_handlers(immediate=True)``.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module state — kept intentionally small
# ---------------------------------------------------------------------------
_shutdown_requested = threading.Event()
_journal_ref = None   # type: ignore  # set via install_signal_handlers(journal=...)
_installed = False


def shutdown_requested() -> bool:
    """Return True once a SIGTERM or SIGINT has been observed."""
    return _shutdown_requested.is_set()


def reset_shutdown_flag() -> None:
    """Clear the shutdown flag — used by tests."""
    _shutdown_requested.clear()


def install_signal_handlers(
    journal=None,
    immediate: bool = False,
) -> None:
    """
    Install SIGTERM / SIGINT handlers.

    Parameters
    ----------
    journal   : optional JournalKB-like object with ``log_shutdown()``;
                stashed so signal handlers can write a shutdown marker.
    immediate : if True, the handler calls ``graceful_exit`` directly
                on signal receipt.  If False (default), the handler
                only sets the ``_shutdown_requested`` flag and the
                caller is expected to poll ``shutdown_requested()``
                between units of work.

    Safe to call multiple times — repeated calls update the journal
    reference but don't register duplicate handlers.
    """
    global _journal_ref, _installed
    _journal_ref = journal

    if _installed:
        return

    def _handler(signum, frame):
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        # ``logger`` is safe here — stdlib logging is signal-safe for
        # already-initialized handlers.
        logger.warning("Received %s — requesting graceful shutdown", sig_name)
        _shutdown_requested.set()
        if immediate:
            graceful_exit(code=0, reason=f"signal_{sig_name}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError) as exc:
            # ``signal.signal`` can only be called from the main thread.
            # Streamlit workers and some test harnesses run in threads —
            # skip installation there, not a fatal error.
            logger.debug("Could not install %s handler: %s", sig, exc)

    _installed = True
    logger.info("Graceful shutdown handlers installed (immediate=%s)", immediate)


def graceful_exit(
    code: int = 0,
    reason: str = "",
    context: Optional[dict] = None,
) -> None:
    """
    Controlled exit: journal the reason, flush logs, then ``sys.exit``.

    Use this for *decided* terminations:
      • after-hours guard
      • daily drawdown circuit breaker
      • signal-handler-initiated shutdown

    Do NOT use for unrecoverable states (hung cycle, corrupt memory).
    Those should call ``hard_exit`` instead.
    """
    msg = f"graceful_exit({code}): {reason or 'no reason given'}"
    try:
        logger.info(msg)
    except Exception:
        pass

    if _journal_ref is not None:
        try:
            _journal_ref.log_shutdown(reason or f"exit_{code}", context or {})
        except Exception as exc:
            # Journal write may race with a half-torn-down logging
            # system; swallow so we can still exit cleanly.
            try:
                logger.warning("Journal shutdown write failed: %s", exc)
            except Exception:
                pass

    # Flush + close all logging handlers in the correct order.
    try:
        logging.shutdown()
    except Exception:
        pass

    sys.exit(code)


def hard_exit(
    code: int = 1,
    reason: str = "",
    context: Optional[dict] = None,
) -> None:
    """
    Unrecoverable exit — used for cycle-timeout / deadlock paths.

    Tries to journal the reason and flush logs, but assumes the process
    may be in a bad state and uses ``os._exit`` to bypass atexit / thread
    cleanup.  Prefer ``graceful_exit`` whenever possible.
    """
    msg = f"hard_exit({code}): {reason or 'no reason given'}"
    try:
        logger.critical(msg)
    except Exception:
        pass

    if _journal_ref is not None:
        try:
            _journal_ref.log_shutdown(reason or f"hard_exit_{code}", context or {})
        except Exception:
            pass

    try:
        logging.shutdown()
    except Exception:
        pass

    os._exit(code)   # noqa: SLF001  intentional hard kill
