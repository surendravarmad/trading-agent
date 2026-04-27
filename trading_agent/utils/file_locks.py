"""
file_locks — POSIX advisory locking primitives for concurrent writes
====================================================================

Context
-------
The agent writes to several shared files from multiple possible
processes:

  • trade_journal/signals.jsonl  — appended every cycle, every ticker
  • trade_journal/signals.md     — appended alongside JSONL
  • trade_plans/daily_state.json — read-modify-write for drawdown + debounce

Although a single scheduler is expected, overlapping runs can happen:
  – cron fires the next cycle before the previous exits
  – a manual ``python -m trading_agent.agent`` runs alongside cron
  – the Streamlit backtester writes to trade_journal concurrently

Without locking, two appends can interleave inside a single JSON line
(corrupting the JSONL parser) and read-modify-write on daily_state can
drop updates.

Design choices
--------------
• ``fcntl.flock`` (advisory, POSIX) — sufficient for processes that
  cooperate.  No kernel enforcement against a rogue writer that
  bypasses flock; that's an acceptable trade for simplicity.

• ``locked_append`` opens in ``"a"`` mode, acquires an exclusive lock,
  writes, fsyncs, then releases.  Guarantees each caller's bytes are
  contiguous in the file.

• ``atomic_write`` writes to a sibling ``.tmp`` file then ``os.replace``
  — atomic on POSIX.  Use for read-modify-write state files.

• Windows compatibility — falls back to a no-op lock.  The live agent
  is POSIX-only (deployed on Linux / macOS); Windows users get best
  effort without hard crashes.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform probe — fcntl is POSIX-only
# ---------------------------------------------------------------------------
try:
    import fcntl   # type: ignore
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


@contextlib.contextmanager
def _flock(fh, exclusive: bool = True) -> Iterator[None]:
    """
    Acquire an advisory lock on *fh*.  No-op on non-POSIX platforms.

    The lock is released in the ``finally`` branch even if the body
    raises.  On platforms without fcntl this degrades to a no-op, which
    matches Python file write semantics (concurrent append still risks
    interleaving but the process won't crash).
    """
    if _HAS_FCNTL:
        op = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(fh.fileno(), op)
        except OSError as exc:
            # Some filesystems (NFS, certain tmpfs configs) don't support
            # flock.  Warn once, then proceed unlocked.
            logger.warning("flock unavailable (%s) — proceeding unlocked", exc)
    try:
        yield
    finally:
        if _HAS_FCNTL:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass   # lock may have been released by process exit


@contextlib.contextmanager
def locked_append(path: str, encoding: str = "utf-8") -> Iterator:
    """
    Open *path* in append mode with an exclusive advisory lock.

    Ensures the caller's writes land contiguously even when multiple
    processes append to the same file.  On exit the buffer is flushed
    and fsynced to disk before the lock is released.

    Usage::

        with locked_append("signals.jsonl") as fh:
            fh.write(json.dumps(record) + "\n")
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fh = open(path, "a", encoding=encoding)
    try:
        with _flock(fh, exclusive=True):
            yield fh
            try:
                fh.flush()
                os.fsync(fh.fileno())
            except OSError as exc:
                # fsync can fail on some network filesystems — not fatal.
                logger.debug("fsync failed on %s: %s", path, exc)
    finally:
        fh.close()


def atomic_write_json(path: str, data: Any, indent: Optional[int] = None) -> None:
    """
    Write *data* as JSON to *path* atomically.

    Strategy: write to ``{path}.tmp.{pid}`` in the same directory, fsync,
    then ``os.replace()`` onto the target.  ``os.replace`` is atomic on
    POSIX, so readers either see the old file or the new one — never a
    partial write.

    Pairing ``atomic_write_json`` (writer) with normal ``open(path)``
    (reader) is safe.  For concurrent writers, callers should hold an
    external lock; the ``update_json_locked`` helper below handles that
    pattern end-to-end.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent, default=str)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup of the tmp file if replace didn't happen.
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def update_json_locked(
    path: str,
    mutator: Callable[[dict], dict],
    default: Optional[dict] = None,
) -> dict:
    """
    Read-modify-write a JSON file under an exclusive advisory lock.

    The lock is held on a sidecar ``.lock`` file so the lock lifetime
    is independent of the target file's replacement.  This is the
    standard pattern for ``os.replace``-based atomic writes.

    Parameters
    ----------
    path     : target JSON file
    mutator  : function (state_dict) -> updated_state_dict
    default  : returned by mutator's input when file is missing/corrupt

    Returns the post-mutation state.
    """
    default = default if default is not None else {}
    lock_path = path + ".lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)

    # Open/create the lock sentinel
    lock_fh = open(lock_path, "a+")
    try:
        with _flock(lock_fh, exclusive=True):
            # Read current state (tolerate missing / corrupt)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    state = json.load(fh)
            except (FileNotFoundError, json.JSONDecodeError):
                state = dict(default)

            new_state = mutator(state)
            atomic_write_json(path, new_state)
            return new_state
    finally:
        lock_fh.close()
