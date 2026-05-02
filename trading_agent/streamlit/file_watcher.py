"""
file_watcher — watchdog-backed change notifier for Streamlit dashboards
======================================================================

Why
---
Streamlit reruns the whole script on every interaction (a checkbox click,
a slider drag, an auto-refresh tick). The Live Monitor and LLM Extension
panels both call ``_load_journal_df()`` / ``_load_recent_signals()`` on
each rerun, which today re-reads ``signals_live.jsonl`` from byte zero —
O(n) work even when the journal hasn't changed.

This module provides a single ``Observer`` (one per Python process, via
``@st.cache_resource``) that watches the trade-journal and trade-plans
directories. Each filesystem event bumps a thread-safe per-file ``version``
counter. Loaders use that counter as a cache key:

    @st.cache_data
    def _load_journal_df(version: int, path: str) -> pd.DataFrame: ...

Reruns from unrelated UI actions hit the cache (zero I/O); only a real
write to the journal triggers a re-parse + dashboard refresh.

Kill switch
-----------
Set ``WATCHDOG_DISABLE=1`` to fall back to today's behaviour (the
``get_version`` function will always return 0, so cache keys never
change and Streamlit's normal rerun semantics drive refresh).

Force-polling
-------------
On NFS / network mounts inotify is unavailable. Set
``WATCHDOG_FORCE_POLLING=1`` to use ``PollingObserver`` (~1 s resolution,
slower but correct).
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import — watchdog is a runtime dep but we degrade gracefully
# if it's missing or disabled.
# ---------------------------------------------------------------------------

_DISABLED = os.environ.get("WATCHDOG_DISABLE", "0") == "1"
_FORCE_POLLING = os.environ.get("WATCHDOG_FORCE_POLLING", "0") == "1"

try:
    if _DISABLED:
        raise ImportError("watchdog disabled by WATCHDOG_DISABLE=1")
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
    from watchdog.observers.polling import PollingObserver
    _HAS_WATCHDOG = True
except ImportError as exc:
    logger.info("watchdog unavailable (%s) — dashboards will rerun without "
                "event-driven cache invalidation", exc)
    _HAS_WATCHDOG = False
    FileSystemEventHandler = object  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Watcher implementation
# ---------------------------------------------------------------------------

class JournalWatcher:
    """
    Per-process singleton that maps absolute file paths to monotonic
    version counters incremented on each filesystem event.

    Thread-safety: ``_versions`` is mutated only under ``_lock``.
    Streamlit reruns read via ``get_version`` which acquires the lock
    briefly — cheap (~µs), called once per rerun per file.
    """

    def __init__(self) -> None:
        self._versions: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._observer = None
        self._watched_dirs: set[str] = set()

        if not _HAS_WATCHDOG:
            return

        ObserverCls = PollingObserver if _FORCE_POLLING else Observer
        self._observer = ObserverCls()
        self._observer.daemon = True
        self._observer.start()
        logger.info("JournalWatcher: %s started",
                    ObserverCls.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def watch(self, path: str | Path) -> None:
        """
        Register *path* (file) for change notifications. Watching the
        parent directory is sufficient — filename filtering happens in
        the event handler. Idempotent: re-watching a directory is a
        no-op.
        """
        if self._observer is None:
            return
        p = Path(path).resolve()
        directory = str(p.parent)
        # Ensure the parent dir exists so watchdog's schedule() doesn't
        # raise. Loaders may be called before the writer has produced
        # the file for the first time.
        os.makedirs(directory, exist_ok=True)
        with self._lock:
            if directory in self._watched_dirs:
                return
            self._watched_dirs.add(directory)
            self._versions.setdefault(str(p), 0)
        handler = _BumpHandler(self, watched_files={str(p)})
        self._observer.schedule(handler, directory, recursive=False)
        logger.debug("JournalWatcher: scheduled %s under %s",
                     p.name, directory)

    def get_version(self, path: str | Path) -> int:
        """
        Return the current version counter for *path*. Returns 0 when
        watchdog is disabled / unavailable so cache keys stay stable
        and loaders fall back to per-rerun reads.
        """
        if self._observer is None:
            return 0
        with self._lock:
            return self._versions.get(str(Path(path).resolve()), 0)

    def _bump(self, abspath: str) -> None:
        """Internal — called by event handler when a watched file changes."""
        with self._lock:
            self._versions[abspath] = self._versions.get(abspath, 0) + 1
        logger.debug("JournalWatcher: bumped %s to v%d",
                     abspath, self._versions[abspath])


class _BumpHandler(FileSystemEventHandler):
    """
    Filters watchdog events to only the files we actually care about
    inside a watched directory, then increments the version counter
    on each modification or creation.
    """

    def __init__(self, watcher: JournalWatcher, watched_files: set[str]):
        super().__init__()
        self._watcher = watcher
        self._watched = watched_files

    def _maybe_bump(self, src_path: str) -> None:
        try:
            resolved = str(Path(src_path).resolve())
        except OSError:
            return
        if resolved in self._watched:
            self._watcher._bump(resolved)

    # watchdog dispatches by event type; we treat created/modified/moved
    # all as "content might have changed". Deletion is intentionally
    # ignored — loaders already handle FileNotFoundError gracefully.
    def on_modified(self, event):
        if not event.is_directory:
            self._maybe_bump(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._maybe_bump(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._maybe_bump(event.dest_path)


# ---------------------------------------------------------------------------
# Streamlit-aware accessors
# ---------------------------------------------------------------------------
# We avoid importing streamlit at module top so this file remains
# importable from non-Streamlit contexts (the test harness, the CLI
# agent, etc.). The ``get_watcher`` accessor lazily wraps the singleton
# in ``st.cache_resource`` so exactly one Observer exists per worker.

_singleton: Optional[JournalWatcher] = None
_singleton_lock = threading.Lock()


def get_watcher() -> JournalWatcher:
    """
    Return the per-process JournalWatcher. Uses ``st.cache_resource``
    when running under Streamlit; otherwise a plain module-level
    singleton (sufficient for tests and the CLI).
    """
    global _singleton
    try:
        import streamlit as st

        @st.cache_resource(show_spinner=False)
        def _cached_watcher() -> JournalWatcher:
            return JournalWatcher()

        return _cached_watcher()
    except Exception:
        # Streamlit not in scope (e.g. unit tests) — use plain singleton.
        with _singleton_lock:
            if _singleton is None:
                _singleton = JournalWatcher()
            return _singleton


def watch(path: str | Path) -> int:
    """
    Convenience: register *path* with the singleton watcher and return
    its current version. Loaders typically call this once per rerun:

        v = file_watcher.watch(JOURNAL_PATH)
        df = _load_journal_df_cached(v, str(JOURNAL_PATH))
    """
    w = get_watcher()
    w.watch(path)
    return w.get_version(path)
