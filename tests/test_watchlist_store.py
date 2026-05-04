"""Tests for trading_agent/watchlist_store.py."""

import json

import pytest

from trading_agent.watchlist_store import (
    SCHEMA_VERSION,
    Watchlist,
    WatchlistEntry,
    add_ticker,
    load_watchlist,
    remove_ticker,
    save_watchlist,
    update_note,
)


# ----------------------------------------------------------------------
# WatchlistEntry normalisation
# ----------------------------------------------------------------------
class TestEntryNormalisation:
    def test_symbol_uppercased_and_stripped(self):
        e = WatchlistEntry(symbol="  spy  ")
        assert e.symbol == "SPY"

    def test_added_at_auto_filled_when_empty(self):
        e = WatchlistEntry(symbol="SPY")
        assert e.added_at  # ISO string, non-empty
        assert e.added_at.endswith("Z")

    def test_added_at_preserved_when_provided(self):
        e = WatchlistEntry(symbol="SPY", added_at="2026-01-01T00:00:00Z")
        assert e.added_at == "2026-01-01T00:00:00Z"


# ----------------------------------------------------------------------
# Round-trip: save → load preserves data
# ----------------------------------------------------------------------
class TestRoundtrip:
    def test_empty_file_returns_empty_watchlist(self, tmp_path):
        wl = load_watchlist(tmp_path / "missing.json")
        assert wl.tickers == []

    def test_save_then_load(self, tmp_path):
        path = tmp_path / "wl.json"
        wl = Watchlist(tickers=[
            WatchlistEntry(symbol="SPY", note="benchmark"),
            WatchlistEntry(symbol="QQQ", note=""),
        ])
        save_watchlist(wl, path)

        loaded = load_watchlist(path)
        assert loaded.symbols() == ["SPY", "QQQ"]
        assert loaded.tickers[0].note == "benchmark"
        assert loaded.schema_version == SCHEMA_VERSION

    def test_atomic_write_no_tmp_left_behind(self, tmp_path):
        path = tmp_path / "wl.json"
        save_watchlist(Watchlist([WatchlistEntry("SPY")]), path)
        assert path.exists()
        assert not path.with_suffix(".json.tmp").exists()


# ----------------------------------------------------------------------
# CRUD
# ----------------------------------------------------------------------
class TestCRUD:
    def test_add_ticker_creates_file(self, tmp_path):
        path = tmp_path / "wl.json"
        wl = add_ticker("SPY", note="benchmark", path=path)
        assert path.exists()
        assert wl.symbols() == ["SPY"]

    def test_add_is_idempotent(self, tmp_path):
        path = tmp_path / "wl.json"
        add_ticker("SPY", path=path)
        add_ticker("spy", path=path)  # case-insensitive dedup
        wl = load_watchlist(path)
        assert wl.symbols() == ["SPY"]

    def test_add_duplicate_updates_note(self, tmp_path):
        path = tmp_path / "wl.json"
        add_ticker("SPY", note="first", path=path)
        add_ticker("SPY", note="second", path=path)
        wl = load_watchlist(path)
        assert wl.tickers[0].note == "second"

    def test_remove(self, tmp_path):
        path = tmp_path / "wl.json"
        add_ticker("SPY", path=path)
        add_ticker("QQQ", path=path)
        remove_ticker("SPY", path=path)
        wl = load_watchlist(path)
        assert wl.symbols() == ["QQQ"]

    def test_remove_missing_is_noop(self, tmp_path):
        path = tmp_path / "wl.json"
        add_ticker("SPY", path=path)
        remove_ticker("AAPL", path=path)  # not present
        wl = load_watchlist(path)
        assert wl.symbols() == ["SPY"]

    def test_update_note_present(self, tmp_path):
        path = tmp_path / "wl.json"
        add_ticker("SPY", note="old", path=path)
        update_note("SPY", "new", path=path)
        assert load_watchlist(path).tickers[0].note == "new"

    def test_update_note_missing_returns_none(self, tmp_path):
        path = tmp_path / "wl.json"
        assert update_note("SPY", "x", path=path) is None

    def test_add_empty_symbol_raises(self, tmp_path):
        with pytest.raises(ValueError):
            add_ticker("", path=tmp_path / "wl.json")


# ----------------------------------------------------------------------
# Schema robustness
# ----------------------------------------------------------------------
class TestSchemaRobustness:
    def test_unknown_schema_version_loads_best_effort(self, tmp_path, caplog):
        path = tmp_path / "wl.json"
        path.write_text(json.dumps({
            "schema_version": 99,
            "tickers": [{"symbol": "SPY", "added_at": "2026-01-01T00:00:00Z"}],
        }))
        wl = load_watchlist(path)
        assert wl.symbols() == ["SPY"]
        # current writer normalises the version on next save
        assert wl.schema_version == SCHEMA_VERSION

    def test_corrupt_json_returns_empty(self, tmp_path):
        path = tmp_path / "wl.json"
        path.write_text("{not json")
        wl = load_watchlist(path)
        assert wl.tickers == []

    def test_skips_malformed_rows(self, tmp_path):
        path = tmp_path / "wl.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "tickers": [
                {"symbol": "SPY"},
                {"no_symbol_key": True},
                {"symbol": ""},          # empty after normalise → skip
                {"symbol": "QQQ"},
            ],
        }))
        wl = load_watchlist(path)
        assert wl.symbols() == ["SPY", "QQQ"]
