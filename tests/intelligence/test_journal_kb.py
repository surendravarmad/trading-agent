"""Tests for the JournalKB always-on signal logger."""

import json
import os
import pytest

from trading_agent.intelligence.journal_kb import JournalKB


@pytest.fixture
def jkb(tmp_path):
    return JournalKB(str(tmp_path / "journal"))


class TestJournalKBInit:
    def test_creates_directory(self, tmp_path):
        path = tmp_path / "new_journal"
        JournalKB(str(path))
        assert path.exists()

    def test_creates_markdown_header_on_first_run(self, jkb):
        assert os.path.exists(jkb.md_path)
        content = open(jkb.md_path).read()
        assert "Timestamp" in content
        assert "Ticker" in content

    def test_does_not_overwrite_existing_md(self, tmp_path):
        """Re-instantiating JournalKB must not wipe existing entries."""
        path = tmp_path / "j"
        jkb1 = JournalKB(str(path))
        jkb1.log_signal("SPY", "dry_run", 500.0, {})
        lines_before = open(jkb1.md_path).readlines()

        jkb2 = JournalKB(str(path))   # re-init
        lines_after = open(jkb2.md_path).readlines()
        assert lines_after == lines_before


class TestLogSignal:
    def test_writes_jsonl_record(self, jkb):
        jkb.log_signal(
            ticker="AAPL", action="dry_run", price=178.5,
            raw_signal={"regime": "bullish", "strategy": "Bull Put Spread",
                        "risk_approved": True, "net_credit": 1.60},
        )
        lines = open(jkb.jsonl_path).readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["ticker"] == "AAPL"
        assert record["action"] == "dry_run"
        assert record["price"] == pytest.approx(178.5)
        assert record["raw_signal"]["regime"] == "bullish"

    def test_writes_markdown_row(self, jkb):
        jkb.log_signal("SPY", "rejected", 655.0,
                       {"regime": "bearish", "strategy": "Bear Call Spread",
                        "risk_approved": False})
        md = open(jkb.md_path).read()
        assert "SPY" in md
        assert "rejected" in md

    def test_multiple_signals_all_appended(self, jkb):
        for i in range(5):
            jkb.log_signal(f"T{i}", "dry_run", float(100 + i), {})
        lines = open(jkb.jsonl_path).readlines()
        assert len(lines) == 5

    def test_exec_status_defaults_to_action(self, jkb):
        jkb.log_signal("SPY", "submitted", 500.0, {})
        record = json.loads(open(jkb.jsonl_path).readline())
        assert record["exec_status"] == "submitted"

    def test_custom_exec_status(self, jkb):
        jkb.log_signal("SPY", "dry_run", 500.0, {}, exec_status="dry_run_ok")
        record = json.loads(open(jkb.jsonl_path).readline())
        assert record["exec_status"] == "dry_run_ok"

    def test_notes_truncated_to_120_chars(self, jkb):
        long_note = "x" * 200
        jkb.log_signal("SPY", "error", 0.0, {}, notes=long_note)
        record = json.loads(open(jkb.jsonl_path).readline())
        assert len(record["notes"]) <= 120

    def test_pipe_chars_escaped_in_markdown(self, jkb):
        jkb.log_signal("SPY", "dry_run", 500.0, {},
                       notes="credit|ratio|test")
        md_rows = open(jkb.md_path).readlines()
        data_rows = [l for l in md_rows if "SPY" in l]
        assert len(data_rows) == 1
        # Pipe in notes must be escaped so it doesn't break the table
        assert "\\|" in data_rows[0]


class TestLogError:
    def test_log_error_writes_error_action(self, jkb):
        jkb.log_error("MSFT", "API timeout")
        record = json.loads(open(jkb.jsonl_path).readline())
        assert record["action"] == "error"
        assert record["ticker"] == "MSFT"
        assert "API timeout" in record["raw_signal"]["error"]

    def test_log_error_with_context(self, jkb):
        jkb.log_error("QQQ", "chain empty", context={"expiry": "2026-05-15"})
        record = json.loads(open(jkb.jsonl_path).readline())
        assert record["raw_signal"]["expiry"] == "2026-05-15"


class TestLogCycleError:
    def test_cycle_error_written_to_jsonl(self, jkb):
        jkb.log_cycle_error("cycle_timeout", {"timeout_seconds": 270})
        lines = open(jkb.jsonl_path).readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == "cycle_error"
        assert record["error"] == "cycle_timeout"

    def test_cycle_error_does_not_write_md_row(self, jkb):
        """Cycle errors are JSONL-only — no malformed MD row."""
        jkb.log_cycle_error("timeout")
        md_lines = open(jkb.md_path).readlines()
        # Only the header lines should be present
        data_lines = [l for l in md_lines if "|" in l and "---" not in l
                      and "Timestamp" not in l]
        assert len(data_lines) == 0


class TestAutoNotes:
    def test_auto_notes_includes_strategy(self, jkb):
        jkb.log_signal("SPY", "dry_run", 500.0,
                       {"strategy": "Bull Put Spread", "net_credit": 1.70,
                        "credit_to_width_ratio": 0.34})
        record = json.loads(open(jkb.jsonl_path).readline())
        assert "Bull Put Spread" in record["notes"]
        assert "1.70" in record["notes"]

    def test_auto_notes_includes_rejection_reason(self, jkb):
        jkb.log_signal("SPY", "rejected", 500.0,
                       {"rejection_reason": "No call contracts"})
        record = json.loads(open(jkb.jsonl_path).readline())
        assert "No call contracts" in record["notes"]
