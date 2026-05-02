"""
Tests for visualize_logs.py — the Agent Performance Dashboard generator.

Coverage
--------
- load_signals    : happy path, empty file, missing file, date/ticker filters
- load_trade_plans: happy path, missing dir, no-approved-trades, bad JSON
- build_heartbeat_timeline  : with data, empty DataFrame
- build_safety_buffer_chart : with data, no price, approved short strike
- build_logic_distribution  : with data, empty DataFrame
- generate_dashboard        : full pipeline, empty logs, HTML output validity
- CLI _parse_args           : default and custom arguments
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

# Import the module under test
import visualize_logs as vl


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _ts(dt_str: str) -> str:
    """Return an ISO-8601 UTC timestamp string from a naive datetime string."""
    return datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc).isoformat()


def _write_signals(path: Path, records: list) -> None:
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _make_signal(
    ticker="SPY",
    action="dry_run",
    price=500.0,
    ts="2026-04-03T14:00:00",
    notes="test note",
    rejection_reason=None,
    regime="bullish",
    strategy="Bull Put Spread",
    risk_approved=True,
    net_credit=0.45,
    rsi_14=55.0,
    sma_50=498.0,
    sma_200=490.0,
) -> dict:
    return {
        "timestamp": _ts(ts),
        "ticker": ticker,
        "action": action,
        "price": price,
        "exec_status": action,
        "notes": notes,
        "raw_signal": {
            "regime": regime,
            "strategy": strategy,
            "plan_valid": True,
            "risk_approved": risk_approved,
            "net_credit": net_credit,
            "max_loss": 455.0,
            "credit_to_width_ratio": 0.09,
            "spread_width": 5.0,
            "expiration": "2026-05-22",
            "dte": 49,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "rsi_14": rsi_14,
            "account_balance": 100_000.0,
            "checks_passed": ["Max loss", "Delta"],
            "checks_failed": [],
            "llm_decision": None,
            "llm_confidence": None,
            "rejection_reason": rejection_reason,
            "error": None,
        },
    }


def _make_trade_plan_file(path: Path, ticker="SPY", entries=None) -> None:
    if entries is None:
        entries = [_make_history_entry(ticker=ticker)]
    data = {
        "ticker": ticker,
        "created": _ts("2026-04-03T09:00:00"),
        "last_updated": _ts("2026-04-03T14:00:00"),
        "state_history": entries,
    }
    with path.open("w") as fh:
        json.dump(data, fh)


def _make_history_entry(
    ticker="SPY",
    run_id="20260403_140000",
    approved=True,
    short_strike=480.0,
    strategy="Bull Put Spread",
    ts="2026-04-03T14:00:00",
) -> dict:
    return {
        "run_id": run_id,
        "timestamp": _ts(ts),
        "mode": "dry_run",
        "trade_plan": {
            "ticker": ticker,
            "strategy": strategy,
            "regime": "bullish",
            "legs": [
                {
                    "symbol": f"{ticker}260522P00{int(short_strike):08d}",
                    "strike": short_strike,
                    "action": "sell",
                    "type": "put",
                    "delta": -0.18,
                    "bid": 1.20,
                    "ask": 1.40,
                },
                {
                    "symbol": f"{ticker}260522P00{int(short_strike - 5):08d}",
                    "strike": short_strike - 5.0,
                    "action": "buy",
                    "type": "put",
                    "delta": -0.10,
                    "bid": 0.80,
                    "ask": 1.00,
                },
            ],
            "spread_width": 5.0,
            "net_credit": 0.45,
            "max_loss": 455.0,
            "credit_to_width_ratio": 0.09,
            "expiration": "2026-05-22",
            "valid": True,
        },
        "risk_verdict": {
            "approved": approved,
            "account_balance": 100_000.0,
            "max_allowed_loss": 2_000.0,
            "checks_passed": ["Max loss", "Delta"],
            "checks_failed": [],
            "summary": "APPROVED: 8 passed, 0 failed",
        },
    }


# ---------------------------------------------------------------------------
# load_signals
# ---------------------------------------------------------------------------


class TestLoadSignals:

    def test_happy_path_returns_correct_columns(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        _write_signals(jfile, [_make_signal()])
        df = vl.load_signals(str(jfile))
        assert not df.empty
        for col in ("timestamp", "ticker", "action", "price", "status", "color", "reason"):
            assert col in df.columns

    def test_status_mapped_for_dry_run(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        _write_signals(jfile, [_make_signal(action="dry_run")])
        df = vl.load_signals(str(jfile))
        assert df.iloc[0]["status"] == "Trade Executed"

    def test_status_mapped_for_skip(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        _write_signals(jfile, [_make_signal(action="skip")])
        df = vl.load_signals(str(jfile))
        assert df.iloc[0]["status"] == "Monitoring/No Action"

    def test_status_mapped_for_error(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        _write_signals(jfile, [_make_signal(action="error")])
        df = vl.load_signals(str(jfile))
        assert df.iloc[0]["status"] == "Error/Timeout"

    def test_defense_first_action_maps_correctly(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        _write_signals(jfile, [_make_signal(action="skipped_defense_first")])
        df = vl.load_signals(str(jfile))
        assert df.iloc[0]["status"] == "Defense First"

    def test_unknown_action_maps_to_unknown(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        record = _make_signal(action="totally_unknown_action")
        _write_signals(jfile, [record])
        df = vl.load_signals(str(jfile))
        assert df.iloc[0]["status"] == "Unknown"

    def test_filter_date_keeps_matching_rows(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        _write_signals(jfile, [
            _make_signal(ts="2026-04-03T10:00:00"),
            _make_signal(ts="2026-04-02T10:00:00"),
        ])
        df = vl.load_signals(str(jfile), filter_date="2026-04-03")
        assert len(df) == 1
        assert df.iloc[0]["timestamp"].date().isoformat() == "2026-04-03"

    def test_filter_date_excludes_other_days(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        _write_signals(jfile, [_make_signal(ts="2026-04-01T09:00:00")])
        df = vl.load_signals(str(jfile), filter_date="2026-04-03")
        assert df.empty

    def test_ticker_filter(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        _write_signals(jfile, [
            _make_signal(ticker="SPY"),
            _make_signal(ticker="QQQ"),
        ])
        df = vl.load_signals(str(jfile), tickers=["SPY"])
        assert list(df["ticker"].unique()) == ["SPY"]

    def test_missing_file_returns_empty_dataframe(self, tmp_path):
        df = vl.load_signals(str(tmp_path / "nonexistent.jsonl"))
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_empty_file_returns_empty_dataframe(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        jfile.write_text("")
        df = vl.load_signals(str(jfile))
        assert df.empty

    def test_cycle_error_events_without_ticker_are_skipped(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        cycle_err = {
            "timestamp": _ts("2026-04-03T10:00:00"),
            "event": "cycle_error",
            "error": "some error",
        }
        _write_signals(jfile, [cycle_err, _make_signal()])
        df = vl.load_signals(str(jfile))
        # Only the signal with a ticker should be loaded
        assert len(df) == 1

    def test_malformed_json_line_is_skipped(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        with jfile.open("w") as fh:
            fh.write("{invalid json}\n")
            fh.write(json.dumps(_make_signal()) + "\n")
        df = vl.load_signals(str(jfile))
        assert len(df) == 1

    def test_reason_falls_back_to_rejection_reason(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        sig = _make_signal(notes="", rejection_reason="Plan invalid: no chain")
        _write_signals(jfile, [sig])
        df = vl.load_signals(str(jfile))
        assert "Plan invalid" in df.iloc[0]["reason"]

    def test_sorted_by_timestamp(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        _write_signals(jfile, [
            _make_signal(ts="2026-04-03T15:00:00"),
            _make_signal(ts="2026-04-03T09:00:00"),
        ])
        df = vl.load_signals(str(jfile))
        timestamps = list(df["timestamp"])
        assert timestamps == sorted(timestamps)

    def test_multiple_tickers_all_loaded(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        _write_signals(jfile, [
            _make_signal(ticker="SPY"),
            _make_signal(ticker="QQQ"),
            _make_signal(ticker="AAPL"),
        ])
        df = vl.load_signals(str(jfile))
        assert set(df["ticker"]) == {"SPY", "QQQ", "AAPL"}


# ---------------------------------------------------------------------------
# _resolve_journal_path — May-2026 journal-split fallback
# ---------------------------------------------------------------------------


class TestResolveJournalPath:
    """
    Pins the back-compat fallback so an old archive that still has
    ``signals.jsonl`` (pre-split) keeps rendering with the new default.
    Explicit user-supplied paths must NEVER be silently rewritten.
    """

    def test_returns_signals_live_when_present(self, tmp_path):
        live = tmp_path / "signals_live.jsonl"
        live.write_text("{}\n")
        resolved = vl._resolve_journal_path(str(live))
        assert resolved == live

    def test_falls_back_to_legacy_when_default_missing(self, tmp_path):
        # No signals_live.jsonl, only the pre-split signals.jsonl
        legacy = tmp_path / "signals.jsonl"
        legacy.write_text("{}\n")
        resolved = vl._resolve_journal_path(str(tmp_path / "signals_live.jsonl"))
        assert resolved == legacy

    def test_does_not_rewrite_explicit_backtest_path(self, tmp_path):
        # User asked for signals_backtest.jsonl explicitly — never fall
        # back to the live legacy; that would silently render the wrong
        # data set.
        bt = tmp_path / "signals_backtest.jsonl"
        # Even with a sibling signals.jsonl present, the explicit path
        # must come back unchanged so load_signals' missing-file branch
        # produces an empty DataFrame instead of legacy data.
        (tmp_path / "signals.jsonl").write_text("{}\n")
        resolved = vl._resolve_journal_path(str(bt))
        assert resolved == bt
        assert not resolved.exists()

    def test_returns_missing_default_when_nothing_present(self, tmp_path):
        # Neither file exists → return the requested default unchanged
        # so load_signals' .exists() check fires its empty-DataFrame path.
        requested = tmp_path / "signals_live.jsonl"
        resolved = vl._resolve_journal_path(str(requested))
        assert resolved == requested
        assert not resolved.exists()


# ---------------------------------------------------------------------------
# load_trade_plans
# ---------------------------------------------------------------------------


class TestLoadTradePlans:

    def test_happy_path_loads_short_strike(self, tmp_path):
        plan_file = tmp_path / "trade_plan_SPY.json"
        _make_trade_plan_file(plan_file, ticker="SPY", entries=[
            _make_history_entry(ticker="SPY", short_strike=480.0, approved=True)
        ])
        plans = vl.load_trade_plans(str(tmp_path))
        assert "SPY" in plans
        df = plans["SPY"]
        assert float(df.iloc[0]["short_strike"]) == 480.0

    def test_approved_flag_loaded(self, tmp_path):
        plan_file = tmp_path / "trade_plan_SPY.json"
        _make_trade_plan_file(plan_file, ticker="SPY", entries=[
            _make_history_entry(approved=True),
            _make_history_entry(approved=False, run_id="20260403_150000",
                                ts="2026-04-03T15:00:00"),
        ])
        plans = vl.load_trade_plans(str(tmp_path))
        df = plans["SPY"]
        approved = df[df["approved"] == True]
        rejected = df[df["approved"] == False]
        assert len(approved) == 1
        assert len(rejected) == 1

    def test_missing_directory_returns_empty(self):
        plans = vl.load_trade_plans("/nonexistent/plans/dir")
        assert plans == {}

    def test_ticker_filter_excludes_unwanted_tickers(self, tmp_path):
        for ticker in ("SPY", "QQQ", "AAPL"):
            f = tmp_path / f"trade_plan_{ticker}.json"
            _make_trade_plan_file(f, ticker=ticker)
        plans = vl.load_trade_plans(str(tmp_path), tickers=["SPY"])
        assert list(plans.keys()) == ["SPY"]

    def test_malformed_json_file_is_skipped(self, tmp_path):
        bad = tmp_path / "trade_plan_BAD.json"
        bad.write_text("not valid json")
        plans = vl.load_trade_plans(str(tmp_path))
        assert "BAD" not in plans

    def test_empty_state_history_is_skipped(self, tmp_path):
        f = tmp_path / "trade_plan_SPY.json"
        f.write_text(json.dumps({
            "ticker": "SPY",
            "state_history": [],
        }))
        plans = vl.load_trade_plans(str(tmp_path))
        assert "SPY" not in plans

    def test_no_sell_leg_gives_none_short_strike(self, tmp_path):
        """If all legs are 'buy', short_strike should be None (no sold leg)."""
        entry = _make_history_entry(short_strike=480.0)
        # Remove all sell legs
        for leg in entry["trade_plan"]["legs"]:
            leg["action"] = "buy"
        f = tmp_path / "trade_plan_SPY.json"
        _make_trade_plan_file(f, ticker="SPY", entries=[entry])
        plans = vl.load_trade_plans(str(tmp_path))
        assert plans["SPY"].iloc[0]["short_strike"] is None

    def test_timestamps_are_sorted(self, tmp_path):
        entries = [
            _make_history_entry(run_id="20260403_150000", ts="2026-04-03T15:00:00"),
            _make_history_entry(run_id="20260403_090000", ts="2026-04-03T09:00:00"),
        ]
        f = tmp_path / "trade_plan_SPY.json"
        _make_trade_plan_file(f, ticker="SPY", entries=entries)
        plans = vl.load_trade_plans(str(tmp_path))
        ts = list(plans["SPY"]["timestamp"])
        assert ts == sorted(ts)

    def test_daily_state_json_is_not_loaded(self, tmp_path):
        """daily_state.json should not match the trade_plan_*.json glob."""
        (tmp_path / "daily_state.json").write_text(json.dumps({"date": "2026-04-03"}))
        plans = vl.load_trade_plans(str(tmp_path))
        assert plans == {}


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


class TestBuildHeartbeatTimeline:

    def _make_df(self, n=3):
        records = [
            _make_signal(action="dry_run",  ts="2026-04-03T09:30:00"),
            _make_signal(action="skip",     ts="2026-04-03T09:35:00"),
            _make_signal(action="error",    ts="2026-04-03T09:40:00"),
        ]
        return vl.load_signals.__wrapped__(records) if hasattr(
            vl.load_signals, "__wrapped__"
        ) else _signals_from_records(records)

    def test_empty_dataframe_returns_figure_with_annotation(self):
        fig = vl.build_heartbeat_timeline(pd.DataFrame())
        titles = [a.text for a in fig.layout.annotations]
        assert any("No signals" in t for t in titles)

    def test_data_produces_scatter_traces(self):
        df = _signals_from_records([
            _make_signal(action="dry_run", ts="2026-04-03T09:30:00"),
            _make_signal(action="skip",    ts="2026-04-03T09:35:00"),
        ])
        fig = vl.build_heartbeat_timeline(df)
        assert len(fig.data) >= 1

    def test_distinct_statuses_produce_separate_traces(self):
        df = _signals_from_records([
            _make_signal(action="dry_run"),
            _make_signal(action="skip"),
            _make_signal(action="error"),
        ])
        fig = vl.build_heartbeat_timeline(df)
        trace_names = {t.name for t in fig.data}
        assert "Trade Executed" in trace_names
        assert "Monitoring/No Action" in trace_names
        assert "Error/Timeout" in trace_names

    def test_dark_background_applied(self):
        df = _signals_from_records([_make_signal()])
        fig = vl.build_heartbeat_timeline(df)
        assert fig.layout.plot_bgcolor == "#1e1e2e"


class TestBuildSafetyBufferChart:

    def test_empty_inputs_return_no_data_figure(self):
        fig = vl.build_safety_buffer_chart(pd.DataFrame(), {})
        titles = [a.text for a in fig.layout.annotations]
        assert any("No price" in t for t in titles)

    def test_price_trace_added_for_ticker(self, tmp_path):
        df = _signals_from_records([
            _make_signal(ticker="SPY", price=500.0, ts="2026-04-03T10:00:00"),
            _make_signal(ticker="SPY", price=501.0, ts="2026-04-03T10:05:00"),
        ])
        fig = vl.build_safety_buffer_chart(df, {}, tickers=["SPY"])
        trace_names = [t.name for t in fig.data]
        assert any("SPY" in n for n in trace_names)

    def test_short_strike_shape_added_for_approved_trade(self, tmp_path):
        df = _signals_from_records([
            _make_signal(ticker="SPY", price=500.0, ts="2026-04-03T10:00:00"),
        ])
        plan_entry = _make_history_entry(ticker="SPY", short_strike=480.0, approved=True)
        plan_df = pd.DataFrame([{
            "run_id": plan_entry["run_id"],
            "timestamp": pd.Timestamp(plan_entry["timestamp"]),
            "mode": plan_entry["mode"],
            "approved": True,
            "account_balance": 100_000.0,
            "short_strike": 480.0,
            "strategy": "Bull Put Spread",
            "net_credit": 0.45,
            "spread_width": 5.0,
            "expiration": "2026-05-22",
            "valid": True,
        }])
        plans = {"SPY": plan_df}
        fig = vl.build_safety_buffer_chart(df, plans, tickers=["SPY"])
        # At least one shape (the horizontal danger line) should be present
        assert len(fig.layout.shapes) >= 1

    def test_no_approved_trade_means_no_shape(self):
        df = _signals_from_records([_make_signal(ticker="SPY", price=500.0)])
        plan_df = pd.DataFrame([{
            "run_id": "1",
            "timestamp": pd.Timestamp(_ts("2026-04-03T10:00:00")),
            "mode": "dry_run",
            "approved": False,       # not approved
            "account_balance": 100_000.0,
            "short_strike": 480.0,
            "strategy": "Bull Put Spread",
            "net_credit": 0.45,
            "spread_width": 5.0,
            "expiration": "2026-05-22",
            "valid": True,
        }])
        fig = vl.build_safety_buffer_chart(df, {"SPY": plan_df}, tickers=["SPY"])
        assert len(fig.layout.shapes) == 0


class TestBuildLogicDistribution:

    def test_empty_dataframe_returns_no_data_figure(self):
        fig = vl.build_logic_distribution(pd.DataFrame())
        titles = [a.text for a in fig.layout.annotations]
        assert any("No data" in t for t in titles)

    def test_executed_bucket_present(self):
        df = _signals_from_records([_make_signal(action="dry_run")])
        fig = vl.build_logic_distribution(df)
        labels = list(fig.data[0].labels)
        assert "Active Trade" in labels

    def test_monitoring_bucket_present(self):
        df = _signals_from_records([_make_signal(action="skip")])
        fig = vl.build_logic_distribution(df)
        labels = list(fig.data[0].labels)
        assert any("Monitoring" in lb for lb in labels)

    def test_rsi_bucket_from_notes(self):
        sig = _make_signal(action="skip", notes="RSI > 70, skipping entry")
        df = _signals_from_records([sig])
        fig = vl.build_logic_distribution(df)
        labels = list(fig.data[0].labels)
        assert any("RSI" in lb or "Overbought" in lb for lb in labels)

    def test_macro_guard_notes_maps_to_sma_bucket(self):
        sig = _make_signal(
            action="skipped_defense_first",
            notes="defense_first: MacroGuard: price < SMA-200"
        )
        df = _signals_from_records([sig])
        fig = vl.build_logic_distribution(df)
        labels = list(fig.data[0].labels)
        assert any("SMA" in lb or "Trend" in lb for lb in labels)

    def test_pie_chart_percentages_sum_to_100(self):
        records = [
            _make_signal(action="dry_run"),
            _make_signal(action="skip"),
            _make_signal(action="error"),
            _make_signal(action="rejected"),
        ]
        df = _signals_from_records(records)
        fig = vl.build_logic_distribution(df)
        values = fig.data[0].values
        total = sum(values)
        assert total == len(records)


# ---------------------------------------------------------------------------
# generate_dashboard (full pipeline)
# ---------------------------------------------------------------------------


class TestGenerateDashboard:

    def test_creates_html_file(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        _write_signals(jfile, [_make_signal(ts="2026-04-03T14:00:00")])
        out = tmp_path / "report.html"
        result = vl.generate_dashboard(
            journal_path=str(jfile),
            plans_dir=str(tmp_path),
            filter_date="2026-04-03",
            output_path=str(out),
        )
        assert os.path.exists(result)
        assert result.endswith(".html")

    def test_html_contains_plotly_script(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        _write_signals(jfile, [_make_signal(ts="2026-04-03T14:00:00")])
        out = tmp_path / "report.html"
        vl.generate_dashboard(
            journal_path=str(jfile),
            plans_dir=str(tmp_path),
            filter_date="2026-04-03",
            output_path=str(out),
        )
        content = out.read_text(encoding="utf-8")
        assert "plotly" in content.lower()

    def test_html_contains_three_chart_sections(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        _write_signals(jfile, [_make_signal(ts="2026-04-03T14:00:00")])
        out = tmp_path / "report.html"
        vl.generate_dashboard(
            journal_path=str(jfile),
            plans_dir=str(tmp_path),
            filter_date="2026-04-03",
            output_path=str(out),
        )
        content = out.read_text(encoding="utf-8")
        assert "Heartbeat Timeline" in content
        assert "Safety Buffer" in content
        assert "Logic Distribution" in content

    def test_empty_logs_produces_valid_html(self, tmp_path):
        """When there are no signals, the report should still render cleanly."""
        jfile = tmp_path / "empty.jsonl"
        jfile.write_text("")
        out = tmp_path / "report.html"
        vl.generate_dashboard(
            journal_path=str(jfile),
            plans_dir=str(tmp_path),
            filter_date="2026-04-03",
            output_path=str(out),
        )
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "No signals" in content or "No signal data" in content

    def test_missing_journal_produces_valid_html(self, tmp_path):
        """A missing signals.jsonl should not raise; it produces an empty report."""
        out = tmp_path / "report.html"
        vl.generate_dashboard(
            journal_path=str(tmp_path / "nonexistent.jsonl"),
            plans_dir=str(tmp_path),
            filter_date="2026-04-03",
            output_path=str(out),
        )
        assert out.exists()

    def test_output_path_creates_parent_dirs(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        _write_signals(jfile, [_make_signal(ts="2026-04-03T14:00:00")])
        nested_out = tmp_path / "a" / "b" / "report.html"
        vl.generate_dashboard(
            journal_path=str(jfile),
            plans_dir=str(tmp_path),
            filter_date="2026-04-03",
            output_path=str(nested_out),
        )
        assert nested_out.exists()

    def test_ticker_filter_reflected_in_subtitle(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        _write_signals(jfile, [_make_signal(ticker="SPY", ts="2026-04-03T14:00:00")])
        out = tmp_path / "report.html"
        vl.generate_dashboard(
            journal_path=str(jfile),
            plans_dir=str(tmp_path),
            tickers=["SPY"],
            filter_date="2026-04-03",
            output_path=str(out),
        )
        content = out.read_text(encoding="utf-8")
        assert "SPY" in content

    def test_stats_bar_shows_trade_count(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        _write_signals(jfile, [
            _make_signal(action="dry_run", ts="2026-04-03T10:00:00"),
            _make_signal(action="dry_run", ts="2026-04-03T10:05:00"),
            _make_signal(action="skip",    ts="2026-04-03T10:10:00"),
        ])
        out = tmp_path / "report.html"
        vl.generate_dashboard(
            journal_path=str(jfile),
            plans_dir=str(tmp_path),
            filter_date="2026-04-03",
            output_path=str(out),
        )
        content = out.read_text(encoding="utf-8")
        # 2 executed trades → "2" should appear in the stats bar
        assert ">2<" in content or "val\">2" in content or ">2</span>" in content

    def test_returns_absolute_path(self, tmp_path):
        jfile = tmp_path / "signals.jsonl"
        jfile.write_text("")
        out = tmp_path / "report.html"
        result = vl.generate_dashboard(
            journal_path=str(jfile),
            plans_dir=str(tmp_path),
            filter_date="2026-04-03",
            output_path=str(out),
        )
        assert os.path.isabs(result)

    def test_plans_dir_integrated_into_safety_chart(self, tmp_path):
        """Approved trade plan should appear as a short_strike shape in the HTML."""
        jfile = tmp_path / "signals.jsonl"
        _write_signals(jfile, [
            _make_signal(ticker="SPY", price=500.0, ts="2026-04-03T10:00:00"),
        ])
        plan_file = tmp_path / "trade_plan_SPY.json"
        _make_trade_plan_file(
            plan_file, ticker="SPY",
            entries=[_make_history_entry(ticker="SPY", short_strike=480.0, approved=True)]
        )
        out = tmp_path / "report.html"
        vl.generate_dashboard(
            journal_path=str(jfile),
            plans_dir=str(tmp_path),
            filter_date="2026-04-03",
            output_path=str(out),
        )
        content = out.read_text(encoding="utf-8")
        # Short strike annotation text should appear somewhere in the Plotly JSON
        assert "480" in content


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:

    def test_defaults(self):
        args = vl._parse_args([])
        # Post-May-2026 journal split: live-mode is the default; legacy
        # signals.jsonl is auto-detected as a fallback inside load_signals
        # (see _resolve_journal_path).
        assert args.journal == "trade_journal/signals_live.jsonl"
        assert args.journal == vl.DEFAULT_JOURNAL
        assert args.plans_dir == "trade_plans"
        assert args.tickers is None
        assert args.date is None
        assert args.output == "daily_report.html"
        assert args.all_dates is False

    def test_custom_args(self):
        args = vl._parse_args([
            "--journal", "/tmp/signals.jsonl",
            "--plans-dir", "/tmp/plans",
            "--tickers", "SPY", "QQQ",
            "--date", "2026-04-03",
            "--output", "/tmp/report.html",
        ])
        assert args.journal == "/tmp/signals.jsonl"
        assert args.plans_dir == "/tmp/plans"
        assert args.tickers == ["SPY", "QQQ"]
        assert args.date == "2026-04-03"
        assert args.output == "/tmp/report.html"

    def test_all_dates_flag(self):
        args = vl._parse_args(["--all-dates"])
        assert args.all_dates is True


# ---------------------------------------------------------------------------
# Internal helper — not part of the module's public API
# ---------------------------------------------------------------------------


def _signals_from_records(records: list) -> pd.DataFrame:
    """Build a signals DataFrame directly from raw record dicts (no file I/O)."""
    import io
    lines = "\n".join(json.dumps(r) for r in records)
    tmp = Path("/tmp/_test_signals_tmp.jsonl")
    tmp.write_text(lines)
    try:
        return vl.load_signals(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
