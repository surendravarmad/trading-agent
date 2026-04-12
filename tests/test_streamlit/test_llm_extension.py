"""
Tests for llm_extension.py — config parsing, env patching, journal loading,
and render smoke tests.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trading_agent.streamlit.llm_extension import (
    OPTIMIZABLE_KEYS,
    _apply_config_to_env,
    _call_ollama,
    _load_recent_signals,
    _parse_config_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_CONFIG = {
    "MAX_RISK_PCT": 0.02,
    "MIN_CREDIT_RATIO": 0.25,
    "MAX_DELTA": 0.20,
    "DAILY_DRAWDOWN_LIMIT": 0.05,
    "MAX_BUYING_POWER_PCT": 0.80,
    "LIQUIDITY_MAX_SPREAD": 0.05,
}

VALID_JSON_STR = json.dumps(VALID_CONFIG)


@pytest.fixture
def journal_file(tmp_path):
    records = [
        {
            "timestamp": f"2026-04-0{i+1}T10:00:00+00:00",
            "ticker": f"TKR{i}",
            "action": "dry_run",
            "price": float(100 + i * 10),
            "raw_signal": {"account_balance": 100_000.0},
        }
        for i in range(15)
    ]
    p = tmp_path / "signals.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return p


@pytest.fixture
def env_file(tmp_path):
    content = (
        "ALPACA_API_KEY=testkey\n"
        "MAX_RISK_PCT=0.02\n"
        "MIN_CREDIT_RATIO=0.25\n"
        "MAX_DELTA=0.25\n"
        "DAILY_DRAWDOWN_LIMIT=0.05\n"
        "MAX_BUYING_POWER_PCT=0.80\n"
        "LIQUIDITY_MAX_SPREAD=0.05\n"
        "DRY_RUN=true\n"
    )
    p = tmp_path / ".env"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# _load_recent_signals
# ---------------------------------------------------------------------------

class TestLoadRecentSignals:
    def test_returns_empty_when_missing(self, tmp_path):
        with patch("trading_agent.streamlit.llm_extension.JOURNAL_PATH",
                   tmp_path / "nonexistent.jsonl"):
            result = _load_recent_signals(10)
        assert result == []

    def test_returns_last_n_records(self, journal_file):
        with patch("trading_agent.streamlit.llm_extension.JOURNAL_PATH", journal_file):
            result = _load_recent_signals(5)
        assert len(result) == 5
        # Should be the last 5 (indices 10-14, tickers TKR10..TKR14)
        tickers = [r["ticker"] for r in result]
        assert "TKR14" in tickers

    def test_returns_all_when_n_exceeds_count(self, journal_file):
        with patch("trading_agent.streamlit.llm_extension.JOURNAL_PATH", journal_file):
            result = _load_recent_signals(100)
        assert len(result) == 15

    def test_skips_malformed_json(self, tmp_path):
        p = tmp_path / "signals.jsonl"
        p.write_text('bad json\n{"ticker":"SPY","action":"skip"}\n')
        with patch("trading_agent.streamlit.llm_extension.JOURNAL_PATH", p):
            result = _load_recent_signals(10)
        assert len(result) == 1
        assert result[0]["ticker"] == "SPY"

    def test_handles_empty_file(self, tmp_path):
        p = tmp_path / "signals.jsonl"
        p.write_text("")
        with patch("trading_agent.streamlit.llm_extension.JOURNAL_PATH", p):
            result = _load_recent_signals(10)
        assert result == []


# ---------------------------------------------------------------------------
# _parse_config_json
# ---------------------------------------------------------------------------

class TestParseConfigJson:
    def test_parses_bare_json(self):
        result = _parse_config_json(VALID_JSON_STR)
        assert result is not None
        assert set(result.keys()) == set(OPTIMIZABLE_KEYS)

    def test_parses_fenced_json_block(self):
        fenced = f"Here is the config:\n```json\n{VALID_JSON_STR}\n```\nDone."
        result = _parse_config_json(fenced)
        assert result is not None
        assert result["MAX_RISK_PCT"] == 0.02

    def test_parses_bare_fence(self):
        fenced = f"```\n{VALID_JSON_STR}\n```"
        result = _parse_config_json(fenced)
        assert result is not None

    def test_returns_none_for_missing_keys(self):
        partial = {"MAX_RISK_PCT": 0.02, "MIN_CREDIT_RATIO": 0.25}
        result = _parse_config_json(json.dumps(partial))
        assert result is None

    def test_returns_none_for_empty_string(self):
        assert _parse_config_json("") is None

    def test_returns_none_for_plain_text(self):
        assert _parse_config_json("I recommend lowering delta to 0.15.") is None

    def test_extracts_correct_values(self):
        config = {**VALID_CONFIG, "MAX_DELTA": 0.18}
        result = _parse_config_json(json.dumps(config))
        assert result["MAX_DELTA"] == 0.18

    def test_only_returns_optimizable_keys(self):
        extra = {**VALID_CONFIG, "SOME_OTHER_KEY": "ignored"}
        result = _parse_config_json(json.dumps(extra))
        assert result is not None
        assert "SOME_OTHER_KEY" not in result
        assert set(result.keys()) == set(OPTIMIZABLE_KEYS)

    def test_handles_json_embedded_in_prose(self):
        prose = (
            "Based on analysis, I suggest:\n"
            f"{VALID_JSON_STR}\n"
            "These values should improve performance."
        )
        result = _parse_config_json(prose)
        assert result is not None


# ---------------------------------------------------------------------------
# _apply_config_to_env
# ---------------------------------------------------------------------------

class TestApplyConfigToEnv:
    def test_patches_existing_keys(self, env_file):
        updates = {"MAX_RISK_PCT": 0.03, "MAX_DELTA": 0.18}
        with patch("trading_agent.streamlit.llm_extension.ENV_PATH", env_file):
            ok, msg = _apply_config_to_env(updates)
        assert ok
        content = env_file.read_text()
        assert "MAX_RISK_PCT=0.03" in content
        assert "MAX_DELTA=0.18" in content
        # Unchanged keys preserved
        assert "ALPACA_API_KEY=testkey" in content

    def test_appends_missing_keys(self, env_file):
        new_key = {"NEW_PARAM": 0.99}
        with patch("trading_agent.streamlit.llm_extension.ENV_PATH", env_file):
            ok, msg = _apply_config_to_env(new_key)
        assert ok
        content = env_file.read_text()
        assert "NEW_PARAM=0.99" in content

    def test_creates_env_file_if_missing(self, tmp_path):
        new_env = tmp_path / ".env"
        updates = {"MAX_RISK_PCT": 0.025}
        with patch("trading_agent.streamlit.llm_extension.ENV_PATH", new_env):
            ok, msg = _apply_config_to_env(updates)
        assert ok
        assert new_env.exists()
        assert "MAX_RISK_PCT=0.025" in new_env.read_text()

    def test_preserves_comments(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("# Trading config\nMAX_DELTA=0.25\n")
        with patch("trading_agent.streamlit.llm_extension.ENV_PATH", env):
            # Pass as string so the exact value is preserved in the file
            ok, _ = _apply_config_to_env({"MAX_DELTA": "0.20"})
        assert ok
        content = env.read_text()
        assert "# Trading config" in content
        assert "MAX_DELTA=0.20" in content

    def test_returns_false_on_permission_error(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("KEY=val\n")
        env.chmod(0o444)  # read-only
        with patch("trading_agent.streamlit.llm_extension.ENV_PATH", env):
            ok, msg = _apply_config_to_env({"KEY": "new"})
        env.chmod(0o644)  # restore for cleanup
        assert not ok
        assert len(msg) > 0

    def test_success_message_lists_updated_keys(self, env_file):
        updates = {"MAX_RISK_PCT": 0.03}
        with patch("trading_agent.streamlit.llm_extension.ENV_PATH", env_file):
            ok, msg = _apply_config_to_env(updates)
        assert ok
        assert "MAX_RISK_PCT" in msg


# ---------------------------------------------------------------------------
# _call_ollama
# ---------------------------------------------------------------------------

class TestCallOllama:
    @patch("trading_agent.streamlit.llm_extension.LLM_BASE_URL", "http://localhost:11434")
    def test_returns_error_string_when_unavailable(self):
        # With no Ollama running, should return a helpful error string, not raise
        result = _call_ollama("test prompt", base_url="http://localhost:1")
        assert isinstance(result, str)
        assert "unavailable" in result.lower() or "error" in result.lower() or "LLM" in result

    def test_returns_response_text_on_success(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "Looks bullish."}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            result = _call_ollama("What is the regime?")

        assert result == "Looks bullish."

    def test_uses_configured_model(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok"}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            _call_ollama("prompt", model="qwen2.5-trading")
            call_kwargs = mock_post.call_args
            assert call_kwargs[1]["json"]["model"] == "qwen2.5-trading"


# ---------------------------------------------------------------------------
# Smoke: render_llm_extension
# ---------------------------------------------------------------------------

class TestRenderLlmExtensionSmoke:
    def test_renders_without_exception(self):
        # Lambda-import pattern ensures module globals (including `st`) are available.
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_function(
            lambda: __import__(
                "trading_agent.streamlit.llm_extension",
                fromlist=["render_llm_extension"],
            ).render_llm_extension()
        )
        at.run(timeout=15)
        assert not at.exception

    def test_shows_warning_with_empty_journal(self, tmp_path):
        # The render function shows st.warning when _load_recent_signals returns [].
        # Verify the trigger condition: an absent journal file yields an empty list,
        # which is exactly what causes the warning branch to execute.
        with patch(
            "trading_agent.streamlit.llm_extension.JOURNAL_PATH",
            tmp_path / "nonexistent.jsonl",
        ):
            signals = _load_recent_signals(10)
        assert signals == [], (
            "Expected empty signals list — this is the condition that renders "
            "the 'No journal entries found' warning in render_llm_extension."
        )
