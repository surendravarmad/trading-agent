"""
llm_extension.py — LLM Extension tab.

Provides a Streamlit chat interface backed by a local Ollama instance,
pre-built analysis buttons, and a one-click strategy config optimizer
that patches .env with LLM-suggested parameter values.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

JOURNAL_PATH = Path("trade_journal/signals.jsonl")
ENV_PATH = Path(".env")

# Read from environment so tests can override
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")

OPTIMIZABLE_KEYS = [
    "MAX_RISK_PCT",
    "MIN_CREDIT_RATIO",
    "MAX_DELTA",
    "DAILY_DRAWDOWN_LIMIT",
    "MAX_BUYING_POWER_PCT",
    "LIQUIDITY_MAX_SPREAD",
]

# ── Preset prompt templates ─────────────────────────────────────────────────

PRESET_PROMPTS: Dict[str, str] = {
    "Analyze Last 10 Trades + Guardrail Suggestions": (
        "You are a quantitative trading analyst. Analyze the following 10 most recent "
        "credit-spread trade signals and provide:\n"
        "1. A concise performance summary (win rate, common failure modes).\n"
        "2. Specific guardrail parameter recommendations (max_delta, min_credit_ratio, "
        "max_risk_pct, daily_drawdown_limit) with numeric values.\n"
        "3. One-sentence rationale for each recommendation.\n\n"
        "Trade signals:\n{trades_json}"
    ),
    "Generate Next-Week Trade Plan": (
        "Based on the following recent trade signals, generate a prioritized trade plan "
        "for next week. Include:\n"
        "1. Top 3 tickers ranked by expected edge.\n"
        "2. Recommended strategy per ticker (Bull Put / Bear Call / Iron Condor) "
        "and key price levels.\n"
        "3. Risk management notes specific to current regime.\n\n"
        "Recent signals:\n{trades_json}"
    ),
    "VIX +20% What-If Analysis": (
        "Assume VIX spikes +20% from its current level. Analyze the impact on this "
        "options spread trading system:\n"
        "1. Which open positions are most at risk (reference tickers/strategies below).\n"
        "2. How regime classification would shift.\n"
        "3. Which of the 8 risk guardrails would trip first.\n"
        "4. Three recommended defensive actions in priority order.\n\n"
        "Recent signals (includes open positions context):\n{trades_json}"
    ),
}

OPTIMIZE_PROMPT = (
    "You are a trading system optimizer. Based on the following recent trade signals, "
    "suggest improved risk parameter values that maximize win rate while preserving "
    "capital. Return ONLY a valid JSON object with exactly these keys:\n"
    "{\n"
    '  "MAX_RISK_PCT": <float 0.01-0.05>,\n'
    '  "MIN_CREDIT_RATIO": <float 0.20-0.40>,\n'
    '  "MAX_DELTA": <float 0.15-0.30>,\n'
    '  "DAILY_DRAWDOWN_LIMIT": <float 0.03-0.08>,\n'
    '  "MAX_BUYING_POWER_PCT": <float 0.50-0.90>,\n'
    '  "LIQUIDITY_MAX_SPREAD": <float 0.02-0.10>\n'
    "}\n"
    "Return only the JSON object — no explanation, no markdown fences.\n\n"
    "Recent trade data:\n{trades_json}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_recent_signals(n: int = 10) -> List[Dict]:
    """Return the last n records from signals.jsonl."""
    if not JOURNAL_PATH.exists():
        return []
    records: List[Dict] = []
    with open(JOURNAL_PATH) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records[-n:]


def _call_ollama(prompt: str, base_url: str = LLM_BASE_URL, model: str = LLM_MODEL) -> str:
    """POST to local Ollama /api/generate and return the response text."""
    try:
        import requests  # already in requirements via alpaca-py's transitive dep
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json().get("response", "No response from model.")
    except Exception as exc:
        return f"LLM unavailable ({exc}). Is Ollama running at {base_url}?"


def _parse_config_json(response: str) -> Optional[Dict]:
    """
    Extract a valid config JSON object from an LLM response string.

    Tries three patterns: bare object, fenced ```json block, bare ``` block.
    Validates that the parsed object contains all OPTIMIZABLE_KEYS.
    """
    candidate_texts = [response]

    # Also extract fenced blocks
    for fence_re in (r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"):
        for match in re.finditer(fence_re, response, re.DOTALL):
            candidate_texts.append(match.group(1))

    # Also try every {...} blob in the text
    for match in re.finditer(r"\{[^{}]*\}", response, re.DOTALL):
        candidate_texts.append(match.group(0))

    required = set(OPTIMIZABLE_KEYS)
    for text in candidate_texts:
        text = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and required.issubset(parsed.keys()):
                return {k: parsed[k] for k in OPTIMIZABLE_KEYS}
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _apply_config_to_env(updates: Dict) -> Tuple[bool, str]:
    """
    Patch .env file with the given key=value pairs.
    Overwrites existing keys; appends missing ones.
    """
    try:
        lines: List[str] = []
        if ENV_PATH.exists():
            lines = ENV_PATH.read_text().splitlines(keepends=True)

        patched_keys: set = set()
        new_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                new_lines.append(line)
                continue
            if "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in updates:
                    new_lines.append(f"{key}={updates[key]}\n")
                    patched_keys.add(key)
                    continue
            new_lines.append(line)

        for key, value in updates.items():
            if key not in patched_keys:
                new_lines.append(f"{key}={value}\n")

        ENV_PATH.write_text("".join(new_lines))
        return True, f"Updated {len(updates)} keys: {', '.join(updates.keys())}"
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_llm_extension() -> None:
    """Render the LLM Extension tab."""
    st.subheader("LLM Trade Analyst")

    # Init session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_config" not in st.session_state:
        st.session_state.pending_config = None

    recent_signals = _load_recent_signals(10)
    trades_json = json.dumps(recent_signals, indent=2, default=str)

    if not recent_signals:
        st.warning(
            "No journal entries found at trade_journal/signals.jsonl. "
            "Run the agent at least once to populate the journal."
        )

    # ── Model info banner ───────────────────────────────────────────────────
    st.caption(f"Model: **{LLM_MODEL}** · Endpoint: `{LLM_BASE_URL}`")
    st.divider()

    # ── Pre-built analysis buttons ─────────────────────────────────────────
    st.subheader("Quick Analysis")
    btn_cols = st.columns(3)
    for col, (label, template) in zip(btn_cols, PRESET_PROMPTS.items()):
        with col:
            if st.button(label, use_container_width=True):
                prompt = template.format(trades_json=trades_json)
                with st.spinner("Analyzing…"):
                    response = _call_ollama(prompt)
                st.session_state.chat_history.append({"role": "user", "content": label})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()

    st.divider()

    # ── Strategy optimizer ──────────────────────────────────────────────────
    st.subheader("Strategy Optimizer")
    opt_col, apply_col = st.columns([3, 2])

    with opt_col:
        if st.button("Optimize Strategy Config", use_container_width=True):
            prompt = OPTIMIZE_PROMPT.format(trades_json=trades_json)
            with st.spinner("Generating optimized parameters…"):
                response = _call_ollama(prompt)
            parsed = _parse_config_json(response)
            if parsed:
                st.session_state.pending_config = parsed
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": f"Suggested config:\n```json\n{json.dumps(parsed, indent=2)}\n```",
                    }
                )
            else:
                st.warning("Could not extract valid JSON config from LLM response.")
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
            st.rerun()

    if st.session_state.pending_config:
        with apply_col:
            st.json(st.session_state.pending_config)
        col_save, col_discard = st.columns(2)
        with col_save:
            if st.button("Apply to .env", type="primary", use_container_width=True):
                ok, msg = _apply_config_to_env(st.session_state.pending_config)
                if ok:
                    st.success(f"Saved — {msg}")
                    st.session_state.pending_config = None
                else:
                    st.error(f"Save failed: {msg}")
        with col_discard:
            if st.button("Discard", use_container_width=True):
                st.session_state.pending_config = None
                st.rerun()

    st.divider()

    # ── Chat interface ──────────────────────────────────────────────────────
    st.subheader("Chat with Analyst")

    # Render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box
    user_input = st.chat_input(
        "Ask about your trades, strategy, or market conditions…"
    )
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        context = (
            f"You are a trading analyst assistant for a credit-spread options agent. "
            f"The agent runs 5-minute cycles trading Bull Put, Bear Call, and Iron Condor "
            f"spreads on US equities. Here are the {len(recent_signals)} most recent "
            f"trade signals for context:\n{trades_json}\n\nUser question: {user_input}"
        )
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                response = _call_ollama(context)
            st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Clear button
    if st.session_state.chat_history:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
