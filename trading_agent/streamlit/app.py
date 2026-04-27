"""
app.py — Streamlit dashboard entry point.

Run with:
    streamlit run trading_agent/streamlit/app.py

Logging
-------
We initialise logging at import time so every ``logger.info(...)`` call
inside the agent / backtester reaches the terminal Streamlit was launched
from.  Without this, Python's root logger defaults to WARNING level and
INFO-level diagnostics (rate-limiter sleeps, per-ticker progress, strike
selection, etc.) are silently dropped — which made long backtests
indistinguishable from genuine hangs.

The level can be overridden via the ``LOG_LEVEL`` env var:

    LOG_LEVEL=DEBUG streamlit run trading_agent/streamlit/app.py
"""

import os
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `trading_agent` is importable
# regardless of how Streamlit sets the working directory.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# IMPORTANT: setup_logging() must run BEFORE any other trading_agent module
# is imported.  Sub-modules grab their loggers at import time via
# ``logging.getLogger(__name__)``; if the root logger isn't configured yet
# those loggers inherit Python's default WARNING level and INFO calls
# from the backtester are silently dropped.
from trading_agent.logger_setup import setup_logging  # noqa: E402

setup_logging(log_level=os.environ.get("LOG_LEVEL", "INFO"))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Trading Agent Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Trading Agent Dashboard")
st.caption("Credit-spread options agent · Paper trading · Alpaca Markets")

tab_live, tab_backtest, tab_llm = st.tabs(
    ["📡 Live Monitoring", "📊 Backtesting", "🤖 LLM Extension"]
)

with tab_live:
    from trading_agent.streamlit.live_monitor import render_live_monitor
    render_live_monitor()

with tab_backtest:
    from trading_agent.streamlit.backtest_ui import render_backtest_ui
    render_backtest_ui()

with tab_llm:
    from trading_agent.streamlit.llm_extension import render_llm_extension
    render_llm_extension()
