"""
app.py — Streamlit dashboard entry point.

Run with:
    streamlit run trading_agent/streamlit/app.py
"""

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `trading_agent` is importable
# regardless of how Streamlit sets the working directory.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import streamlit as st

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
