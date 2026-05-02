"""
backtest_ui.py — Backtesting tab.

Provides a Backtester class that simulates credit-spread P&L on
historical prices, then renders the results with metrics, charts,
a sortable trade log, and CSV/JSON/Journal export.

Agent-parity contract
---------------------
The backtester is intentionally a *subset* of the live-agent decision
path, but every entry filter and exit rule it implements MUST use the
same defaults and same comparison semantics as the corresponding agent
module.  The spine of parity is:

   SPREAD_WIDTH, TARGET_DTE   ← imported from strategy.StrategyPlanner
   min_credit_ratio = 0.33    ← strategy.StrategyPlanner default
   max_delta        = 0.20    ← strategy.StrategyPlanner default (≈80% POP)
   max_risk_pct     = 0.02    ← risk_manager.RiskManager default
   hard_stop × 3.0 OR stop_loss_pct × 0.50  ← position_monitor.ExitMonitor
   EarningsCalendar lookahead = 7 days      ← sentiment_pipeline Tier-0
   auto_adjust=False                         ← market_data.MarketDataProvider

Residual drift (documented, not yet closed):
  • Regime classification still uses home-grown SMA (drift #1 / #14).
  • Sentiment / LLM gate absent — backtest skips the Tier-2 chain (drift #11).
  • ETF macro signals (leadership Z + VIX gate, Items 1-3) are CLOSED:
    use_macro_signals=True wires the same gates the live agent uses.
    Residual: bar timescale (live=5min, backtest=whatever timeframe
    selects); open-bar-skip not re-implemented per session day.

Live Quote Refresh — gating contract
------------------------------------
The Phase VI ``_refresh_live_quotes`` stage is intentionally a no-op
in two scenarios.  Both gates live in ``run()`` (search for
``refresh_eligible``) — DO NOT bypass them without updating
``tests/test_streamlit/test_backtest_ui.py::TestRefreshGating``:

  1. ``use_alpaca_historical=True`` → the historical plan IS the
     truthful quote for that bar; refreshing against today's snapshot
     overwrites honest economics with a stale-vs-actual quote.
  2. ``(today - entry_dt) > _SNAPSHOT_FRESH_DAYS`` → today's snapshot
     is structurally meaningless as a proxy for an old entry's quote.

Toggles are **opt-in** on the Backtester class (defaults = None/False so
existing unit tests keep passing) and default-**on** in the Streamlit UI
so interactive runs are apples-to-apples with the live agent.
"""

import collections
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from scipy.stats import percentileofscore

from trading_agent.config import load_config
from trading_agent.streamlit.components import (
    drawdown_chart,
    equity_curve_chart,
    regime_bar_chart,
)

# --- Shared agent constants -------------------------------------------------
# Import the live-agent defaults rather than re-declaring them so a future
# change in strategy.py / risk_manager.py / position_monitor.py flows straight
# into the backtester.  If any of these imports ever fail (e.g. the module
# rename), we'd rather crash loudly than silently drift.
from trading_agent.strategy import StrategyPlanner as _Planner
from trading_agent.regime import (
    LEADERSHIP_ANCHORS,
    VIX_INHIBIT_ZSCORE,
)
from trading_agent.market_data import MarketDataProvider as _MDP

# Z-score thresholds + window — sourced from the live agent so changes in
# strategy.py / market_data.py automatically flow through to the backtester.
RS_ZSCORE_THRESHOLD = _Planner.RS_ZSCORE_THRESHOLD
LEADERSHIP_WINDOW_BARS = _MDP.LEADERSHIP_WINDOW_BARS
VIX_WINDOW_BARS = _MDP.VIX_WINDOW_BARS

logger = logging.getLogger(__name__)

# Option chain cache TTL (seconds) - matches market_data.py
OPTION_CHAIN_TTL = 180  # 3 minutes

STARTING_EQUITY = 100_000.0
SPREAD_WIDTH = _Planner.SPREAD_WIDTH                     # 5.0
COMMISSION_ROUND_TRIP = 2.60  # 4 legs × $0.65

# Agent-parity defaults (read via _Planner / hand-coded where the agent uses
# runtime config instead of a class constant):
AGENT_MIN_CREDIT_RATIO = 0.33       # strategy.StrategyPlanner(__init__ default)
AGENT_MAX_DELTA        = 0.20       # strategy.StrategyPlanner(__init__ default)
AGENT_MAX_RISK_PCT     = 0.02       # risk_manager.RiskManager(__init__ default)
AGENT_HARD_STOP_MULT   = 3.0        # position_monitor.ExitMonitor default
AGENT_STOP_LOSS_PCT    = 0.50       # position_monitor.ExitMonitor default
AGENT_PROFIT_TARGET    = 0.50       # position_monitor.ExitMonitor default
AGENT_HIGH_IV_THRESH   = 95.0       # regime.RegimeClassifier._compute_iv_rank
AGENT_EARNINGS_LOOKAHEAD = 7        # IntelligenceConfig default

DEFAULT_TICKERS = ["SPY", "QQQ", "IWM"]
DEFAULT_START = date.today() - timedelta(days=365)   # needs 200+ bars for SMA warmup
DEFAULT_END = date.today() - timedelta(days=1)

# Yahoo Finance hard limits for intraday data
INTRADAY_MAX_DAYS = 29          # 5m data only available for last ~30 days
INTRADAY_WARMUP_BARS = 20       # 20 × 5-min bars ≈ 1.5 hours warmup for intraday SMA
INTRADAY_HOLD_BARS = 12         # 12 × 5-min bars = 1 hour hold per intraday trade

# OTM % for short strike placement (legacy fixed-% path — kept for backward
# compatibility with test_backtest_ui.py and as a fallback when realized-vol
# cannot be computed).
# Daily: 3% is realistic over a 45-day hold (SPY moves ~1% per day)
# Intraday: 3% in 60 minutes is nearly impossible → use 0.5% so losses actually occur
DAILY_OTM_PCT   = 0.03
INTRADAY_OTM_PCT = 0.005

# --- Sigma-based (delta-proxy) strike placement ----------------------------
# Sigma multiplier defaults approximate standard short-delta targets:
#   1.0σ ≈ 16 Δ   (too aggressive for intraday theta)
#   1.5σ ≈  7 Δ   (balanced intraday default)
#   2.0σ ≈  2 Δ   (very conservative, small credits)
DEFAULT_SIGMA_MULT_DAILY    = 1.0
DEFAULT_SIGMA_MULT_INTRADAY = 1.5

# Bars per year — used to annualize the stdev of log-returns.
BARS_PER_YEAR_DAILY    = 252
BARS_PER_YEAR_INTRADAY = 252 * 78     # 6.5h × 12 five-minute bars per session

# Vol window used to estimate realized σ at entry (uses warmup bars by default).
VOL_WINDOW_DAILY    = 20      # ~1 month of daily bars
VOL_WINDOW_INTRADAY = 20      # same as warmup → ~1.5h

# --- Early loss-cut ---------------------------------------------------------
# On first breach of the short strike, close at ``-LOSS_CUT_MULTIPLIER × credit``
# instead of the full max-loss payoff. This reshapes the loss:win ratio from
# (width − credit)/(credit × profit_target) ≈ 4.9:1 down to ~4.0:1 at k=2,
# dropping the breakeven win rate from ~83% to ~80%.
DEFAULT_LOSS_CUT_MULTIPLIER = 2.0

ALL_TICKERS = [
    "SPY", "QQQ", "IWM", "GOOG", "AAPL",
    "MSFT", "AMZN", "META", "SOFI", "TSLA", "JPM",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SimTrade:
    ticker: str
    strategy: str
    regime: str
    entry_date: date
    expiry_date: date
    credit: float
    max_loss: float
    outcome: str = ""   # "win" | "loss"
    pnl: float = 0.0
    hold_days: int = 0


@dataclass
class RejectionRecord:
    """Per-candidate diagnostic capture when a gate rejects an entry.

    The goal is to let the user answer "why was *that specific bar* on *that
    specific ticker* rejected?" without re-running the backtest — the same
    question you'd ask of a live-agent log line.  We cap how many records we
    keep per gate (``REJECTION_SAMPLE_CAP`` below) so long backtests don't
    explode memory.
    """
    ticker: str
    entry_date: date
    gate: str              # rejection reason token ("iv_rank>threshold", etc.)
    phase: str             # pipeline phase label ("2. Event-risk gate", …)
    price: float
    regime: str
    strategy: str
    # Gate-specific measurement + threshold (both floats when meaningful,
    # else None).  Example for iv_rank: measured=87.4, threshold=95.0
    measured: Optional[float] = None
    threshold: Optional[float] = None
    # Free-form one-line explanation
    reason: str = ""


# Cap number of rejection records kept per gate.  Aggregate counts are
# *always* exact; only the detailed records are subsampled.
REJECTION_SAMPLE_CAP = 50


@dataclass
class PhaseFunnel:
    """
    Candidate-survival funnel, phase-by-phase, mirroring the live agent's
    decision path.  Each entry is a ``(phase_label, surviving_count)`` pair.
    """
    considered: int = 0
    after_earnings: int = 0
    after_iv_rank: int = 0
    after_max_delta: int = 0
    after_credit_ratio: int = 0
    after_max_risk: int = 0
    simulated: int = 0
    # Phase → gate skipped (e.g. "credit_ratio" skipped under σ-path).
    skipped_phases: List[str] = field(default_factory=list)

    def as_rows(self) -> List[Dict]:
        """Serialize as ordered rows for the UI funnel table."""
        rows = [
            ("1. Candidates considered",     self.considered),
            ("2. After earnings gate",       self.after_earnings),
            ("3. After IV-rank gate",        self.after_iv_rank),
            ("4. After max-Δ gate",          self.after_max_delta),
            ("5. After credit-ratio gate",   self.after_credit_ratio),
            ("6. After max-risk gate",       self.after_max_risk),
            ("7. Trades simulated",          self.simulated),
        ]
        out = []
        prev = None
        for label, count in rows:
            dropped = None if prev is None else max(0, prev - count)
            out.append({"phase": label, "surviving": count, "dropped": dropped})
            prev = count
        return out


# ---------------------------------------------------------------------------
# Alpaca rate limiter — 60-second sliding-window token bucket
# ---------------------------------------------------------------------------
#
# Alpaca Market Data API caps standard accounts at **200 req/min** (per
# account, shared across endpoints).  Without a limiter, a multi-ticker
# backtest blows the budget in seconds and Alpaca replies with 429.  This
# class enforces a soft budget (default 180/min — 10 % headroom) by
# blocking the caller until the window has a free slot.
#
# Module-level singleton so multiple Backtester instances (e.g. Streamlit
# reruns during a session) share the same budget instead of each
# independently hammering the limit.

# Default soft budget.  The Alpaca docs state standard is 200 rpm;
# override via ALPACA_MAX_RPM env var when on a higher tier.
DEFAULT_ALPACA_MAX_RPM = 180


class AlpacaRateLimiter:
    """
    Sliding-window rate limiter: keeps timestamps of the last ``max_rpm``
    requests inside a 60-second window and blocks ``acquire()`` callers
    until a slot frees.  Thread-safe.
    """

    def __init__(self, max_rpm: int = DEFAULT_ALPACA_MAX_RPM):
        self.max_rpm = max(1, int(max_rpm))
        self._window: Deque[float] = collections.deque()
        self._lock = threading.Lock()
        # Diagnostics counters (read by the UI expander).
        self.sleep_count = 0
        self.total_sleep_s = 0.0
        self.requests_total = 0

    def set_budget(self, max_rpm: int) -> None:
        with self._lock:
            self.max_rpm = max(1, int(max_rpm))

    def acquire(self) -> None:
        """Block until a request slot is available in the 60-s window."""
        while True:
            with self._lock:
                now = time.monotonic()
                # Drop timestamps older than 60 s.
                while self._window and now - self._window[0] >= 60.0:
                    self._window.popleft()
                if len(self._window) < self.max_rpm:
                    self._window.append(now)
                    self.requests_total += 1
                    return
                # Compute the wait needed for the oldest slot to age out.
                wait = 60.0 - (now - self._window[0]) + 0.05
            self.sleep_count += 1
            self.total_sleep_s += wait
            logger.info(
                "Alpaca rate limiter: sleeping %.2fs (budget %d/min full)",
                wait, self.max_rpm,
            )
            time.sleep(max(0.05, wait))

    def stats(self) -> Dict[str, float]:
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "sleep_count": self.sleep_count,
                "total_sleep_s": round(self.total_sleep_s, 2),
                "budget_rpm": self.max_rpm,
                "in_window": len(self._window),
            }


# Module-level singleton.  Env override: ALPACA_MAX_RPM=9000 for Algo
# Trader Plus subscribers.
_ALPACA_RATE_LIMITER = AlpacaRateLimiter(
    max_rpm=int(os.getenv("ALPACA_MAX_RPM", str(DEFAULT_ALPACA_MAX_RPM)))
)


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: Dict
    regime_stats: pd.DataFrame
    skipped: List[str] = field(default_factory=list)
    # Decision-path diagnostics (populated by run()); default-empty so
    # existing callers that only read trades/metrics keep working.
    funnel: PhaseFunnel = field(default_factory=PhaseFunnel)
    rejection_counts: Dict[str, int] = field(default_factory=dict)
    rejection_samples: List[RejectionRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Simulate credit-spread strategy on historical OHLCV data.

    Regime classification mirrors the real RegimeClassifier rules:
      price > SMA-200 AND slope > 0  → bullish  → Bull Put Spread
      price < SMA-200 AND slope < 0  → bearish  → Bear Call Spread
      otherwise                      → sideways → Iron Condor

    P&L simulation:
      - Credit = spread_width × credit_pct  (default 30 % of width)
      - Win if short strike is never breached within hold period
      - Short strike placed 3 % OTM for puts/calls, ±3 % band for condors
      - Win: capture profit_target_pct of credit minus commission
      - Loss: pay max_loss (width − credit) minus commission
    """

    def __init__(
        self,
        starting_equity: float = STARTING_EQUITY,
        spread_width: float = SPREAD_WIDTH,
        credit_pct: float = 0.30,
        target_dte: int = _Planner.TARGET_DTE,
        profit_target_pct: float = AGENT_PROFIT_TARGET,
        commission: float = COMMISSION_ROUND_TRIP,
        # --- New (delta-proxy / loss-cut) knobs ---
        # When ``sigma_mult`` is None the legacy fixed-% OTM path is used and
        # ``credit`` stays at ``spread_width × credit_pct``. When it is a
        # positive float, short-strike distance and per-trade credit are both
        # derived from a realized-vol projection (see _sigma_strike_distance
        # and _credit_from_sigma).
        sigma_mult: Optional[float] = None,
        # When ``loss_cut_multiplier`` is None, breaches pay full max-loss.
        # When it is a positive float k, breaches pay only ``-k × credit``.
        loss_cut_multiplier: Optional[float] = None,
        # --- Agent-parity gates (opt-in; default None/False to keep the
        # existing test suite green; the Streamlit UI default-enables them
        # so interactive runs are apples-to-apples with the live agent). ---
        min_credit_ratio: Optional[float] = None,
        max_delta: Optional[float] = None,
        max_risk_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        use_iv_gate: bool = False,
        iv_high_threshold: float = AGENT_HIGH_IV_THRESH,
        use_earnings_gate: bool = False,
        earnings_lookahead_days: int = AGENT_EARNINGS_LOOKAHEAD,
        earnings_calendar=None,            # injectable for tests
        # --- Option chain support (uses config if not explicitly provided) ---
        alpaca_api_key: Optional[str] = None,
        alpaca_secret_key: Optional[str] = None,
        alpaca_data_url: Optional[str] = None,
        # --- Real historical Alpaca option bars mode (30-day parity) ---
        # When True, planning and exit use Alpaca's /options/contracts +
        # historical /options/bars endpoints.  Only honest for dates
        # inside Alpaca's options retention window (~30 days).  When the
        # historical data isn't available for a given bar, that bar is
        # skipped (counted in the funnel) rather than silently falling
        # back to the synthetic σ-credit — the whole point of this mode
        # is apples-to-apples live parity.
        use_alpaca_historical: bool = False,
        # --- ETF macro signals (parity with live agent's Items 1-3) ---
        # When True, run() pre-loads anchor + ^VIX series via yfinance and
        # applies the same z-scored gates the live regime classifier uses:
        #   * leadership_z > RS_ZSCORE_THRESHOLD on a SIDEWAYS bar promotes
        #     to BULLISH (parity with strategy.plan() Priority 3)
        #   * vix_z > VIX_INHIBIT_ZSCORE on a BULLISH/SIDEWAYS bar demotes
        #     to BEARISH (parity with strategy.plan() Priority 2)
        # Default False so the existing test suite stays apples-to-apples
        # with the pre-patch backtester; the Streamlit UI default-enables
        # so interactive runs stay aligned with the live agent.
        #
        # Caveat: the live agent computes these z-scores on rolling 5-min
        # bars.  When the backtest timeframe is "1Day" the same arithmetic
        # is applied to daily return diffs — directionally aligned with
        # the live signal but at a different timescale.  Set timeframe
        # = "5Min" for true parity (subject to yfinance's 30-day limit).
        use_macro_signals: bool = False,
        # --- Unified decision-engine path (Phase 2 of live↔backtest unify) ---
        # When ``use_unified_engine=True`` AND ``preset`` is provided,
        # ``_alpaca_historical_plan`` delegates strike-selection, credit
        # pricing, EV scoring, and the C/W floor gate to
        # ``trading_agent.decision_engine.decide()`` — the *same* call the
        # live ``ChainScanner`` makes. Drift between live and backtest is
        # impossible here by construction: any change to
        # ``_score_candidate_with_reason`` or ``_quote_credit`` in
        # ``chain_scanner.py`` automatically flows through. Defaults are
        # ``preset=None`` / ``use_unified_engine=False`` so the existing
        # test suite stays byte-identical with the legacy σ-path; the
        # Streamlit UI flips the toggle on for interactive runs so they
        # match the live agent's economics.
        preset: Optional[object] = None,
        use_unified_engine: bool = False,
    ) -> None:
        self.starting_equity = starting_equity
        self.spread_width = spread_width
        self.credit_pct = credit_pct
        self.target_dte = target_dte
        self.profit_target_pct = profit_target_pct
        self.commission = commission
        self.sigma_mult = sigma_mult
        self.loss_cut_multiplier = loss_cut_multiplier
        self.min_credit_ratio = min_credit_ratio
        self.max_delta = max_delta
        self.max_risk_pct = max_risk_pct
        self.stop_loss_pct = stop_loss_pct
        self.use_iv_gate = use_iv_gate
        self.iv_high_threshold = iv_high_threshold
        self.use_earnings_gate = use_earnings_gate
        self.earnings_lookahead_days = earnings_lookahead_days
        self._earnings_calendar = earnings_calendar
        # Use config values if not explicitly provided
        self._alpaca_api_key = alpaca_api_key or os.getenv("ALPACA_API_KEY", "")
        self._alpaca_secret_key = alpaca_secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self._alpaca_data_url = alpaca_data_url or os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets/v2")
        self.use_alpaca_historical = bool(use_alpaca_historical)
        self.use_macro_signals = bool(use_macro_signals)
        self.preset = preset
        self.use_unified_engine = bool(use_unified_engine)
        # Last ScanDiagnostics returned by decide() — populated only on the
        # unified path. The Streamlit UI surfaces this in the same diagnostics
        # panel the live monitor uses, so reject-reason taxonomy stays unified.
        self.last_decide_diagnostics = None
        # Option chain cache: {(ticker, expiry, type): (contracts, epoch)}
        self._option_cache: Dict[Tuple[str, str, str], Tuple[List[Dict], float]] = {}
        # Alpaca API call counters surfaced to the diagnostics expander.
        self.alpaca_calls_made = 0
        self.alpaca_429_hits = 0
        self.alpaca_failures = 0
        # Per-run rejection counters (surface to UI as skipped reasons)
        self.rejections: Dict[str, int] = {}
        # Per-run detailed diagnostics — filled by _record_rejection.
        self._rejection_samples: List[RejectionRecord] = []
        self._funnel = PhaseFunnel()
        # ETF macro signal counters — surfaced as informational metrics
        # in the diagnostics panel.  They count *gate firings*, not trades:
        # a vix_inhibited bar still flows through to Bear-Call simulation.
        self.leadership_biased = 0     # SIDEWAYS → BULLISH promotions
        self.vix_inhibited = 0         # BULLISH/SIDEWAYS → BEARISH demotions

    # ── Agent-parity helpers ────────────────────────────────────────────────

    def _bump_rejection(self, reason: str) -> None:
        """Legacy shim — counts only.  Prefer ``_record_rejection`` for new
        call sites so we capture per-candidate context for the UI."""
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def _record_rejection(
        self,
        *,
        ticker: str,
        entry_date: date,
        gate: str,
        phase: str,
        price: float,
        regime: str,
        strategy: str,
        measured: Optional[float] = None,
        threshold: Optional[float] = None,
        reason: str = "",
    ) -> None:
        """Record a candidate rejection with full context.

        Always bumps the aggregate counter; additionally captures a
        ``RejectionRecord`` until ``REJECTION_SAMPLE_CAP`` records have been
        collected *for that gate*.  This keeps memory bounded on multi-year
        backtests while still giving the user enough evidence to understand
        why the gate fired.
        """
        self.rejections[gate] = self.rejections.get(gate, 0) + 1
        # Per-gate sample cap so one chatty gate can't crowd others out
        cap_reached = (
            sum(1 for r in self._rejection_samples if r.gate == gate)
            >= REJECTION_SAMPLE_CAP
        )
        if not cap_reached:
            self._rejection_samples.append(RejectionRecord(
                ticker=ticker,
                entry_date=entry_date,
                gate=gate,
                phase=phase,
                price=float(price),
                regime=regime,
                strategy=strategy,
                measured=measured,
                threshold=threshold,
                reason=reason,
            ))

    @staticmethod
    def _delta_from_sigma_distance(sigma_mult: float) -> float:
        """
        Approximate |Δ| for a short strike placed at ``sigma_mult`` × σ_hold.

        Uses the standard normal-CDF identity: for a strike N standard
        deviations OTM under a log-normal price model, |Δ| ≈ Φ(-N).  This is
        the same heuristic the README (and the σ-mult slider help text) uses
        to map "1.0σ ≈ 16Δ", "1.5σ ≈ 7Δ", "2.0σ ≈ 2Δ".

        Returns 0.0 for non-positive sigma_mult.
        """
        if sigma_mult <= 0.0:
            return 0.0
        from math import erf, sqrt
        # Φ(-x) = 0.5 * (1 - erf(x / √2))
        return 0.5 * (1.0 - erf(sigma_mult / sqrt(2.0)))

    @staticmethod
    def _iv_rank_from_returns(close: pd.Series, idx: int, window: int = 20) -> float:
        """
        Mirror of regime.RegimeClassifier._compute_iv_rank for the backtest.

        Uses ``scipy.stats.percentileofscore`` (kind='mean') over rolling
        20-bar annualised realised vols sampled every 5 bars — this is the
        exact formulation the live agent uses (regime.py:194), so the IV
        rank a trade is gated on at backtest entry equals what the agent
        would have computed at that same bar in production.

        Returns 0.0 when there's not enough history.
        """
        if idx < window + 5:
            return 0.0
        sub = close.iloc[: idx + 1]
        returns = sub.pct_change().dropna()
        if len(returns) < window + 5:
            return 0.0
        current_vol = float(returns.tail(window).std() * np.sqrt(252)) * 100.0
        hist_vols = []
        for i in range(0, len(returns) - window, 5):
            v = float(returns.iloc[i : i + window].std() * np.sqrt(252)) * 100.0
            hist_vols.append(v)
        if not hist_vols:
            return 0.0
        return float(percentileofscore(hist_vols, current_vol, kind="mean"))

    def _earnings_blocks_entry(self, ticker: str, entry_dt: date) -> bool:
        """
        Return True iff ``ticker`` has scheduled earnings within
        ``earnings_lookahead_days`` of ``entry_dt``.

        Uses the live EarningsCalendar module so the same yfinance lookup,
        the same caching policy, and the same lookahead-window semantics as
        the agent's Tier-0 short-circuit (sentiment_pipeline._earnings_short_circuit)
        apply to backtest entries too.
        """
        if not self.use_earnings_gate:
            return False
        if self._earnings_calendar is None:
            try:
                from trading_agent.earnings_calendar import EarningsCalendar
                self._earnings_calendar = EarningsCalendar(
                    enabled=True,
                    lookahead_days=self.earnings_lookahead_days,
                )
            except Exception as exc:
                logger.debug("EarningsCalendar unavailable for backtest (%s)", exc)
                return False
        # Note: EarningsCalendar.has_earnings_within compares against today,
        # not the synthetic backtest entry_dt — but for a 7-day lookahead
        # over a year of bars the per-ticker "is there an earnings event in
        # the near future from the calendar fetch's perspective" gate still
        # captures the dominant effect (no entries inside the issuer's known
        # quiet period at *backtest run time*).  A fully-historical earnings
        # filter would require a second yfinance call per bar, which is too
        # heavy; this is a documented simplification.
        try:
            return bool(self._earnings_calendar.has_earnings_within(
                ticker, days=self.earnings_lookahead_days,
            ))
        except Exception:
            return False

    # ── Volatility & credit helpers (sigma-based strike model) ──────────────

    @staticmethod
    def _realized_vol_annual(
        prices: pd.Series, idx: int, window: int, bars_per_year: int
    ) -> float:
        """
        Annualized stdev of log returns over the last ``window`` bars ending
        at (and including) bar ``idx``.

        Returns 0.0 if there's not enough history or if the window is
        degenerate (flat prices → σ=0).
        """
        start = max(0, idx - window)
        segment = prices.iloc[start: idx + 1]
        if len(segment) < 3:
            return 0.0
        log_ret = np.log(segment / segment.shift(1)).dropna()
        sd = float(log_ret.std())
        if not np.isfinite(sd) or sd <= 0.0:
            return 0.0
        return sd * float(np.sqrt(bars_per_year))

    @staticmethod
    def _sigma_strike_distance(
        sigma_annual: float, hold_bars: int, bars_per_year: int, sigma_mult: float,
    ) -> float:
        """
        Return the fractional distance from entry price to the short strike,
        projecting annualized σ over the hold horizon and scaling by
        ``sigma_mult`` (a σ-count proxy for delta placement).
        """
        if sigma_annual <= 0.0 or bars_per_year <= 0 or hold_bars <= 0:
            return 0.0
        sigma_hold = sigma_annual * float(np.sqrt(hold_bars / bars_per_year))
        return max(0.0, sigma_mult * sigma_hold)

    @staticmethod
    def _credit_from_sigma(
        sigma_mult: float, min_frac: float = 0.05, max_frac: float = 0.45,
    ) -> float:
        """
        Approximate credit as a fraction of width, as a function of strike
        distance in σ.

        Heuristic (roughly matches listed SPY vertical quotes):
            0.5σ → 0.40 × width     (close to ATM, big premium)
            1.0σ → 0.30 × width     (≈ current default 30%)
            1.5σ → 0.225 × width
            2.0σ → 0.15 × width
            2.5σ → 0.075 × width    (clipped to min_frac)

        This prevents the common backtest cheat of moving strikes further
        OTM while holding credit constant (which is free risk reduction).
        """
        return float(np.clip(0.45 - 0.15 * sigma_mult, min_frac, max_frac))

    # ── Option chain helpers (Alpaca API) ──────────────────────────────────
    #
    # IMPORTANT HISTORICAL-DATA CAVEAT
    # --------------------------------
    # Alpaca's ``/v1beta1/options/snapshots/{underlying}`` endpoint and
    # yfinance's ``tk.option_chain(expiry)`` BOTH return the *current*
    # market snapshot — not a historical chain as-of ``entry_date``.
    # That means this helper is only honest for recent entries (roughly
    # today).  For multi-month daily backtests the same snapshot would
    # be applied to every bar, which is a known drift we flag in logs.
    #
    # The proper 30-day parity path is in the alpaca_historical methods
    # below (``_fetch_alpaca_option_contracts`` + ``_fetch_alpaca_option_bars``)
    # which query Alpaca's ``/v2/options/contracts`` + historical
    # ``/v1beta1/options/bars`` endpoints for real as-of-date pricing.

    # Below this date-age threshold the current-snapshot path is a
    # reasonable approximation of "as-of today"; above it, we log loudly
    # so the user knows the credits they're seeing are snapshot-biased.
    _SNAPSHOT_FRESH_DAYS = 3

    def _alpaca_headers(self) -> Dict[str, str]:
        """Return Alpaca API headers."""
        return {
            "APCA-API-KEY-ID": self._alpaca_api_key or "",
            "APCA-API-SECRET-KEY": self._alpaca_secret_key or "",
            "Accept": "application/json",
        }

    def _alpaca_request(
        self,
        url: str,
        params: Optional[Dict] = None,
        timeout: int = 15,
        max_retries: int = 3,
    ) -> Optional["requests.Response"]:
        """
        Rate-limited + retry-aware Alpaca GET.

        • Blocks on the module-level ``_ALPACA_RATE_LIMITER`` so we never
          exceed the 60-second request budget (default 180/min).
        • On HTTP 429, reads ``Retry-After`` (seconds) and re-queues with
          capped exponential backoff.
        • Returns a ``requests.Response`` on success, or ``None`` if all
          retries were exhausted (callers already handle None/raise).
        """
        delay = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            _ALPACA_RATE_LIMITER.acquire()
            self.alpaca_calls_made += 1
            try:
                resp = requests.get(
                    url, headers=self._alpaca_headers(),
                    params=params, timeout=timeout,
                )
            except requests.RequestException as exc:
                last_exc = exc
                if attempt == max_retries:
                    self.alpaca_failures += 1
                    logger.warning(
                        "Alpaca request failed (%s) — giving up after %d attempts",
                        exc, attempt + 1,
                    )
                    return None
                logger.info(
                    "Alpaca request exception (%s) — retrying in %.1fs",
                    exc, delay,
                )
                time.sleep(delay)
                delay = min(30.0, delay * 2)
                continue

            if resp.status_code == 429:
                self.alpaca_429_hits += 1
                retry_after_raw = resp.headers.get("Retry-After")
                try:
                    retry_after = float(retry_after_raw) if retry_after_raw else delay
                except ValueError:
                    retry_after = delay
                retry_after = min(60.0, max(1.0, retry_after))
                logger.warning(
                    "Alpaca 429 Too Many Requests (attempt %d/%d) — "
                    "Retry-After=%.1fs (url=%s)",
                    attempt + 1, max_retries + 1, retry_after, url,
                )
                if attempt == max_retries:
                    self.alpaca_failures += 1
                    return None
                time.sleep(retry_after)
                delay = min(30.0, delay * 2)
                continue

            return resp

        # Unreachable under normal control flow, but keeps linters happy.
        if last_exc is not None:
            logger.warning("Alpaca request gave up after retries: %s", last_exc)
        return None

    def _fetch_option_chain(
        self, underlying: str, expiration_date: str, option_type: str = "put"
    ) -> Optional[List[Dict]]:
        """
        Fetch option chain snapshot from Alpaca (cached OPTION_CHAIN_TTL s).

        Returns a list of option contract dicts with Greeks, or None on
        HTTP failure.  An empty list is NOT cached so a transient 404
        doesn't poison the whole session.
        """
        cache_key = (underlying, expiration_date, option_type)
        now = time.monotonic()
        if cache_key in self._option_cache:
            contracts, cached_at = self._option_cache[cache_key]
            if (now - cached_at) < OPTION_CHAIN_TTL:
                logger.debug(
                    "[%s] Option chain cache HIT (%s %s)",
                    underlying, option_type, expiration_date
                )
                return contracts

        # Data URL is https://data.alpaca.markets/v2 by env default; the
        # options endpoints live under /v1beta1/options, so trim the /v2
        # suffix to avoid doubled version path segments.
        data_base = (self._alpaca_data_url or "").rstrip("/")
        if data_base.endswith("/v2"):
            data_base = data_base[: -len("/v2")]
        url = f"{data_base}/v1beta1/options/snapshots/{underlying}"
        params = {
            "type": option_type,
            "expiration_date": expiration_date,
            # Alpaca default is 100; raising to 1000 avoids truncating
            # wide SPY/QQQ chains that would otherwise miss OTM strikes.
            "limit": 1000,
        }
        logger.info(
            "Fetching %s option chain for %s exp %s",
            option_type, underlying, expiration_date
        )
        try:
            resp = self._alpaca_request(url, params=params, timeout=15)
            if resp is None:
                return None
            resp.raise_for_status()
            data = resp.json()
            snapshots = data.get("snapshots", {}) or {}
            contracts: List[Dict] = []
            for symbol, snap in snapshots.items():
                greeks = snap.get("greeks") or {}
                quote = snap.get("latestQuote") or {}
                bid = float(quote.get("bp", 0) or 0)
                ask = float(quote.get("ap", 0) or 0)
                contracts.append({
                    "symbol": symbol,
                    "bid": bid,
                    "ask": ask,
                    "mid": round((bid + ask) / 2, 4),
                    "delta": float(greeks.get("delta", 0) or 0),
                    "theta": float(greeks.get("theta", 0) or 0),
                    "vega": float(greeks.get("vega", 0) or 0),
                    "gamma": float(greeks.get("gamma", 0) or 0),
                    "iv": float(greeks.get("impliedVolatility", 0) or 0),
                    "strike": self._extract_strike(symbol),
                    "expiration": expiration_date,
                    "type": option_type,
                })
            logger.info(
                "Received %d %s contracts for %s",
                len(contracts), option_type, underlying
            )
            if contracts:   # never cache empty results — allow a fresh retry
                self._option_cache[cache_key] = (contracts, time.monotonic())
            return contracts
        except requests.RequestException as exc:
            logger.error("Alpaca option chain request failed: %s", exc)
            return None

    @staticmethod
    def _extract_strike(option_symbol: str) -> float:
        """Extract the strike price from an OCC option symbol."""
        try:
            return int(option_symbol[-8:]) / 1000.0
        except (ValueError, IndexError):
            return 0.0

    @staticmethod
    def _yfinance_chain_to_contracts(
        df: "pd.DataFrame", option_type: str, expiration_date: str,
    ) -> List[Dict]:
        """
        Normalise a yfinance puts/calls DataFrame into the same list-of-dicts
        shape ``_fetch_option_chain`` returns.  This lets the rest of the
        backtester treat both sources uniformly and avoids the DataFrame
        truth-value-is-ambiguous ValueError the previous code tripped over.
        """
        if df is None or df.empty:
            return []
        out: List[Dict] = []
        for _, row in df.iterrows():
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            out.append({
                "symbol": str(row.get("contractSymbol", "")),
                "bid": bid,
                "ask": ask,
                "mid": round((bid + ask) / 2, 4),
                # yfinance does not publish delta; downstream must estimate
                # it from σ-distance rather than from an "inTheMoney" flag.
                "delta": float("nan"),
                "theta": 0.0,
                "vega": 0.0,
                "gamma": 0.0,
                "iv": float(row.get("impliedVolatility", 0) or 0),
                "strike": float(row.get("strike", 0) or 0),
                "in_the_money": bool(row.get("inTheMoney", False)),
                "expiration": expiration_date,
                "type": option_type,
            })
        return out

    def _get_option_chain_for_date(
        self, ticker: str, entry_date: date, target_dte: int
    ) -> Optional[Tuple[float, float, float]]:
        """
        Resolve a short-strike distance, credit, and |Δ| estimate for the
        given (ticker, entry_date) using a *current-snapshot* chain.

        Source precedence:
          1. Alpaca /v1beta1/options/snapshots (real bid/ask + Greeks)
          2. yfinance tk.option_chain(expiry) (real bid/ask, no Greeks)

        CAVEAT — both sources return the CURRENT market snapshot, not a
        historical as-of-``entry_date`` chain.  For honest historical
        backtests, use the alpaca_historical mode (see
        ``_fetch_alpaca_option_bars``).  We log loudly when ``entry_date``
        is older than ``_SNAPSHOT_FRESH_DAYS`` so the user knows the
        credit is today's snapshot applied to a past bar.

        Returns ``(strike_distance_pct, credit, |delta|)`` or ``None``
        when no usable chain could be resolved.
        """
        today = date.today()
        age_days = (today - entry_date).days
        if age_days > self._SNAPSHOT_FRESH_DAYS:
            logger.warning(
                "[%s] snapshot chain applied to %s (%d days stale) — "
                "credits are current-market, NOT historical. Use the "
                "alpaca_historical mode for honest 30-day parity.",
                ticker, entry_date, age_days,
            )

        try:
            tk = yf.Ticker(ticker)
            # yfinance's `.options` is the canonical source of expiry
            # strings — Alpaca's chain endpoints require one, too.
            try:
                expirations = list(tk.options or [])
            except Exception as exc:
                logger.warning("[%s] yfinance expirations unavailable: %s", ticker, exc)
                return None
            if not expirations:
                logger.warning("[%s] No expirations available", ticker)
                return None

            # Pick the expiry whose DTE (from entry_date) is closest to target.
            best_expiry: Optional[str] = None
            best_diff = float("inf")
            for expiry_str in expirations:
                try:
                    expiry_dt = date.fromisoformat(expiry_str)
                except ValueError:
                    continue
                dte = (expiry_dt - entry_date).days
                if dte <= 0:
                    continue
                diff = abs(dte - target_dte)
                if diff < best_diff:
                    best_diff = diff
                    best_expiry = expiry_str
            if best_expiry is None:
                logger.warning("[%s] No future expirations found", ticker)
                return None

            # ── Source 1: Alpaca snapshots ───────────────────────────
            put_chain: List[Dict] = []
            call_chain: List[Dict] = []
            if self._alpaca_api_key and self._alpaca_secret_key:
                alp_puts = self._fetch_option_chain(ticker, best_expiry, "put") or []
                alp_calls = self._fetch_option_chain(ticker, best_expiry, "call") or []
                put_chain = list(alp_puts)
                call_chain = list(alp_calls)
                if put_chain or call_chain:
                    logger.info(
                        "[%s] Alpaca snapshot: %d puts, %d calls (exp %s)",
                        ticker, len(put_chain), len(call_chain), best_expiry,
                    )

            # ── Source 2: yfinance fallback (also current snapshot) ──
            if not put_chain and not call_chain:
                try:
                    option_chain = tk.option_chain(best_expiry)
                except Exception as exc:
                    logger.warning("[%s] yfinance fallback failed: %s", ticker, exc)
                    return None
                put_chain = self._yfinance_chain_to_contracts(
                    getattr(option_chain, "puts", None), "put", best_expiry,
                )
                call_chain = self._yfinance_chain_to_contracts(
                    getattr(option_chain, "calls", None), "call", best_expiry,
                )
                logger.info(
                    "[%s] yfinance snapshot: %d puts, %d calls (exp %s)",
                    ticker, len(put_chain), len(call_chain), best_expiry,
                )

            if not put_chain and not call_chain:
                logger.error("[%s] No option chains available", ticker)
                return None

            # Underlying price on entry_date (for strike-distance computation)
            current_price = self._get_price_for_date(ticker, entry_date)
            if not current_price or current_price <= 0:
                logger.warning("[%s] No price data for %s", ticker, entry_date)
                return None

            # ── Short-strike selection ───────────────────────────────
            # Pick the richer wing (more contracts) and take the OTM
            # strike closest to spot — this is a conservative short
            # leg that the σ-gate will reject when it's too close.
            use_puts = len(put_chain) >= len(call_chain)
            chain = put_chain if use_puts else call_chain
            opt_type = "put" if use_puts else "call"

            if opt_type == "put":
                otm = [c for c in chain if c["strike"] > 0 and c["strike"] < current_price]
                # Highest-OTM put strike = closest to spot from below.
                otm.sort(key=lambda c: c["strike"], reverse=True)
            else:
                otm = [c for c in chain if c["strike"] > 0 and c["strike"] > current_price]
                # Lowest-OTM call strike = closest to spot from above.
                otm.sort(key=lambda c: c["strike"])

            if not otm:
                logger.warning("[%s] No OTM %ss available (spot=$%.2f)",
                               ticker, opt_type, current_price)
                return None
            short_leg = otm[0]
            strike = float(short_leg["strike"])
            # Prefer the mid if both sides populated; fall back to bid
            # (what we'd receive on a market sell order).
            bid = float(short_leg.get("bid") or 0)
            ask = float(short_leg.get("ask") or 0)
            if bid > 0 and ask > 0:
                credit = round((bid + ask) / 2, 4)
            elif bid > 0:
                credit = bid
            else:
                logger.warning(
                    "[%s] Short %s strike $%.2f has no quote (bid=%.2f, ask=%.2f)",
                    ticker, opt_type, strike, bid, ask,
                )
                return None

            # ── |Delta| estimate ─────────────────────────────────────
            # Prefer the broker-supplied delta (Alpaca).  When it's
            # missing (yfinance) or zero, approximate from the strike
            # distance expressed in σ-units — this is the same heuristic
            # used by the σ-path and correctly places the short leg on
            # the Φ(-N) curve instead of the old binary 0.5-or-0.0 hack.
            delta_raw = short_leg.get("delta", float("nan"))
            try:
                delta_val = abs(float(delta_raw))
            except (TypeError, ValueError):
                delta_val = float("nan")
            if not np.isfinite(delta_val) or delta_val == 0.0:
                # Fall back to σ-distance proxy using the bar's own realised
                # vol over the hold window.
                try:
                    close_px = float(current_price)
                    sigma_guess = float(short_leg.get("iv") or 0.0)
                    dte_days = max(1, (date.fromisoformat(best_expiry) - entry_date).days)
                    if sigma_guess > 0 and close_px > 0:
                        sigma_hold = sigma_guess * np.sqrt(dte_days / 252.0)
                        n_sigma = abs(strike - close_px) / (close_px * sigma_hold)
                        delta_val = self._delta_from_sigma_distance(n_sigma)
                    else:
                        delta_val = 0.15   # neutral default for OTM nearest-to-ATM
                except Exception:
                    delta_val = 0.15

            strike_distance_pct = abs(strike - current_price) / current_price
            logger.info(
                "[%s] Selected short %s: strike $%.2f (dist %.2f%%, "
                "credit $%.2f, |Δ| %.3f, source=%s)",
                ticker, opt_type, strike, strike_distance_pct * 100.0,
                credit, delta_val,
                "alpaca" if put_chain and put_chain[0].get("delta", float("nan")) == put_chain[0].get("delta", float("nan")) and short_leg.get("delta") is not None else "yfinance",
            )
            return (strike_distance_pct, credit, delta_val)

        except Exception as exc:
            logger.error(
                "Option chain fetch for %s on %s failed: %s",
                ticker, entry_date, exc,
            )
            return None

    def _refresh_live_quotes(
        self,
        ticker: str,
        entry_date: date,
        strategy: str,
        strike_distance_pct: Optional[float],
        credit: float,
        approx_abs_delta: Optional[float],
        equity: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
        """
        Refresh live option quotes immediately before trade execution.

        Mirrors the live agent's executor._refresh_limit_price() pattern:
        fetch fresh option chain from Alpaca, recalculate credit using
        live bid/ask, and re-validate the two economics-bearing
        guardrails before committing to the trade.

        Parameters
        ----------
        equity: current account equity (used for the max-risk guardrail).
            Defaults to ``self.starting_equity`` when the caller omits it
            — but the backtest loop should always pass the *current*
            equity so the budget rebase reflects compounded P&L.

        Returns ``(strike_distance, credit, delta, status)``.  Statuses:
            "success"               — fresh quote within drift threshold
            "drift_warning"         — fresh quote differs ≥10 % from plan
            "rejected_credit_ratio" — post-refresh credit/width < floor
            "rejected_max_loss"     — post-refresh max-loss > risk budget
            "failed" / "no_api_config" — refresh itself didn't complete
        """
        if not self._alpaca_api_key or not self._alpaca_secret_key:
            return (strike_distance_pct, credit, approx_abs_delta, "no_api_config")

        eq_for_budget = float(equity) if equity is not None else float(self.starting_equity)

        try:
            # Mirror the live agent: bust the chain cache for this ticker
            # so the "refresh" is actually a fresh HTTP call rather than
            # re-reading the same snapshot that planning just used.
            stale_keys = [k for k in self._option_cache if k[0] == ticker]
            for k in stale_keys:
                self._option_cache.pop(k, None)

            result = self._get_option_chain_for_date(ticker, entry_date, self.target_dte)
            if result is None:
                logger.warning(
                    "[%s] Live quote refresh failed — no option chain for %s (DTE %d)",
                    ticker, entry_date, self.target_dte,
                )
                return (strike_distance_pct, credit, approx_abs_delta, "failed")

            new_strike_distance, new_credit, new_delta = result

            drift = abs(new_credit - credit)
            drift_pct = drift / credit if credit > 0 else 0.0
            drift_warn_pct = 0.10  # 10 % threshold for warning

            credit_to_width = new_credit / self.spread_width if self.spread_width else 0.0
            max_loss = self.spread_width - new_credit

            # Guardrail 1: credit-to-width ratio (same gate the planning
            # phase enforces, re-checked against the refreshed quote).
            if self.min_credit_ratio is not None and credit_to_width < self.min_credit_ratio:
                logger.warning(
                    "[%s] Live quote REJECTED: credit/width %.4f < %.2f floor "
                    "(plan=%.2f → live=%.2f, width=%.2f)",
                    ticker, credit_to_width, self.min_credit_ratio,
                    credit, new_credit, self.spread_width,
                )
                return (new_strike_distance, new_credit, new_delta, "rejected_credit_ratio")

            # Guardrail 2: max loss per contract ≤ risk budget
            # (RiskManager.max_risk_pct × *current* equity — not starting).
            if self.max_risk_pct is not None:
                allowed_loss = eq_for_budget * self.max_risk_pct
                if max_loss * 100.0 > allowed_loss:
                    logger.warning(
                        "[%s] Live quote REJECTED: max_loss $%.2f > $%.2f budget "
                        "(%.1f%% of $%.2f equity)",
                        ticker, max_loss * 100.0, allowed_loss,
                        self.max_risk_pct * 100.0, eq_for_budget,
                    )
                    return (new_strike_distance, new_credit, new_delta, "rejected_max_loss")

            if drift_pct > drift_warn_pct:
                logger.warning(
                    "[%s] Credit drifted %.1f%% since planning "
                    "(plan=$%.2f → live=$%.2f)",
                    ticker, drift_pct * 100.0, credit, new_credit,
                )
                return (new_strike_distance, new_credit, new_delta, "drift_warning")

            logger.info(
                "[%s] Live quote refreshed: credit $%.2f (plan was $%.2f)",
                ticker, new_credit, credit,
            )
            return (new_strike_distance, new_credit, new_delta, "success")

        except Exception as exc:
            logger.warning(
                "[%s] Live quote refresh exception: %s — using planned values",
                ticker, exc,
            )
            return (strike_distance_pct, credit, approx_abs_delta, "failed")

    def _get_price_for_date(self, ticker: str, entry_date: date) -> Optional[float]:
        """
        Return the underlying close on or immediately before ``entry_date``.

        Used by the option-chain helpers for strike-distance arithmetic.
        Caches nothing — call sites already sit inside the per-bar path of
        ``run()``, which only invokes this once per entry.
        """
        try:
            tk = yf.Ticker(ticker)
            # A small window around the entry keeps the response tiny while
            # tolerating weekends / holidays (no bar on the exact date).
            start = (entry_date - timedelta(days=7)).isoformat()
            end = (entry_date + timedelta(days=2)).isoformat()
            df = tk.history(start=start, end=end, auto_adjust=False)
            if df is None or df.empty:
                return None
            # Drop timezone so index.date comparisons against `entry_date`
            # (a naive datetime.date) don't raise on tz-aware indexes.
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_localize(None)
            # Pick the most recent bar on-or-before entry_date — mirrors the
            # live agent which uses the last-completed bar at decision time.
            dates = np.array([d.date() for d in df.index])
            mask = dates <= entry_date
            if mask.any():
                closest = df[mask].iloc[-1]
            else:
                # entry_date is earlier than the window's first bar — take
                # the nearest future bar so downstream math doesn't div/0.
                closest = df.iloc[0]
            return float(closest["Close"])
        except Exception as exc:
            logger.debug("[%s] _get_price_for_date(%s) failed: %s",
                         ticker, entry_date, exc)
            return None

    # ── Alpaca historical option bars (30-day live-parity mode) ────────────
    #
    # This path answers the user's "match live trading with real options
    # data for the last 30 days" ask.  Unlike the snapshot path above,
    # these methods query Alpaca's HISTORICAL bars endpoints so the
    # credit at entry and the spread value at exit reflect real market
    # prices as-of the backtest dates.
    #
    # Data flow per simulated trade:
    #   1. ``/v2/options/contracts`` → list contracts for (ticker, target
    #      expiry) with strike/type metadata
    #   2. ``/v1beta1/options/bars`` → daily OHLCV for both legs over
    #      [entry_date, exit_date]
    #   3. Credit at entry ≈ short.close − long.close
    #   4. Simulate exit via the live-agent's first-to-fire dual-stop
    #      logic applied to the underlying, then mark-to-market the
    #      spread using the *option* bars on the exit date.
    #
    # Alpaca retains historical options data for roughly the last 30–90
    # days (depends on subscription).  Outside that window this path
    # degrades to "no data" and the caller falls back.

    # Alpaca's trading API hosts the contract catalogue; default to the
    # paper endpoint used by .env.  Broker URL is derived by stripping
    # "/v2" from the data_url and substituting the trading host, but we
    # read ALPACA_BASE_URL directly to stay consistent with config.py.
    _ALPACA_BROKER_URL_ENV = "ALPACA_BASE_URL"
    _ALPACA_BROKER_URL_DEFAULT = "https://paper-api.alpaca.markets/v2"

    def _alpaca_broker_url(self) -> str:
        base = os.getenv(self._ALPACA_BROKER_URL_ENV, self._ALPACA_BROKER_URL_DEFAULT)
        return base.rstrip("/")

    def _alpaca_data_base(self) -> str:
        """Return the Alpaca data host (without /v2 suffix) for /v1beta1/…"""
        data = (self._alpaca_data_url or "").rstrip("/")
        if data.endswith("/v2"):
            data = data[: -len("/v2")]
        return data

    def _fetch_alpaca_option_contracts(
        self,
        underlying: str,
        expiration_date: str,
        option_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        List active option contracts for ``underlying`` expiring on
        ``expiration_date``.  Returns ``[{symbol, strike, type,
        expiration}]``.  Empty list on error.

        Cached
        ------
        Result is memoized in ``self._contracts_cache`` keyed by
        ``(underlying, expiration_date, option_type)``.  The catalogue
        for a (ticker, expiration, type) tuple does not change within
        a backtest session, so re-fetching for every 5-min bar wastes
        thousands of API calls.  Cache lives for the lifetime of the
        Backtester instance — long-running sessions that span multiple
        trading days should construct a new instance.
        """
        cache_key = (underlying, expiration_date, option_type or "")
        if not hasattr(self, "_contracts_cache"):
            self._contracts_cache: Dict[Tuple[str, str, str], List[Dict]] = {}
        cached = self._contracts_cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{self._alpaca_broker_url()}/options/contracts"
        params: Dict[str, object] = {
            "underlying_symbols": underlying,
            "expiration_date": expiration_date,
            "status": "active",
            "limit": 10000,
        }
        if option_type:
            params["type"] = option_type
        out: List[Dict] = []
        try:
            next_page_token: Optional[str] = None
            for _ in range(10):   # bounded pagination
                if next_page_token:
                    params["page_token"] = next_page_token
                resp = self._alpaca_request(url, params=params, timeout=15)
                if resp is None:
                    break
                resp.raise_for_status()
                data = resp.json() or {}
                for c in data.get("option_contracts", []) or []:
                    try:
                        strike = float(c.get("strike_price") or 0)
                    except (TypeError, ValueError):
                        strike = 0.0
                    out.append({
                        "symbol": c.get("symbol", ""),
                        "strike": strike,
                        "type": (c.get("type") or "").lower(),  # "call" | "put"
                        "expiration": c.get("expiration_date", expiration_date),
                    })
                next_page_token = data.get("next_page_token")
                if not next_page_token:
                    break
            # Only cache non-empty results — an empty list usually
            # indicates a transient API error and we want a fresh retry
            # on the next call rather than memoizing the failure.
            if out:
                self._contracts_cache[cache_key] = out
            return out
        except requests.RequestException as exc:
            logger.warning(
                "Alpaca /options/contracts failed for %s exp %s: %s",
                underlying, expiration_date, exc,
            )
            return []

    def _fetch_alpaca_option_bars(
        self,
        symbols: List[str],
        start: date,
        end: date,
        timeframe: str = "1Day",
    ) -> Dict[str, List[Dict]]:
        """
        Fetch historical OHLCV bars for a set of option contract symbols.
        Returns ``{symbol: [{t, o, h, l, c, v}, …]}``.
        Empty dict on error; missing symbols are simply absent from the
        result map.
        """
        if not symbols:
            return {}
        # Alpaca allows ≤100 symbols per request — chunk defensively.
        out: Dict[str, List[Dict]] = {}
        url = f"{self._alpaca_data_base()}/v1beta1/options/bars"
        chunk_size = 100
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i : i + chunk_size]
            params = {
                "symbols": ",".join(chunk),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "timeframe": timeframe,
                "limit": 10000,
            }
            try:
                resp = self._alpaca_request(url, params=params, timeout=30)
                if resp is None:
                    continue
                resp.raise_for_status()
                data = resp.json() or {}
                bars_map = data.get("bars") or {}
                for sym, bars in bars_map.items():
                    if not bars:
                        continue
                    out.setdefault(sym, []).extend(bars)
            except requests.RequestException as exc:
                logger.warning(
                    "Alpaca /options/bars failed for %d symbols "
                    "(%s…%s): %s",
                    len(chunk), start, end, exc,
                )
                # Continue with other chunks — partial data is still useful.
                continue
        return out

    def _pick_alpaca_expiration(
        self,
        ticker: str,
        entry_date: date,
        target_dte: int,
        exclude: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """
        Pick the expiration date in Alpaca's contract catalogue closest
        to ``entry_date + target_dte``.  Returns an ISO-date string or
        ``None`` when the catalogue lookup fails.

        ``exclude`` lets the caller exclude expirations that have
        already been tried and failed (e.g. ``no_bars_on_entry_day``)
        so the fallback loop in ``_alpaca_historical_plan`` can request
        the next-best candidate without re-picking the dud.

        Cached
        ------
        The full expiration *set* for ``(ticker, entry_date, target_dte)``
        is memoized in ``self._expiration_set_cache`` so 5-min-bar runs
        within the same session don't re-scan the catalogue 78 times
        per day.  The cache stores the unfiltered set; ``exclude`` is
        applied at lookup time.  Without this cache the new fallback
        loop multiplied catalogue scans by 3-4×, pushing 30-day 5-min
        runtimes from minutes into multi-hour territory.
        """
        if not hasattr(self, "_expiration_set_cache"):
            self._expiration_set_cache: Dict[
                Tuple[str, str, int], Set[str]
            ] = {}
        cache_key = (ticker, entry_date.isoformat(), int(target_dte))
        expiries = self._expiration_set_cache.get(cache_key)
        if expiries is None:
            # Alpaca's /options/contracts supports expiration_date_gte/_lte
            # filters so we can pull just a small window around the target.
            window_lo = entry_date + timedelta(days=max(1, target_dte - 10))
            window_hi = entry_date + timedelta(days=target_dte + 20)
            url = f"{self._alpaca_broker_url()}/options/contracts"
            params = {
                "underlying_symbols": ticker,
                "expiration_date_gte": window_lo.isoformat(),
                "expiration_date_lte": window_hi.isoformat(),
                "status": "active",
                "type": "put",  # only need the expiry set, one type is enough
                "limit": 10000,
            }
            try:
                resp = self._alpaca_request(url, params=params, timeout=15)
                if resp is None:
                    return None
                resp.raise_for_status()
                data = resp.json() or {}
            except requests.RequestException as exc:
                logger.warning(
                    "[%s] Alpaca expiry lookup failed (%s..%s): %s",
                    ticker, window_lo, window_hi, exc,
                )
                return None

            expiries = set()
            for c in data.get("option_contracts", []) or []:
                exp = c.get("expiration_date")
                if exp:
                    expiries.add(exp)
            # Only cache populated sets — empty result usually means
            # transient catalogue failure and we want a fresh retry.
            if expiries:
                self._expiration_set_cache[cache_key] = expiries

        if exclude:
            expiries = expiries - set(exclude)
        if not expiries:
            return None

        target_dt = entry_date + timedelta(days=target_dte)

        # Friday-weekly preference
        # ------------------------
        # Catalogue contains Mon/Wed/Thu/Fri weeklies on QQQ/SPY/IWM.
        # Friday weeklies are *materially* more liquid: they receive
        # 5-10× the volume of Mon/Wed expiries, and the bars endpoint
        # only returns bars where trades printed.  When the picker's
        # nearest-DTE choice falls on a Mon/Wed weekly we routinely get
        # ``no_bars_on_entry_day`` rejections at deep-OTM strikes even
        # though the contract is "listed" — there were simply no trades.
        #
        # Resolution: add a non-Friday penalty to the diff metric.
        # Penalty = 4 days means a Friday weekly within 4 days of the
        # target wins over a non-Friday at exact target.  Math:
        #
        #   target = Mon (entry+35d)
        #   Mon @ 35d → diff = 0 + 4 = 4
        #   Wed @ 37d → diff = 2 + 4 = 6
        #   Fri @ 32d → diff = 3 + 0 = 3 ← wins
        #   Fri @ 39d → diff = 4 + 0 = 4 (ties with Mon)
        #
        # Tiebreaker on equal effective diff: prefer earlier expiry
        # (less theta exposure on the long leg) and Friday over non-
        # Friday (so the Mon vs Fri+39 tie above goes to Friday).
        NON_FRIDAY_PENALTY = 4

        best: Optional[str] = None
        best_diff: float = float("inf")
        best_is_friday: bool = False
        best_date: Optional[date] = None
        for exp in expiries:
            try:
                d = date.fromisoformat(exp)
            except ValueError:
                continue
            if d <= entry_date:
                continue
            is_friday = d.weekday() == 4
            raw_diff = abs((d - target_dt).days)
            eff_diff = raw_diff + (0 if is_friday else NON_FRIDAY_PENALTY)
            # Strict improvement on effective diff, then prefer Friday
            # on ties, then prefer the earlier expiration.
            better = False
            if eff_diff < best_diff:
                better = True
            elif eff_diff == best_diff:
                if is_friday and not best_is_friday:
                    better = True
                elif is_friday == best_is_friday and (
                    best_date is None or d < best_date
                ):
                    better = True
            if better:
                best_diff = eff_diff
                best_is_friday = is_friday
                best_date = d
                best = exp
        return best

    def _alpaca_historical_plan(
        self,
        ticker: str,
        entry_date: date,
        regime: str,
        target_dte: int,
        underlying_price: float,
        sigma_annual: float,
        effective_sigma_mult: float,
        hold_bars: Optional[int] = None,
        bars_per_year: Optional[int] = None,
    ) -> Tuple[Optional[Dict], str]:
        """
        Build a real-data trade plan for (ticker, entry_date) using
        Alpaca's historical contracts + bars endpoints.

        Returns
        -------
        (plan, reason) : Tuple[Optional[Dict], str]
            On success ``plan`` is a dict with keys::

                short_symbol, long_symbol, short_strike, long_strike,
                expiration, option_type, credit, strike_distance_pct,
                approx_abs_delta

            and ``reason`` is the empty string ``""``.

            On failure ``plan`` is ``None`` and ``reason`` is a
            structured one-line token of the form ``"<gate>: <context>"``
            naming the *specific* upstream failure (no_expiration_in_window,
            no_bars_on_entry_day, long_leg_off_grid, …).  The token before
            the first ``:`` is also used as the funnel-gate suffix so per-
            cause counters appear separately in ``self.rejections`` instead
            of collapsing into one generic bucket.

        σ-horizon contract
        ------------------
        The short-leg distance target is computed as ``effective_sigma_mult ×
        sigma_annual × √(hold_bars / bars_per_year)``.  This matches the
        synthetic σ-path (``_sigma_strike_distance``) and the live agent's
        per-hold-horizon thinking.

        When ``hold_bars`` / ``bars_per_year`` are not provided (legacy
        callers), we fall back to the original DTE-horizon projection
        (``√(dte_days / 252)``).  For daily backtests both forms are
        numerically identical because ``hold_bars == target_dte`` and
        ``bars_per_year == 252``.

        For **intraday** backtests they differ dramatically:
          * legacy DTE-horizon — projects σ over ~30 days regardless of
            actual hold (1 hour), pushing strikes ~13% OTM with |Δ|≈0.02.
            Live agent never picks anything below the 0.15 MIN_DELTA floor
            so the resulting trades are not apples-to-apples.
          * hold-horizon (this path) — projects σ over the real 1-hour
            hold, picking strikes ~1% OTM with |Δ|≈0.20-0.30, matching
            what a 5-min live agent would actually trade.
        """
        if not self._alpaca_api_key or not self._alpaca_secret_key:
            return None, "missing_api_keys: ALPACA_API_KEY/SECRET not set"
        if underlying_price <= 0:
            return None, f"invalid_underlying_price: {underlying_price:.4f}"

        # Expiration-fallback loop
        # ------------------------
        # Most ``no_bars_on_entry_day`` failures are not "the data is gone"
        # but "the picker landed on a less-liquid weekly that didn't print
        # any trades at the chosen strikes that day".  Retry with the
        # next-best expiration — typically the next Friday weekly — before
        # giving up.  Capped at MAX_FALLBACK_ATTEMPTS to bound API calls
        # per candidate bar.
        #
        # Only data-availability failures (no_contracts_for_expiry,
        # no_bars_on_entry_day, non_positive_credit) trigger a fallback.
        # Contract-shape failures (no_otm_near_target, long_leg_off_grid,
        # degenerate_bar_close) are deterministic given the expiration's
        # strike grid — retrying a different expiration won't help.
        #
        # Cap is intentionally small: the Friday-weekly preference in
        # ``_pick_alpaca_expiration`` lands on a liquid expiry on the
        # first try ~95%+ of the time, so one retry catches the rare
        # edge case where even the Friday weekly has no bars at the
        # chosen strikes.  Keeping the cap low bounds API call volume
        # under intraday backtests where we'd otherwise pay the
        # catalogue-scan cost N× per candidate bar.
        MAX_FALLBACK_ATTEMPTS = 2
        excluded_expirations: Set[str] = set()
        last_reason = ""

        for attempt in range(MAX_FALLBACK_ATTEMPTS):
            expiration = self._pick_alpaca_expiration(
                ticker, entry_date, target_dte,
                exclude=excluded_expirations or None,
            )
            if not expiration:
                if attempt == 0:
                    logger.info(
                        "[%s] alpaca_historical: no expiration near %s+%dd",
                        ticker, entry_date, target_dte,
                    )
                    return None, (
                        f"no_expiration_in_window: {ticker} "
                        f"target={entry_date}+{target_dte}d"
                    )
                # Exhausted all candidates after one+ retries.
                return None, (
                    f"no_bars_after_fallbacks: {ticker} {entry_date} "
                    f"tried={sorted(excluded_expirations)} "
                    f"last={last_reason or 'unknown'}"
                )

            plan, reason, retry = self._build_alpaca_plan_for_expiration(
                ticker=ticker,
                entry_date=entry_date,
                regime=regime,
                expiration=expiration,
                underlying_price=underlying_price,
                sigma_annual=sigma_annual,
                effective_sigma_mult=effective_sigma_mult,
                hold_bars=hold_bars,
                bars_per_year=bars_per_year,
            )
            if plan is not None:
                if attempt > 0:
                    logger.info(
                        "[%s] alpaca_historical: succeeded on fallback "
                        "exp=%s after skipping %s",
                        ticker, expiration, sorted(excluded_expirations),
                    )
                return plan, reason
            if not retry:
                # Non-fallbackable failure — return immediately.
                return None, reason
            # Fallbackable failure (no contracts / no bars / non-positive
            # credit on this expiry) — exclude and try the next-best.
            last_reason = reason
            excluded_expirations.add(expiration)
            logger.info(
                "[%s] alpaca_historical: %s — falling back to next expiry",
                ticker, reason.split(":", 1)[0],
            )

        # Exhausted MAX_FALLBACK_ATTEMPTS without success.
        return None, (
            f"no_bars_after_fallbacks: {ticker} {entry_date} "
            f"tried={sorted(excluded_expirations)} "
            f"last={last_reason or 'unknown'}"
        )

    # ── Unified decision-engine bridge (Phase 2) ─────────────────────────────
    def _synth_chain_slice_for_decide(
        self,
        *,
        expiration: str,
        entry_date: date,
        contracts: List[Dict],
        bars_by_symbol: Dict[str, List[Dict]],
        spot: float,
        sigma_hold: float,
    ) -> Optional[object]:
        """
        Build a ``decision_engine.ChainSlice`` from Alpaca-historical inputs.

        Alpaca's historical endpoints don't return bid/ask or Greeks — only
        contracts metadata + OHLCV bars. We fabricate the dict-shaped chain
        ``decide()`` expects by:

          * setting ``bid = ask = bar_close`` so ``_quote_credit`` collapses
            to ``(short_close − long_close − fill_haircut)``, matching the
            legacy historical credit minus a one-tick fill haircut.
          * approximating ``Δ`` from a one-period BS with σ_hold (no rate,
            no carry — fine for short-dated index spreads). The σ-path
            already does this for ``approx_abs_delta``; we apply the same
            trick *per-strike* so the engine can run its Δ-grid sweep.

        Returns ``None`` when no contract has bars on entry_date or when
        the inputs are degenerate. Caller treats this as a fallbackable
        data-availability failure.
        """
        from math import erf, log, sqrt
        from trading_agent.decision_engine import ChainSlice
        if spot <= 0 or sigma_hold <= 0 or not contracts:
            return None
        try:
            dte_days = max(1, (date.fromisoformat(expiration) - entry_date).days)
        except (TypeError, ValueError):
            return None
        rows: List[Dict] = []
        for c in contracts:
            sym = (c.get("symbol") or "").strip()
            strike = float(c.get("strike") or 0.0)
            if not sym or strike <= 0:
                continue
            bars = bars_by_symbol.get(sym) or []
            if not bars:
                continue
            try:
                close = float(bars[0].get("c") or 0.0)
            except (TypeError, ValueError):
                continue
            if close <= 0:
                continue
            try:
                d1 = (log(spot / strike) + 0.5 * sigma_hold * sigma_hold) / sigma_hold
            except (ValueError, ZeroDivisionError):
                continue
            cdf_d1 = 0.5 * (1.0 + erf(d1 / sqrt(2.0)))
            opt_type = (c.get("type") or "").lower()
            # Δ_call = Φ(d1); Δ_put = Φ(d1) − 1.  Sign matters: live uses
            # signed delta so the C/W floor (|Δ|×(1+edge_buffer)) is the
            # *same* expression on both sides.
            delta = cdf_d1 if opt_type == "call" else (cdf_d1 - 1.0)
            rows.append({
                "strike": strike,
                "delta":  delta,
                "bid":    close,
                "ask":    close,
                "symbol": sym,
            })
        if not rows:
            return None
        return ChainSlice(expiration=expiration, dte=dte_days, contracts=rows)

    def _build_alpaca_plan_via_decide(
        self,
        *,
        ticker: str,
        entry_date: date,
        regime: str,
        expiration: str,
        underlying_price: float,
        sigma_annual: float,
        effective_sigma_mult: float,
        hold_bars: Optional[int],
        bars_per_year: Optional[int],
    ) -> Tuple[Optional[Dict], str, bool]:
        """
        Resolve a trade plan through ``decision_engine.decide()`` — the
        same scoring path the live ``ChainScanner`` runs. This is the
        parity seam: ``_score_candidate_with_reason``, ``_quote_credit``,
        and the C/W floor formula all live in ``chain_scanner.py``,
        and changes there automatically flow into the backtester here.

        Returns the ``(plan, reason, retry)`` tuple shape
        ``_build_alpaca_plan_for_expiration`` expects so the caller's
        expiration-fallback loop is untouched.
        """
        from trading_agent.decision_engine import DecisionInput, decide
        side = "bear_call" if regime == "bearish" else "bull_put"
        option_type = "call" if side == "bear_call" else "put"
        contracts = self._fetch_alpaca_option_contracts(
            ticker, expiration, option_type,
        )
        if not contracts:
            return None, (
                f"no_contracts_for_type: {ticker} {option_type}s "
                f"exp={expiration}"
            ), True
        symbols = [c["symbol"] for c in contracts if c.get("symbol")]
        bars_by_symbol = self._fetch_alpaca_option_bars(
            symbols,
            start=entry_date,
            end=entry_date + timedelta(days=1),
        )
        if not bars_by_symbol:
            return None, (
                f"no_bars_on_entry_day: {ticker} {entry_date} "
                f"exp={expiration}"
            ), True
        # σ_hold matches the legacy horizon contract so the two paths are
        # comparable when sweeping side-by-side.
        try:
            dte_days = max(1, (date.fromisoformat(expiration) - entry_date).days)
        except (TypeError, ValueError):
            dte_days = 1
        if hold_bars and bars_per_year and hold_bars > 0 and bars_per_year > 0:
            horizon_frac = hold_bars / bars_per_year
        else:
            horizon_frac = dte_days / 252.0
        sigma_hold = (
            sigma_annual * float(np.sqrt(horizon_frac))
            if sigma_annual > 0 else 0.01
        )
        chain_slice = self._synth_chain_slice_for_decide(
            expiration=expiration,
            entry_date=entry_date,
            contracts=contracts,
            bars_by_symbol=bars_by_symbol,
            spot=underlying_price,
            sigma_hold=sigma_hold,
        )
        if chain_slice is None or not chain_slice.contracts:
            return None, (
                f"empty_synth_chain: {ticker} {entry_date} "
                f"exp={expiration}"
            ), True
        output = decide(
            DecisionInput(side=side, chain_slices=[chain_slice], preset=self.preset),
            max_candidates=1,
        )
        # Stash diagnostics so the Streamlit panel can show why the
        # engine rejected (or what the near-miss was) on the same UI
        # the live monitor uses.
        self.last_decide_diagnostics = output.diagnostics
        if not output.candidates:
            top_reason = max(
                output.diagnostics.rejects_by_reason.items(),
                key=lambda kv: kv[1],
                default=("unknown", 0),
            )[0]
            return None, (
                f"decide_no_candidate_{top_reason}: {ticker} "
                f"{entry_date} exp={expiration}"
            ), False
        pick = output.candidates[0]
        strike_distance_pct = abs(pick.short_strike - underlying_price) / underlying_price
        return {
            "short_symbol":         pick.short_symbol,
            "long_symbol":          pick.long_symbol,
            "short_strike":         pick.short_strike,
            "long_strike":          pick.long_strike,
            "expiration":           pick.expiration,
            "option_type":          option_type,
            "credit":               pick.credit,
            "strike_distance_pct":  strike_distance_pct,
            "approx_abs_delta":     abs(pick.short_delta),
        }, "", False

    def _build_alpaca_plan_for_expiration(
        self,
        *,
        ticker: str,
        entry_date: date,
        regime: str,
        expiration: str,
        underlying_price: float,
        sigma_annual: float,
        effective_sigma_mult: float,
        hold_bars: Optional[int],
        bars_per_year: Optional[int],
    ) -> Tuple[Optional[Dict], str, bool]:
        """
        Inner per-expiration builder used by ``_alpaca_historical_plan``.

        Returns ``(plan, reason, retry)`` where ``retry`` is ``True``
        when the failure mode is data-availability (caller should try
        the next-best expiration) and ``False`` when the failure is
        deterministic given the expiration's strike grid (caller should
        return immediately).

        When ``self.use_unified_engine`` is ``True`` and ``self.preset``
        is provided, this method delegates to
        ``_build_alpaca_plan_via_decide`` — the parity-critical seam
        that runs the same ``decide()`` the live scanner runs. The
        legacy σ-distance heuristic below is preserved for the
        existing test suite and as a fallback when no preset is
        wired in.
        """
        if self.use_unified_engine and self.preset is not None:
            return self._build_alpaca_plan_via_decide(
                ticker=ticker,
                entry_date=entry_date,
                regime=regime,
                expiration=expiration,
                underlying_price=underlying_price,
                sigma_annual=sigma_annual,
                effective_sigma_mult=effective_sigma_mult,
                hold_bars=hold_bars,
                bars_per_year=bars_per_year,
            )
        # Pull both sides for iron condor; for directional regimes we only
        # need one but fetching both is cheap and simplifies logic.
        contracts_p = self._fetch_alpaca_option_contracts(ticker, expiration, "put")
        contracts_c = self._fetch_alpaca_option_contracts(ticker, expiration, "call")
        if not contracts_p and not contracts_c:
            logger.info(
                "[%s] alpaca_historical: no contracts for exp %s",
                ticker, expiration,
            )
            return None, f"no_contracts_for_expiry: {ticker} exp={expiration}", True

        # σ-distance target for the short leg.  See class docstring above
        # for the hold-horizon vs DTE-horizon discussion.
        dte_days = max(1, (date.fromisoformat(expiration) - entry_date).days)
        if hold_bars and bars_per_year and hold_bars > 0 and bars_per_year > 0:
            horizon_frac = hold_bars / bars_per_year
            horizon_label = f"hold={hold_bars}/{bars_per_year}"
        else:
            # Legacy DTE-horizon fallback (kept for backward compatibility
            # with callers / tests that don't pass the new kwargs).
            horizon_frac = dte_days / 252.0
            horizon_label = f"dte={dte_days}/252"
        if sigma_annual > 0:
            sigma_hold = sigma_annual * np.sqrt(horizon_frac)
        else:
            sigma_hold = 0.01
        target_distance = max(0.001, effective_sigma_mult * sigma_hold)
        logger.debug(
            "[%s] alpaca_historical horizon: %s σ_annual=%.4f → "
            "σ_hold=%.4f, target_dist=%.2f%%",
            ticker, horizon_label, sigma_annual, sigma_hold,
            target_distance * 100,
        )
        target_short_strike = (
            underlying_price * (1 - target_distance)
            if regime == "bullish"
            else underlying_price * (1 + target_distance)
            if regime == "bearish"
            else underlying_price * (1 - target_distance)  # IC: use put wing
        )
        option_type = "call" if regime == "bearish" else "put"

        pool = contracts_c if option_type == "call" else contracts_p
        if not pool:
            logger.info(
                "[%s] alpaca_historical: no %ss for exp %s",
                ticker, option_type, expiration,
            )
            # Different option type may exist on a different expiration —
            # treat as fallbackable.
            return None, (
                f"no_contracts_for_type: {ticker} {option_type}s exp={expiration}"
            ), True

        # Pick the strike closest to target on the OTM side.
        if option_type == "put":
            otm = [c for c in pool if 0 < c["strike"] <= underlying_price]
            otm.sort(key=lambda c: abs(c["strike"] - target_short_strike))
        else:
            otm = [c for c in pool if c["strike"] >= underlying_price]
            otm.sort(key=lambda c: abs(c["strike"] - target_short_strike))
        if not otm:
            logger.info(
                "[%s] alpaca_historical: no OTM %ss near target strike %.2f",
                ticker, option_type, target_short_strike,
            )
            # Strike-grid issue — same grid likely on next expiry too.
            return None, (
                f"no_otm_near_target: {ticker} {option_type} "
                f"target={target_short_strike:.2f} spot={underlying_price:.2f}"
            ), False
        short_leg = otm[0]
        short_strike = float(short_leg["strike"])

        # Long strike is ``spread_width`` further OTM.
        long_strike_target = (
            short_strike - self.spread_width
            if option_type == "put"
            else short_strike + self.spread_width
        )
        # Find the contract whose strike is closest to long_strike_target.
        long_pool = pool
        long_candidates = sorted(
            long_pool, key=lambda c: abs(c["strike"] - long_strike_target),
        )
        long_leg = None
        for c in long_candidates:
            if option_type == "put" and c["strike"] < short_strike:
                long_leg = c
                break
            if option_type == "call" and c["strike"] > short_strike:
                long_leg = c
                break
        if long_leg is None:
            logger.info(
                "[%s] alpaca_historical: no long %s strike near %.2f",
                ticker, option_type, long_strike_target,
            )
            # Strike-grid issue — won't change on next expiry.
            return None, (
                f"long_leg_off_grid: {ticker} {option_type} "
                f"target={long_strike_target:.2f} short={short_strike:.2f} "
                f"width={self.spread_width}"
            ), False

        # Fetch entry-day bars for both legs to price the credit.
        bars = self._fetch_alpaca_option_bars(
            [short_leg["symbol"], long_leg["symbol"]],
            start=entry_date,
            end=entry_date + timedelta(days=1),
        )
        short_bars = bars.get(short_leg["symbol"]) or []
        long_bars = bars.get(long_leg["symbol"]) or []
        if not short_bars or not long_bars:
            logger.info(
                "[%s] alpaca_historical: no bars on %s for %s / %s",
                ticker, entry_date, short_leg["symbol"], long_leg["symbol"],
            )
            # Pure data-availability failure — different expiry may have
            # bars where this one doesn't.  Mark as fallbackable.
            return None, (
                f"no_bars_on_entry_day: {ticker} {entry_date} "
                f"short={short_leg['symbol']}({len(short_bars)}b) "
                f"long={long_leg['symbol']}({len(long_bars)}b)"
            ), True

        # Use the first bar on-or-after entry_date.  Alpaca returns UTC
        # timestamps so the entry-day bar is normally index 0.
        short_close = float(short_bars[0].get("c") or 0)
        long_close = float(long_bars[0].get("c") or 0)
        if short_close <= 0 or long_close < 0:
            logger.info(
                "[%s] alpaca_historical: degenerate bar close "
                "(short=%.4f, long=%.4f)",
                ticker, short_close, long_close,
            )
            # Bar exists but its close is broken — treat as fallbackable.
            return None, (
                f"degenerate_bar_close: {ticker} {entry_date} "
                f"short={short_close:.4f} long={long_close:.4f}"
            ), True
        credit = max(0.0, short_close - long_close)
        if credit <= 0.0:
            logger.info(
                "[%s] alpaca_historical: non-positive credit %.4f on %s "
                "(short=%.4f, long=%.4f)",
                ticker, credit, entry_date, short_close, long_close,
            )
            # Stale / illiquid quote on this expiry — try the next one.
            return None, (
                f"non_positive_credit: {ticker} {entry_date} "
                f"credit={credit:.4f} short={short_close:.4f} long={long_close:.4f}"
            ), True

        strike_distance_pct = abs(short_strike - underlying_price) / underlying_price
        # Estimate |Δ| from the chosen σ-distance (matches the σ-path).
        n_sigma = (
            effective_sigma_mult if sigma_hold > 0 and target_distance > 0
            else 1.0
        )
        approx_abs_delta = self._delta_from_sigma_distance(n_sigma)

        return {
            "short_symbol": short_leg["symbol"],
            "long_symbol": long_leg["symbol"],
            "short_strike": short_strike,
            "long_strike": float(long_leg["strike"]),
            "expiration": expiration,
            "option_type": option_type,
            "credit": credit,
            "strike_distance_pct": strike_distance_pct,
            "approx_abs_delta": approx_abs_delta,
        }, "", False

    def _simulate_alpaca_historical(
        self,
        plan: Dict,
        entry_date: date,
        hold_bars: int,
        underlying_prices: pd.Series,
        entry_idx: int,
        loss_cut_multiplier: Optional[float],
        stop_loss_pct: Optional[float],
    ) -> Tuple[str, float, int]:
        """
        Walk forward with real Alpaca option bars to compute the realistic
        P&L of the planned spread.

        Exit logic mirrors ``_simulate``:
          * First breach of short_strike → exit at that day's spread value,
            capped by the smaller of (loss_cut_multiplier × credit) and
            (stop_loss_pct × max_loss), then max-loss.
          * No breach before hold ends → exit at the last bar; if that's
            the expiry we use the intrinsic value.
        """
        short_sym = plan["short_symbol"]
        long_sym = plan["long_symbol"]
        short_strike = float(plan["short_strike"])
        long_strike = float(plan["long_strike"])
        opt_type = plan["option_type"]
        credit = float(plan["credit"])
        expiration = date.fromisoformat(plan["expiration"])

        # Exit window = min(hold_bars days after entry, expiration)
        end_idx = min(entry_idx + hold_bars, len(underlying_prices) - 1)
        window_end = min(
            expiration,
            underlying_prices.index[end_idx].date()
            if hasattr(underlying_prices.index[end_idx], "date")
            else expiration,
        )
        # Pull both legs' daily bars for the whole hold window up-front.
        bars_by_sym = self._fetch_alpaca_option_bars(
            [short_sym, long_sym],
            start=entry_date,
            end=window_end + timedelta(days=1),
        )
        short_bars = {
            self._bar_date(b): b for b in (bars_by_sym.get(short_sym) or [])
        }
        long_bars = {
            self._bar_date(b): b for b in (bars_by_sym.get(long_sym) or [])
        }

        # Underlying prices over the hold window.
        fwd = underlying_prices.iloc[entry_idx : end_idx + 1]
        entry_p = float(underlying_prices.iloc[entry_idx])

        # Breach detection uses the real short strike from the plan — not
        # a σ-distance synthetic — so losses reflect real market moves.
        if opt_type == "put":
            breach_mask = (fwd < short_strike).to_numpy()
        else:
            breach_mask = (fwd > short_strike).to_numpy()
        breach_positions = np.flatnonzero(breach_mask)
        max_loss = self.spread_width - credit

        def _spread_value_on(dt: date) -> Optional[float]:
            sb = short_bars.get(dt)
            lb = long_bars.get(dt)
            if not sb or not lb:
                return None
            s_close = float(sb.get("c") or 0)
            l_close = float(lb.get("c") or 0)
            # Spread value to CLOSE the trade = short_close - long_close
            # (we buy back the short leg, sell back the long).
            return max(0.0, s_close - l_close)

        def _intrinsic_at_expiry(under_px: float) -> float:
            if opt_type == "put":
                short_intrinsic = max(0.0, short_strike - under_px)
                long_intrinsic = max(0.0, long_strike - under_px)
            else:
                short_intrinsic = max(0.0, under_px - short_strike)
                long_intrinsic = max(0.0, under_px - long_strike)
            return max(0.0, short_intrinsic - long_intrinsic)

        def _pnl_from_spread_value(close_value: float) -> float:
            # We RECEIVED credit upfront; we PAY close_value to exit.
            # Net cashflow per share of underlying = credit - close_value.
            # × 100 for contract multiplier, minus round-trip commissions.
            return (credit - close_value) * 100.0 - self.commission

        if breach_positions.size == 0:
            # No breach → hold to the last bar in the window.
            last_idx = end_idx
            last_dt = (
                underlying_prices.index[last_idx].date()
                if hasattr(underlying_prices.index[last_idx], "date")
                else expiration
            )
            close_val = _spread_value_on(last_dt)
            if close_val is None:
                # No option bar at the last day → assume expiration value.
                under_px = float(underlying_prices.iloc[last_idx])
                close_val = _intrinsic_at_expiry(under_px)
            pnl = _pnl_from_spread_value(close_val)
            outcome = "win" if pnl > 0 else "loss"
            return outcome, round(pnl, 2), last_idx - entry_idx

        # Breach → exit on the first breach day, applying the live-agent
        # dual-stop "first to fire" rule in *dollar* space.
        breach_bar = int(breach_positions[0])
        breach_idx = entry_idx + breach_bar
        breach_dt = (
            underlying_prices.index[breach_idx].date()
            if hasattr(underlying_prices.index[breach_idx], "date")
            else entry_date
        )
        close_val = _spread_value_on(breach_dt)
        if close_val is None:
            under_px = float(underlying_prices.iloc[breach_idx])
            close_val = _intrinsic_at_expiry(under_px)

        raw_loss = max(0.0, close_val - credit) * 100.0
        candidates: List[float] = [raw_loss]
        if loss_cut_multiplier is not None and loss_cut_multiplier > 0:
            candidates.append(loss_cut_multiplier * credit * 100.0)
        if stop_loss_pct is not None and 0.0 < stop_loss_pct < 1.0:
            candidates.append(stop_loss_pct * max_loss * 100.0)
        candidates.append(max_loss * 100.0)
        effective_loss = min(candidates)
        pnl = -effective_loss - self.commission
        return "loss", round(pnl, 2), breach_bar

    @staticmethod
    def _bar_date(bar: Dict) -> Optional[date]:
        """Normalise an Alpaca bar's timestamp field ``t`` to a date."""
        t = bar.get("t")
        if not t:
            return None
        try:
            # RFC3339 → strip "Z" / offset, take date portion.
            return datetime.fromisoformat(str(t).replace("Z", "+00:00")).date()
        except Exception:
            try:
                return date.fromisoformat(str(t)[:10])
            except Exception:
                return None

    # ── Regime helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _classify(prices: pd.Series, idx: int) -> str:
        """Legacy daily-bar classifier (200-bar window). Kept for test compatibility."""
        return Backtester._classify_bars(prices, idx, warmup=200)

    @staticmethod
    def _classify_bars(prices: pd.Series, idx: int, warmup: int = 200) -> str:
        """
        Generalized regime classifier.

        Uses `warmup` bars as the long SMA window and warmup//4 as the short
        SMA window. Works for both daily bars (warmup=200) and intraday bars
        (warmup=20, short window=5).
        """
        if idx < warmup:
            return "sideways"
        short_window = max(5, warmup // 4)
        window = prices.iloc[max(0, idx - warmup): idx + 1]
        sma_short = window.iloc[-min(short_window, len(window)):].mean()
        sma_long = window.mean()
        lookback = min(10, len(window))
        slope = (
            window.iloc[-lookback // 2:].mean() - window.iloc[-lookback: -lookback // 2].mean()
            if lookback >= 4
            else 0.0
        )
        price = window.iloc[-1]
        if price > sma_long and slope > 0:
            return "bullish"
        if price < sma_long and slope < 0:
            return "bearish"
        return "sideways"

    @staticmethod
    def _strategy(regime: str) -> str:
        return {
            "bullish": "Bull Put Spread",
            "bearish": "Bear Call Spread",
            "sideways": "Iron Condor",
        }.get(regime, "Iron Condor")

    # ── ETF macro signals (parity with regime.py / market_data.py) ─────────

    @staticmethod
    def _zscore_last(values: List[float]) -> Optional[float]:
        """
        Population-stdev Z-score of the *last* element of ``values`` against
        the entire series.  Mirrors the math in
        ``MarketDataProvider.get_leadership_zscore`` /
        ``get_vix_zscore`` — population (not sample) variance because we
        treat the rolling window as the full intraday distribution.

        Returns ``None`` for series shorter than 2 or with degenerate
        (≤1e-9) stdev to match the live no-signal branch.
        """
        n = len(values)
        if n < 2:
            return None
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        std = var ** 0.5
        if std <= 1e-9:
            return None
        return (values[-1] - mean) / std

    @staticmethod
    def _leadership_zscore_at(ticker_prices: pd.Series,
                              anchor_prices: pd.Series,
                              idx: int,
                              window: int = LEADERSHIP_WINDOW_BARS,
                              ) -> Optional[Tuple[float, float]]:
        """
        Compute ``(raw_diff, zscore)`` of (ticker - anchor) returns at
        bar ``idx``, using a trailing ``window``-bar window.  Returns
        ``None`` when either series lacks data at ``idx`` or the window
        is too short.

        Tail-aligned to the shorter of the two windows after intersecting
        on timestamps (anchor data may have gaps the ticker doesn't,
        e.g. a missing 5-min bar).
        """
        if idx <= 0:
            return None
        # Start at idx - window (inclusive of the entry bar itself)
        start_idx = max(0, idx - window)
        ticker_slice = ticker_prices.iloc[start_idx: idx + 1]
        # Align anchor to the same timestamps the ticker has.
        try:
            anchor_slice = anchor_prices.reindex(ticker_slice.index).dropna()
        except Exception:
            return None
        # Re-align ticker to anchor's surviving timestamps so both share
        # the exact same index when we diff returns.
        ticker_slice = ticker_slice.reindex(anchor_slice.index).dropna()
        if len(ticker_slice) < 2 or len(anchor_slice) < 2:
            return None
        t_returns = ticker_slice.pct_change().dropna().tolist()
        a_returns = anchor_slice.pct_change().dropna().tolist()
        n = min(len(t_returns), len(a_returns))
        if n < 2:
            return None
        diffs = [t - a for t, a in zip(t_returns[-n:], a_returns[-n:])]
        z = Backtester._zscore_last(diffs)
        if z is None:
            return None
        return (diffs[-1], z)

    @staticmethod
    def _vix_zscore_at(vix_close: pd.Series,
                       ts,
                       window: int = VIX_WINDOW_BARS,
                       ) -> Optional[Tuple[float, float]]:
        """
        Z-score the latest VIX 5-min *level change* (point delta, not %)
        in a trailing ``window``-bar window ending at ``ts`` (inclusive).
        Mirrors ``MarketDataProvider.get_vix_zscore`` semantics.

        Returns ``(raw_change, zscore)`` or ``None`` when the window is
        too short or the stdev degenerates to zero.
        """
        if vix_close is None or vix_close.empty:
            return None
        # Slice up to and including ts.  Use searchsorted-equivalent via
        # boolean mask so this works for both DatetimeIndex and date.
        try:
            mask = vix_close.index <= ts
        except TypeError:
            return None
        sub = vix_close.loc[mask].tail(window)
        if len(sub) < 2:
            return None
        closes = sub.astype(float).tolist()
        diffs = [b - a for a, b in zip(closes, closes[1:])]
        z = Backtester._zscore_last(diffs)
        if z is None:
            return None
        return (diffs[-1], z)

    @staticmethod
    def _load_anchor_series(tickers: List[str],
                            start: date,
                            end: date,
                            yf_interval: str,
                            ) -> Dict[str, pd.Series]:
        """
        Pre-download the *unique anchors* for the given tickers in a
        single yfinance batch and return a {anchor → Close series} dict.

        Used by ``run()`` to avoid repeating per-bar downloads.  yfinance
        is called with ``auto_adjust=False`` for parity with the agent
        ``MarketDataProvider`` (raw closes, not split/dividend-adjusted).
        Failures are silently dropped — the caller should treat a missing
        anchor as "no leadership signal available for this ticker".
        """
        anchors_needed = sorted({
            LEADERSHIP_ANCHORS[t] for t in tickers if t in LEADERSHIP_ANCHORS
        })
        out: Dict[str, pd.Series] = {}
        if not anchors_needed:
            return out
        try:
            raw = yf.download(
                anchors_needed,
                start=start.isoformat(),
                end=end.isoformat(),
                interval=yf_interval,
                progress=False,
                auto_adjust=False,
                group_by="ticker",
            )
        except Exception as exc:
            logger.warning("Anchor download failed (%s) — leadership "
                           "signal disabled for this run", exc)
            return out
        if raw is None or raw.empty:
            return out
        # yf returns either a flat DataFrame (single ticker) or a
        # MultiIndex(columns=[ticker, field]) DataFrame (multi-ticker).
        for anchor in anchors_needed:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    sub = raw[anchor] if anchor in raw.columns.get_level_values(0) else None
                    if sub is None or sub.empty:
                        continue
                    series = sub["Close"].dropna()
                else:
                    series = raw["Close"].dropna()
                if not series.empty:
                    out[anchor] = series
            except Exception:
                continue
        return out

    @staticmethod
    def _load_vix_series(start: date,
                        end: date,
                        yf_interval: str,
                        ) -> Optional[pd.Series]:
        """
        Pre-download the ^VIX close series for the backtest window.

        Returns ``None`` when yfinance has no data for the requested
        range (5m intraday is capped at the last ~30 days; daily covers
        the full history).  Caller treats ``None`` as "no VIX gate
        available — bar passes through with vix_inhibit=False".
        """
        try:
            raw = yf.download(
                "^VIX",
                start=start.isoformat(),
                end=end.isoformat(),
                interval=yf_interval,
                progress=False,
                auto_adjust=False,
            )
        except Exception as exc:
            logger.warning("VIX download failed (%s) — VIX gate disabled "
                           "for this run", exc)
            return None
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                raw = raw.xs("^VIX", axis=1, level=1)
            except Exception:
                pass
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        if "Close" not in raw.columns:
            return None
        return raw["Close"].dropna()

    # ── Outcome simulation ──────────────────────────────────────────────────

    def _simulate(
        self, prices: pd.Series, entry_idx: int, regime: str, credit: float,
        hold_bars: int = None, otm_pct: float = DAILY_OTM_PCT,
        strike_distance_pct: Optional[float] = None,
        loss_cut_multiplier: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
    ) -> tuple:
        """
        Walk forward from ``entry_idx`` and return ``(outcome, pnl, hold_count)``.

        Strike placement
        ----------------
        - If ``strike_distance_pct`` is provided, it overrides ``otm_pct`` and
          is the sigma-based placement supplied by ``run()``.
        - Otherwise the legacy fixed-% OTM model applies (preserves tests).

        Loss model (matches position_monitor.ExitMonitor "first to fire")
        ---------------------------------------------------------------
        On the first bar that breaches the short strike, two stops are
        evaluated:

           A. hard_stop = ``loss_cut_multiplier × credit × 100``  (≈ 3× credit)
           B. legacy   = ``stop_loss_pct × max_loss × 100``       (≈ 50% max-loss)

        The smaller of A and B is the effective dollar loss, mirroring the
        live agent which exits at whichever threshold trips first.  When a
        knob is None it's treated as +∞ (i.e. ignored).  Both None → fall
        back to full max-loss payoff (legacy backtester behaviour).
        """
        if hold_bars is None:
            hold_bars = self.target_dte
        end_idx = min(entry_idx + hold_bars, len(prices) - 1)
        fwd = prices.iloc[entry_idx: end_idx + 1]
        entry_p = prices.iloc[entry_idx]

        distance = strike_distance_pct if strike_distance_pct is not None else otm_pct
        lower = entry_p * (1 - distance)
        upper = entry_p * (1 + distance)

        if regime == "bullish":
            breach_mask = (fwd < lower).to_numpy()
        elif regime == "bearish":
            breach_mask = (fwd > upper).to_numpy()
        else:  # sideways / iron condor — either wing
            breach_mask = ((fwd < lower) | (fwd > upper)).to_numpy()

        # breach_mask[0] is the entry bar itself — by construction entry_p
        # lies inside [lower, upper] so it's always False. First True is the
        # first bar that touched the strike.
        breach_positions = np.flatnonzero(breach_mask)
        full_hold = end_idx - entry_idx

        if breach_positions.size == 0:
            # Winner — no touch for the whole hold window
            pnl = credit * 100.0 * self.profit_target_pct - self.commission
            return "win", round(pnl, 2), full_hold

        # Loser — close at the first breach using "first to fire" semantics.
        breach_bar = int(breach_positions[0])
        max_loss_dollars = (self.spread_width - credit) * 100.0

        # Build candidate stops (in dollar loss terms, positive number).
        candidates: List[float] = []
        if loss_cut_multiplier is not None and loss_cut_multiplier > 0:
            candidates.append(loss_cut_multiplier * credit * 100.0)
        if stop_loss_pct is not None and 0.0 < stop_loss_pct < 1.0:
            candidates.append(stop_loss_pct * max_loss_dollars)
        # Always cap at max-loss — you can't lose more than the spread allows.
        candidates.append(max_loss_dollars)

        effective_loss = min(candidates)
        pnl = -effective_loss - self.commission
        return "loss", round(pnl, 2), breach_bar

    # ── Main run ────────────────────────────────────────────────────────────

    def run(
        self,
        tickers: List[str],
        start: date,
        end: date,
        timeframe: str = "1Day",
        use_alpaca: bool = False,
    ) -> BacktestResult:
        """
        Execute the backtest and return a BacktestResult.

        Parameters
        ----------
        tickers     : list of ticker symbols
        start / end : date range
        timeframe   : "1Day" or "5Min"
        use_alpaca  : reserved for future Alpaca data source (currently no-op)

        Notes on 5Min timeframe
        -----------------------
        Yahoo Finance only serves 5-minute data for the last ~30 days.
        Any start date older than that returns an empty DataFrame.
        When timeframe="5Min", start is automatically clamped to today-29 days.
        The warmup period is reduced from 200 daily bars to 20 intraday bars
        (≈ 1.5 hours) and the hold period from 45 bars to 12 bars (≈ 1 hour).
        """
        is_intraday = timeframe == "5Min"
        yf_interval = "1d" if not is_intraday else "5m"

        # ── Yahoo Finance 5m hard limit: clamp start to last 29 days ──────
        warnings: List[str] = []
        if is_intraday:
            earliest_allowed = date.today() - timedelta(days=INTRADAY_MAX_DAYS)
            if start < earliest_allowed:
                warnings.append(
                    f"5-minute data is only available for the last {INTRADAY_MAX_DAYS} days "
                    f"(Yahoo Finance limitation). Start date clamped from {start} "
                    f"to {earliest_allowed}."
                )
                start = earliest_allowed

        warmup_bars = INTRADAY_WARMUP_BARS if is_intraday else 200
        hold_bars = INTRADAY_HOLD_BARS if is_intraday else self.target_dte
        bars_per_year = BARS_PER_YEAR_INTRADAY if is_intraday else BARS_PER_YEAR_DAILY
        vol_window = VOL_WINDOW_INTRADAY if is_intraday else VOL_WINDOW_DAILY

        # If the caller didn't set sigma_mult on the instance, fall back to
        # the timeframe-appropriate default (intraday is tighter because
        # theta over 1 hour is a small fraction of daily theta).
        effective_sigma_mult = (
            self.sigma_mult
            if self.sigma_mult is not None
            else (DEFAULT_SIGMA_MULT_INTRADAY if is_intraday else DEFAULT_SIGMA_MULT_DAILY)
        )
        # A value ≤ 0 disables the sigma path → legacy fixed-% OTM model.
        use_sigma_path = effective_sigma_mult > 0.0

        # One-shot notice when parity's credit-ratio gate would be
        # structurally incompatible with the synthetic credit model.
        if use_sigma_path and self.min_credit_ratio is not None:
            logger.info(
                "Agent-parity credit-ratio floor (%.2f) skipped: backtester "
                "uses synthetic _credit_from_sigma model, which produces "
                "credits below 0.33 for σ>0.8. Other parity gates remain "
                "active (earnings, IV-rank, max_delta, max_risk_pct).",
                self.min_credit_ratio,
            )

        all_trades: List[SimTrade] = []
        equity = self.starting_equity
        equity_curve: List[Dict] = [{"timestamp": pd.Timestamp(start), "account_balance": equity}]

        skipped: List[str] = warnings.copy()

        # ── ETF macro signals: pre-load anchor + VIX series once ────────
        # Live agent fetches these once per cycle from Alpaca/yfinance.
        # In the backtester we batch-download the entire window so the
        # per-bar lookups are O(1) slices rather than network calls.
        anchor_series_by_anchor: Dict[str, pd.Series] = {}
        vix_series: Optional[pd.Series] = None
        if self.use_macro_signals:
            anchor_series_by_anchor = self._load_anchor_series(
                tickers, start, end, yf_interval,
            )
            vix_series = self._load_vix_series(start, end, yf_interval)
            if not anchor_series_by_anchor:
                skipped.append(
                    "macro signals: no anchor data available "
                    "(yfinance returned empty) — leadership gate disabled."
                )
            if vix_series is None:
                skipped.append(
                    "macro signals: no ^VIX data available "
                    "(yfinance returned empty) — VIX gate disabled."
                )

        # ── Run-level startup banner ──────────────────────────────────────
        # Single INFO line so the operator can confirm the run actually
        # started (and with what configuration) the moment they hit
        # "Run Backtest" — even before the first ticker download finishes.
        # Pairs with the per-ticker startup line below to make a hung run
        # visually distinct from a slow-but-healthy one.
        mode_label = (
            "alpaca-historical" if self.use_alpaca_historical
            else ("snapshot+sigma" if use_sigma_path else "fixed-%-OTM")
        )
        logger.info(
            "Backtest starting: %d ticker(s) %s, %s..%s, timeframe=%s, "
            "mode=%s, macro_signals=%s, agent_parity=%s",
            len(tickers), list(tickers), start, end, timeframe,
            mode_label, self.use_macro_signals,
            any([
                self.min_credit_ratio is not None,
                self.max_delta is not None,
                self.max_risk_pct is not None,
                self.use_iv_gate,
                self.use_earnings_gate,
            ]),
        )

        # Progress signpost cadence — emit every ``progress_every`` bars
        # (per ticker).  Tuned so daily runs (~250 bars/yr) emit a few
        # lines per ticker and 5-min runs (~1500 bars/30d) emit ~15.
        progress_every = max(1, 100 if not is_intraday else 200)

        for t_idx, ticker in enumerate(tickers, start=1):
            try:
                logger.info(
                    "[%s] (%d/%d) Downloading %s bars for %s..%s",
                    ticker, t_idx, len(tickers), timeframe, start, end,
                )
                # auto_adjust=False mirrors market_data.MarketDataProvider
                # (task #2) — the live agent reasons on raw closes, not
                # dividend-/split-adjusted closes, so the backtester must
                # do the same to stay apples-to-apples around corporate
                # actions.
                raw = yf.download(
                    ticker,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    interval=yf_interval,
                    progress=False,
                    auto_adjust=False,
                )
                if raw.empty:
                    skipped.append(
                        f"{ticker} (no data returned — "
                        + ("Yahoo only serves 5m data for the last 30 days; "
                           "try a more recent date range" if is_intraday
                           else "check ticker symbol or try a wider date range")
                        + ")"
                    )
                    logger.warning(
                        "[%s] yfinance returned no rows — skipping ticker", ticker,
                    )
                    continue
                # Handle multi-level columns returned by recent yfinance versions
                if isinstance(raw.columns, pd.MultiIndex):
                    raw = raw.xs(ticker, axis=1, level=1) if ticker in raw.columns.get_level_values(1) else raw
                    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
                prices: pd.Series = raw["Close"].dropna()
            except Exception as exc:
                skipped.append(f"{ticker} (download error: {exc})")
                logger.warning(
                    "[%s] yfinance download error: %s — skipping ticker",
                    ticker, exc,
                )
                continue

            min_bars = warmup_bars + hold_bars + 1
            if len(prices) < min_bars:
                unit = "intraday bars" if is_intraday else "daily bars"
                skipped.append(
                    f"{ticker} (only {len(prices)} {unit} downloaded — "
                    f"need {min_bars}+ for {warmup_bars}-bar warmup; "
                    + ("Yahoo Finance only provides ~1560 bars of 5m data over 30 days"
                       if is_intraday
                       else "extend your date range to at least 1 year")
                    + ")"
                )
                logger.warning(
                    "[%s] insufficient bars (%d < %d minimum) — skipping ticker",
                    ticker, len(prices), min_bars,
                )
                continue

            # Per-ticker startup signpost so the operator can see WHICH
            # ticker is currently being processed.  Pairs with the
            # periodic progress line below for stuck-vs-slow diagnosis.
            ticker_entries_before = len(all_trades)
            ticker_t0 = time.monotonic()
            logger.info(
                "[%s] Beginning simulation: %d bars (warmup=%d, hold=%d) "
                "→ ~%d candidate windows",
                ticker, len(prices), warmup_bars, hold_bars,
                max(0, (len(prices) - warmup_bars) // max(1, hold_bars)),
            )

            last_entry_idx = -hold_bars
            last_progress_log_i = warmup_bars
            for i in range(warmup_bars, len(prices)):
                # Periodic progress signpost (per-ticker, INFO level).
                # Lets the operator confirm the loop is alive and see
                # roughly how far through the bars we are.
                if i - last_progress_log_i >= progress_every:
                    logger.info(
                        "[%s] Progress: bar %d/%d (%.0f%%), "
                        "%d entries so far this ticker, elapsed %.1fs",
                        ticker, i, len(prices),
                        100.0 * i / max(1, len(prices)),
                        len(all_trades) - ticker_entries_before,
                        time.monotonic() - ticker_t0,
                    )
                    last_progress_log_i = i

                if i - last_entry_idx < hold_bars:
                    continue
                if equity <= 0:
                    logger.info(
                        "[%s] Equity exhausted ($%.2f) — halting at bar %d",
                        ticker, equity, i,
                    )
                    break

                regime = self._classify_bars(prices, i, warmup_bars)

                # ── ETF macro signals (Items 1-3 parity) ───────────────
                # Apply the same gate ordering the live planner uses:
                # VIX inhibit (Priority 2) before leadership bias
                # (Priority 3) so a fear spike can demote a "leading"
                # SIDEWAYS to BEARISH.  Both gates are no-ops when
                # use_macro_signals is False or when the underlying
                # series isn't loaded.
                if self.use_macro_signals:
                    bar_ts = prices.index[i]
                    # VIX inter-market gate
                    if vix_series is not None:
                        vix_z_pair = self._vix_zscore_at(vix_series, bar_ts)
                        if (vix_z_pair is not None
                                and vix_z_pair[1] > VIX_INHIBIT_ZSCORE
                                and regime in ("bullish", "sideways")):
                            regime = "bearish"
                            self.vix_inhibited += 1
                    # Leadership bias (only meaningful when the regime
                    # would otherwise have been SIDEWAYS — bullish/bearish
                    # already have a directional signal so the live planner
                    # doesn't override them on RS alone).
                    if (regime == "sideways"
                            and ticker in LEADERSHIP_ANCHORS):
                        anchor = LEADERSHIP_ANCHORS[ticker]
                        anchor_series = anchor_series_by_anchor.get(anchor)
                        if anchor_series is not None:
                            lead_pair = self._leadership_zscore_at(
                                prices, anchor_series, i,
                            )
                            if (lead_pair is not None
                                    and lead_pair[1] > RS_ZSCORE_THRESHOLD):
                                regime = "bullish"
                                self.leadership_biased += 1

                strategy = self._strategy(regime)

                # --- Strike + credit: use option chain if available, else sigma/fixed ---
                strike_distance_pct: Optional[float] = None
                credit: float = 0.0
                approx_abs_delta: Optional[float] = None
                alpaca_historical_plan: Optional[Dict] = None

                # Try to fetch option chain for this entry date
                raw_date = prices.index[i]
                entry_dt = (
                    raw_date.date()
                    if hasattr(raw_date, "date") else start
                )

                # ── Alpaca historical (30-day live parity) ─────────────
                # When this mode is enabled, fetch a REAL historical
                # option chain + bars for this (ticker, entry_dt) and
                # derive the credit / strike / delta from actual Alpaca
                # data.  If the data isn't available (outside the
                # retention window, illiquid strike, etc.) we SKIP the
                # bar entirely — falling back to the synthetic σ-credit
                # would defeat the purpose of the mode.
                if self.use_alpaca_historical:
                    sigma_annual = self._realized_vol_annual(
                        prices, i, vol_window, bars_per_year,
                    )
                    underlying_px = float(prices.iloc[i])
                    alpaca_historical_plan, alpaca_reject_reason = (
                        self._alpaca_historical_plan(
                            ticker=ticker,
                            entry_date=entry_dt,
                            regime=regime,
                            # ``target_dte`` controls EXPIRATION pick; for
                            # intraday we still want a 30-40 DTE option to
                            # collect theta over the position lifetime.
                            target_dte=(
                                hold_bars if not is_intraday else self.target_dte
                            ),
                            underlying_price=underlying_px,
                            sigma_annual=sigma_annual,
                            effective_sigma_mult=(
                                effective_sigma_mult
                                if effective_sigma_mult > 0
                                else 1.0
                            ),
                            # ``hold_bars`` / ``bars_per_year`` control
                            # σ-DISTANCE projection.  Decoupling DTE from
                            # hold horizon is the core of the fix:
                            # intraday trades hold for ~1 hour even though
                            # the option lives for 30+ days, and σ should
                            # be projected over the actual hold.
                            hold_bars=hold_bars,
                            bars_per_year=bars_per_year,
                        )
                    )
                    if alpaca_historical_plan is None:
                        # Record the skip in the funnel so the UI shows
                        # why this bar produced no trade.  ``alpaca_reject_reason``
                        # is a structured "<token>: <context>" string from
                        # ``_alpaca_historical_plan``; we use the token before
                        # the first ``:`` as a sub-gate so per-cause counters
                        # show up separately in ``self.rejections`` instead of
                        # collapsing into one generic bucket.
                        self._funnel.considered += 1
                        sub_gate = (
                            alpaca_reject_reason.split(":", 1)[0].strip()
                            or "unspecified"
                        )
                        self._record_rejection(
                            ticker=ticker, entry_date=entry_dt,
                            gate=f"alpaca_historical:{sub_gate}",
                            phase="1. Real-data availability",
                            price=underlying_px, regime=regime,
                            strategy=strategy,
                            measured=None, threshold=None,
                            reason=(
                                alpaca_reject_reason
                                or "alpaca_historical: unspecified failure"
                            ),
                        )
                        continue
                    strike_distance_pct = alpaca_historical_plan["strike_distance_pct"]
                    credit = alpaca_historical_plan["credit"]
                    approx_abs_delta = alpaca_historical_plan["approx_abs_delta"]
                    logger.info(
                        "[%s] alpaca_historical: %s short=$%.2f long=$%.2f "
                        "credit=$%.2f |Δ|=%.3f (exp %s)",
                        ticker, alpaca_historical_plan["option_type"],
                        alpaca_historical_plan["short_strike"],
                        alpaca_historical_plan["long_strike"],
                        credit, approx_abs_delta,
                        alpaca_historical_plan["expiration"],
                    )
                    option_chain_data = (strike_distance_pct, credit, approx_abs_delta)
                else:
                    option_chain_data = None

                option_chain_error = None

                if (
                    alpaca_historical_plan is None
                    and self._alpaca_api_key and self._alpaca_secret_key
                    and not self.use_alpaca_historical   # snapshot-only when not in historical mode
                    and (date.today() - entry_dt).days <= self._SNAPSHOT_FRESH_DAYS
                ):
                    # Only use the current-snapshot chain for recent bars
                    # where "today's quotes" are a reasonable proxy for
                    # "as-of entry_dt quotes".  For older dates, stick
                    # with the σ-path to avoid applying today's IV to a
                    # months-old trade.
                    try:
                        option_chain_data = self._get_option_chain_for_date(
                            ticker, entry_dt, hold_bars
                         )
                        if option_chain_data is None:
                            option_chain_error = "Alpaca API returned None"
                    except Exception as exc:
                        option_chain_error = f"Alpaca API exception: {exc}"
                        logger.warning(
                            "[%s] Option chain fetch failed: %s — using sigma-based fallback",
                            ticker, option_chain_error
                         )

                if option_chain_data:
                    # Use real option chain data from Alpaca
                    strike_distance_pct, credit, approx_abs_delta = option_chain_data
                    logger.debug(
                        "[%s] Real option chain: strike_dist=%.2f%%, credit=%.2f, delta=%.3f",
                        ticker, strike_distance_pct * 100, credit, approx_abs_delta
                    )
                elif use_sigma_path:
                    # Use sigma-based calculation when API fails
                    sigma_annual = self._realized_vol_annual(
                        prices, i, vol_window, bars_per_year
                    )
                    strike_distance_pct = self._sigma_strike_distance(
                        sigma_annual, hold_bars, bars_per_year, effective_sigma_mult
                    )
                    # Degenerate σ (flat prices, insufficient history) — fall
                    # back to the legacy fixed-% OTM so we don't inadvertently
                    # place the strike on top of spot.
                    if strike_distance_pct <= 0.0:
                        strike_distance_pct = None
                        credit = self.spread_width * self.credit_pct
                        logger.debug(
                            "[%s] Degenerate σ (flat prices) — using fixed credit %.2f",
                            ticker, credit
                         )
                    else:
                        credit_frac = self._credit_from_sigma(effective_sigma_mult)
                        credit = self.spread_width * credit_frac
                        approx_abs_delta = self._delta_from_sigma_distance(
                            effective_sigma_mult
                         )
                        logger.debug(
                            "[%s] Sigma-based: σ=%.2f, strike_dist=%.2f%%, credit=%.2f, delta=%.3f",
                            ticker, sigma_annual, strike_distance_pct * 100, credit, approx_abs_delta
                         )
                else:
                    # Legacy fixed-% OTM (only when sigma_path is disabled)
                    credit = self.spread_width * self.credit_pct
                    logger.warning(
                        "[%s] Using legacy fixed-% OTM (credit=%.2f) — Alpaca API unavailable "
                         "and sigma_path disabled",
                        ticker, credit
                     )

                max_loss = self.spread_width - credit

                # ── Agent-parity entry gates (in the agent's own order) ────
                # Each gate below mirrors a specific live-agent phase.  The
                # ``_record_rejection`` calls capture per-candidate context
                # so the UI can show *why* a bar was dropped, not just a
                # count.  Phase labels below align with the live agent's
                # pipeline:
                #
                #   1. Candidate considered          (every bar past warmup)
                #   2. Event-risk gate               (EarningsCalendar Tier-0)
                #   3. IV-rank gate                  (RegimeClassifier.high_iv)
                #   4. Max-Δ gate                    (StrategyPlanner.max_delta)
                #   5. Credit-ratio gate             (StrategyPlanner.min_credit_ratio)
                #   6. Max-risk gate                 (RiskManager.max_risk_pct)
                #   7. Position simulated            (→ ExitMonitor loop)
                self._funnel.considered += 1

                raw_date_for_gate = prices.index[i]
                entry_dt = (
                    raw_date_for_gate.date()
                    if hasattr(raw_date_for_gate, "date") else start
                )
                cur_price = float(prices.iloc[i])

                # 2. Event risk: earnings calendar (sentiment_pipeline Tier-0).
                if self._earnings_blocks_entry(ticker, entry_dt):
                    self._record_rejection(
                        ticker=ticker, entry_date=entry_dt,
                        gate="earnings_window",
                        phase="2. Event-risk gate (EarningsCalendar)",
                        price=cur_price, regime=regime, strategy=strategy,
                        measured=None,
                        threshold=float(self.earnings_lookahead_days),
                        reason=(
                            f"Scheduled earnings for {ticker} within "
                            f"{self.earnings_lookahead_days} trading days — "
                            "matches live agent's Tier-0 short-circuit."
                        ),
                    )
                    continue
                self._funnel.after_earnings += 1

                # 3. IV-rank high-vol guard (regime.high_iv_warning).
                if self.use_iv_gate:
                    iv_rank = self._iv_rank_from_returns(prices, i, vol_window)
                    if iv_rank > self.iv_high_threshold:
                        self._record_rejection(
                            ticker=ticker, entry_date=entry_dt,
                            gate="iv_rank>threshold",
                            phase="3. IV-rank gate (RegimeClassifier)",
                            price=cur_price, regime=regime, strategy=strategy,
                            measured=float(iv_rank),
                            threshold=float(self.iv_high_threshold),
                            reason=(
                                f"Realized-vol percentile {iv_rank:.1f} > "
                                f"{self.iv_high_threshold:.1f} threshold — "
                                "live agent suppresses entries when "
                                "RegimeAnalysis.high_iv_warning fires."
                            ),
                        )
                        continue
                self._funnel.after_iv_rank += 1

                # 4. Max-delta clamp on short leg (StrategyPlanner + RiskManager).
                if (
                    self.max_delta is not None
                    and approx_abs_delta is not None
                    and approx_abs_delta > self.max_delta
                ):
                    self._record_rejection(
                        ticker=ticker, entry_date=entry_dt,
                        gate="|delta|>max_delta",
                        phase="4. Max-Δ gate (StrategyPlanner)",
                        price=cur_price, regime=regime, strategy=strategy,
                        measured=float(approx_abs_delta),
                        threshold=float(self.max_delta),
                        reason=(
                            f"Estimated |Δ|={approx_abs_delta:.3f} > "
                            f"{self.max_delta:.2f} → short strike too close "
                            f"to spot (σ-distance too small) to hit the "
                            "agent's POP≥80% target."
                        ),
                    )
                    continue
                self._funnel.after_max_delta += 1

                # 5. Credit-to-width ratio floor (StrategyPlanner.min_credit_ratio).
                #
                # IMPORTANT parity caveat — this gate is designed to reject
                # *bad real Alpaca quotes* (stale chains, wide spreads,
                # missing data).  When the backtester is using its synthetic
                # ``_credit_from_sigma`` model, the credit is a deterministic
                # function of σ-distance and *cannot* exhibit the data-
                # quality failures the gate exists to catch.  Worse, the
                # model's calibration (0.45 − 0.15·σ) puts credits below the
                # 0.33 agent floor for any σ ≳ 0.8 — so leaving the gate on
                # silently rejects nearly every candidate.
                #
                # Resolution: skip the gate when credit is **synthetic σ
                # only** (i.e. σ-path strike-picking AND no real Alpaca
                # chain).  Keep it active when:
                #   * the user opts into a fixed-credit backtest
                #     (``use_sigma_model=False``) where credit_pct *is*
                #     a quote-like input the user might mis-set; or
                #   * Alpaca-historical mode is on and the credit is
                #     derived from real bar closes (in which case the
                #     gate's data-quality rationale fully applies and
                #     the synthetic-calibration argument does not).
                #
                # Bug history (Apr 2026): this branch used to skip on
                # ``not use_sigma_path`` alone, which silently disabled
                # the gate in alpaca-historical+σ mode and let trades
                # with C/W < 0.10 reach the trade journal even though
                # ``min_credit_ratio=0.33`` was configured.  See
                # ``TestCreditRatioGateAlpacaHistorical``.
                credit_is_synthetic = use_sigma_path and not self.use_alpaca_historical
                credit_to_width = credit / self.spread_width if self.spread_width else 0.0
                if self.min_credit_ratio is not None and credit_is_synthetic:
                    # Record the skip exactly once per run so the funnel
                    # reflects that the gate is intentionally bypassed.
                    if "credit_ratio" not in self._funnel.skipped_phases:
                        self._funnel.skipped_phases.append("credit_ratio")
                if (
                    self.min_credit_ratio is not None
                    and not credit_is_synthetic      # only skip when σ-credit synthetic
                    and credit_to_width < self.min_credit_ratio
                ):
                    self._record_rejection(
                        ticker=ticker, entry_date=entry_dt,
                        gate="credit_ratio<floor",
                        phase="5. Credit-ratio gate (StrategyPlanner)",
                        price=cur_price, regime=regime, strategy=strategy,
                        measured=float(credit_to_width),
                        threshold=float(self.min_credit_ratio),
                        reason=(
                            f"Credit/width {credit_to_width:.3f} < "
                            f"{self.min_credit_ratio:.2f} floor — "
                            "live agent rejects quotes with insufficient "
                            "premium relative to max-loss."
                        ),
                    )
                    continue
                self._funnel.after_credit_ratio += 1

                # 6. Max-risk-per-trade (RiskManager.max_risk_pct × equity).
                if self.max_risk_pct is not None:
                    allowed_loss = equity * self.max_risk_pct
                    if max_loss * 100.0 > allowed_loss:
                        self._record_rejection(
                            ticker=ticker, entry_date=entry_dt,
                            gate="max_loss>risk_budget",
                            phase="6. Max-risk gate (RiskManager)",
                            price=cur_price, regime=regime, strategy=strategy,
                            measured=float(max_loss * 100.0),
                            threshold=float(allowed_loss),
                            reason=(
                                f"Max-loss ${max_loss*100:.2f} > "
                                f"{self.max_risk_pct*100:.1f}% of ${equity:.2f} "
                                f"equity (=${allowed_loss:.2f}) — live agent "
                                "sizes trades against this same cap."
                            ),
                        )
                        continue
                self._funnel.after_max_risk += 1

                otm_pct = INTRADAY_OTM_PCT if is_intraday else DAILY_OTM_PCT

                # ── Live Quote Refresh (Phase VI) ─────────────────────
                # Mirror the live agent's executor._refresh_limit_price() pattern:
                # fetch fresh option quotes immediately before execution and
                # re-validate economics-bearing guardrails.  This prevents
                # simulating trades on stale quotes that would be rejected
                # in live trading.
                #
                # Two gates govern when this stage actually runs:
                #
                #   1. NOT in alpaca_historical mode.  When historical mode
                #      is on, the planning stage already fetched the REAL
                #      option chain for entry_dt — there is no truer quote
                #      we could refresh against.  Re-fetching today's
                #      snapshot here would *replace* honest historical
                #      economics with current-market quotes and produce
                #      garbage drift warnings (e.g. $6 credits on $5-wide
                #      spreads when the underlying has moved since entry).
                #
                #   2. Entry date within _SNAPSHOT_FRESH_DAYS.  Outside
                #      that window, "today's quote" is structurally
                #      meaningless as a proxy for "the quote at entry_dt"
                #      — same reason the snapshot *planning* path is
                #      disabled for stale dates (line ~2271 below).
                #
                # If you change either gate, update the corresponding
                # tests in test_backtest_ui.py::TestRefreshGating.
                refresh_age_days = (date.today() - entry_dt).days
                refresh_eligible = (
                    self._alpaca_api_key
                    and self._alpaca_secret_key
                    and not self.use_alpaca_historical
                    and refresh_age_days <= self._SNAPSHOT_FRESH_DAYS
                )
                if refresh_eligible:
                    (
                        refreshed_strike_distance,
                        refreshed_credit,
                        refreshed_delta,
                        refresh_status,
                    ) = self._refresh_live_quotes(
                        ticker=ticker,
                        entry_date=entry_dt,
                        strategy=strategy,
                        strike_distance_pct=strike_distance_pct,
                        credit=credit,
                        approx_abs_delta=approx_abs_delta,
                        equity=equity,
                    )
                    
                    if refresh_status in ("rejected_credit_ratio", "rejected_max_loss"):
                        # Guardrail failed — skip this trade
                        self._record_rejection(
                            ticker=ticker,
                            entry_date=entry_dt,
                            gate=f"live_quote_{refresh_status}",
                            phase="6. Live Quote Refresh (Executor)",
                            price=cur_price,
                            regime=regime,
                            strategy=strategy,
                            measured=refreshed_credit if refreshed_credit else None,
                            threshold=(
                                self.min_credit_ratio * self.spread_width
                                if refresh_status == "rejected_credit_ratio"
                                else equity * self.max_risk_pct
                            ),
                            reason=(
                                f"Live quote refresh {refresh_status} — "
                                f"trade would be rejected in live trading"
                            ),
                        )
                        continue
                    elif refresh_status == "drift_warning":
                        # Log significant drift but continue with refreshed values
                        logger.info(
                            "[%s] Live quote drift detected: %s — using refreshed values",
                            ticker, refresh_status,
                        )
                    
                    # Use refreshed values if available
                    if refreshed_credit is not None:
                        credit = refreshed_credit
                        if refreshed_strike_distance is not None:
                            strike_distance_pct = refreshed_strike_distance
                        if refreshed_delta is not None:
                            approx_abs_delta = refreshed_delta
                    
                    logger.debug(
                        "[%s] Live quote refresh: %s (credit: $%.2f→$%.2f)",
                        ticker, refresh_status, credit, credit,
                    )
                
                if alpaca_historical_plan is not None:
                    # Real-data simulation — use Alpaca option bars for
                    # the exit value instead of the synthetic payoff model.
                    outcome, pnl, hold_count = self._simulate_alpaca_historical(
                        plan=alpaca_historical_plan,
                        entry_date=entry_dt,
                        hold_bars=hold_bars,
                        underlying_prices=prices,
                        entry_idx=i,
                        loss_cut_multiplier=self.loss_cut_multiplier,
                        stop_loss_pct=self.stop_loss_pct,
                    )
                else:
                    outcome, pnl, hold_count = self._simulate(
                        prices, i, regime, credit, hold_bars, otm_pct,
                        strike_distance_pct=strike_distance_pct,
                        loss_cut_multiplier=self.loss_cut_multiplier,
                        stop_loss_pct=self.stop_loss_pct,
                    )
                equity = round(equity + pnl, 2)
                last_entry_idx = i

                raw_date = prices.index[i]
                entry_date = raw_date.date() if hasattr(raw_date, "date") else start

                # For intraday, hold_days represents bars (each = 5 min), not calendar days.
                #
                # Trade-journal expiry_date contract
                # ----------------------------------
                # Bug history (Apr 2026): this column used to record
                # ``entry_date + 1 day`` for every intraday trade,
                # making it look like the agent was trading 0-DTE
                # options when the planner actually picks ~30-DTE
                # contracts (TARGET_DTE=35).  The value should reflect
                # the OPTION's expiration, not the holding-period exit.
                #
                # Resolution:
                #   * alpaca-historical mode → use the actual exp from
                #     the plan dict (real OCC contract expiration).
                #   * synthetic-σ daily mode → ``entry+hold_count`` is
                #     a reasonable proxy because the synthetic model
                #     has no notion of contract expiration distinct
                #     from the hold horizon.
                #   * synthetic-σ intraday mode → fall back to
                #     ``entry + target_dte`` (the planner's intended
                #     DTE), not entry+1, so the journal column reflects
                #     the option being modeled, not the 1-hour hold.
                if alpaca_historical_plan is not None:
                    try:
                        plan_expiry = date.fromisoformat(
                            alpaca_historical_plan["expiration"]
                        )
                    except (KeyError, ValueError, TypeError):
                        plan_expiry = entry_date + timedelta(days=self.target_dte)
                    expiry_for_journal = plan_expiry
                elif is_intraday:
                    expiry_for_journal = entry_date + timedelta(days=self.target_dte)
                else:
                    expiry_for_journal = entry_date + timedelta(days=hold_count)

                all_trades.append(
                    SimTrade(
                        ticker=ticker,
                        strategy=strategy,
                        regime=regime,
                        entry_date=entry_date,
                        expiry_date=expiry_for_journal,
                        credit=credit,
                        max_loss=max_loss,
                        outcome=outcome,
                        pnl=pnl,
                        hold_days=hold_count,
                    )
                )
                equity_curve.append(
                    {"timestamp": pd.Timestamp(raw_date), "account_balance": equity}
                )
                self._funnel.simulated += 1

            # Per-ticker completion signpost — counterpart to the per-ticker
            # startup line above.  Lets the operator see the sweep finish
            # ticker-by-ticker rather than waiting for the whole run.
            ticker_entries = len(all_trades) - ticker_entries_before
            logger.info(
                "[%s] Done: %d entries, equity now $%.2f, ticker took %.1fs",
                ticker, ticker_entries, equity,
                time.monotonic() - ticker_t0,
            )

        # Patch the funnel for gates that were intentionally skipped this
        # run (e.g. credit_ratio under σ-path).  Without this, the UI
        # funnel would show an artificial drop from max_delta → credit_ratio
        # of zero "survivors" even though no rejection happened.
        if "credit_ratio" in self._funnel.skipped_phases:
            self._funnel.after_credit_ratio = self._funnel.after_max_delta

        # Surface rejection counts so the UI can show why candidates were
        # dropped by the agent-parity gates.  Keeps "skipped" human-readable.
        for reason, n in sorted(self.rejections.items()):
            skipped.append(f"{n} candidate(s) rejected by gate: {reason}")

        # Surface rate-limit / call-volume stats so the user can tell
        # at a glance whether the run was throttled.
        limiter_stats = _ALPACA_RATE_LIMITER.stats()
        skipped.append(
            f"Alpaca calls: {self.alpaca_calls_made} "
            f"(429 retries: {self.alpaca_429_hits}, "
            f"rate-limit sleeps: {limiter_stats['sleep_count']} "
            f"totaling {limiter_stats['total_sleep_s']:.1f}s, "
            f"failures after retry: {self.alpaca_failures}, "
            f"budget: {limiter_stats['budget_rpm']}/min)"
        )

        # Snapshot diagnostics for the result.  Copy (not alias) so a
        # subsequent .run() on the same Backtester instance doesn't mutate
        # an already-returned BacktestResult.
        rejection_counts_snap = dict(self.rejections)
        rejection_samples_snap = list(self._rejection_samples)
        funnel_snap = PhaseFunnel(
            considered=self._funnel.considered,
            after_earnings=self._funnel.after_earnings,
            after_iv_rank=self._funnel.after_iv_rank,
            after_max_delta=self._funnel.after_max_delta,
            after_credit_ratio=self._funnel.after_credit_ratio,
            after_max_risk=self._funnel.after_max_risk,
            simulated=self._funnel.simulated,
            skipped_phases=list(self._funnel.skipped_phases),
        )

        if not all_trades:
            empty_trades = pd.DataFrame(
                columns=["ticker", "strategy", "regime", "entry_date",
                         "expiry_date", "credit", "max_loss", "outcome", "pnl", "hold_days"]
            )
            empty_eq = pd.DataFrame(
                [{"timestamp": pd.Timestamp(start), "account_balance": self.starting_equity}]
            )
            return BacktestResult(
                trades=empty_trades,
                equity_curve=empty_eq,
                metrics=self._metrics([], self.starting_equity),
                regime_stats=pd.DataFrame(columns=["regime", "pnl", "trade_count"]),
                skipped=skipped,
                funnel=funnel_snap,
                rejection_counts=rejection_counts_snap,
                rejection_samples=rejection_samples_snap,
            )

        trades_df = pd.DataFrame([vars(t) for t in all_trades])
        equity_df = pd.DataFrame(equity_curve)

        return BacktestResult(
            trades=trades_df,
            equity_curve=equity_df,
            metrics=self._metrics(all_trades, self.starting_equity),
            regime_stats=self._regime_stats(trades_df),
            skipped=skipped,
            funnel=funnel_snap,
            rejection_counts=rejection_counts_snap,
            rejection_samples=rejection_samples_snap,
        )

    # ── Stats helpers ───────────────────────────────────────────────────────

    def _metrics(self, trades: List[SimTrade], starting: float) -> Dict:
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe": 0.0,
                "avg_hold_days": 0.0,
            }
        wins = [t for t in trades if t.outcome == "win"]
        losses = [t for t in trades if t.outcome == "loss"]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

        pnl_arr = np.array([t.pnl for t in trades])
        sharpe = (
            float(pnl_arr.mean() / pnl_arr.std() * np.sqrt(252))
            if pnl_arr.std() > 0
            else 0.0
        )

        # Max drawdown from trade-by-trade equity
        eq = starting
        peak = starting
        max_dd = 0.0
        for t in sorted(trades, key=lambda x: x.entry_date):
            eq += t.pnl
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        return {
            "total_trades": len(trades),
            "win_rate": round(len(wins) / len(trades) * 100, 1),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe": round(sharpe, 2),
            "avg_hold_days": round(sum(t.hold_days for t in trades) / len(trades), 1),
        }

    @staticmethod
    def _regime_stats(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["regime", "pnl", "trade_count"])
        return (
            df.groupby("regime")
            .agg(pnl=("pnl", "sum"), trade_count=("pnl", "count"))
            .reset_index()
        )


# ---------------------------------------------------------------------------
# Cached runner
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running backtest…")
def _run_cached(
    tickers: tuple,
    start: date,
    end: date,
    timeframe: str,
    use_alpaca: bool,
    sigma_mult: Optional[float] = None,
    loss_cut_multiplier: Optional[float] = None,
    # --- Agent-parity gates (propagated from the UI panel) ---
    min_credit_ratio: Optional[float] = None,
    max_delta: Optional[float] = None,
    max_risk_pct: Optional[float] = None,
    stop_loss_pct: Optional[float] = None,
    use_iv_gate: bool = False,
    use_earnings_gate: bool = False,
    earnings_lookahead_days: int = AGENT_EARNINGS_LOOKAHEAD,
    use_alpaca_historical: bool = False,
    use_macro_signals: bool = False,
    # Unified Decision Engine — preset is passed by *name* (str, hashable)
    # so the @st.cache_data wrapper stays happy. We resolve the actual
    # PresetConfig inside this function.
    preset_name: Optional[str] = None,
    use_unified_engine: bool = False,
) -> BacktestResult:
    preset_obj = None
    if use_unified_engine and preset_name:
        try:
            from trading_agent.strategy_presets import PRESETS
            preset_obj = PRESETS.get(preset_name)
        except Exception:
            preset_obj = None
    return Backtester(
        sigma_mult=sigma_mult,
        loss_cut_multiplier=loss_cut_multiplier,
        min_credit_ratio=min_credit_ratio,
        max_delta=max_delta,
        max_risk_pct=max_risk_pct,
        stop_loss_pct=stop_loss_pct,
        use_iv_gate=use_iv_gate,
        use_earnings_gate=use_earnings_gate,
        earnings_lookahead_days=earnings_lookahead_days,
        use_alpaca_historical=use_alpaca_historical,
        use_macro_signals=use_macro_signals,
        preset=preset_obj,
        use_unified_engine=use_unified_engine and preset_obj is not None,
    ).run(list(tickers), start, end, timeframe, use_alpaca)


# ---------------------------------------------------------------------------
# Journal export helper
# ---------------------------------------------------------------------------

def _export_to_journal(result: BacktestResult) -> None:
    try:
        from trading_agent.journal_kb import JournalKB
        # ``run_mode="backtest"`` routes the export to
        # ``signals_backtest.jsonl`` so backtest summaries never mingle
        # with the live ``signals_live.jsonl`` corpus that the LLM and
        # diagnostics panels consume.
        journal = JournalKB(journal_dir="trade_journal", run_mode="backtest")
        m = result.metrics
        journal.log_signal(
            ticker="BACKTEST",
            action="dry_run",
            price=0.0,
            raw_signal={
                "regime": "backtest",
                "strategy": "Backtest Export",
                "plan_valid": True,
                "risk_approved": False,
                "account_balance": STARTING_EQUITY,
                "checks_passed": [],
                "checks_failed": [],
                "backtest_metrics": m,
                "backtest_trade_count": len(result.trades),
            },
            notes=(
                f"Backtest: {m['total_trades']} trades, "
                f"WR={m['win_rate']}%, Sharpe={m['sharpe']:.2f}"
            ),
        )
        st.success("Backtest summary exported to trade journal.")
    except Exception as exc:
        st.error(f"Journal export failed: {exc}")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_backtest_ui() -> None:
    """Render the Backtesting tab."""

    # ── Two-column layout: settings panel (left) + results (right) ────────
    # All controls are in the left column, never in st.sidebar, so they only
    # appear when the Backtesting tab is active.
    settings_col, results_col = st.columns([1, 3], gap="large")

    with settings_col:
        st.subheader("Settings")

        timeframe = st.selectbox(
            "Timeframe", options=["1Day", "5Min"], index=0, key="bt_timeframe"
        )
        is_intraday = timeframe == "5Min"

        if is_intraday:
            st.info(
                f"Yahoo Finance only serves 5m data for the last "
                f"{INTRADAY_MAX_DAYS} days. Start date is auto-clamped.",
                icon="⚠️",
            )

        # Reset defaults when timeframe changes (also seeds session state
        # on first render, since prev_tf is None ≠ timeframe).
        prev_tf = st.session_state.get("_bt_prev_timeframe")
        if prev_tf != timeframe:
            st.session_state["_bt_prev_timeframe"] = timeframe
            st.session_state.pop("backtest_result", None)
            st.session_state.pop("backtest_timeframe", None)
            st.session_state["bt_start_date"] = (
                date.today() - timedelta(days=INTRADAY_MAX_DAYS)
                if is_intraday else DEFAULT_START
            )

        # NOTE: do NOT pass ``value=`` here — the widget's key is bound to
        # ``st.session_state["bt_start_date"]``, which the branch above
        # populates. Passing both ``value=`` and a populated session-state
        # key triggers Streamlit's "default value but also set via Session
        # State API" warning.
        start_date = st.date_input(
            "Start Date",
            key="bt_start_date",
        )
        end_date = st.date_input("End Date", value=DEFAULT_END, key="bt_end_date")
        tickers = st.multiselect(
            "Tickers", options=ALL_TICKERS, default=DEFAULT_TICKERS, key="bt_tickers"
        )
        use_alpaca = st.toggle(
            "Use Alpaca Data",
            value=False,
            key="bt_use_alpaca",
            help="Prefer Alpaca historical bars over yfinance (requires ALPACA_API_KEY and ALPACA_SECRET_KEY in .env).",
        )

        use_alpaca_historical = st.toggle(
            "Match live trading (real Alpaca options, ~30d)",
            value=False,
            key="bt_use_alpaca_historical",
            help=(
                "Use REAL historical Alpaca option bars for both entry "
                "credit AND exit P&L. Mirrors live-trading economics "
                "exactly for dates inside Alpaca's options retention "
                "window (~30 days).\n\n"
                "When this is ON, the synthetic σ-credit model is "
                "bypassed. Bars that don't have Alpaca option data are "
                "skipped and reported in the funnel as "
                "'alpaca_historical_no_data' — that's expected for dates "
                "outside the retention window.\n\n"
                "Requires ALPACA_API_KEY / ALPACA_SECRET_KEY in .env."
            ),
        )
        if use_alpaca_historical:
            st.caption(
                "Real-data mode: set Start Date within the last ~30 days "
                "for meaningful coverage."
            )

        # ── Strategy-realism knobs ────────────────────────────────────────
        st.markdown("**Strategy Realism**")

        use_sigma = st.toggle(
            "Sigma-based strikes",
            value=True,
            key="bt_use_sigma",
            help=(
                "Place short strikes at N × realized σ projected over the "
                "hold window (delta-proxy). Credit is scaled down as strikes "
                "move further OTM — prevents the fixed-% free-lunch bias. "
                "Off = legacy fixed 0.5% / 3% OTM."
            ),
        )
        if use_sigma:
            default_sigma = (
                DEFAULT_SIGMA_MULT_INTRADAY if is_intraday else DEFAULT_SIGMA_MULT_DAILY
            )
            sigma_mult_value: Optional[float] = st.slider(
                "σ multiplier (short strike distance)",
                min_value=0.5, max_value=3.0,
                value=float(default_sigma), step=0.1,
                key="bt_sigma_mult",
                help=(
                    "1.0σ ≈ 16Δ (aggressive), 1.5σ ≈ 7Δ (balanced), "
                    "2.0σ ≈ 2Δ (conservative, tiny credits)."
                ),
            )
        else:
            sigma_mult_value = 0.0   # explicit disable → legacy fixed-% OTM

        use_loss_cut = st.toggle(
            "Early loss cut",
            value=True,
            key="bt_use_loss_cut",
            help=(
                "On first breach of the short strike, close at "
                "−k × credit instead of the full max-loss. Reshapes "
                "loss:win from ~4.9:1 to ~4.0:1 and drops breakeven "
                "win rate from 83% to ~80%."
            ),
        )
        if use_loss_cut:
            loss_cut_value: Optional[float] = st.slider(
                "Loss-cut multiplier (× credit)",
                min_value=1.0, max_value=4.0,
                value=float(AGENT_HARD_STOP_MULT), step=0.25,
                key="bt_loss_cut_mult",
                help=(
                    f"Matches the agent's hard_stop_multiplier (default "
                    f"{AGENT_HARD_STOP_MULT:.1f}×). The legacy "
                    f"{DEFAULT_LOSS_CUT_MULTIPLIER:.1f}× breakeven-shaper "
                    "is still available — just drag the slider."
                ),
            )
        else:
            loss_cut_value = None

        # ── Agent-parity gates ────────────────────────────────────────────
        # Default-ON so interactive backtests use the same entry filters,
        # risk budget, and exit thresholds as the live agent.
        st.markdown("**Agent Parity**")
        agent_parity = st.toggle(
            "Enforce agent-parity filters",
            value=True,
            key="bt_agent_parity",
            help=(
                "When enabled, the backtester applies the live agent's "
                "entry gates and exit rules with their production defaults:\n\n"
                f"• Min credit/width ≥ {AGENT_MIN_CREDIT_RATIO} "
                "(StrategyPlanner) — auto-skipped when σ-credit model "
                "is active (synthetic credit can't fail this gate the way "
                "bad real quotes can)\n"
                f"• Short-leg |Δ| ≤ {AGENT_MAX_DELTA} "
                "(StrategyPlanner, ≈80% POP)\n"
                f"• Max loss ≤ {AGENT_MAX_RISK_PCT*100:.0f}% of equity "
                "(RiskManager)\n"
                f"• Exit at min(hard_stop × {AGENT_HARD_STOP_MULT}×credit, "
                f"stop_loss × {AGENT_STOP_LOSS_PCT*100:.0f}%×max-loss) "
                "(ExitMonitor)\n"
                "• Skip entries inside the earnings-calendar lookahead\n"
                "• Skip entries when IV-rank > 95 "
                "(RegimeClassifier.high_iv_warning)\n"
                "• Apply leadership Z-score bias (>+1.5σ) and VIX "
                "inter-market gate (>+2σ) — Items 1-3 of the ETF macro "
                "patch (Backtester.use_macro_signals)"
            ),
        )
        if agent_parity:
            min_credit_ratio_val: Optional[float] = AGENT_MIN_CREDIT_RATIO
            max_delta_val: Optional[float] = AGENT_MAX_DELTA
            max_risk_pct_val: Optional[float] = AGENT_MAX_RISK_PCT
            stop_loss_pct_val: Optional[float] = AGENT_STOP_LOSS_PCT
            use_iv_gate_val = True
            use_earnings_gate_val = True
            earnings_lookahead_val = AGENT_EARNINGS_LOOKAHEAD
            use_macro_signals_val = True
        else:
            min_credit_ratio_val = None
            max_delta_val = None
            max_risk_pct_val = None
            stop_loss_pct_val = None
            use_iv_gate_val = False
            use_earnings_gate_val = False
            earnings_lookahead_val = AGENT_EARNINGS_LOOKAHEAD
            use_macro_signals_val = False

        # ── Unified Decision Engine ───────────────────────────────────────
        # Routes the backtest's strike-selection through the same
        # ``decision_engine.decide()`` function the live agent calls. When
        # OFF the legacy %-OTM / sigma-distance picker is used (kept for
        # apples-to-apples comparison against historical runs).
        st.markdown("**Unified Decision Engine**")
        try:
            from trading_agent.strategy_presets import (
                PRESETS as _PRESETS,
                load_active_preset as _load_active_preset,
            )
            _preset_choices = list(_PRESETS.keys())
            try:
                _default_preset_name = _load_active_preset().name
            except Exception:
                _default_preset_name = "balanced"
            _default_idx = (_preset_choices.index(_default_preset_name)
                            if _default_preset_name in _preset_choices else 0)
            _presets_loaded = True
        except Exception:
            _preset_choices = ["balanced"]
            _default_idx = 0
            _presets_loaded = False

        use_unified_engine_val = st.toggle(
            "Use unified decision engine",
            value=False,
            key="bt_use_unified_engine",
            disabled=not _presets_loaded,
            help=(
                "When ON, the backtester picks strikes via "
                "``decision_engine.decide()`` — the same function the live "
                "agent calls — using the preset selected below. This makes "
                "the backtest a true dry-run of live behaviour: same C/W "
                "floor, same |Δ|×(1+edge_buffer) rule, same scoring.\n\n"
                "When OFF, the backtester uses its legacy %-OTM / "
                "sigma-distance heuristic (kept for back-compat with "
                "earlier runs).\n\n"
                "**Requires** ``use_alpaca_historical=True`` — the unified "
                "path needs real option contracts + bars to synthesize the "
                "chain."
            ),
        )
        if use_unified_engine_val and _presets_loaded:
            preset_name_val: Optional[str] = st.selectbox(
                "Preset",
                options=_preset_choices,
                index=_default_idx,
                key="bt_unified_preset",
                help=(
                    "Strategy preset whose `dte_grid`, `delta_grid`, "
                    "`width_grid_pct`, `edge_buffer`, and `min_pop` will "
                    "drive the unified scanner."
                ),
            )
            if not use_alpaca_historical:
                st.warning(
                    "⚠️  Unified engine requires `use_alpaca_historical=ON` "
                    "(it builds the synth chain from real Alpaca contracts). "
                    "Enable Alpaca-historical above or the toggle has no "
                    "effect."
                )
        else:
            preset_name_val = None

        st.divider()
        run_btn = st.button("Run Backtest", type="primary", width='stretch')

    # ── Results rendered in the right column ───────────────────────────────
    with results_col:
        st.subheader("Results")

        if not run_btn and "backtest_result" not in st.session_state:
            if is_intraday:
                st.info(
                    "**5-Min timeframe selected.**\n\n"
                    f"Yahoo Finance only provides 5-minute bars for the last {INTRADAY_MAX_DAYS} days. "
                    "Start date has been set automatically. Click **Run Backtest** to continue.\n\n"
                    "Use **1Day** timeframe for multi-year historical analysis."
                )
            else:
                st.info("Configure parameters on the left and click **Run Backtest**.")
            return

        if run_btn:
            if not tickers:
                st.error("Select at least one ticker.")
                return
            if start_date >= end_date:
                st.error("Start date must be before end date.")
                return
            # ── User-visible run feedback ──────────────────────────────────
            # Without the spinner the UI sits frozen for the entire run
            # (often many minutes in alpaca_historical mode), making a
            # healthy run indistinguishable from a hang.  The status
            # caption echoes the run config so the operator can sanity-
            # check what's actually executing.
            mode_label = (
                "alpaca-historical (real chains, ~30-day window)"
                if use_alpaca_historical
                else (
                    "snapshot+sigma (synthetic credit)" if sigma_mult_value
                    else "fixed-%-OTM (legacy)"
                )
            )
            run_caption = (
                f"Running backtest · {len(tickers)} ticker(s) · "
                f"{start_date} → {end_date} · {timeframe} · "
                f"mode={mode_label}"
                + (" · macro-signals=ON" if use_macro_signals_val else "")
                + (" · agent-parity=ON" if agent_parity else "")
                + (f" · unified-engine=ON (preset={preset_name_val})"
                   if (use_unified_engine_val and preset_name_val) else "")
            )
            run_status = st.empty()
            run_status.info(
                f"⏳ {run_caption}\n\n"
                "Per-ticker progress is logged to the terminal Streamlit was "
                "launched from (LOG_LEVEL=INFO by default). In `use_alpaca_"
                "historical` mode this can take several minutes per ticker "
                "due to the 180 req/min Alpaca rate limit — see the terminal "
                "for live signposts."
            )
            run_t0 = time.monotonic()
            with st.spinner(run_caption):
                result = _run_cached(
                    tuple(sorted(tickers)), start_date, end_date, timeframe, use_alpaca,
                    sigma_mult=sigma_mult_value,
                    loss_cut_multiplier=loss_cut_value,
                    min_credit_ratio=min_credit_ratio_val,
                    max_delta=max_delta_val,
                    max_risk_pct=max_risk_pct_val,
                    stop_loss_pct=stop_loss_pct_val,
                    use_iv_gate=use_iv_gate_val,
                    use_earnings_gate=use_earnings_gate_val,
                    earnings_lookahead_days=earnings_lookahead_val,
                    use_alpaca_historical=use_alpaca_historical,
                    use_macro_signals=use_macro_signals_val,
                    preset_name=preset_name_val,
                    use_unified_engine=use_unified_engine_val,
                )
            elapsed_s = time.monotonic() - run_t0
            n_trades = 0 if result is None or result.trades is None else len(result.trades)
            run_status.success(
                f"✅ Backtest complete in {elapsed_s:.1f}s — "
                f"{n_trades} trade(s) simulated. See terminal log for details."
            )
            st.session_state["backtest_result"] = result
            st.session_state["backtest_timeframe"] = timeframe
            st.session_state["backtest_config"] = {
                "sigma_mult": sigma_mult_value,
                "use_alpaca_historical": use_alpaca_historical,
                "loss_cut_multiplier": loss_cut_value,
                "agent_parity": agent_parity,
            }

        result: Optional[BacktestResult] = st.session_state.get("backtest_result")
        if result is None:
            return

        result_timeframe = st.session_state.get("backtest_timeframe", timeframe)
        is_result_intraday = result_timeframe == "5Min"

        m = result.metrics

        # ── Skipped / clamped warnings ─────────────────────────────────────
        if result.skipped:
            for msg in result.skipped:
                st.warning(msg)

        if result.trades.empty:
            if is_result_intraday:
                st.error(
                    "No trades were simulated for 5-Min timeframe. Possible causes:\n\n"
                    "- **Yahoo Finance 30-day limit** — 5-minute data is only available "
                    f"for the last {INTRADAY_MAX_DAYS} days. Dates older than that return empty data.\n"
                    "- **Not enough bars** — the intraday backtester needs at least "
                    f"{INTRADAY_WARMUP_BARS + INTRADAY_HOLD_BARS + 1} bars "
                    f"({INTRADAY_WARMUP_BARS}-bar warmup + {INTRADAY_HOLD_BARS}-bar hold).\n"
                    "- **Try 1Day timeframe** for multi-year historical analysis."
                )
            else:
                st.error(
                    "No trades were simulated. The most common cause is a date range that is "
                    "too short — the backtester needs at least **201 daily bars** (≈ 1 year) "
                    "to compute the SMA-200 warmup before it can place the first trade. "
                    "Try setting Start Date to at least 1 year ago."
                )
            return

        # ── Active-config caption (so numbers can be tied to knobs) ───────
        cfg = st.session_state.get("backtest_config", {})
        cfg_sigma = cfg.get("sigma_mult")
        cfg_lc = cfg.get("loss_cut_multiplier")
        cfg_parity = cfg.get("agent_parity", False)
        sigma_label = (
            f"σ×{cfg_sigma:.1f}" if cfg_sigma and cfg_sigma > 0 else "fixed-%-OTM"
        )
        lc_label = (
            f"loss-cut@{cfg_lc:.1f}×credit" if cfg_lc else "full-max-loss"
        )
        parity_label = "agent-parity ON" if cfg_parity else "agent-parity OFF"
        data_source_label = (
            "alpaca-historical-30d"
            if cfg.get("use_alpaca_historical")
            else f"sigma-model"
        )
        st.caption(
            f"Strike model: **{sigma_label}** · Loss model: **{lc_label}** · "
            f"Data: **{data_source_label}** · **{parity_label}**"
        )

        # ── Decision-path diagnostics (funnel + per-candidate samples) ────
        funnel = getattr(result, "funnel", None)
        rej_counts = getattr(result, "rejection_counts", {}) or {}
        rej_samples = getattr(result, "rejection_samples", []) or []
        if funnel is not None and (funnel.considered > 0 or rej_counts):
            with st.expander(
                "Decision-path diagnostics (why each candidate was accepted / rejected)",
                expanded=False,
            ):
                st.markdown("**Candidate funnel**")
                funnel_df = pd.DataFrame(funnel.as_rows())
                st.dataframe(funnel_df, width='stretch', hide_index=True)
                if funnel.skipped_phases:
                    st.info(
                        "Phases intentionally skipped this run: "
                        + ", ".join(funnel.skipped_phases)
                    )

                if rej_counts:
                    st.markdown("**Rejection counts by gate**")
                    rc_df = (
                        pd.DataFrame(
                            [{"gate": k, "count": v} for k, v in rej_counts.items()]
                        )
                        .sort_values("count", ascending=False)
                        .reset_index(drop=True)
                    )
                    st.dataframe(rc_df, width='stretch', hide_index=True)

                if rej_samples:
                    st.markdown(
                        f"**Per-candidate rejection samples** "
                        f"(capped at {REJECTION_SAMPLE_CAP} per gate)"
                    )
                    samp_df = pd.DataFrame([
                        {
                            "entry_date": r.entry_date,
                            "ticker": r.ticker,
                            "phase": r.phase,
                            "gate": r.gate,
                            "price": r.price,
                            "regime": r.regime,
                            "strategy": r.strategy,
                            "measured": r.measured,
                            "threshold": r.threshold,
                            "reason": r.reason,
                        }
                        for r in rej_samples
                    ])
                    gate_choices = sorted(samp_df["gate"].unique().tolist())
                    gate_filter = st.multiselect(
                        "Filter by gate",
                        options=gate_choices,
                        default=gate_choices,
                        key="bt_rej_gate_filter",
                    )
                    if gate_filter:
                        samp_df = samp_df[samp_df["gate"].isin(gate_filter)]
                    st.dataframe(
                        samp_df.sort_values("entry_date").reset_index(drop=True),
                        width='stretch',
                        hide_index=True,
                    )

        # ── Summary metric cards ───────────────────────────────────────────
        hold_label = "Avg Hold (bars)" if is_result_intraday else "Avg Hold (days)"
        pf = m["profit_factor"]
        pf_display = "∞" if pf == float("inf") else f"{pf:.2f}"
        metric_cols = st.columns(6)
        for col, (label, value) in zip(
            metric_cols,
            [
                ("Trades", m["total_trades"]),
                ("Win Rate", f"{m['win_rate']:.1f}%"),
                ("Profit Factor", pf_display),
                ("Max DD", f"{m['max_drawdown_pct']:.1f}%"),
                ("Sharpe", f"{m['sharpe']:.2f}"),
                (hold_label, m["avg_hold_days"]),
            ],
        ):
            col.metric(label, value)

        st.divider()

        # ── Per-regime table + bar chart ───────────────────────────────────
        if not result.regime_stats.empty:
            left, right = st.columns([2, 3])
            with left:
                st.subheader("Results by Regime")
                st.dataframe(
                    result.regime_stats.assign(
                        pnl=result.regime_stats["pnl"].map("${:,.2f}".format)
                    ),
                    width='stretch',
                    hide_index=True,
                )
            with right:
                st.plotly_chart(regime_bar_chart(result.regime_stats), width='stretch')
            st.divider()

        # ── Equity + drawdown charts ───────────────────────────────────────
        if not result.equity_curve.empty:
            st.plotly_chart(equity_curve_chart(result.equity_curve), width='stretch')
            st.plotly_chart(drawdown_chart(result.equity_curve), width='stretch')
            st.divider()

        # ── Sortable trade log ─────────────────────────────────────────────
        st.subheader("Trade Log")
        if not result.trades.empty:
            sort_col = st.selectbox(
                "Sort by", options=["entry_date", "pnl", "hold_days", "ticker"], index=0,
                key="bt_sort_col",
            )
            ascending = st.checkbox("Ascending", value=False, key="bt_ascending")
            st.dataframe(
                result.trades.sort_values(sort_col, ascending=ascending),
                width='stretch',
                hide_index=True,
            )

        st.divider()

        # ── Export row ─────────────────────────────────────────────────────
        col_csv, col_json, col_journal = st.columns(3)

        with col_csv:
            st.download_button(
                "Export CSV",
                data=result.trades.to_csv(index=False).encode(),
                file_name="backtest_trades.csv",
                mime="text/csv",
                width='stretch',
            )

        with col_json:
            st.download_button(
                "Export JSON",
                data=result.trades.to_json(orient="records", date_format="iso", indent=2).encode(),
                file_name="backtest_trades.json",
                mime="application/json",
                width='stretch',
            )

        with col_journal:
            if st.button("Export to Journal", width='stretch'):
                _export_to_journal(result)
