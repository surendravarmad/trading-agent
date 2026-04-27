"""
ports — vendor-agnostic interfaces for the agent core
=====================================================

This module is the *hexagonal core boundary*.  Strategy, risk, and
orchestration code (``agent.py``, ``regime.py``, ``strategy.py``,
``risk.py``) depend only on the Protocol classes defined here — never
on Alpaca / Yahoo / IBKR concretes.

Vendor adapters
---------------
Concrete adapters satisfying these ports live in:

  • ``market_data.py``    — ``MarketDataProvider``  (Yahoo historical +
                            Alpaca real-time). Satisfies
                            :class:`MarketDataPort` and
                            :class:`AccountPort`.
  • ``executor.py``       — ``OrderExecutor``  (Alpaca). Satisfies
                            :class:`ExecutionPort`.
  • ``position_monitor.py`` — ``PositionMonitor`` (Alpaca). Satisfies
                            :class:`PositionsPort`.
  • ``order_tracker.py``  — ``OrderTracker`` (Alpaca). Satisfies
                            :class:`OrdersPort`.

Why five ports, not one "BrokerPort"?
-------------------------------------
The agent already holds four separate instances (data_provider,
executor, position_monitor, order_tracker).  Wrapping them in a single
facade would add a class and shuffle call sites without adding test
isolation.  Keeping one port per logical concern mirrors the existing
class shape while still gating the agent core against Alpaca specifics.

Why ``@runtime_checkable``?
---------------------------
Tests can now do ``assert isinstance(provider, MarketDataPort)`` to
verify a fake / mock satisfies the contract — useful during the
week 7 IBKR-adapter validation round when a skeletal adapter will be
built to prove the seam holds.

Scope limits (intentional, per week 5-6 charter)
------------------------------------------------
- No multi-venue support yet (US equity / options only).
- No attempt to normalize exotic greeks, multi-leg options beyond verticals,
  or crypto/forex idioms.
- Indicator math kept on the port rather than split into a separate
  ``IndicatorsPort`` — a future GPU / numba adapter can override by
  implementing the same port.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import pandas as pd


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
@runtime_checkable
class MarketDataPort(Protocol):
    """Everything the regime classifier, strategy planner, and cycle
    orchestration need in order to read the market.

    An adapter conforming to this port is responsible for:
      * Historical OHLCV (cached, TTL-bounded)
      * Real-time prices, batched if possible
      * Option chain snapshots with Greeks and live-quote refresh
      * Stateless indicator helpers (SMA/RSI/Bollinger/sma_slope)
      * A cached-price query — lets the core ask "what's the last price
        you saw for X?" without reaching into private caches.
    """

    # ----- Historical OHLCV ----------------------------------------------
    def fetch_historical_prices(self, ticker: str,
                                period_days: int = 200) -> pd.DataFrame: ...

    def prefetch_historical_parallel(self, tickers: List[str],
                                     period_days: int = 200) -> None: ...

    # ----- Real-time price & quotes --------------------------------------
    def get_current_price(self, ticker: str) -> float: ...

    def fetch_batch_snapshots(self, tickers: List[str]) -> Dict[str, float]: ...

    def get_underlying_bid_ask(self,
                               ticker: str) -> Optional[Tuple[float, float]]: ...

    def get_5min_return(self, ticker: str) -> Optional[float]: ...

    # ----- Options --------------------------------------------------------
    def fetch_option_chain(self, underlying: str,
                           expiration_date: str,
                           option_type: str = "put") -> Optional[list]: ...

    def fetch_option_quotes(self,
                            symbols: List[str]) -> Dict[str, Dict]: ...

    # ----- Indicators (stateless) ----------------------------------------
    def compute_sma(self, prices: pd.Series,
                    window: int) -> pd.Series: ...

    def compute_rsi(self, prices: pd.Series,
                    window: int = 14) -> pd.Series: ...

    def compute_bollinger_bands(
        self, prices: pd.Series, window: int = 20, num_std: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]: ...

    def sma_slope(self, sma_series: pd.Series,
                  lookback: int = 5) -> float: ...

    # ----- Cached-price query --------------------------------------------
    # Public replacement for the prior `provider._snapshot_cache.get(...)`
    # access in agent.py.  Returns the most recently observed price for
    # the ticker, or None if the adapter has never seen it.
    def get_cached_price(self, ticker: str) -> Optional[float]: ...


# ---------------------------------------------------------------------------
# Account / clock
# ---------------------------------------------------------------------------
@runtime_checkable
class AccountPort(Protocol):
    """Broker account equity, buying power, and market-clock queries.

    The port intentionally takes NO base_url argument — the adapter owns
    its endpoint configuration internally.  This fixes the cross-config
    leak where the agent passed ``AlpacaConfig.base_url`` into
    ``MarketDataProvider`` methods on every call.
    """

    def get_account_info(self) -> Optional[Dict]: ...

    def is_market_open(self) -> bool: ...


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
@runtime_checkable
class ExecutionPort(Protocol):
    """Submit new spread orders; close existing spreads.

    Implementations must return a dict describing the outcome.  The
    agent core never inspects vendor-specific fields — it only needs to
    know whether the submission succeeded and what order/leg IDs were
    assigned for downstream tracking.
    """

    def execute(self, verdict: Any) -> Dict: ...

    def close_spread(self, spread: Any) -> Dict: ...


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------
@runtime_checkable
class PositionsPort(Protocol):
    """Fetch broker-held positions and decide whether any should exit."""

    def fetch_open_positions(self) -> List[Any]: ...

    def group_into_spreads(self, positions: List[Any],
                           trade_plans: List[Any]) -> List[Any]: ...

    def evaluate(self, spreads: List[Any],
                 current_regimes: Dict[str, Any],
                 underlying_prices: Dict[str, float]) -> List[Any]: ...

    def summary(self, spreads: List[Any]) -> Dict: ...


# ---------------------------------------------------------------------------
# Orders (observability)
# ---------------------------------------------------------------------------
@runtime_checkable
class OrdersPort(Protocol):
    """Recent order history for operational logging / dashboards."""

    def fetch_open_orders(self) -> List[Any]: ...

    def fetch_recent_fills(self, limit: int = 20) -> List[Any]: ...

    def summarize_orders(self, orders: List[Any]) -> Dict: ...


# ---------------------------------------------------------------------------
# Sentiment readout — unified surface for LLM analyst consumption
# ---------------------------------------------------------------------------
#
# Both :class:`trading_agent.fingpt_analyser.SentimentReport` (raw FinGPT
# output) and :class:`trading_agent.sentiment_verifier.VerifiedSentimentReport`
# (reasoning-model verified wrapper) must be consumable by
# ``LLMAnalyst.analyze_trade`` without the analyst caring which stage of
# the pipeline produced it.  Previously the analyst was typed as
# ``Optional[SentimentReport]`` but received either — a type lie that
# would have bitten us once a verifier fallback path needed to short
# circuit the pipeline (Tier-0 cache, earnings calendar authoritative
# event_risk, etc.).
#
# Requirements of the port:
#   • Every field is ultimately derived from verified evidence when the
#     verifier ran; from raw FinGPT output otherwise.
#   • Passing the same Protocol-typed value into the analyst prompt
#     builder always pulls the most authoritative view of the sentiment.
@runtime_checkable
class SentimentReadout(Protocol):
    """Canonical sentiment surface consumed by the LLM Analyst prompt.

    Implementations:
      • ``SentimentReport`` — direct FinGPT output.
      • ``VerifiedSentimentReport`` — exposes the verified_* scores
        through the same property names via ``@property`` aliases.
    """
    ticker: str
    sentiment_score: float
    event_risk: float
    confidence: float
    recommendation: str
    reasoning: str
    key_themes: List[str]
    headlines: List[str]


__all__ = [
    "MarketDataPort",
    "AccountPort",
    "ExecutionPort",
    "PositionsPort",
    "OrdersPort",
    "SentimentReadout",
]
