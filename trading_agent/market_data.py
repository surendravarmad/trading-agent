"""
Phase I — PERCEIVE
Fetches historical price data from Yahoo Finance and real-time
option snapshots from the Alpaca Market Data API.

5-minute cycle optimisations
------------------------------
1. TTL-based caches prevent redundant network calls within the same
   session or across close-spaced runs:
     • Historical prices  — 4-hour TTL   (PRICE_HISTORY_TTL)
     • Stock snapshots    — 60-second TTL (SNAPSHOT_TTL)
     • Option chains      — 3-minute TTL  (OPTION_CHAIN_TTL)

2. prefetch_historical_parallel(tickers) fetches all tickers'
   price history concurrently using a ThreadPoolExecutor, so the
   cache is warm before the per-ticker processing loop starts.

3. fetch_batch_snapshots(tickers) retrieves current prices for all
   tickers in a single Alpaca API call instead of N separate calls.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    yf = None  # Graceful fallback — tests use mocked data

logger = logging.getLogger(__name__)

# Cache TTLs (seconds)
PRICE_HISTORY_TTL = 14_400   # 4 hours — historical bars don't change intraday
SNAPSHOT_TTL = 60            # 1 minute — real-time price
OPTION_CHAIN_TTL = 180       # 3 minutes — Greeks move but not millisecond-fast

# Max workers for parallel historical fetches
_MAX_PREFETCH_WORKERS = 5


class MarketDataProvider:
    """Fetches and caches market data from Yahoo Finance and Alpaca."""

    def __init__(self, alpaca_api_key: str, alpaca_secret_key: str,
                 alpaca_data_url: str = "https://data.alpaca.markets/v2"):
        self.alpaca_api_key = alpaca_api_key
        self.alpaca_secret_key = alpaca_secret_key
        self.alpaca_data_url = alpaca_data_url

        # price cache: ticker → DataFrame
        self._price_cache: Dict[str, pd.DataFrame] = {}
        # price cache timestamps: ticker → epoch float
        self._price_cache_ts: Dict[str, float] = {}

        # snapshot cache: ticker → (price, epoch float)
        self._snapshot_cache: Dict[str, Tuple[float, float]] = {}

        # option chain cache: "{underlying}_{expiry}_{type}" → (contracts, epoch)
        self._option_cache: Dict[str, Tuple[list, float]] = {}

    # ------------------------------------------------------------------
    # Yahoo Finance — historical OHLCV
    # ------------------------------------------------------------------

    def fetch_historical_prices(self, ticker: str,
                                period_days: int = 200) -> pd.DataFrame:
        """
        Download daily OHLCV for *ticker* covering at least *period_days*
        trading days.  Returns a DataFrame indexed by date.

        Results are cached for PRICE_HISTORY_TTL seconds.
        """
        now = time.monotonic()
        cached_ts = self._price_cache_ts.get(ticker, 0.0)
        if ticker in self._price_cache and (now - cached_ts) < PRICE_HISTORY_TTL:
            logger.debug("[%s] Price history cache HIT (age=%.0fs)",
                         ticker, now - cached_ts)
            return self._price_cache[ticker]

        logger.info("Fetching %d days of price history for %s", period_days, ticker)
        cal_days = int(period_days * 1.6)
        end = datetime.now()
        start = end - timedelta(days=cal_days)

        if yf is None:
            raise ImportError("yfinance is required for live data. "
                              "Install with: pip install yfinance")

        tk = yf.Ticker(ticker)
        df = tk.history(start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"))

        if df.empty:
            raise ValueError(f"No price data returned for {ticker}")

        df = df.tail(period_days).copy()
        self._price_cache[ticker] = df
        self._price_cache_ts[ticker] = time.monotonic()
        logger.info("Received %d rows for %s (cached)", len(df), ticker)
        return df

    def prefetch_historical_parallel(self, tickers: List[str],
                                     period_days: int = 200) -> None:
        """
        Pre-populate the price-history cache for *tickers* concurrently.

        Call this once before the per-ticker processing loop so that
        regime classification reads from memory, not the network.
        """
        if not tickers:
            return
        workers = min(len(tickers), _MAX_PREFETCH_WORKERS)
        logger.info("Pre-fetching price history for %d tickers (workers=%d)",
                    len(tickers), workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self.fetch_historical_prices, t, period_days): t
                for t in tickers
            }
            for fut in as_completed(futures):
                ticker = futures[fut]
                try:
                    fut.result()
                    logger.debug("[%s] Historical pre-fetch done", ticker)
                except Exception as exc:
                    logger.warning("[%s] Historical pre-fetch failed: %s",
                                   ticker, exc)

    def get_current_price(self, ticker: str) -> float:
        """
        Return the real-time price for *ticker* (cached SNAPSHOT_TTL s).
        Falls back to the most recent cached closing price.
        """
        realtime = self._fetch_alpaca_snapshot_price(ticker)
        if realtime is not None:
            return realtime

        logger.warning("[%s] Alpaca snapshot unavailable, using cached close",
                       ticker)
        if ticker not in self._price_cache:
            self.fetch_historical_prices(ticker)
        return float(self._price_cache[ticker]["Close"].iloc[-1])

    def fetch_batch_snapshots(self, tickers: List[str]) -> Dict[str, float]:
        """
        Retrieve current prices for all *tickers* in a **single** Alpaca
        API call and populate the snapshot cache.

        Returns a dict of {ticker: price}.
        """
        if not tickers:
            return {}

        url = f"{self.alpaca_data_url}/stocks/snapshots"
        params = {"symbols": ",".join(tickers)}
        now = time.monotonic()

        try:
            resp = requests.get(url, headers=self._alpaca_headers(),
                                params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.warning("Batch snapshot request failed: %s", exc)
            return {}

        prices: Dict[str, float] = {}
        for ticker, snap in data.items():
            price = None
            latest_trade = snap.get("latestTrade", {})
            if latest_trade and latest_trade.get("p"):
                price = float(latest_trade["p"])
            else:
                daily_bar = snap.get("dailyBar", {})
                if daily_bar and daily_bar.get("c"):
                    price = float(daily_bar["c"])
            if price is not None:
                prices[ticker] = price
                self._snapshot_cache[ticker] = (price, now)

        logger.info("Batch snapshot: fetched prices for %d/%d tickers",
                    len(prices), len(tickers))
        return prices

    def _fetch_alpaca_snapshot_price(self, ticker: str) -> Optional[float]:
        """
        Fetch real-time price from Alpaca snapshot endpoint (with cache).
        Returns cached value if fresher than SNAPSHOT_TTL seconds.
        """
        now = time.monotonic()
        if ticker in self._snapshot_cache:
            cached_price, cached_at = self._snapshot_cache[ticker]
            if (now - cached_at) < SNAPSHOT_TTL:
                logger.debug("[%s] Snapshot cache HIT ($%.2f)", ticker, cached_price)
                return cached_price

        url = f"{self.alpaca_data_url}/stocks/snapshots"
        params = {"symbols": ticker}
        try:
            resp = requests.get(url, headers=self._alpaca_headers(),
                                params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            snap = data.get(ticker, {})
            latest_trade = snap.get("latestTrade", {})
            if latest_trade and latest_trade.get("p"):
                price = float(latest_trade["p"])
                logger.info("[%s] Alpaca real-time price: $%.2f (latest trade)",
                            ticker, price)
                self._snapshot_cache[ticker] = (price, time.monotonic())
                return price
            daily_bar = snap.get("dailyBar", {})
            if daily_bar and daily_bar.get("c"):
                price = float(daily_bar["c"])
                logger.info("[%s] Alpaca real-time price: $%.2f (daily bar close)",
                            ticker, price)
                self._snapshot_cache[ticker] = (price, time.monotonic())
                return price
            logger.warning("[%s] Snapshot returned but no price data found", ticker)
            return None
        except requests.RequestException as exc:
            logger.warning("[%s] Alpaca snapshot request failed: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Technical indicators (stateless — no caching needed)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_sma(prices: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return prices.rolling(window=window).mean()

    @staticmethod
    def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def compute_bollinger_bands(prices: pd.Series, window: int = 20,
                                 num_std: float = 2.0):
        """Return (upper, middle, lower) Bollinger Bands."""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + num_std * std
        lower = middle - num_std * std
        return upper, middle, lower

    @staticmethod
    def sma_slope(sma_series: pd.Series, lookback: int = 5) -> float:
        """Average daily slope of an SMA over the last *lookback* periods."""
        recent = sma_series.dropna().tail(lookback)
        if len(recent) < 2:
            return 0.0
        return float((recent.iloc[-1] - recent.iloc[0]) / len(recent))

    # ------------------------------------------------------------------
    # Alpaca — option snapshots & Greeks
    # ------------------------------------------------------------------

    def _alpaca_headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.alpaca_secret_key,
            "Accept": "application/json",
        }

    def fetch_option_chain(self, underlying: str,
                           expiration_date: str,
                           option_type: str = "put") -> Optional[list]:
        """
        Fetch option chain snapshot from Alpaca (cached OPTION_CHAIN_TTL s).
        Returns a list of option contract dicts with Greeks.
        """
        cache_key = f"{underlying}_{expiration_date}_{option_type}"
        now = time.monotonic()
        if cache_key in self._option_cache:
            contracts, cached_at = self._option_cache[cache_key]
            if (now - cached_at) < OPTION_CHAIN_TTL:
                logger.debug("[%s] Option chain cache HIT (%s %s)",
                             underlying, option_type, expiration_date)
                return contracts

        url = (f"https://data.alpaca.markets/v1beta1"
               f"/options/snapshots/{underlying}")
        params = {
            "type": option_type,
            "expiration_date": expiration_date,
            "limit": 100,
        }
        logger.info("Fetching %s option chain for %s exp %s",
                     option_type, underlying, expiration_date)
        try:
            resp = requests.get(url, headers=self._alpaca_headers(),
                                params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            snapshots = data.get("snapshots", {})
            contracts = []
            for symbol, snap in snapshots.items():
                greeks = snap.get("greeks", {})
                quote = snap.get("latestQuote", {})
                contracts.append({
                    "symbol": symbol,
                    "bid": quote.get("bp", 0),
                    "ask": quote.get("ap", 0),
                    "mid": round(
                        (quote.get("bp", 0) + quote.get("ap", 0)) / 2, 2),
                    "delta": greeks.get("delta", 0),
                    "theta": greeks.get("theta", 0),
                    "vega": greeks.get("vega", 0),
                    "gamma": greeks.get("gamma", 0),
                    "iv": greeks.get("impliedVolatility", 0),
                    "strike": self._extract_strike(symbol),
                    "expiration": expiration_date,
                    "type": option_type,
                })
            logger.info("Received %d %s contracts for %s",
                        len(contracts), option_type, underlying)
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

    def fetch_option_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch a fresh bid/ask quote for specific option symbols — no cache.

        Called right before order submission so the limit_price reflects
        the current market, not the snapshot taken during Phase III planning.

        Returns a dict keyed by OCC symbol::

            {
              "SPY260515P00550000": {"bid": 1.45, "ask": 1.55, "mid": 1.50},
              ...
            }
        """
        if not symbols:
            return {}

        url = "https://data.alpaca.markets/v1beta1/options/snapshots"
        params = {"symbols": ",".join(symbols)}
        try:
            resp = requests.get(url, headers=self._alpaca_headers(),
                                params=params, timeout=10)
            resp.raise_for_status()
            snapshots = resp.json().get("snapshots", {})
            quotes = {}
            for sym, snap in snapshots.items():
                q = snap.get("latestQuote", {})
                bid = float(q.get("bp", 0))
                ask = float(q.get("ap", 0))
                quotes[sym] = {
                    "bid": bid,
                    "ask": ask,
                    "mid": round((bid + ask) / 2, 2),
                }
            logger.info("Live quotes fetched for %d/%d symbols",
                        len(quotes), len(symbols))
            return quotes
        except requests.RequestException as exc:
            logger.warning("Live option quote fetch failed: %s", exc)
            return {}

    def get_underlying_bid_ask(self, ticker: str) -> Optional[Tuple[float, float]]:
        """
        Fetch the current bid/ask for the underlying stock from Alpaca.
        Returns (bid, ask) or None if unavailable.

        Used by the liquidity check — only trade tickers whose underlying
        bid/ask spread is below the configured threshold.
        """
        url = f"{self.alpaca_data_url}/stocks/snapshots"
        params = {"symbols": ticker}
        try:
            resp = requests.get(url, headers=self._alpaca_headers(),
                                params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            snap = data.get(ticker, {})
            quote = snap.get("latestQuote", {})
            bid = quote.get("bp")
            ask = quote.get("ap")
            if bid is not None and ask is not None:
                logger.debug("[%s] Underlying bid=%.4f ask=%.4f spread=%.4f",
                             ticker, bid, ask, ask - bid)
                return float(bid), float(ask)
            logger.warning("[%s] No bid/ask in snapshot latestQuote", ticker)
            return None
        except requests.RequestException as exc:
            logger.warning("[%s] Failed to fetch underlying bid/ask: %s", ticker, exc)
            return None

    def get_5min_return(self, ticker: str) -> Optional[float]:
        """
        Fetch the most recent 5-minute bar return for *ticker* from Alpaca.
        Returns (last_close / prev_close) - 1, or None on failure.

        Used for relative strength comparison: if a ticker's 5-min return
        exceeds the benchmark (SPY/QQQ), it is showing relative strength.
        """
        url = f"{self.alpaca_data_url}/stocks/{ticker}/bars"
        params = {"timeframe": "5Min", "limit": 2, "adjustment": "raw"}
        try:
            resp = requests.get(url, headers=self._alpaca_headers(),
                                params=params, timeout=10)
            resp.raise_for_status()
            bars = resp.json().get("bars") or []
            if len(bars) < 2:
                logger.debug("[%s] Not enough 5-min bars for RS calculation", ticker)
                return None
            prev_close = float(bars[-2]["c"])
            last_close = float(bars[-1]["c"])
            if prev_close == 0:
                return None
            ret = (last_close / prev_close) - 1.0
            logger.debug("[%s] 5-min return=%.4f%%", ticker, ret * 100)
            return ret
        except requests.RequestException as exc:
            logger.warning("[%s] 5-min bar fetch failed: %s", ticker, exc)
            return None

    def get_account_info(self, base_url: str) -> Optional[Dict]:
        """Fetch paper trading account information from Alpaca."""
        url = f"{base_url}/account"
        try:
            resp = requests.get(url, headers=self._alpaca_headers(), timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("Failed to fetch account info: %s", exc)
            return None

    def is_market_open(self, base_url: str) -> bool:
        """Check if the market is currently open via Alpaca clock API."""
        url = f"{base_url}/clock"
        try:
            resp = requests.get(url, headers=self._alpaca_headers(), timeout=10)
            resp.raise_for_status()
            return resp.json().get("is_open", False)
        except requests.RequestException as exc:
            logger.error("Failed to check market clock: %s", exc)
            return False
