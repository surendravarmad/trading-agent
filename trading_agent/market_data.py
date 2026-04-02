"""
Phase I — PERCEIVE
Fetches historical price data from Yahoo Finance and real-time
option snapshots from the Alpaca Market Data API.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    yf = None  # Graceful fallback — tests use mocked data

logger = logging.getLogger(__name__)


class MarketDataProvider:
    """Fetches and caches market data from Yahoo Finance and Alpaca."""

    def __init__(self, alpaca_api_key: str, alpaca_secret_key: str,
                 alpaca_data_url: str = "https://data.alpaca.markets/v2"):
        self.alpaca_api_key = alpaca_api_key
        self.alpaca_secret_key = alpaca_secret_key
        self.alpaca_data_url = alpaca_data_url
        self._price_cache: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Yahoo Finance — historical OHLCV
    # ------------------------------------------------------------------

    def fetch_historical_prices(self, ticker: str, period_days: int = 200) -> pd.DataFrame:
        """
        Download daily OHLCV for *ticker* covering at least *period_days*
        trading days.  Returns a DataFrame indexed by date.
        """
        logger.info("Fetching %d days of price history for %s", period_days, ticker)
        # Request extra calendar days to guarantee enough trading days
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

        # Keep only the last *period_days* rows
        df = df.tail(period_days).copy()
        self._price_cache[ticker] = df
        logger.info("Received %d rows for %s", len(df), ticker)
        return df

    def get_current_price(self, ticker: str) -> float:
        """
        Return the real-time price for *ticker* using Alpaca's snapshot API.
        Falls back to the most recent cached closing price if the API call fails.
        """
        realtime = self._fetch_alpaca_snapshot_price(ticker)
        if realtime is not None:
            return realtime

        # Fallback: use cached historical close
        logger.warning("[%s] Alpaca snapshot unavailable, using cached close", ticker)
        if ticker not in self._price_cache:
            self.fetch_historical_prices(ticker)
        return float(self._price_cache[ticker]["Close"].iloc[-1])

    def _fetch_alpaca_snapshot_price(self, ticker: str) -> Optional[float]:
        """
        Fetch real-time price from Alpaca's stock snapshot endpoint.
        GET https://data.alpaca.markets/v2/stocks/snapshots?symbols={ticker}

        Returns the latest trade price, or None on failure.
        """
        url = f"{self.alpaca_data_url}/stocks/snapshots"
        params = {"symbols": ticker}
        try:
            resp = requests.get(url, headers=self._alpaca_headers(),
                                params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            snap = data.get(ticker, {})
            # Prefer latest trade price, fall back to daily bar close
            latest_trade = snap.get("latestTrade", {})
            if latest_trade and latest_trade.get("p"):
                price = float(latest_trade["p"])
                logger.info("[%s] Alpaca real-time price: $%.2f (latest trade)",
                            ticker, price)
                return price
            daily_bar = snap.get("dailyBar", {})
            if daily_bar and daily_bar.get("c"):
                price = float(daily_bar["c"])
                logger.info("[%s] Alpaca real-time price: $%.2f (daily bar close)",
                            ticker, price)
                return price
            logger.warning("[%s] Snapshot returned but no price data found", ticker)
            return None
        except requests.RequestException as exc:
            logger.warning("[%s] Alpaca snapshot request failed: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Technical indicators
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
        rsi = 100 - (100 / (1 + rs))
        return rsi

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
        """
        Average daily slope of an SMA over the last *lookback* periods.
        Positive → upward, negative → downward.
        """
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
        Fetch option chain snapshot from Alpaca for a given underlying,
        expiration, and type ('put' or 'call').
        Returns a list of option contract dicts with Greeks.
        """
        url = f"https://data.alpaca.markets/v1beta1/options/snapshots/{underlying}"
        params = {

            "type": option_type,
            "expiration_date": expiration_date,
            "limit": 100,
        }
        logger.info("Fetching %s option chain for %s exp %s",
                     option_type, underlying, expiration_date)
        try:
            resp = requests.get(url, headers=self._alpaca_headers(), params=params,
                                timeout=15)
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
                    "mid": round((quote.get("bp", 0) + quote.get("ap", 0)) / 2, 2),
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
            return contracts
        except requests.RequestException as exc:
            logger.error("Alpaca option chain request failed: %s", exc)
            return None

    @staticmethod
    def _extract_strike(option_symbol: str) -> float:
        """
        Extract the strike price from an OCC option symbol.
        Example: SPY250404P00550000 → 550.00
        """
        try:
            # Last 8 digits represent the strike × 1000
            strike_raw = option_symbol[-8:]
            return int(strike_raw) / 1000.0
        except (ValueError, IndexError):
            return 0.0

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
