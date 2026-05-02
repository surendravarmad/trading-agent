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

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
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
PRICE_HISTORY_TTL = 14_400      # 4 hours — historical bars don't change intraday
SNAPSHOT_TTL = 60               # 1 minute — real-time price
OPTION_CHAIN_TTL = 180          # 3 minutes — Greeks move but not millisecond-fast
INTRADAY_RETURN_TTL = 60        # 1 minute — 5-min bar return; long enough to
                                # dedupe SPY/QQQ benchmark calls within one
                                # cycle (~1-3 min), short enough to roll over
                                # to the next bar between cycles

# Max workers for parallel historical fetches
_MAX_PREFETCH_WORKERS = 5

# (connect, read) timeouts for Alpaca HTTP calls. The connect side stays
# tight so an unreachable host (DNS, VPN, firewall) fails in 2s instead of
# blocking a whole cycle / test for the full read-budget. The read side
# stays generous since some Alpaca endpoints take a few seconds under load.
ALPACA_CONNECT_TIMEOUT = 2
ALPACA_READ_TIMEOUT = 10
ALPACA_READ_TIMEOUT_LONG = 15
ALPACA_TIMEOUT = (ALPACA_CONNECT_TIMEOUT, ALPACA_READ_TIMEOUT)
ALPACA_TIMEOUT_LONG = (ALPACA_CONNECT_TIMEOUT, ALPACA_READ_TIMEOUT_LONG)


def _truncate_json(obj: object, limit: int = 400) -> str:
    """Compact, length-capped JSON repr for log lines — diagnostic only."""
    try:
        text = json.dumps(obj, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        text = repr(obj)
    return text if len(text) <= limit else text[:limit] + f"…(+{len(text)-limit}ch)"


class InsufficientDataError(ValueError):
    """
    Raised when a historical fetch returns fewer bars than the caller
    requires. Classifying a regime with < 200 bars leaves SMA-200 as NaN,
    which silently falls through to SIDEWAYS. Callers should catch this
    and skip the ticker rather than trade on a broken signal.
    """


class MarketDataProvider:
    """Fetches and caches market data from Yahoo Finance and Alpaca."""

    def __init__(self, alpaca_api_key: str, alpaca_secret_key: str,
                 alpaca_data_url: str = "https://data.alpaca.markets/v2",
                 alpaca_base_url: str = "https://paper-api.alpaca.markets/v2"):
        """
        Parameters
        ----------
        alpaca_api_key, alpaca_secret_key
            Alpaca credentials.
        alpaca_data_url
            Market-data endpoint (snapshots, bars, option chains).
        alpaca_base_url
            Trading/account endpoint (clock, account info).  Stored on
            the instance so :meth:`get_account_info` and
            :meth:`is_market_open` no longer need the caller to pass it
            in on every invocation.  This is the week 5-6 fix for the
            AccountPort base_url leak.
        """
        self.alpaca_api_key = alpaca_api_key
        self.alpaca_secret_key = alpaca_secret_key
        self.alpaca_data_url = alpaca_data_url
        self.alpaca_base_url = alpaca_base_url

        # price cache: ticker → DataFrame
        self._price_cache: Dict[str, pd.DataFrame] = {}
        # price cache timestamps: ticker → epoch float
        self._price_cache_ts: Dict[str, float] = {}

        # snapshot cache: ticker → (price, epoch float)
        self._snapshot_cache: Dict[str, Tuple[float, float]] = {}

        # option chain cache: "{underlying}_{expiry}_{type}" → (contracts, epoch)
        self._option_cache: Dict[str, Tuple[list, float]] = {}

        # intraday 5-min return cache: ticker → (return, epoch float).
        # Collapses the N-1 redundant SPY/QQQ benchmark fetches that the
        # relative-strength calculation otherwise produces per cycle.
        self._intraday_return_cache: Dict[str, Tuple[float, float]] = {}

        # intraday 5-min return SERIES cache: ticker → (returns_list, epoch).
        # The Z-score leadership signal needs ~20 bars of history per ticker
        # to compute a meaningful rolling stdev; without per-cycle caching
        # the regime classifier would re-fetch the same window N times.
        self._intraday_return_series_cache: Dict[str, Tuple[List[float], float]] = {}

        # VIX (volatility-index) cache — one global signal per cycle.
        # ^VIX is sourced via yfinance because Alpaca doesn't carry it as a
        # tradable symbol on the IEX/SIP feeds.  Cached for INTRADAY_RETURN_TTL.
        self._vix_zscore_cache: Optional[Tuple[float, float, float]] = None  # (raw_change, zscore, epoch)

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

        # Per-ticker per-cycle on cache miss — DEBUG. Cache-hit path is
        # already DEBUG above; agent.py logs the cycle-level pre-fetch
        # summary at INFO so the operator still sees aggregate progress.
        logger.debug("Fetching %d days of price history for %s", period_days, ticker)
        cal_days = int(period_days * 1.6)
        end = datetime.now()
        start = end - timedelta(days=cal_days)

        if yf is None:
            raise ImportError("yfinance is required for live data. "
                              "Install with: pip install yfinance")

        tk = yf.Ticker(ticker)
        # auto_adjust=False returns raw (unadjusted) closes so that our SMAs
        # and Bollinger Bands are comparable to Alpaca's real-time price
        # (latestTrade.p), which is also raw. Mixing adjusted and unadjusted
        # series causes spurious regime flips around dividends and splits.
        df = tk.history(start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"),
                        auto_adjust=False)

        if df.empty:
            raise InsufficientDataError(
                f"No price data returned for {ticker} "
                f"(requested {period_days} bars)")

        df = df.tail(period_days).copy()

        # Regime classification needs period_days bars (default 200) for
        # SMA-200 to be non-NaN. Without this guard, sma_200.iloc[-1] is NaN,
        # `price > NaN` is False, and the regime silently falls through to
        # SIDEWAYS — a quiet misclassification. Fail loud instead.
        if len(df) < period_days:
            raise InsufficientDataError(
                f"{ticker}: only {len(df)} bars returned, "
                f"need {period_days} for reliable classification")

        self._price_cache[ticker] = df
        self._price_cache_ts[ticker] = time.monotonic()
        logger.debug("Received %d rows for %s (cached)", len(df), ticker)
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

        ``feed`` is read from the ``ALPACA_STOCKS_FEED`` env var
        (default ``iex``).  Free/basic accounts cannot read SIP and the
        endpoint silently returns 403 without an explicit feed; IEX is
        the correct free-tier choice.  Operators on a paid SIP plan can
        override via the env var.
        """
        if not tickers:
            return {}

        url = f"{self.alpaca_data_url}/stocks/snapshots"
        feed = os.getenv("ALPACA_STOCKS_FEED", "iex").strip() or "iex"
        params = {"symbols": ",".join(tickers), "feed": feed}
        now = time.monotonic()

        try:
            resp = requests.get(url, headers=self._alpaca_headers(),
                                params=params, timeout=ALPACA_TIMEOUT)
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

        # Per-cycle batch fetch — agent.py logs the cycle start at INFO
        # which already implies "we're about to fetch prices". DEBUG.
        logger.debug("Batch snapshot: fetched prices for %d/%d tickers",
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
        feed = os.getenv("ALPACA_STOCKS_FEED", "iex").strip() or "iex"
        params = {"symbols": ticker, "feed": feed}
        try:
            resp = requests.get(url, headers=self._alpaca_headers(),
                                params=params, timeout=ALPACA_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            snap = data.get(ticker, {})
            latest_trade = snap.get("latestTrade", {})
            if latest_trade and latest_trade.get("p"):
                price = float(latest_trade["p"])
                # Per-ticker per-cycle quote — DEBUG. Phase I/III logs in
                # agent.py + strategy.py already report the price actually
                # used for planning and execution.
                logger.debug("[%s] Alpaca real-time price: $%.2f (latest trade)",
                             ticker, price)
                self._snapshot_cache[ticker] = (price, time.monotonic())
                return price
            daily_bar = snap.get("dailyBar", {})
            if daily_bar and daily_bar.get("c"):
                price = float(daily_bar["c"])
                logger.debug("[%s] Alpaca real-time price: $%.2f (daily bar close)",
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
        """Simple Moving Average — pandas rolling mean."""
        return prices.rolling(window=window).mean()

    @staticmethod
    def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index — Wilder's smoothing (RMA).

        Uses exponentially-weighted mean with ``alpha = 1 / window`` — the
        mathematically correct RSI as defined by J. Welles Wilder,
        matching the convention used by TA-Lib, pandas-ta, TradingView,
        and ThinkOrSwim. This differs from a naive rolling-mean RSI (the
        prior implementation): values are smoother, converge more slowly
        after price shocks, and persist farther into overbought /
        oversold territory during sustained trends.

        Why not depend on pandas-ta? pandas-ta ≥0.3.14b unconditionally
        imports ``numba`` at module load time via
        ``pandas_ta/utils/_math.py``. On platforms where no numba wheel
        exists (e.g. Python 3.14 at certain points), the whole package
        fails to import. pandas' own ``ewm`` produces identical smoothing
        for the three indicators we need, so we stay lean.
        """
        delta = prices.diff()
        # clip (not where) preserves the bar-0 NaN so the first ewm
        # window isn't biased by a spurious 0 on the leading position.
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        # alpha = 1/window reproduces Wilder's RMA. adjust=True (pandas
        # default) matches pandas-ta's rma() and TradingView's RSI output.
        # min_periods=window keeps the first window-1 values NaN, so the
        # first reported RSI appears at bar ``window`` — TA-Lib convention.
        avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def compute_bollinger_bands(prices: pd.Series, window: int = 20,
                                 num_std: float = 2.0):
        """Return (upper, middle, lower) Bollinger Bands.

        Uses pandas' default sample standard deviation (``ddof=1``) —
        the unbiased estimator appropriate for finite windows. Note that
        TA-Lib and many charting platforms default to population std
        (``ddof=0``), which produces bands ~2.5% narrower at the 20-day
        default. The regime-detection logic and its tests were
        calibrated against the sample-std version, so we keep it here;
        external chart comparisons should account for the ddof choice.
        """
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()   # ddof=1 (sample)
        upper = middle + num_std * std
        lower = middle - num_std * std
        return upper, middle, lower

    @staticmethod
    def sma_slope(sma_series: pd.Series, lookback: int = 5) -> float:
        """Average daily change of an SMA over the last *lookback* periods.

        Units — **dollars per day** (not a percentage).  Returns
        ``(SMA[t] − SMA[t−lookback]) / lookback`` as a raw price delta.

        A reading of ``0.50`` for ``lookback=5`` means the SMA rose by
        an average of 50 cents per day over the five trading days ending
        at ``t``.  To interpret magnitude as a fraction of the underlying,
        divide by ``SMA[t]`` (``slope / sma.iloc[-1]``) at the call site.

        All current downstream consumers use the **sign** of this value
        (``slope > 0`` → bullish trend continuation; ``slope < 0`` →
        bearish; see ``RegimeClassifier._determine_regime``), and the
        sign is unit-invariant so the dimensional form is harmless.
        Code or LLM prompts that treat this number as a percentage would
        misread it — see ``RegimeAnalysis.sma_50_slope`` and the
        "SMA-50 slope units" note in ``README.md``.
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
        Fetch option chain snapshot from Alpaca (cached OPTION_CHAIN_TTL s).
        Returns a list of option contract dicts with Greeks.

        Notes
        -----
        * ``limit`` is raised to 1000 (Alpaca's max) so wide SPY/QQQ chains
          aren't truncated to 100 — which would drop the OTM wings where
          the 0.15-0.20 delta target sits.
        * ``feed`` is set explicitly. Free/basic accounts must use
          ``indicative`` or the endpoint silently returns an empty
          snapshots dict. Accounts with the OPRA real-time subscription
          can override via the ``ALPACA_OPTIONS_FEED`` env var.
        * If the exact ``expiration_date`` yields zero contracts, the
          request is retried with ``expiration_date_gte``/``_lte`` set
          to the same date. Some Alpaca endpoint revisions treat these
          filters more leniently than the exact-match ``expiration_date``
          (e.g. off-by-one around weekly settlement conventions).
        * When the chain is empty, the raw response body and the
          effective URL are logged at WARNING so the operator can see
          whether the endpoint returned ``snapshots: {}`` (data-plan
          issue), ``snapshots: null``, or something entirely different.
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
        feed = os.getenv("ALPACA_OPTIONS_FEED", "indicative").strip() or "indicative"
        base_params = {
            "type": option_type,
            "feed": feed,
            "limit": 1000,
        }
        # Per-(ticker, expiration) chain fetch — chain scanner's grid
        # sweep alone produces |DTE_grid| chain fetches per ticker per
        # cycle. DEBUG. The "Received N contracts" summary below stays
        # at DEBUG too; failures are still WARNING / ERROR.
        logger.debug("Fetching %s option chain for %s exp %s (feed=%s)",
                     option_type, underlying, expiration_date, feed)

        def _parse_snapshots(snapshots: Dict) -> List[Dict]:
            parsed: List[Dict] = []
            for symbol, snap in (snapshots or {}).items():
                if not isinstance(snap, dict):
                    continue
                greeks = snap.get("greeks") or {}
                quote = snap.get("latestQuote") or {}
                bid = float(quote.get("bp", 0) or 0)
                ask = float(quote.get("ap", 0) or 0)
                # Prefer the expiration extracted from the OCC symbol so
                # fallback-retry paths (where the effective expiration
                # may not equal ``expiration_date``) still stamp the real
                # listed date on the contract dict.
                sym_exp = self._extract_expiration(symbol) or expiration_date
                parsed.append({
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
                    "expiration": sym_exp,
                    "type": option_type,
                })
            return parsed

        def _request(extra_params: Dict) -> Tuple[Optional[List[Dict]], Optional[Dict], Optional[str]]:
            """Single GET, returns (contracts, raw_body, effective_url)."""
            params = {**base_params, **extra_params}
            try:
                resp = requests.get(url, headers=self._alpaca_headers(),
                                    params=params, timeout=ALPACA_TIMEOUT_LONG)
                resp.raise_for_status()
                data = resp.json() if resp.content else {}
                snaps = data.get("snapshots") if isinstance(data, dict) else None
                parsed = _parse_snapshots(snaps or {})
                return parsed, data, resp.url
            except requests.RequestException as exc:
                logger.error("Alpaca option chain request failed: %s", exc)
                return None, None, None

        # Primary: exact expiration_date match
        contracts, body, effective_url = _request(
            {"expiration_date": expiration_date}
        )

        # Fallback: same day via _gte/_lte window
        if contracts == [] and body is not None:
            logger.warning(
                "[%s] exact expiration_date=%s returned 0 %s snapshots; "
                "retrying with _gte/_lte window. Raw body (truncated): %s",
                underlying, expiration_date, option_type,
                _truncate_json(body),
            )
            retry_contracts, retry_body, retry_url = _request({
                "expiration_date_gte": expiration_date,
                "expiration_date_lte": expiration_date,
            })
            if retry_contracts:
                logger.info(
                    "[%s] _gte/_lte retry recovered %d %s contracts "
                    "(effective_url=%s)",
                    underlying, len(retry_contracts), option_type, retry_url,
                )
                contracts = retry_contracts
            elif retry_contracts == [] and retry_body is not None:
                logger.warning(
                    "[%s] _gte/_lte retry also empty. effective_url=%s body=%s",
                    underlying, retry_url, _truncate_json(retry_body),
                )

        # Final fallback: ask the contracts catalogue what's actually
        # listed near the target date, then refetch the snapshot using
        # the nearest real expiration.  This saves us when the indicative
        # feed doesn't publish some weekly expirations even though they
        # exist — a case we've hit empirically on free-tier accounts.
        if contracts == []:
            alt_exp = self._nearest_listed_expiration(
                underlying, expiration_date, option_type
            )
            if alt_exp and alt_exp != expiration_date:
                logger.warning(
                    "[%s] Requested expiration %s not available in %s feed; "
                    "swapping to nearest listed expiration %s",
                    underlying, expiration_date, feed, alt_exp,
                )
                alt_contracts, _alt_body, alt_url = _request(
                    {"expiration_date": alt_exp}
                )
                if alt_contracts:
                    logger.info(
                        "[%s] Catalog fallback recovered %d %s contracts "
                        "at %s (effective_url=%s)",
                        underlying, len(alt_contracts), option_type,
                        alt_exp, alt_url,
                    )
                    # Stamp every dict with the effective expiration so
                    # downstream risk-check / assemble logic references
                    # the actual listed date, not the originally-requested
                    # (unavailable) one.
                    for c in alt_contracts:
                        c["expiration"] = (
                            self._extract_expiration(c["symbol"]) or alt_exp
                        )
                    contracts = alt_contracts
                else:
                    logger.warning(
                        "[%s] Catalog fallback: listed expiration %s also "
                        "returned no snapshots (feed=%s).",
                        underlying, alt_exp, feed,
                    )

        if contracts is None:
            return None

        # Per-chain summary; same volume as the "Fetching chain" log
        # above. DEBUG. Recovery / fallback paths above stay at INFO /
        # WARNING because they're actionable signals.
        logger.debug("Received %d %s contracts for %s",
                     len(contracts), option_type, underlying)

        if not contracts and effective_url:
            logger.warning(
                "[%s] Empty option chain from Alpaca. "
                "effective_url=%s — check (a) account has options data "
                "entitlement, (b) expiration_date is a real listed "
                "expiration, (c) ALPACA_OPTIONS_FEED env var "
                "(current=%s) matches your subscription (free→indicative, "
                "paid→opra).",
                underlying, effective_url, feed,
            )

        if contracts:   # never cache empty results — allow a fresh retry
            self._option_cache[cache_key] = (contracts, time.monotonic())
        return contracts

    @staticmethod
    def _extract_strike(option_symbol: str) -> float:
        """Extract the strike price from an OCC option symbol."""
        try:
            return int(option_symbol[-8:]) / 1000.0
        except (ValueError, IndexError):
            return 0.0

    @staticmethod
    def _extract_expiration(option_symbol: str) -> Optional[str]:
        """
        Extract the expiration date from an OCC option symbol.

        OCC format is ``ROOT + YYMMDD + C/P + STRIKE*1000`` (21 chars
        total for common equity options).  The 6-char date lives at
        positions ``[-15:-9]``.  Returns an ISO ``YYYY-MM-DD`` string or
        ``None`` if parsing fails.
        """
        try:
            yymmdd = option_symbol[-15:-9]
            yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
            # 2-digit year → assume 2000-2099 window (fine until 2100).
            return f"20{yy:02d}-{mm:02d}-{dd:02d}"
        except (ValueError, IndexError):
            return None

    def _nearest_listed_expiration(
        self, underlying: str, target_date: str, option_type: str = "put",
    ) -> Optional[str]:
        """
        Query Alpaca's ``/v2/options/contracts`` catalog and return the
        listed expiration nearest to ``target_date``.

        Falls back gracefully to ``None`` on any error — callers should
        treat ``None`` as "no alternative found, keep original behavior".

        This endpoint is on the trading tier, not the market-data tier,
        so it works on free/basic accounts even when the snapshot
        endpoint's ``indicative`` feed is sparse.
        """
        try:
            target = datetime.strptime(target_date, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return None

        # ±14-day window around the requested expiration captures weekly
        # and monthly neighbours without pulling the whole catalogue.
        lo = (target - timedelta(days=14)).isoformat()
        hi = (target + timedelta(days=14)).isoformat()
        url = f"{self.alpaca_base_url}/options/contracts"
        params = {
            "underlying_symbols": underlying,
            "expiration_date_gte": lo,
            "expiration_date_lte": hi,
            "status": "active",
            "type": option_type,
            "limit": 10000,
        }
        try:
            resp = requests.get(url, headers=self._alpaca_headers(),
                                params=params, timeout=ALPACA_TIMEOUT_LONG)
            resp.raise_for_status()
            body = resp.json() or {}
        except requests.RequestException as exc:
            logger.warning(
                "[%s] Options catalog lookup failed (%s..%s): %s",
                underlying, lo, hi, exc,
            )
            return None

        listed = set()
        for c in (body.get("option_contracts") or []):
            exp = c.get("expiration_date")
            if exp:
                listed.add(exp)
        if not listed:
            logger.warning(
                "[%s] Options catalog returned no active contracts "
                "in %s..%s — check underlying_symbols filter.",
                underlying, lo, hi,
            )
            return None

        best: Optional[str] = None
        best_diff = 10**9
        for exp in listed:
            try:
                d = datetime.strptime(exp, "%Y-%m-%d").date()
            except ValueError:
                continue
            if d < target:                     # prefer ≥ target, fallback to closest
                diff = (target - d).days + 1000
            else:
                diff = (d - target).days
            if diff < best_diff:
                best_diff = diff
                best = exp
        if best:
            logger.info(
                "[%s] Catalog lookup: %d listed expirations in %s..%s "
                "(sample: %s); nearest to %s = %s",
                underlying, len(listed), lo, hi,
                ",".join(sorted(listed)[:6]) + ("…" if len(listed) > 6 else ""),
                target_date, best,
            )
        return best

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
        feed = os.getenv("ALPACA_OPTIONS_FEED", "indicative").strip() or "indicative"
        params = {"symbols": ",".join(symbols), "feed": feed}
        try:
            resp = requests.get(url, headers=self._alpaca_headers(),
                                params=params, timeout=ALPACA_TIMEOUT)
            resp.raise_for_status()
            snapshots = resp.json().get("snapshots", {}) or {}
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
        feed = os.getenv("ALPACA_STOCKS_FEED", "iex").strip() or "iex"
        params = {"symbols": ticker, "feed": feed}
        try:
            resp = requests.get(url, headers=self._alpaca_headers(),
                                params=params, timeout=ALPACA_TIMEOUT)
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

    @staticmethod
    def _last_completed_5min_end(reference: Optional[datetime] = None) -> str:
        """
        Return the RFC-3339 (UTC) timestamp representing the upper bound of
        the most recently *completed* 5-minute bar.

        We floor the wall-clock time to a 5-minute boundary and subtract
        one second. Alpaca filters bars whose start timestamp is <= `end`,
        so passing (boundary - 1s) guarantees the bar whose start equals
        the current boundary (still forming) is excluded.

        Example: at 10:37:12 UTC → boundary 10:35 → end = 10:34:59.
        Bars returned: 10:30 (completed) and 10:25 (completed).
        """
        now = reference or datetime.now(timezone.utc)
        # Ensure UTC; if a naive datetime was passed in, assume it's already UTC
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)
        floored_minute = (now.minute // 5) * 5
        boundary = now.replace(minute=floored_minute, second=0, microsecond=0)
        end = boundary - timedelta(seconds=1)
        # RFC-3339 with trailing Z for UTC (Alpaca accepts both Z and +00:00)
        return end.strftime("%Y-%m-%dT%H:%M:%SZ")

    def get_5min_return(self, ticker: str) -> Optional[float]:
        """
        Fetch the most recent 5-minute bar return for *ticker* from Alpaca.
        Returns (last_close / prev_close) - 1, or None on failure.

        Used for relative strength comparison: if a ticker's 5-min return
        exceeds the benchmark (SPY/QQQ), it is showing relative strength.

        Only **completed** bars are requested — the `end` parameter is set
        to (floor(now, 5min) - 1s), which excludes the bar that is still
        forming at query time. Without this guard, bars[-1] can be a partial
        bar and the return mixes half-a-window of data with a full window.

        Results are cached for INTRADAY_RETURN_TTL seconds so that the
        SPY/QQQ benchmark calls (invoked once per non-benchmark ticker by
        RegimeClassifier) collapse to a single fetch per cycle. Failed
        fetches (None) are not cached — the next caller retries.
        """
        now = time.monotonic()
        cached = self._intraday_return_cache.get(ticker)
        if cached is not None:
            cached_ret, cached_at = cached
            if (now - cached_at) < INTRADAY_RETURN_TTL:
                logger.debug("[%s] 5-min return cache HIT (%.4f%%)",
                             ticker, cached_ret * 100)
                return cached_ret

        url = f"{self.alpaca_data_url}/stocks/{ticker}/bars"
        feed = os.getenv("ALPACA_STOCKS_FEED", "iex").strip() or "iex"
        params = {
            "timeframe": "5Min",
            "limit": 2,
            "adjustment": "raw",
            # Free/basic Alpaca subscriptions cannot read SIP — they get
            # 403 on stock bars without an explicit feed. IEX is the
            # correct free-tier choice; paid SIP users override via env.
            "feed": feed,
            # Restrict to completed bars only — see _last_completed_5min_end
            "end": self._last_completed_5min_end(),
        }
        try:
            resp = requests.get(url, headers=self._alpaca_headers(),
                                params=params, timeout=ALPACA_TIMEOUT)
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
            # Cache only successful results — failed fetches (None) stay
            # uncached so the next caller retries.
            self._intraday_return_cache[ticker] = (ret, time.monotonic())
            return ret
        except requests.RequestException as exc:
            logger.warning("[%s] 5-min bar fetch failed: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Rolling 5-min return series & Z-scored leadership signal
    # ------------------------------------------------------------------

    # Skip the first ``OPEN_BAR_SKIP`` 5-min bars after the 9:30 ET open
    # because they typically print 3-4× the rest of the day's vol and
    # would dominate any rolling stdev calculation.  Two bars = first
    # 10 minutes of the session.
    OPEN_BAR_SKIP = 2
    LEADERSHIP_WINDOW_BARS = 21          # 20 returns + 1 anchor close
    VIX_WINDOW_BARS = 21

    def get_5min_return_series(self, ticker: str,
                               window: int = LEADERSHIP_WINDOW_BARS
                               ) -> Optional[List[float]]:
        """
        Return up to ``window-1`` consecutive 5-minute log-style returns
        for *ticker* (most recent last).  Used by the Z-scored leadership
        signal which needs a rolling stdev, not just the latest bar.

        Cached for ``INTRADAY_RETURN_TTL`` seconds.  Returns ``None`` if
        the bar feed yields fewer than ``OPEN_BAR_SKIP + 2`` bars (e.g.
        very early in the session, before the rolling stdev is reliable).
        """
        now = time.monotonic()
        cached = self._intraday_return_series_cache.get(ticker)
        if cached is not None:
            cached_series, cached_at = cached
            if (now - cached_at) < INTRADAY_RETURN_TTL:
                logger.debug("[%s] 5-min series cache HIT (%d bars)",
                             ticker, len(cached_series))
                return cached_series

        url = f"{self.alpaca_data_url}/stocks/{ticker}/bars"
        feed = os.getenv("ALPACA_STOCKS_FEED", "iex").strip() or "iex"
        params = {
            "timeframe": "5Min",
            "limit": int(window),
            "adjustment": "raw",
            "feed": feed,
            "end": self._last_completed_5min_end(),
        }
        try:
            resp = requests.get(url, headers=self._alpaca_headers(),
                                params=params, timeout=ALPACA_TIMEOUT)
            resp.raise_for_status()
            bars = resp.json().get("bars") or []
        except requests.RequestException as exc:
            logger.warning("[%s] 5-min series fetch failed: %s", ticker, exc)
            return None

        # Drop the first OPEN_BAR_SKIP bars to suppress the open-print spike
        # (the open auction often prints 3-4× the rest of the day's vol).
        bars = bars[self.OPEN_BAR_SKIP:]
        if len(bars) < 2:
            logger.debug("[%s] Only %d bars after open-skip — series too short",
                         ticker, len(bars))
            return None

        closes = [float(b["c"]) for b in bars if b.get("c")]
        returns: List[float] = []
        for prev, curr in zip(closes, closes[1:]):
            if prev > 0:
                returns.append((curr / prev) - 1.0)

        if len(returns) < 2:
            return None

        self._intraday_return_series_cache[ticker] = (returns, time.monotonic())
        logger.debug("[%s] 5-min series fetched (%d returns)", ticker, len(returns))
        return returns

    def get_leadership_zscore(self, ticker: str, anchor: str,
                              window: int = LEADERSHIP_WINDOW_BARS
                              ) -> Optional[Tuple[float, float]]:
        """
        Compute the Z-scored 5-minute return differential of
        ``ticker - anchor`` over a rolling ``window``-bar window.

        Returns ``(raw_diff, zscore)`` or ``None`` when either series
        is too short or the rolling stdev is degenerate (zero variance).

        Z-score interpretation:
          * ``zscore > 0``  → ticker is currently leading the anchor
          * ``zscore > 1.5`` → leadership is statistically significant
          * ``zscore < 0``  → ticker is lagging
        """
        if ticker == anchor:
            return None  # Self-comparison is always 0 — useless signal.

        ticker_series = self.get_5min_return_series(ticker, window)
        anchor_series = self.get_5min_return_series(anchor, window)
        if not ticker_series or not anchor_series:
            return None

        # Align tail-aligned (most recent N items where N = min length)
        n = min(len(ticker_series), len(anchor_series))
        if n < 2:
            return None
        diffs = [t - a for t, a in zip(ticker_series[-n:], anchor_series[-n:])]

        # Population stdev — we have the full intraday window, not a sample
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / n
        std = var ** 0.5
        if std <= 1e-9:                       # degenerate: zero variance
            return None

        raw_diff = diffs[-1]
        zscore = (raw_diff - mean) / std
        return (raw_diff, zscore)

    # ------------------------------------------------------------------
    # Inter-market gate (VIX volatility index) — yfinance source
    # ------------------------------------------------------------------

    def get_vix_zscore(self, window: int = VIX_WINDOW_BARS,
                       symbol: str = "^VIX"
                       ) -> Optional[Tuple[float, float]]:
        """
        Fetch the latest 5-min ``^VIX`` change and Z-score it over a
        rolling window.  Returns ``(raw_change, zscore)`` or ``None``
        when yfinance data is unavailable.

        ``zscore > 0`` means VIX is rising vs its rolling mean — i.e.
        an inter-market fear spike.  Strategies that open new short
        premium (Bull Put, Iron Condor put wing) should be inhibited
        when ``zscore > VIX_INHIBIT_ZSCORE`` (default +2.0σ).

        Cached at ``self._vix_zscore_cache`` for ``INTRADAY_RETURN_TTL``
        so the per-cycle classifier loop only fetches once.

        ^VIX is *not* on the Alpaca tradable feed, so we source it from
        yfinance.  If yfinance isn't installed or the call fails we
        return ``None`` and the gate becomes a soft no-op (callers
        should treat ``None`` as "no inhibit signal available").
        """
        now = time.monotonic()
        if self._vix_zscore_cache is not None:
            raw, zscore, ts = self._vix_zscore_cache
            if (now - ts) < INTRADAY_RETURN_TTL:
                logger.debug("VIX z-score cache HIT (raw=%.4f z=%.2f)", raw, zscore)
                return (raw, zscore)

        if yf is None:
            return None

        try:
            ticker = yf.Ticker(symbol)
            # 1d / 5m gives us up to ~78 5-min bars in a regular session
            df = ticker.history(period="1d", interval="5m", auto_adjust=False)
            if df is None or df.empty or len(df) < self.OPEN_BAR_SKIP + 2:
                return None
        except Exception as exc:
            logger.warning("VIX yfinance fetch failed (%s): %s", symbol, exc)
            return None

        # Skip the open spike for VIX too — the 9:30 print is dominated
        # by overnight gap math and is structurally noisy.
        df = df.iloc[self.OPEN_BAR_SKIP:].tail(int(window))
        closes = df["Close"].astype(float).tolist()
        if len(closes) < 2:
            return None

        # 5-min change in VIX *level* (not %) — a 0.5-point VIX move in
        # 5 minutes is meaningful regardless of starting VIX value.
        diffs = [b - a for a, b in zip(closes, closes[1:])]
        if len(diffs) < 2:
            return None
        mean = sum(diffs) / len(diffs)
        var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
        std = var ** 0.5
        if std <= 1e-6:
            return None
        raw = diffs[-1]
        zscore = (raw - mean) / std
        self._vix_zscore_cache = (raw, zscore, time.monotonic())
        logger.info("VIX 5-min change %.3f pts, z-score %+.2f σ", raw, zscore)
        return (raw, zscore)

    # ------------------------------------------------------------------
    # Cached-price query — AccountPort / MarketDataPort seam
    # ------------------------------------------------------------------
    def get_cached_price(self, ticker: str) -> Optional[float]:
        """
        Return the most recently observed price for *ticker*, or None if
        the cache has never seen it.

        This is the public replacement for the prior
        ``provider._snapshot_cache.get(...)`` and
        ``provider._price_cache.get(...)`` access that leaked across
        the module boundary.  Callers (notably ``agent.py`` when
        assembling underlying_prices for the position evaluator) should
        use this method instead.

        Lookup order:
          1. Snapshot cache (fresh real-time price from Alpaca)
          2. Most recent close from the historical price cache

        No network calls are made — this is a pure in-memory query.
        """
        snap = self._snapshot_cache.get(ticker)
        if snap is not None:
            price, _ts = snap
            return float(price)
        frame = self._price_cache.get(ticker)
        if frame is not None and not frame.empty:
            return float(frame["Close"].iloc[-1])
        return None

    # ------------------------------------------------------------------
    # AccountPort — no base_url parameter (adapter owns its endpoints)
    # ------------------------------------------------------------------
    def get_account_info(self) -> Optional[Dict]:
        """Fetch paper trading account information from Alpaca.

        Reads the trading endpoint from ``self.alpaca_base_url`` — the
        caller no longer passes it in (week 5-6 AccountPort refactor).
        """
        url = f"{self.alpaca_base_url}/account"
        try:
            resp = requests.get(url, headers=self._alpaca_headers(), timeout=ALPACA_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("Failed to fetch account info: %s", exc)
            return None

    def is_market_open(self) -> bool:
        """Check if the market is currently open via Alpaca clock API.

        Reads the trading endpoint from ``self.alpaca_base_url`` — the
        caller no longer passes it in (week 5-6 AccountPort refactor).
        """
        url = f"{self.alpaca_base_url}/clock"
        try:
            resp = requests.get(url, headers=self._alpaca_headers(), timeout=ALPACA_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("is_open", False)
        except requests.RequestException as exc:
            logger.error("Failed to check market clock: %s", exc)
            return False
