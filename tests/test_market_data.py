"""Tests for market data provider — technical indicator calculations."""

import numpy as np
import pandas as pd
import pytest

from trading_agent.market_data import MarketDataProvider, InsufficientDataError


class TestSMA:
    def test_sma_correct_values(self):
        prices = pd.Series([10, 20, 30, 40, 50])
        sma = MarketDataProvider.compute_sma(prices, window=3)
        assert np.isnan(sma.iloc[0])
        assert np.isnan(sma.iloc[1])
        assert sma.iloc[2] == pytest.approx(20.0)
        assert sma.iloc[3] == pytest.approx(30.0)
        assert sma.iloc[4] == pytest.approx(40.0)

    def test_sma_window_larger_than_data(self):
        prices = pd.Series([1, 2, 3])
        sma = MarketDataProvider.compute_sma(prices, window=10)
        assert sma.isna().all()


class TestRSI:
    def test_rsi_range(self, bullish_prices):
        rsi = MarketDataProvider.compute_rsi(bullish_prices["Close"], 14)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_bullish_trend_is_high(self, bullish_prices):
        """In a strong uptrend, RSI should generally be above 50."""
        rsi = MarketDataProvider.compute_rsi(bullish_prices["Close"], 14)
        recent = rsi.tail(20).mean()
        assert recent > 50


class TestBollingerBands:
    def test_bands_ordering(self, bullish_prices):
        upper, middle, lower = MarketDataProvider.compute_bollinger_bands(
            bullish_prices["Close"], 20, 2.0)
        valid_idx = upper.dropna().index
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()


class TestSMASlope:
    def test_positive_slope_for_uptrend(self):
        sma = pd.Series([100, 101, 102, 103, 104])
        slope = MarketDataProvider.sma_slope(sma, lookback=5)
        assert slope > 0

    def test_negative_slope_for_downtrend(self):
        sma = pd.Series([104, 103, 102, 101, 100])
        slope = MarketDataProvider.sma_slope(sma, lookback=5)
        assert slope < 0


class TestTTLCaching:
    def test_snapshot_cache_hit_within_ttl(self, monkeypatch):
        """A cached price within TTL is returned without hitting the network."""
        import time
        provider = MarketDataProvider("k", "s")
        provider._snapshot_cache["SPY"] = (500.0, time.monotonic() - 5)

        # Any network call should raise — a cache hit must not reach requests
        monkeypatch.setattr(
            "requests.get",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("network must not be called on cache hit")),
        )
        price = provider._fetch_alpaca_snapshot_price("SPY")
        assert price == pytest.approx(500.0)

    def test_snapshot_cache_miss_after_ttl(self, monkeypatch):
        import time
        from unittest.mock import MagicMock
        provider = MarketDataProvider("k", "s")
        # Cache entry older than SNAPSHOT_TTL
        from trading_agent.market_data import SNAPSHOT_TTL
        provider._snapshot_cache["SPY"] = (400.0, time.monotonic() - SNAPSHOT_TTL - 10)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "SPY": {"latestTrade": {"p": 505.0}, "dailyBar": {}}}
        monkeypatch.setattr("requests.get", lambda *a, **kw: mock_resp)

        price = provider._fetch_alpaca_snapshot_price("SPY")
        assert price == pytest.approx(505.0)

    def test_option_chain_cache_hit(self, monkeypatch):
        import time
        provider = MarketDataProvider("k", "s")
        fake_chain = [{"symbol": "SPY250425P00480000", "bid": 1.2}]
        provider._option_cache["SPY_2025-04-25_put"] = (fake_chain, time.monotonic() - 30)

        # If cache is hit, requests.get should never be called
        monkeypatch.setattr("requests.get",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                AssertionError("network should not be called")))
        result = provider.fetch_option_chain("SPY", "2025-04-25", "put")
        assert result == fake_chain

    def test_empty_option_chain_not_cached(self, monkeypatch):
        """An empty result must not be cached so a fresh retry is possible."""
        import time
        provider = MarketDataProvider("k", "s")
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"snapshots": {}}   # zero contracts
        monkeypatch.setattr("requests.get", lambda *a, **kw: mock_resp)

        provider.fetch_option_chain("SPY", "2025-04-25", "put")
        # Cache should NOT contain this key
        assert "SPY_2025-04-25_put" not in provider._option_cache

    def test_price_history_cache_hit_within_ttl(self, bullish_prices):
        import time
        provider = MarketDataProvider("k", "s")
        provider._price_cache["SPY"] = bullish_prices
        provider._price_cache_ts["SPY"] = time.monotonic() - 60   # 1 min old

        # fetch_historical_prices should return cache without calling yfinance
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("yfinance.Ticker",
                       lambda *a: (_ for _ in ()).throw(
                           AssertionError("yfinance should not be called")))
            result = provider.fetch_historical_prices("SPY")
        assert result is bullish_prices


class TestBatchSnapshots:
    def test_batch_populates_snapshot_cache(self, monkeypatch):
        from unittest.mock import MagicMock
        provider = MarketDataProvider("k", "s")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "SPY": {"latestTrade": {"p": 555.0}, "dailyBar": {}},
            "QQQ": {"latestTrade": {"p": 440.0}, "dailyBar": {}},
        }
        monkeypatch.setattr("requests.get", lambda *a, **kw: mock_resp)

        prices = provider.fetch_batch_snapshots(["SPY", "QQQ"])

        assert prices["SPY"] == pytest.approx(555.0)
        assert prices["QQQ"] == pytest.approx(440.0)
        assert "SPY" in provider._snapshot_cache
        assert "QQQ" in provider._snapshot_cache

    def test_batch_returns_empty_on_api_failure(self, monkeypatch):
        import requests as req
        provider = MarketDataProvider("k", "s")
        monkeypatch.setattr("requests.get",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                req.RequestException("timeout")))
        result = provider.fetch_batch_snapshots(["SPY"])
        assert result == {}

    def test_batch_empty_ticker_list(self):
        provider = MarketDataProvider("k", "s")
        assert provider.fetch_batch_snapshots([]) == {}


class TestOptionQuotes:
    def test_fetch_option_quotes_no_cache(self, monkeypatch):
        """fetch_option_quotes always hits the network — results are not cached."""
        from unittest.mock import MagicMock
        provider = MarketDataProvider("k", "s")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "snapshots": {
                "SPY250425P00480000": {"latestQuote": {"bp": 1.30, "ap": 1.50}},
                "SPY250425P00475000": {"latestQuote": {"bp": 0.85, "ap": 1.05}},
            }
        }
        monkeypatch.setattr("requests.get", lambda *a, **kw: mock_resp)

        quotes = provider.fetch_option_quotes(
            ["SPY250425P00480000", "SPY250425P00475000"])

        assert quotes["SPY250425P00480000"]["bid"] == pytest.approx(1.30)
        assert quotes["SPY250425P00480000"]["ask"] == pytest.approx(1.50)
        assert quotes["SPY250425P00475000"]["mid"] == pytest.approx(0.95)

    def test_fetch_option_quotes_returns_empty_on_failure(self, monkeypatch):
        import requests as req
        provider = MarketDataProvider("k", "s")
        monkeypatch.setattr("requests.get",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                req.RequestException("timeout")))
        result = provider.fetch_option_quotes(["SPY250425P00480000"])
        assert result == {}


class TestStrikeExtraction:
    def test_standard_occ_symbol(self):
        assert MarketDataProvider._extract_strike("SPY250404P00550000") == 550.0

    def test_decimal_strike(self):
        assert MarketDataProvider._extract_strike("SPY250404C00482500") == 482.5

    def test_invalid_symbol(self):
        assert MarketDataProvider._extract_strike("INVALID") == 0.0


class TestInsufficientDataGuard:
    """fetch_historical_prices must fail loud when too few bars are returned,
    so the regime classifier never silently misclassifies as SIDEWAYS."""

    def _patch_yf(self, monkeypatch, returned_df):
        from unittest.mock import MagicMock
        fake_ticker = MagicMock()
        fake_ticker.history = MagicMock(return_value=returned_df)
        monkeypatch.setattr("yfinance.Ticker", lambda *a, **kw: fake_ticker)

    def test_empty_dataframe_raises(self, monkeypatch):
        provider = MarketDataProvider("k", "s")
        self._patch_yf(monkeypatch, pd.DataFrame())
        with pytest.raises(InsufficientDataError, match="No price data"):
            provider.fetch_historical_prices("NEWCO", period_days=200)
        # Cache must not be populated on failure
        assert "NEWCO" not in provider._price_cache

    def test_too_few_bars_raises(self, monkeypatch):
        provider = MarketDataProvider("k", "s")
        # Build a 50-row df even though caller asks for 200
        idx = pd.date_range("2025-01-01", periods=50, freq="B")
        short_df = pd.DataFrame({"Close": np.linspace(100, 110, 50)}, index=idx)
        self._patch_yf(monkeypatch, short_df)
        with pytest.raises(InsufficientDataError, match="only 50 bars"):
            provider.fetch_historical_prices("NEWCO", period_days=200)
        assert "NEWCO" not in provider._price_cache

    def test_exactly_enough_bars_succeeds(self, monkeypatch):
        provider = MarketDataProvider("k", "s")
        idx = pd.date_range("2024-01-01", periods=200, freq="B")
        ok_df = pd.DataFrame({"Close": np.linspace(100, 200, 200)}, index=idx)
        self._patch_yf(monkeypatch, ok_df)
        result = provider.fetch_historical_prices("SPY", period_days=200)
        assert len(result) == 200
        assert "SPY" in provider._price_cache
