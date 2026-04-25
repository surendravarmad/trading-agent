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


class TestLastCompleted5MinEnd:
    """The RFC-3339 'end' timestamp we send to Alpaca must exclude the
    currently-forming bar. Floor to 5-min boundary, subtract 1 second."""

    def test_floors_to_previous_5min_boundary_minus_one_sec(self):
        from datetime import datetime, timezone
        ref = datetime(2026, 4, 18, 14, 37, 12, tzinfo=timezone.utc)
        end = MarketDataProvider._last_completed_5min_end(ref)
        assert end == "2026-04-18T14:34:59Z"

    def test_exactly_on_boundary(self):
        """At 14:35:00 the 14:35 bar just opened. end must be 14:34:59."""
        from datetime import datetime, timezone
        ref = datetime(2026, 4, 18, 14, 35, 0, tzinfo=timezone.utc)
        end = MarketDataProvider._last_completed_5min_end(ref)
        assert end == "2026-04-18T14:34:59Z"

    def test_naive_datetime_treated_as_utc(self):
        from datetime import datetime
        ref = datetime(2026, 4, 18, 14, 37, 12)  # no tzinfo
        end = MarketDataProvider._last_completed_5min_end(ref)
        assert end == "2026-04-18T14:34:59Z"

    def test_non_utc_datetime_converted(self):
        """A tz-aware ET timestamp must be converted to UTC before flooring."""
        from datetime import datetime, timezone, timedelta
        # 10:37:12 ET (UTC-4 during DST) == 14:37:12 UTC
        et = timezone(timedelta(hours=-4))
        ref = datetime(2026, 4, 18, 10, 37, 12, tzinfo=et)
        end = MarketDataProvider._last_completed_5min_end(ref)
        assert end == "2026-04-18T14:34:59Z"


class TestCompletedBarsRequest:
    """get_5min_return must send an `end` parameter so Alpaca returns only
    completed bars, not the partially-formed current one."""

    def test_request_includes_end_param(self, monkeypatch):
        from unittest.mock import MagicMock
        provider = MarketDataProvider("k", "s")
        captured = {}

        def fake_get(url, headers, params, timeout):
            captured["params"] = params
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "bars": [{"c": 100.0}, {"c": 100.5}]
            }
            return resp

        monkeypatch.setattr("requests.get", fake_get)
        provider.get_5min_return("SPY")

        assert "end" in captured["params"]
        # Format: YYYY-MM-DDTHH:MM:SSZ — and the seconds must be 59
        # (proof of the boundary - 1s construction)
        end_val = captured["params"]["end"]
        assert end_val.endswith("59Z")
        # Minutes digit must end in 4 or 9 (one second before a :00 or :05 boundary)
        minute_str = end_val[14:16]
        assert minute_str[-1] in ("4", "9")


class TestIntradayReturnCache:
    """get_5min_return must dedupe redundant SPY/QQQ benchmark calls within
    a cycle. Successful results cache; failed fetches do not."""

    def _bars_response(self, prev_close: float, last_close: float):
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "bars": [
                {"c": prev_close},
                {"c": last_close},
            ]
        }
        return resp

    def test_cache_hit_within_ttl(self, monkeypatch):
        import time
        provider = MarketDataProvider("k", "s")
        provider._intraday_return_cache["SPY"] = (0.0042, time.monotonic() - 5)

        # Any network call must raise — a cache hit must not reach requests
        monkeypatch.setattr(
            "requests.get",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("network must not be called on cache hit")),
        )
        ret = provider.get_5min_return("SPY")
        assert ret == pytest.approx(0.0042)

    def test_cache_miss_after_ttl(self, monkeypatch):
        import time
        from trading_agent.market_data import INTRADAY_RETURN_TTL
        provider = MarketDataProvider("k", "s")
        # Stale entry — must be ignored
        provider._intraday_return_cache["SPY"] = (
            0.0001, time.monotonic() - INTRADAY_RETURN_TTL - 5)

        monkeypatch.setattr(
            "requests.get",
            lambda *a, **kw: self._bars_response(100.0, 100.5),
        )
        ret = provider.get_5min_return("SPY")
        assert ret == pytest.approx(0.005)

    def test_successful_fetch_populates_cache(self, monkeypatch):
        provider = MarketDataProvider("k", "s")
        monkeypatch.setattr(
            "requests.get",
            lambda *a, **kw: self._bars_response(200.0, 201.0),
        )
        provider.get_5min_return("QQQ")
        assert "QQQ" in provider._intraday_return_cache
        cached_ret, _ = provider._intraday_return_cache["QQQ"]
        assert cached_ret == pytest.approx(0.005)

    def test_failed_fetch_not_cached(self, monkeypatch):
        """A network failure or insufficient-bars response must leave the
        cache empty so the next caller can retry."""
        from unittest.mock import MagicMock
        provider = MarketDataProvider("k", "s")
        # Only one bar returned → not enough for a return calc → returns None
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"bars": [{"c": 100.0}]}
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp)

        ret = provider.get_5min_return("SPY")
        assert ret is None
        assert "SPY" not in provider._intraday_return_cache

    def test_dedupes_repeated_calls_in_one_cycle(self, monkeypatch):
        """Simulates the real call pattern: regime classifier asks for SPY's
        5-min return once per non-benchmark ticker. Should fetch exactly once."""
        provider = MarketDataProvider("k", "s")
        call_count = {"n": 0}

        def fake_get(*a, **kw):
            call_count["n"] += 1
            return self._bars_response(100.0, 100.5)

        monkeypatch.setattr("requests.get", fake_get)

        # Classifier loop: 5 non-benchmark tickers, each asks SPY once
        for _ in range(5):
            provider.get_5min_return("SPY")

        assert call_count["n"] == 1


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


class TestStocksFeedParam:
    """
    Free/basic Alpaca subscriptions cannot read SIP — they 403 on stock
    bars / snapshots without an explicit ``feed`` parameter.  All four
    stock endpoints must send ``feed=iex`` by default and honor the
    ``ALPACA_STOCKS_FEED`` env override for paid SIP users.
    """

    def _capturing_get(self, captured, payload):
        from unittest.mock import MagicMock

        def _get(url, *args, **kwargs):
            captured.append({"url": url, "params": kwargs.get("params", {})})
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = payload
            resp.url = url
            resp.content = b"{}"
            return resp
        return _get

    def test_batch_snapshots_sends_feed_iex(self, monkeypatch):
        captured = []
        monkeypatch.delenv("ALPACA_STOCKS_FEED", raising=False)
        monkeypatch.setattr("requests.get",
                            self._capturing_get(captured, {}))

        provider = MarketDataProvider("k", "s")
        provider.fetch_batch_snapshots(["SPY", "QQQ"])

        assert len(captured) == 1
        assert captured[0]["params"].get("feed") == "iex"

    def test_single_snapshot_sends_feed_iex(self, monkeypatch):
        captured = []
        monkeypatch.delenv("ALPACA_STOCKS_FEED", raising=False)
        monkeypatch.setattr("requests.get",
                            self._capturing_get(
                                captured,
                                {"SPY": {"latestTrade": {"p": 500.0}}}))

        provider = MarketDataProvider("k", "s")
        provider._fetch_alpaca_snapshot_price("SPY")

        assert len(captured) == 1
        assert captured[0]["params"].get("feed") == "iex"

    def test_underlying_bid_ask_sends_feed_iex(self, monkeypatch):
        captured = []
        monkeypatch.delenv("ALPACA_STOCKS_FEED", raising=False)
        monkeypatch.setattr("requests.get",
                            self._capturing_get(
                                captured,
                                {"SPY": {"latestQuote": {"bp": 499.95,
                                                          "ap": 500.05}}}))

        provider = MarketDataProvider("k", "s")
        bid_ask = provider.get_underlying_bid_ask("SPY")

        assert bid_ask == (499.95, 500.05)
        assert len(captured) == 1
        assert captured[0]["params"].get("feed") == "iex"

    def test_5min_return_sends_feed_iex(self, monkeypatch):
        captured = []
        monkeypatch.delenv("ALPACA_STOCKS_FEED", raising=False)
        monkeypatch.setattr("requests.get",
                            self._capturing_get(
                                captured,
                                {"bars": [{"c": 100.0}, {"c": 100.5}]}))

        provider = MarketDataProvider("k", "s")
        provider.get_5min_return("SPY")

        assert len(captured) == 1
        assert captured[0]["params"].get("feed") == "iex"
        # also check end-bar guard wasn't dropped
        assert "end" in captured[0]["params"]

    def test_env_override_uses_sip(self, monkeypatch):
        """Paid SIP customers can opt out of the IEX default."""
        captured = []
        monkeypatch.setenv("ALPACA_STOCKS_FEED", "sip")
        monkeypatch.setattr("requests.get",
                            self._capturing_get(captured, {}))

        provider = MarketDataProvider("k", "s")
        provider.fetch_batch_snapshots(["SPY"])

        assert captured[0]["params"].get("feed") == "sip"


class TestFiveMinReturnSeries:
    """``get_5min_return_series`` — Item 2 dependency.

    Returns up to ``window-1`` 5-minute returns, drops the first
    ``OPEN_BAR_SKIP`` bars, and caches the result for one TTL.
    """

    def _bars_response(self, closes, monkeypatch):
        from unittest.mock import MagicMock

        def _get(url, *args, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "bars": [{"c": float(c)} for c in closes]
            }
            return resp

        monkeypatch.setattr("requests.get", _get)

    def test_drops_open_bars(self, monkeypatch):
        """First ``OPEN_BAR_SKIP=2`` bars are dropped before computing returns."""
        provider = MarketDataProvider("k", "s")
        # 6 bars in → after skipping 2, 4 closes remain → 3 returns
        self._bars_response([90, 95, 100, 101, 102, 103], monkeypatch)
        series = provider.get_5min_return_series("SPY", window=21)
        assert series is not None
        assert len(series) == 3
        # First post-skip return: 101/100 - 1 = 0.01
        assert series[0] == pytest.approx(0.01)

    def test_too_short_after_skip_returns_none(self, monkeypatch):
        """If fewer than 2 bars remain after the open-skip, return None."""
        provider = MarketDataProvider("k", "s")
        # Only 3 bars → after skipping 2, only 1 close remains
        self._bars_response([100, 101, 102], monkeypatch)
        assert provider.get_5min_return_series("SPY", window=21) is None

    def test_cache_hit_within_ttl(self, monkeypatch):
        import time
        provider = MarketDataProvider("k", "s")
        provider._intraday_return_series_cache["SPY"] = (
            [0.001, 0.002], time.monotonic() - 5)
        monkeypatch.setattr(
            "requests.get",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("network must not be called on cache hit")),
        )
        result = provider.get_5min_return_series("SPY")
        assert result == [0.001, 0.002]

    def test_network_failure_returns_none(self, monkeypatch):
        import requests as _requests
        provider = MarketDataProvider("k", "s")
        monkeypatch.setattr(
            "requests.get",
            lambda *a, **kw: (_ for _ in ()).throw(
                _requests.RequestException("boom")),
        )
        assert provider.get_5min_return_series("SPY") is None


class TestLeadershipZScore:
    """``get_leadership_zscore`` — Item 2 main signal."""

    def test_self_anchor_returns_none(self):
        """Self-comparison is degenerate — must return None, not 0.0."""
        provider = MarketDataProvider("k", "s")
        assert provider.get_leadership_zscore("SPY", "SPY") is None

    def test_outperforming_yields_positive_z(self, monkeypatch):
        """Differential equal to its own σ above mean → z = +1.0."""
        provider = MarketDataProvider("k", "s")
        # Build deterministic series where the LAST diff is exactly +1σ above
        # the rolling mean.  Use diff series [0.0, 0.0, 0.0, +stdev_of_set].
        # SPY ret series and QQQ ret series chosen so SPY-QQQ matches above.
        spy_series = [0.001, 0.002, 0.001, 0.005]
        qqq_series = [0.001, 0.002, 0.001, 0.001]
        # Diffs = [0, 0, 0, 0.004], mean = 0.001, stdev (population) = ~0.001732
        # zscore of last diff (0.004) = (0.004 - 0.001) / 0.001732 ≈ +1.732
        provider._intraday_return_series_cache["SPY"] = (
            spy_series, __import__("time").monotonic())
        provider._intraday_return_series_cache["QQQ"] = (
            qqq_series, __import__("time").monotonic())
        result = provider.get_leadership_zscore("SPY", "QQQ")
        assert result is not None
        raw, z = result
        assert raw == pytest.approx(0.004)
        assert z == pytest.approx(1.732, abs=0.01)

    def test_zero_variance_returns_none(self):
        """If the rolling stdev collapses to ~0, the gate is meaningless."""
        provider = MarketDataProvider("k", "s")
        # Diffs all identical → stdev = 0
        provider._intraday_return_series_cache["SPY"] = (
            [0.001, 0.002, 0.003], __import__("time").monotonic())
        provider._intraday_return_series_cache["QQQ"] = (
            [0.001, 0.002, 0.003], __import__("time").monotonic())
        assert provider.get_leadership_zscore("SPY", "QQQ") is None

    def test_missing_series_returns_none(self, monkeypatch):
        """If either side fails to fetch, return None (no signal)."""
        provider = MarketDataProvider("k", "s")
        # Neither cached → both will hit network → cause network to fail
        import requests as _requests
        monkeypatch.setattr(
            "requests.get",
            lambda *a, **kw: (_ for _ in ()).throw(
                _requests.RequestException("boom")),
        )
        assert provider.get_leadership_zscore("SPY", "QQQ") is None


class TestVIXZScore:
    """``get_vix_zscore`` — Item 3 inter-market gate signal."""

    def _patch_yf(self, monkeypatch, closes):
        from unittest.mock import MagicMock
        import pandas as pd
        idx = pd.date_range("2026-04-25 09:30", periods=len(closes),
                            freq="5min")
        df = pd.DataFrame({"Close": [float(c) for c in closes]}, index=idx)
        fake_ticker = MagicMock()
        fake_ticker.history = MagicMock(return_value=df)
        monkeypatch.setattr("yfinance.Ticker", lambda *a, **kw: fake_ticker)

    def test_high_zscore_when_last_change_is_outlier(self, monkeypatch):
        """Final 5-min jump much larger than the rolling mean → high z."""
        provider = MarketDataProvider("k", "s")
        # 6 closes — drop first 2, leaves 4 closes → 3 diffs
        # Diffs = [0.0, 0.0, +1.0]   (mean 0.333, std 0.4714)
        # last z ≈ (1.0 - 0.333) / 0.4714 ≈ +1.414
        self._patch_yf(monkeypatch, [12, 13, 15.0, 15.0, 15.0, 16.0])
        result = provider.get_vix_zscore(window=21)
        assert result is not None
        raw, z = result
        assert raw == pytest.approx(1.0)
        assert z == pytest.approx(1.414, abs=0.01)

    def test_too_few_bars_returns_none(self, monkeypatch):
        provider = MarketDataProvider("k", "s")
        # 3 closes — only 1 left after skipping 2 → cannot compute
        self._patch_yf(monkeypatch, [12, 13, 14])
        assert provider.get_vix_zscore() is None

    def test_yf_failure_returns_none(self, monkeypatch):
        provider = MarketDataProvider("k", "s")
        from unittest.mock import MagicMock
        fake_ticker = MagicMock()
        fake_ticker.history = MagicMock(side_effect=RuntimeError("yf down"))
        monkeypatch.setattr("yfinance.Ticker", lambda *a, **kw: fake_ticker)
        assert provider.get_vix_zscore() is None

    def test_cache_hit_within_ttl(self, monkeypatch):
        import time
        provider = MarketDataProvider("k", "s")
        provider._vix_zscore_cache = (0.4, 1.8, time.monotonic() - 5)
        # yfinance must NOT be called when cache is fresh
        monkeypatch.setattr(
            "yfinance.Ticker",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("yfinance must not be called on cache hit")),
        )
        result = provider.get_vix_zscore()
        assert result == (0.4, 1.8)
