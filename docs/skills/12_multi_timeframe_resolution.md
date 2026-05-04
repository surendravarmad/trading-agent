# Multi-timeframe regime resolution

> **One-line summary:** Compute a `RegimeAnalysis` for any intraday timeframe (1h, 4h, …) by reusing the **identical** `_determine_regime` rule on resampled bars. Macro overlays (VIX-z, Leadership-z, IV rank) are populated on every intraday row too, each wrapped in try/except so a single failed RPC can't blank the row.
> **Source of truth:** [`trading_agent/multi_tf_regime.py:225-361`](../../trading_agent/multi_tf_regime.py)
> **Phase:** 1  •  **Group:** regime
> **Depends on:** [11 Six-regime classifier](11_six_regime_classifier.md), [08 Leadership Z-score](08_leadership_zscore.md), [09 VIX Z-score inhibitor](09_vix_zscore_inhibitor.md), [07 Anchor map for leadership](07_anchor_map_for_leadership.md).
> **Consumed by:** Watchlist UI's per-timeframe rows; future ML feature pipelines.

---

## 1. Theory & Objective

The watchlist tab needs a regime label for the same ticker at multiple timeframes (daily for trend context, 4h for swing setup, 1h for entry timing). A naive design would fork the classifier into per-timeframe variants. That immediately violates the "single source of truth" architectural invariant and creates three places to keep in sync.

Instead, this wrapper **fetches intraday bars, computes the same indicators (SMA, RSI, BB, slope), and calls the same `_determine_regime` rule on them**. The classifier doesn't know or care that the bars are intraday — it sees the same shape of input either way. This is the cleanest expression of the invariant: one rule, many timeframes.

The `_classify_intraday` function additionally **populates macro overlays** (VIX-z, Leadership-z, IV rank, trend-conflict) on every intraday `RegimeAnalysis`. Originally these were left at zero defaults on intraday rows, which silently blanked them in the UI. The fix populates them with the same RPCs the daily path uses, but **wrapped in try/except** — a single failed RPC (e.g. VIX feed hiccup) shouldn't blank an otherwise-good intraday row.

The macro values are *market-wide*, not timeframe-specific. Including them on every intraday `RegimeAnalysis` is a deliberate redundancy: it keeps consumers honest. Anything that reads an intraday object directly (the backtester scaffolding, future ML features) sees real values instead of zeros pretending to be data.

The `last_bar_ts` field is captured here too (best-effort; never raises) so the watchlist's stale-data ⏰ chip can compare against wall clock.

## 2. Mathematical Formula

N/A — this is composition / control flow over primitives that have their own skills. The math is in:

- [11 Six-regime classifier](11_six_regime_classifier.md) — the core rule.
- [10 ADX with Wilder smoothing](10_adx_wilder_smoothing.md) — sibling indicator (separate function in same module).
- [08 Leadership Z-score](08_leadership_zscore.md) — populated overlay.
- [09 VIX Z-score inhibitor](09_vix_zscore_inhibitor.md) — populated overlay.

## 3. Reference Python Implementation

```python
# trading_agent/multi_tf_regime.py:225-361
def _classify_intraday(
    ticker: str,
    interval: str,
    data: MarketDataProvider,
) -> RegimeAnalysis:
    """Fetch intraday bars, compute SMAs/RSI/BB, run _determine_regime."""
    short_w, long_w = _SMA_WINDOWS[interval]
    bars = data.fetch_intraday_bars(ticker, interval)

    close: pd.Series = bars["Close"]
    if len(close) < long_w + 5:
        # Mirror the daily-path failure mode: refuse rather than silently
        # falling through to a degenerate SIDEWAYS classification when
        # the long SMA is mostly NaN.
        raise ValueError(
            f"{ticker} @ {interval}: only {len(close)} bars, "
            f"need at least {long_w + 5} for stable SMA-{long_w}"
        )

    # SAME static helpers used by RegimeClassifier — guaranteed parity.
    sma_short = data.compute_sma(close, short_w)
    sma_long = data.compute_sma(close, long_w)
    rsi = data.compute_rsi(close, 14)
    upper, middle, lower = data.compute_bollinger_bands(close, 20, 2.0)
    upper_3std, _, lower_3std = data.compute_bollinger_bands(close, 20, 3.0)
    sma_short_slope = data.sma_slope(sma_short, lookback=5)
    sma_long_slope = data.sma_slope(sma_long, lookback=5)

    current_price = float(close.iloc[-1])
    bb_width = float((upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1])
    u3 = float(upper_3std.iloc[-1])
    l3 = float(lower_3std.iloc[-1])

    mr_upper = current_price >= u3
    mr_lower = current_price <= l3
    mean_reversion_signal = mr_upper or mr_lower
    mean_reversion_direction = (
        "upper" if mr_upper else "lower" if mr_lower else ""
    )

    # Reuse the static classifier rule — single source of truth.
    classifier = RegimeClassifier(data)
    regime, reasoning = classifier._determine_regime(
        current_price,
        float(sma_short.iloc[-1]),
        float(sma_long.iloc[-1]),
        sma_short_slope,
        bb_width,
        mean_reversion_signal,
        mean_reversion_direction,
        u3,
        l3,
    )

    # ------------------------------------------------------------------
    # Macro / inter-market signals (previously left at defaults).
    # ------------------------------------------------------------------
    # Even though VIX z-score and leadership z-score are *market-wide*
    # signals (not specific to any timeframe), populating them on every
    # RegimeAnalysis keeps consumers honest — anything that reads the
    # intraday object directly (backtester scaffolding, future ML
    # features) sees real values instead of zeros pretending to be data.
    # All three are wrapped in try/except so a single failed RPC can't
    # blank an otherwise-good intraday row.
    leadership_anchor = leadership_anchor_for(ticker)
    leadership_zscore = 0.0
    leadership_raw_diff = 0.0
    leadership_signal_available = False
    if leadership_anchor and hasattr(data, "get_leadership_zscore"):
        try:
            result = data.get_leadership_zscore(ticker, leadership_anchor)
            if result is not None:
                leadership_raw_diff, leadership_zscore = result
                leadership_signal_available = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("[%s] leadership_zscore @ %s failed: %s",
                         ticker, interval, exc)

    vix_zscore = 0.0
    inter_market_inhibit_bullish = False
    if hasattr(data, "get_vix_zscore"):
        try:
            vix_result = data.get_vix_zscore()
            if vix_result is not None:
                _, vix_zscore = vix_result
                inter_market_inhibit_bullish = vix_zscore > VIX_INHIBIT_ZSCORE
        except Exception as exc:  # noqa: BLE001
            logger.debug("[%s] vix_zscore @ %s failed: %s",
                         ticker, interval, exc)

    # IV rank from realized-vol percentile — same algorithm as the daily
    # path, just fed intraday closes.  Returns 0.0 when the bar history
    # is too short for a stable percentile, which is the same fallback
    # the daily classifier uses.
    try:
        iv_rank, high_iv_warning = RegimeClassifier._compute_iv_rank(close)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[%s] iv_rank @ %s failed: %s", ticker, interval, exc)
        iv_rank, high_iv_warning = 0.0, False

    # Trend conflict — same rule the daily path applies.
    trend_conflict = (
        (sma_short_slope < 0 and sma_long_slope > 0) or
        (sma_short_slope > 0 and sma_long_slope < 0)
    )

    # Last bar timestamp — for the watchlist Stale-data chip.
    last_bar_ts = None
    try:
        ts = close.index[-1]
        last_bar_ts = pd.Timestamp(ts).to_pydatetime()
    except Exception:  # noqa: BLE001
        last_bar_ts = None

    return RegimeAnalysis(
        regime=regime,
        current_price=current_price,
        sma_50=float(sma_short.iloc[-1]),   # short window labelled as sma_50
        sma_200=float(sma_long.iloc[-1]),   # long window labelled as sma_200
        sma_50_slope=sma_short_slope,
        rsi_14=float(rsi.iloc[-1]),
        bollinger_width=bb_width,
        reasoning=reasoning,
        mean_reversion_signal=mean_reversion_signal,
        mean_reversion_direction=mean_reversion_direction,
        leadership_anchor=leadership_anchor,
        leadership_zscore=leadership_zscore,
        leadership_raw_diff=leadership_raw_diff,
        leadership_signal_available=leadership_signal_available,
        vix_zscore=vix_zscore,
        inter_market_inhibit_bullish=inter_market_inhibit_bullish,
        iv_rank=iv_rank,
        high_iv_warning=high_iv_warning,
        sma_200_slope=sma_long_slope,
        trend_conflict=trend_conflict,
        last_bar_ts=last_bar_ts,
    )
```

## 4. Edge Cases / Guardrails

- **Insufficient bars** — explicitly raises `ValueError` rather than silently returning a degenerate SIDEWAYS. Mirrors the daily path's failure mode. The caller (watchlist refresh loop) catches this and renders an "insufficient data" chip on the row.
- **Single failed RPC** — each macro overlay is wrapped in its own try/except. A VIX feed hiccup leaves `vix_zscore = 0.0, inter_market_inhibit_bullish = False`. The leadership signal still populates if its RPC succeeds. The row stays useful.
- **Sentinel pattern** — `leadership_signal_available = True` only inside the `if result is not None:` branch. If the RPC throws or returns `None`, the sentinel stays `False` and the UI renders `— (no data vs X)`. Critical: do not move the assignment outside the try/except; that would set `True` even when the RPC failed.
- **Macro overlays on intraday rows** — these values are market-wide, not timeframe-specific. The same VIX-z appears on the 1h, 4h, and daily rows for every ticker. That's intentional. Don't try to compute a "1h VIX-z" — VIX is a single instrument with one Z-score per fetch.
- **`compute_iv_rank` access** — calls a private classmethod (`RegimeClassifier._compute_iv_rank`). This is technically a leaky abstraction. The wrapper module imports the class to reuse one method; if the daily path's IV-rank algorithm changes, intraday automatically follows.
- **`last_bar_ts` swallows all exceptions** — best-effort; if `pd.Timestamp(...)` fails (weird index type), we silently set `None`. The watchlist UI handles `None` gracefully ("no timestamp"). Logging the exception was rejected as too noisy for an opportunistic field.
- **Field names misnomer** — `sma_50` and `sma_200` on intraday rows actually hold the **short** and **long** window for that timeframe (e.g. `sma_50` on a 1h row is really `sma_short` over short_w bars). The field name is daily-anchored but reused for intraday for schema consistency. The inline comment documents this; don't "correct" the naming without updating the watchlist UI and tests.

## 5. Cross-References

- [11 Six-regime classifier](11_six_regime_classifier.md) — the rule this wrapper reuses.
- [10 ADX with Wilder smoothing](10_adx_wilder_smoothing.md) — sibling indicator in the same module.
- [08 Leadership Z-score](08_leadership_zscore.md), [09 VIX Z-score inhibitor](09_vix_zscore_inhibitor.md), [07 Anchor map for leadership](07_anchor_map_for_leadership.md) — overlays populated here.

---

*Last verified against repo HEAD on 2026-05-03.*
