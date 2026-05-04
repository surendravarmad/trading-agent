# ADX with Wilder smoothing

> **One-line summary:** Average Directional Index (Wilder 1978) measuring trend *strength* irrespective of direction. Implemented in pure pandas with `EMA(alpha=1/window, adjust=False)` for Wilder smoothing. ADX < 20 = chop; 20-40 = developing trend; ≥ 40 = strong trend.
> **Source of truth:** [`trading_agent/multi_tf_regime.py:367-423`](../../trading_agent/multi_tf_regime.py)
> **Phase:** 1  •  **Group:** regime
> **Depends on:** nothing — pure indicator math.
> **Consumed by:** Watchlist UI's regime-strength badge; not used in trade routing.

---

## 1. Theory & Objective

ADX is the canonical "is there a trend at all?" indicator. The companion +DI / −DI tell you direction; ADX tells you whether the directional reading is meaningful or noise.

We compute it ourselves in pure pandas/numpy because:

1. The watchlist tab is intentionally dependency-light — it must run without `pandas-ta` installed.
2. Wilder smoothing is just `EMA(alpha = 1/window, adjust=False)`. Trivial to write correctly.
3. We avoid the maintenance cost of pinning a third-party indicator library that gets recompiled on every release.

The *gotcha* this implementation traps: pandas' `replace(0, pd.NA)` coerces a float Series to `object` dtype. Subsequent `.ewm().mean()` on an `object` Series raises `pandas.errors.DataError: No numeric types to aggregate`. The fix is to use `np.nan` instead of `pd.NA`, and force-cast back to float with `pd.to_numeric(..., errors="coerce")`. This bug bit production once; the inline comment exists so it doesn't bite again.

The output is intentionally a single scalar — the most recent ADX reading. We do **not** return the full series because the only consumer is the regime-strength badge, which only needs "now."

## 2. Mathematical Formula

```text
For OHLC bars indexed t = 1..N:
  TR_t       = max(H_t − L_t, |H_t − C_{t−1}|, |L_t − C_{t−1}|)        ← True Range
  +DM_t      = max(H_t − H_{t−1}, 0)  if (H_t − H_{t−1}) > (L_{t−1} − L_t) else 0
  −DM_t      = max(L_{t−1} − L_t, 0)  if (L_{t−1} − L_t) > (H_t − H_{t−1}) else 0

  ATR_t      = WilderEMA(TR, window)
  +DI_t      = 100 × WilderEMA(+DM, window) / ATR_t
  −DI_t      = 100 × WilderEMA(−DM, window) / ATR_t

  DX_t       = 100 × |+DI_t − −DI_t| / (+DI_t + −DI_t)
  ADX_t      = WilderEMA(DX, window)

WilderEMA(x, w) ≡ EMA with alpha = 1/w, adjust=False
                ≡ recursion: y_t = (1 − 1/w) × y_{t−1} + (1/w) × x_t

return ADX_N (most recent)

window = 14 (default)
ADX bands:
  < 20    chop
  20–40   developing trend
  ≥ 40    strong trend
```

**Wilder smoothing ≠ standard EMA.** Standard EMA uses `alpha = 2 / (w + 1)`. Wilder uses `alpha = 1 / w`, which is slower. For window=14: standard `α = 0.133`, Wilder `α = 0.071` — almost half-speed. Mixing them gives ADX values that disagree with TradingView/pandas-ta.

## 3. Reference Python Implementation

```python
# trading_agent/multi_tf_regime.py:367-423
def adx_strength(bars: pd.DataFrame, window: int = 14) -> Optional[float]:
    """
    Compute the latest ADX value from an OHLCV DataFrame.

    ADX (Average Directional Index, Wilder 1978) measures **trend strength**
    irrespective of direction:
      * ADX < 20      → weak / chop
      * 20 ≤ ADX < 40 → developing trend
      * ADX ≥ 40      → strong trend

    Implemented here in pure pandas/numpy so the watchlist tab does not
    require ``pandas-ta`` for the categorical regime-strength badge.
    PR #4 swaps this for the pandas-ta version when the chart panel
    upgrades its infrastructure.
    """
    high = bars["High"]
    low = bars["Low"]
    close = bars["Close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move

    # Wilder smoothing == EMA with alpha=1/window.
    atr = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr)

    # IMPORTANT: replace 0 → np.nan (NOT pd.NA). pd.NA is the masked-array
    # NA scalar; replacing into a float Series with it coerces dtype to
    # `object`, after which `.ewm().mean()` raises
    #   pandas.errors.DataError: No numeric types to aggregate
    # Using np.nan keeps the Series in float64.
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    # Force-cast in case any upstream op (e.g. resample on an empty
    # group, downcast(infer_objects)) leaked an object dtype.
    dx = pd.to_numeric(dx, errors="coerce")
    if dx.dropna().empty:
        return None
    adx = dx.ewm(alpha=1 / window, adjust=False).mean()

    last = adx.iloc[-1]
    if pd.isna(last):
        return None
    return float(last)
```

## 4. Edge Cases / Guardrails

- **Empty / too-short DataFrame** — `dx.dropna().empty` returns `True`, function returns `None`. The badge UI renders "—" instead of crashing.
- **`pd.NA` vs `np.nan`** — using `pd.NA` flips the Series dtype to `object` and breaks `.ewm()`. The inline comment is load-bearing — do not "simplify" it.
- **`pd.to_numeric(..., errors="coerce")`** — defensive force-cast. Some upstream operations (`.resample` on an empty group, `infer_objects()` cleanup) can sneak an `object` dtype past the explicit `np.nan` replacement. The cast keeps things float64.
- **`+DI + −DI = 0`** — would div-by-zero. The `.replace(0, np.nan)` handles it; the resulting `NaN` is dropped by `dropna()`.
- **Last bar `NaN`** — final `pd.isna(last)` guard returns `None` instead of `nan`. Important because the consumer code does `f"ADX={x:.1f}"` and `f"{nan:.1f}"` would print `"nan"` instead of `"—"`.
- **Window = 14 default** — standard for Wilder's original paper. Don't change without re-tuning the 20/40 thresholds. ADX ranges scale with the smoothing window.
- **Not a routing input** — the strategy code never reads ADX. It's purely visual. If you wire it into routing, you also need to wire it into the architectural invariant scan.

## 5. Cross-References

- [11 Six-regime classifier](11_six_regime_classifier.md) — the routing classifier; ADX could be a future input but currently isn't.
- [12 Multi-timeframe regime resolution](12_multi_timeframe_resolution.md) — same module; both were written in pure pandas to keep the watchlist dependency-light.

---

*Last verified against repo HEAD on 2026-05-03.*
