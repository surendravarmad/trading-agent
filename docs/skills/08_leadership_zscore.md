# Leadership Z-score

> **One-line summary:** Z-score the most-recent 5-minute return *differential* of `ticker − anchor` over a 21-bar window. `|z| > 1.5σ` is statistically significant directional leadership.
> **Source of truth:** [`trading_agent/market_data.py:1075, 1138-1176`](../../trading_agent/market_data.py); [`trading_agent/regime.py:274-282`](../../trading_agent/regime.py)
> **Phase:** 1  •  **Group:** bias
> **Depends on:** [07 Anchor map for leadership](07_anchor_map_for_leadership.md).
> **Consumed by:** `strategy.py:268-278` (Bull Put routing in SIDEWAYS regime), `thesis_builder.py:38-65` (justification text), `streamlit/watchlist_ui.py` (read-only badge).

---

## 1. Theory & Objective

Two aligned tickers should move together. When they don't, the divergence is information. We measure how unusual today's divergence is relative to its own recent distribution by Z-scoring the return differential.

For ticker `T` and anchor `A`, on each 5-minute bar `i`:

```
diff_i = return_T_i − return_A_i
```

Compute the rolling mean and standard deviation over the last 21 bars (`LEADERSHIP_WINDOW_BARS = 21`, i.e. ~105 minutes of intraday data). The Z-score of the **latest** differential tells us how many σ today's bar deviates from the recent norm.

`|z| > 1.5σ` is the threshold for "this is leadership, not noise." We picked 1.5 rather than the conventional 2.0 because:

- 5-minute returns are heavy-tailed; 2.0σ is too rare to fire often enough.
- 1.5σ corresponds to ~p < 0.07 under a Gaussian — still a real signal, just less stringent.
- The downstream router uses this only for **directional bias** (bullish vs bearish vertical), not for go/no-go. False positives have low cost.

**Population stdev**, not sample. We have the full intraday window, not a sample drawn from a larger population. Using sample stdev (`/(n-1)`) would inflate the variance and shrink z-scores by a small but systematic amount.

The function returns `None` (not `0.0`) when computation is impossible (insufficient bars, zero variance, self-anchor). Callers must distinguish "no data" from "real 0σ" — that's what `leadership_signal_available: bool` exists for.

## 2. Mathematical Formula

```text
For bars i = 1..n where n = min(len(T_returns), len(A_returns)):
  diff_i = r_T_i − r_A_i

  μ      = (1/n) × Σ diff_i
  σ²     = (1/n) × Σ (diff_i − μ)²       ← population variance, NOT n−1
  σ      = √σ²

  raw_diff = diff_n                       ← most recent bar
  z        = (raw_diff − μ) / σ           ← undefined if σ ≤ 1e-9

return (raw_diff, z)   if σ > 1e-9 and n ≥ 2
return None            otherwise
```

Bias threshold: `|z| > 1.5σ` triggers downstream routing. `z > 0` means `T` is leading `A`; `z < 0` means lagging.

## 3. Reference Python Implementation

```python
# trading_agent/market_data.py:1075
LEADERSHIP_WINDOW_BARS = 21          # 20 returns + 1 anchor close
```

```python
# trading_agent/market_data.py:1138-1176
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
```

```python
# trading_agent/regime.py:274-282 — caller side, sets the sentinel
leadership_anchor = leadership_anchor_for(ticker)
leadership_zscore = 0.0
leadership_raw_diff = 0.0
leadership_signal_available = False
if leadership_anchor and hasattr(self.data, "get_leadership_zscore"):
    result = self.data.get_leadership_zscore(ticker, leadership_anchor)
    if result is not None:
        leadership_raw_diff, leadership_zscore = result
        leadership_signal_available = True
```

## 4. Edge Cases / Guardrails

- **Self-anchor (`ticker == anchor`)** — returns `None`. Z would be 0/0; we refuse to compute.
- **Either return series empty** — returns `None`. Common on weekends, off-hours, or when an anchor is a thinly-traded sector ETF (XLF on IEX feed often has < 21 bars after `OPEN_BAR_SKIP`).
- **`n < 2`** — variance is undefined; returns `None`.
- **Zero variance (`std ≤ 1e-9`)** — happens when both series moved in lockstep (rare but possible on a quiet morning). Returns `None` rather than dividing by ~0.
- **Sentinel pattern (`leadership_signal_available`)** — the caller (regime.py:274-282) sets a boolean to `True` only when the RPC returned real data. Without this, `leadership_zscore = 0.0` is ambiguous: did the anchor lock-step (real 0σ) or did the RPC fail? The watchlist UI uses the sentinel to render `— (no data vs X)` in the failed case.
- **Population vs sample stdev** — we use population (`/n`). A junior reader who "fixes" this to sample (`/(n-1)`) will systematically shrink all z-scores by a few percent. That's a regression, not an improvement.
- **Tail alignment** — when the two series have different lengths, we tail-align: take the most recent `n = min(len_T, len_A)` from each. We do **not** align by timestamp; we trust the upstream `get_5min_return_series` to produce contiguous bars from the same wall-clock window.

## 5. Cross-References

- [07 Anchor map for leadership](07_anchor_map_for_leadership.md) — supplies the `anchor` argument.
- [11 Six-regime classifier](11_six_regime_classifier.md) — populates `leadership_zscore` on the daily `RegimeAnalysis`.
- [12 Multi-timeframe regime resolution](12_multi_timeframe_resolution.md) — populates it on every intraday `RegimeAnalysis` too, with try/except so a single failure doesn't blank the row.
- (Phase 2) `17_signal_availability_sentinel.md` — generalizes the `leadership_signal_available` pattern to other macro overlays.

---

*Last verified against repo HEAD on 2026-05-03.*
