# Six-regime classifier

> **One-line summary:** A priority-ordered rule that classifies a price/SMA/Bollinger snapshot into one of {Bullish, Bearish, Sideways, Mean-Reversion}. Mean-Reversion (3-σ band touch) wins over everything; narrow Bollinger (< 4 %) wins over trend; SMA-200 + slope splits trend bull/bear; everything else falls through to Sideways.
> **Source of truth:** [`trading_agent/regime.py:209, 412-455`](../../trading_agent/regime.py)
> **Phase:** 1  •  **Group:** regime
> **Depends on:** Bollinger Bands, SMA-50/200, slope estimation (utility primitives in `data` module).
> **Consumed by:** `RegimeClassifier.classify` (daily entry point); `multi_tf_regime._classify_intraday` (reuses identical rule for intraday — single source of truth).

---

## 1. Theory & Objective

The classifier exists to drive strategy dispatch:

- **Bullish** → Bull Put.
- **Bearish** → Bear Call.
- **Sideways** → Bull Put or Bear Call depending on leadership Z-score (skill 08), or Iron Condor.
- **Mean-Reversion** → Mean-Reversion strategy (sell credit *against* the move expecting reversal).

We want the rules to be **deterministic**, **priority-ordered**, and **single-sourced**. There must be exactly one function in the codebase that maps `(price, sma, bb_width, …) → Regime`. The architectural invariant scanner (`scripts/checks/scan_invariant_check.py`) enforces this — any second implementation is a CI failure.

The priority order matters:

1. **Mean-Reversion first.** A 3-σ Bollinger touch is a strong signal that overrides any trend reading. Even in a strong bull market, an over-extension above the 3-σ upper band is more actionable as a fade than as a continuation.
2. **Sideways second.** If Bollinger width is below 4 %, vol is too low for a directional view to be reliable. Default to Sideways regardless of where price sits on the SMAs.
3. **Bullish trend.** Price above SMA-200 *and* SMA-50 sloping up. Both conditions; an upward-sloping SMA-50 below SMA-200 is a relief rally, not a trend.
4. **Bearish trend.** Mirror image.
5. **Sideways fallthrough.** Anything that doesn't match the above is mixed-signal → Sideways.

The 4 % Bollinger-width threshold (`BOLLINGER_NARROW_THRESHOLD = 0.04`) was chosen empirically from rolling-window analysis of SPY/QQQ/IWM 2018–2024: bandwidth crossings of this threshold approximated the visual eye-test for "consolidation vs trending."

The function is **stateless and pure** — same inputs always produce the same output. This is what lets `multi_tf_regime` reuse the identical rule across daily and intraday timeframes without forking the logic.

## 2. Mathematical Formula

Boolean priority cascade — short-circuit on the first match:

```text
inputs:
  price                ∈ ℝ⁺
  sma_50               ∈ ℝ⁺
  sma_200              ∈ ℝ⁺
  slope_50             ∈ ℝ        slope of SMA-50 over recent lookback
  bb_width             ∈ ℝ⁺       (upper − lower) / middle of 2-σ Bollinger
  mean_reversion_signal: bool     True iff price ≥ upper_3σ or ≤ lower_3σ
  mean_reversion_direction: str   "upper" | "lower" | ""
  upper_3std, lower_3std ∈ ℝ⁺    3-σ Bollinger bands

constants:
  BOLLINGER_NARROW_THRESHOLD = 0.04

decision:
  if mean_reversion_signal:                      → MEAN_REVERSION
  if bb_width < BOLLINGER_NARROW_THRESHOLD:      → SIDEWAYS
  if price > sma_200 and slope_50 > 0:           → BULLISH
  if price < sma_200 and slope_50 < 0:           → BEARISH
  otherwise:                                     → SIDEWAYS
```

Each branch returns a `(Regime, reasoning_string)` tuple; the reasoning string surfaces in the `RegimeAnalysis.reasoning` field and in the watchlist UI.

## 3. Reference Python Implementation

```python
# trading_agent/regime.py:209
BOLLINGER_NARROW_THRESHOLD = 0.04  # 4 % width considered "narrow"
```

```python
# trading_agent/regime.py:412-455
def _determine_regime(self, price: float, sma50: float, sma200: float,
                      slope_50: float, bb_width: float,
                      mean_reversion_signal: bool = False,
                      mean_reversion_direction: str = "",
                      upper_3std: float = 0.0,
                      lower_3std: float = 0.0):
    """Core classification logic.

    Mean reversion takes highest priority — a 3-std Bollinger Band
    touch overrides any trend signal.
    """
    # Highest priority: 3-std Bollinger Band touch → Mean Reversion
    if mean_reversion_signal:
        dir_label = ("above upper" if mean_reversion_direction == "upper"
                     else "below lower")
        return (Regime.MEAN_REVERSION,
                f"Price ({price:.2f}) is {dir_label} 3-std Bollinger Band "
                f"(upper={upper_3std:.2f}, lower={lower_3std:.2f}). "
                f"Expect mean reversion.")

    # Sideways check: narrow Bollinger Bands
    if bb_width < self.BOLLINGER_NARROW_THRESHOLD:
        return (Regime.SIDEWAYS,
                f"Bollinger Band width ({bb_width:.4f}) is below "
                f"threshold ({self.BOLLINGER_NARROW_THRESHOLD}), "
                f"indicating low volatility / sideways movement.")

    # Bullish: price above SMA-200 and SMA-50 trending up
    if price > sma200 and slope_50 > 0:
        return (Regime.BULLISH,
                f"Price ({price:.2f}) > SMA-200 ({sma200:.2f}) and "
                f"SMA-50 slope is positive ({slope_50:.4f}).")

    # Bearish: price below SMA-200 and SMA-50 trending down
    if price < sma200 and slope_50 < 0:
        return (Regime.BEARISH,
                f"Price ({price:.2f}) < SMA-200 ({sma200:.2f}) and "
                f"SMA-50 slope is negative ({slope_50:.4f}).")

    # Default: sideways
    return (Regime.SIDEWAYS,
            f"Price ({price:.2f}) is between SMAs or slope direction "
            f"doesn't confirm a trend. SMA-50={sma50:.2f}, "
            f"SMA-200={sma200:.2f}, slope={slope_50:.4f}.")
```

## 4. Edge Cases / Guardrails

- **Both `mean_reversion_signal` and narrow BB true** — Mean-Reversion wins (priority order). A narrow-band 3-σ touch is rare but valid; the over-extension still trades against the band.
- **Price exactly at SMA-200** — `>` and `<` are strict, so price `==` SMA-200 falls through to the Sideways default. Intentional: at the line, there's no trend confirmation.
- **`slope_50 == 0` exactly** — strict `>` and `<`. A perfectly flat SMA-50 falls through to Sideways. In practice this almost never happens with real prices.
- **Bullish without SMA-200 confirmation** — price above SMA-200 but slope_50 negative? Falls through to Sideways. Conservative; mixed signals don't get a trend label.
- **Single source of truth** — `multi_tf_regime._classify_intraday` instantiates `RegimeClassifier` and calls this exact method. Don't write a "shadow" intraday classifier; the architectural invariant scanner will flag it.
- **Reasoning string is part of the contract** — the watchlist UI and the LLM thesis builder both render this string. Changing the format (e.g. removing the SMA values) breaks the tooltip text. If you reformat, also update `tests/test_regime.py` and the UI snapshot tests.
- **Append-only outputs** — `Regime` is an `Enum` with five values (BULLISH, BEARISH, SIDEWAYS, MEAN_REVERSION, plus a NEUTRAL alias). Adding a new regime breaks every `match` / `if` block downstream — feasible but requires a coordinated PR across `strategy.py`, `thesis_builder.py`, and the UI.

## 5. Cross-References

- [12 Multi-timeframe regime resolution](12_multi_timeframe_resolution.md) — wraps this rule across timeframes without forking it.
- [09 VIX Z-score inhibitor](09_vix_zscore_inhibitor.md) — overlay that can demote a Bullish/Sideways routing to Bear Call.
- [08 Leadership Z-score](08_leadership_zscore.md) — overlay that influences Sideways routing direction.
- (Phase 2) `19_bollinger_bandwidth_regime.md` — the 4 % cutoff isolated as its own skill.

---

*Last verified against repo HEAD on 2026-05-03.*
