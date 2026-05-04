# VIX Z-score inhibitor

> **One-line summary:** When the 5-minute VIX-change Z-score exceeds `+2.0σ`, demote any new BULLISH or SIDEWAYS-bullish-routed plan to a Bear Call Spread instead of a Bull Put. Volatility spikes kill bullish premium-selling.
> **Source of truth:** [`trading_agent/regime.py:117, 293-299`](../../trading_agent/regime.py); [`trading_agent/strategy.py:255-265`](../../trading_agent/strategy.py)
> **Phase:** 1  •  **Group:** bias
> **Depends on:** nothing — the VIX RPC is the sole input.
> **Consumed by:** `strategy._plan_for_ticker` (demotion routing).

---

## 1. Theory & Objective

A bullish credit spread (Bull Put) sold into a rapidly rising VIX is a textbook way to lose money. Even if the underlying doesn't tank, the put-side IV expansion increases the spread's mark-to-market loss and pressures margin. We want a global "no new bullish premium" inhibitor that fires when implied volatility is *spiking* — not just elevated.

The right signal is **rate of change**, not level. VIX = 25 in a calm regime is alarming; VIX = 25 in a stressed regime is normal. We Z-score the 5-minute VIX changes against their own recent rolling distribution to capture the *unexpected* part of the move.

`VIX_INHIBIT_ZSCORE = 2.0` is set at 2σ rather than 1.5σ (the leadership threshold) because the cost of a false positive is meaningful: skipping an otherwise good Bull Put. We want this to fire only on real spikes — flash crashes, surprise FOMC moves, geopolitical shocks. About 2.3 % of bars under a Gaussian; in practice a few times per month.

The inhibitor is **directional**: it doesn't refuse the trade entirely, it **demotes** to the bearish counterpart. A regime that wanted Bull Put becomes a Bear Call. The thinking: volatility spikes have negative skew — when VIX jumps, equities usually fall. So flipping the directional bias on the demoted spread captures the same underlying view (sell premium) without taking the wrong side.

## 2. Mathematical Formula

```text
For VIX 5-minute closes c_1..c_n over the rolling window:
  Δ_i = c_i − c_{i-1}                        ← absolute change

  μ   = mean(Δ)
  σ_v = stdev(Δ)
  z   = (Δ_n − μ) / σ_v

inhibit_bullish = (z > VIX_INHIBIT_ZSCORE)
                = (z > 2.0)

action when inhibit_bullish AND analysis.regime ∈ {BULLISH, SIDEWAYS}:
  → demote to Bear Call Spread instead of Bull Put

where
  VIX_INHIBIT_ZSCORE = 2.0  (constant)
  z                  ∈ ℝ    can be very large during shocks
```

The exact computation lives in `MarketDataProvider.get_vix_zscore()`; the regime classifier consumes it. A `None` return means "VIX feed unavailable" — we fail-open (no inhibition) rather than fail-closed because the cost of *missing* a Bull Put because the VIX feed hiccupped is worse than the cost of one extra Bull Put on a real spike.

## 3. Reference Python Implementation

```python
# trading_agent/regime.py:117
VIX_INHIBIT_ZSCORE: float = 2.0
```

```python
# trading_agent/regime.py:293-299
vix_zscore = 0.0
inter_market_inhibit_bullish = False
if hasattr(self.data, "get_vix_zscore"):
    vix_result = self.data.get_vix_zscore()
    if vix_result is not None:
        _, vix_zscore = vix_result
        inter_market_inhibit_bullish = vix_zscore > VIX_INHIBIT_ZSCORE
```

```python
# trading_agent/strategy.py:255-265
inter_market_inhibit = getattr(
    analysis, "inter_market_inhibit_bullish", False)
if inter_market_inhibit and analysis.regime in (
        Regime.BULLISH, Regime.SIDEWAYS):
    expiration = self._pick_expiration(self.KIND_VERTICAL)
    logger.info(
        "[%s] VIX inter-market inhibit (z=%.2f σ) → demoting "
        "%s to Bear Call Spread, expiration %s",
        ticker, getattr(analysis, "vix_zscore", 0.0),
        analysis.regime.value, expiration)
    return self._plan_bear_call(ticker, analysis, expiration)
```

## 4. Edge Cases / Guardrails

- **VIX feed `None`** — fail-open. `get_vix_zscore()` returning `None` leaves `inter_market_inhibit_bullish = False`. Bull Put proceeds normally. The cost of a false negative (rare) is lower than the cost of false positives across all tickers (frequent).
- **Asymmetric threshold** — only `z > +2.0` inhibits, not `z < -2.0`. A *falling* VIX is a green light for bullish premium, not a red one. We deliberately don't symmetrize.
- **Demotion, not refusal** — the strategy still trades. Bear Call benefits from the same realized-vol expansion that hurts Bull Put. This preserves trade count under stress.
- **Regime gate** — inhibitor only activates for `BULLISH` and `SIDEWAYS`. A `BEARISH` regime already routes Bear Call; no demotion needed. A `MEAN_REVERSION` regime has its own routing path and isn't touched.
- **Why 2.0σ and not 2.5σ** — historical backtest on 2023-2024 paper data showed 2.0 fires ~3x/month with a measurable improvement in Bull Put outcomes; 2.5 fires ~1x/month and the sample size is too small to evaluate. 2.0 is the lowest threshold where the protective effect was statistically distinguishable from noise.
- **Different from the C/W floor** — the C/W floor (skill 03) is a *trade-by-trade* economic gate. The VIX inhibitor is a *cross-cutting routing* override. Don't try to merge them.

## 5. Cross-References

- [08 Leadership Z-score](08_leadership_zscore.md) — sibling overlay; both apply on top of regime, both use the `*_signal_available` pattern (in `08`'s case, Z-score itself; in this case, `inter_market_inhibit_bullish` being `False` when the RPC fails serves the same role).
- [11 Six-regime classifier](11_six_regime_classifier.md) — the regime that the inhibitor decides whether to override.

---

*Last verified against repo HEAD on 2026-05-03.*
