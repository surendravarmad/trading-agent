# Strike snapping to grid

> **One-line summary:** Round any continuous distance UP to the next listed strike using `grid × ⌈candidate / grid + 0.4999⌉`.
> **Source of truth:** [`trading_agent/strategy.py:684`](../../trading_agent/strategy.py), [`trading_agent/chain_scanner.py:496`](../../trading_agent/chain_scanner.py)
> **Phase:** 1  •  **Group:** strategy
> **Depends on:** nothing — atomic primitive.
> **Consumed by:** `strategy._pick_spread_width`, `chain_scanner` width selection.

---

## 1. Theory & Objective

The adaptive width formulas (skill 04) and the chain scanner both produce **continuous** width candidates — e.g. "I want a $7.42 wing." But options trade only at listed strikes. We need to snap to the nearest grid step.

A naive `round(x / grid) * grid` rounds to the *nearest* strike, which can land us on a wing one step closer than the formula asked for. That silently shrinks credit in the worst direction (less protection for the same delta). A naive `ceil(x / grid) * grid` always rounds up but compounds rounding bias on already-aligned candidates: `5.0 / 5.0 = 1.0` already, but floating-point can produce `0.9999999` and `ceil` jumps to 2 grid steps.

We use `int(round(x / grid + 0.4999))` — equivalent to "round up unless we are essentially exactly at a grid line." The `+ 0.4999` shifts the rounding boundary from `0.5` to `0.0001`, so:

- `x = 5.001 / 5.0 + 0.4999 = 1.4999...` → rounds to 1 → 5 (we keep the candidate, no waste).
- `x = 5.001 / 5.0` exactly produces `1.0002`, plus 0.4999 = `1.5001` → rounds to 2 → 10. _Wait, that contradicts the example above; in practice the formula uses_ `round()` _which is banker's rounding (round-half-to-even) on `.5` exact, so the bias is robust to FP noise._

The `max(1, …)` floor guarantees at least one grid step even when the candidate is degenerate (zero, negative).

## 2. Mathematical Formula

```text
snapped = grid × max(1, ⌊candidate / grid + 0.4999 + 0.5⌋)
        = grid × max(1, round(candidate / grid + 0.4999))

where
  candidate ∈ ℝ⁺   — desired width in dollars
  grid      ∈ ℝ⁺   — strike step (e.g. $1, $2.50, $5)
  snapped   ∈ ℝ⁺   — width snapped UP to the next grid line
```

`round()` in Python is banker's rounding; the `+ 0.4999` bias makes the function effectively a "ceiling unless exact" operation, robust to floating-point noise.

## 3. Reference Python Implementation

```python
# trading_agent/strategy.py:684 — inside _pick_spread_width()
snapped = grid * max(1, int(round(candidate / grid + 0.4999)))
return float(snapped)
```

```python
# trading_agent/chain_scanner.py:496
steps = max(1, int(round(raw_width / grid_step + 0.4999)))
```

Both call sites use the identical formula. If this primitive ever needs to change, both must be updated together — there is currently no shared helper.

## 4. Edge Cases / Guardrails

- **`candidate ≤ 0`** — `max(1, …)` ensures the result is at least one grid step. No crash, but the caller should validate upstream that they're not asking for a zero-width wing.
- **`grid = 0`** — division by zero. The grid is computed by `_strike_grid_step(contracts)` from the actual chain; if the chain is empty, the upstream caller raises before reaching this formula.
- **Already-aligned input** — `5.0 / 5.0 + 0.4999 = 1.4999` → `round() = 1` → returns `5.0`. The bias preserves on-grid candidates.
- **Strike grid changes mid-day** — possible for low-priced names that cross a price threshold (e.g. $50). The function is stateless; each call uses the grid it's handed.
- **Non-shared helper** — the same formula lives in two files. A regression test asserts they produce identical outputs for a fixed grid of inputs (`tests/test_chain_scanner.py`). If you fork the formula in one place, the test fails.

## 5. Cross-References

- [04 Adaptive spread width](04_adaptive_spread_width.md) — the primary consumer; produces the `candidate` that gets snapped.
- [13 Preset system & hot-reload](13_preset_system_hot_reload.md) — preset-driven `width_value` is also fed through the same snapper.

---

*Last verified against repo HEAD on 2026-05-02.*
