# EV per $ risked scoring

> **One-line summary:** Score every spread candidate by `EV_per_$risked = (POP × C/W − (1−POP) × (1−C/W)) / (1−C/W)`, then pick the highest.
> **Source of truth:** [`trading_agent/chain_scanner.py:165-207`](../../trading_agent/chain_scanner.py)
> **Phase:** 1  •  **Group:** strategy
> **Depends on:** [01 POP from short delta](01_pop_from_delta.md), [03 Credit-to-Width floor](03_credit_to_width_floor.md).
> **Consumed by:** `chain_scanner.SpreadCandidate`, `chain_scanner._score_candidate_with_reason`, `strategy._plan_via_scanner` (adaptive mode).

---

## 1. Theory & Objective

Comparing credit spread candidates by absolute credit is wrong — a $1 credit on a $5 wing is a much better risk-adjusted bet than a $1 credit on a $20 wing. We want a single scalar that reflects the **expected dollar return per dollar of capital actually at risk**.

For a credit spread:

- **Win** (probability POP): keep the credit `C`. Gain per $ width = `C/W`.
- **Lose** (probability 1−POP): pay max loss = `W − C`. Loss per $ width = `1 − C/W`.

So the EV per $ of *width* is `POP × C/W − (1−POP) × (1−C/W)`. But the capital actually tied up is `(W − C) × multiplier`, not `W × multiplier`. To convert to "EV per $ at risk," divide by `1 − C/W`. This makes candidates with very different W comparable on a single axis.

Then the scanner annualizes: `annualized = EV_per_$risked × (365 / DTE)`. This last step is what lets a 7-DTE candidate fairly compete with a 45-DTE candidate. A short-DTE spread compounds the same edge more times per year, so should be preferred even at slightly lower per-trade EV.

The scoring is *only* used in adaptive mode (skill 14). Static mode picks a single point and only checks the C/W floor — no scoring needed.

## 2. Mathematical Formula

```text
POP        = 1 − |Δ_short|                                          ← skill 01
C/W        = credit / width
EV_per_$W  = POP × (C/W) − (1 − POP) × (1 − C/W)                    ← per $ of width
EV_per_$R  = EV_per_$W / (1 − C/W)                                   ← per $ at risk
annualized = EV_per_$R × (365 / DTE)                                ← time-normalized

where
  Δ_short ∈ [-1, 1]    — short-leg delta
  credit  ∈ ℝ⁺         — net credit received per share
  width   ∈ ℝ⁺         — distance between strikes
  DTE     ∈ ℤ⁺         — days to expiration
  EV_per_$R, annualized ∈ ℝ — can be negative (gets filtered out)
```

`1 − C/W > 0` is enforced because `C/W ≥ 1` would imply a debit not a credit (violates the credit-spread assumption upstream).

## 3. Reference Python Implementation

```python
# trading_agent/chain_scanner.py:110-112
def _pop_from_delta(short_delta: float) -> float:
    """POP ≈ 1 − |Δshort| for a vertical credit spread."""
    return max(0.0, 1.0 - abs(short_delta))
```

```python
# trading_agent/chain_scanner.py:165-181
def _ev_per_dollar_risked(credit: float, width: float,
                          short_delta: float) -> Optional[float]:
    """
    EV per dollar at risk for a credit spread.

    Returns None when the trade is structurally invalid (zero/negative
    credit, zero/negative width, or credit ≥ width which would imply a
    debit not a credit).
    """
    if width <= 0 or credit <= 0 or credit >= width:
        return None
    cw = credit / width
    pop = _pop_from_delta(short_delta)
    # gain_per_$1_width = cw   ;   loss_per_$1_width = (1 − cw)
    ev_per_width = pop * cw - (1.0 - pop) * (1.0 - cw)
    # Convert to per-$-risked basis: $-at-risk per $1 width = (1 − cw).
    return ev_per_width / (1.0 - cw)
```

```python
# trading_agent/chain_scanner.py:184-207
def _score_candidate(credit: float, width: float, short_delta: float,
                     dte: int, edge_buffer: float, min_pop: float
                     ) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Score a candidate. Returns ``(cw, pop, cw_floor, ev_per_$risked,
    annualized_score)`` or ``None`` if the candidate fails any hard filter.
    """
    if dte <= 0:
        return None
    pop = _pop_from_delta(short_delta)
    if pop < min_pop:
        return None
    floor = _cw_floor(short_delta, edge_buffer)
    if width <= 0 or credit <= 0:
        return None
    cw = credit / width
    if cw < floor:
        return None
    ev = _ev_per_dollar_risked(credit, width, short_delta)
    if ev is None or ev <= 0:
        # ev > 0 is implied by cw > breakeven, but guard against rounding.
        return None
    annualized = ev * (365.0 / dte)
    return cw, pop, floor, ev, annualized
```

## 4. Edge Cases / Guardrails

- **`credit ≥ width`** — would be a debit not a credit; `_ev_per_dollar_risked` returns `None`. Candidate dropped.
- **`width ≤ 0` or `credit ≤ 0`** — same; returns `None`.
- **`dte ≤ 0`** — annualization would divide by zero or go negative; returns `None`.
- **`pop < min_pop`** — hard gate before scoring (default `min_pop = 0.55`). Saves a few floating-point ops on rejected candidates.
- **`cw < floor`** — the C/W floor (skill 03) is applied **before** EV computation. A candidate that barely edges over breakeven won't survive.
- **`ev ≤ 0` after passing `cw ≥ floor`** — mathematically should be impossible (`cw > |Δ|` ⇔ `EV > 0`), but the explicit guard catches floating-point edge cases on the boundary.
- **Annualization warps short-DTE bias** — a 1-DTE candidate gets a 365× multiplier, which can dominate the ranking. The `min_pop` and `dte_grid` floors keep the scanner from picking 1-DTE pin-risk specials. If you change `dte_grid` to include very-short values, also raise `min_pop`.
- **Comparing across symbols** — the scoring is symbol-agnostic (no IV term, no liquidity term). The risk manager's gates (skill 06 and others) handle cross-symbol differences before execution.

## 5. Cross-References

- [01 POP from short delta](01_pop_from_delta.md) — the win-probability term.
- [03 Credit-to-Width floor](03_credit_to_width_floor.md) — the hard filter applied before scoring.
- [14 Adaptive vs static scan modes](14_adaptive_vs_static_scan_modes.md) — only adaptive mode actually uses these scores; static mode picks a single point.
- [13 Preset system & hot-reload](13_preset_system_hot_reload.md) — `min_pop`, `edge_buffer`, and the scan grids that feed this scoring all live on the preset.

---

*Last verified against repo HEAD on 2026-05-03.*
