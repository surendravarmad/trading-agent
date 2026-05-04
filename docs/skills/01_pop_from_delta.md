# POP from short delta

> **One-line summary:** Approximate Probability of Profit for a vertical credit spread as `1 − |Δ_short|`.
> **Source of truth:** [`trading_agent/chain_scanner.py:110-112`](../../trading_agent/chain_scanner.py)
> **Phase:** 1  •  **Group:** strategy
> **Depends on:** nothing — atomic primitive.
> **Consumed by:** `chain_scanner.py:_score_candidate` (gate), `chain_scanner.py:_ev_per_dollar_risked` (EV term).

---

## 1. Theory & Objective

For an out-of-the-money short option with delta `Δ`, the option's delta is approximately the risk-neutral probability that it expires in the money. So the probability that the spread expires worthless (i.e. POP) is approximately `1 − |Δ_short|`.

This is a **textbook approximation**, not a Black-Scholes simulation. It assumes (a) lognormal terminal distribution, (b) zero risk-free rate effect over short DTE, (c) no early-exercise drag for American options near expiry. For 7–60 DTE credit spreads on liquid US ETFs and large-caps, the approximation is good to within ~2–3 percentage points of a full Monte-Carlo simulation, and crucially it is **monotonic** in `Δ` — which is all the scanner needs to rank candidates.

We deliberately do **not** add a more sophisticated POP model because the scanner uses POP only as (a) a hard gate (`pop ≥ min_pop`) and (b) a coefficient in the EV formula. Both are robust to the few-percent error.

## 2. Mathematical Formula

```text
POP = max(0, 1 − |Δ_short|)

where
  Δ_short ∈ [-1, 1]   — the delta of the SHORT (sold) leg
  POP     ∈ [0, 1]    — probability the spread expires worthless
```

The `max(0, …)` clamp is defensive — for any well-formed option chain `|Δ| ≤ 1`, but Alpaca's `indicative` feed has been observed to return sentinel values like `Δ = 1.5` for deep ITM contracts on illiquid expiries.

## 3. Reference Python Implementation

```python
# trading_agent/chain_scanner.py:110-112
def _pop_from_delta(short_delta: float) -> float:
    """POP ≈ 1 − |Δshort| for a vertical credit spread."""
    return max(0.0, 1.0 - abs(short_delta))
```

## 4. Edge Cases / Guardrails

- **`|Δ| > 1` from a bad feed** — the `max(0, …)` clamp returns 0, which then fails any `min_pop` gate. The candidate is silently dropped; no crash.
- **`Δ = 0` (deep OTM)** — returns POP = 1.0. This is mathematically valid but should never produce a viable spread because credit also approaches 0; the EV gate (skill 05) will reject it.
- **`Δ = sign(`expected`)`** — for a put, `Δ_short` is negative; for a call, positive. The `abs()` makes the helper symmetric so callers don't have to remember.
- **Sign convention** — the function takes the **short** leg's delta. Passing the long leg's delta will give a wildly wrong POP. Callers must extract `short_delta` themselves; there is no validation.
- **Not a closed-form POP** — for analyst-grade reporting use a lognormal POP from spot, IV, and DTE. This skill is for ranking, not for client-facing risk numbers.

## 5. Cross-References

- [03 Credit-to-Width floor](03_credit_to_width_floor.md) — uses `|Δ_short|` as the breakeven C/W, exploits the same delta-as-probability reading.
- [05 EV per $ risked scoring](05_ev_per_dollar_risked.md) — POP appears as the win-probability coefficient in the expected-value formula.

---

*Last verified against repo HEAD on 2026-05-02.*
