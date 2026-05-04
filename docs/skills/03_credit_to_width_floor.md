# Credit-to-Width floor

> **One-line summary:** Refuse to open a credit spread unless `C / W ≥ |Δ_short| × (1 + edge_buffer)`. This is the breakeven C/W (`= |Δ|`) plus a configurable margin.
> **Source of truth:** [`trading_agent/chain_scanner.py:160-162`](../../trading_agent/chain_scanner.py), [`trading_agent/risk_manager.py:108-114`](../../trading_agent/risk_manager.py)
> **Phase:** 1  •  **Group:** risk
> **Depends on:** [01 POP from short delta](01_pop_from_delta.md) — the floor is derived from the same delta-as-probability identity.
> **Consumed by:** `chain_scanner._score_candidate` (hard gate), `risk_manager.RiskGate` (pre-execution check), `executor.submit` (final guard).

---

## 1. Theory & Objective

A credit spread has expected value zero when `C / W = |Δ_short|`. The proof is short:

```
EV = POP × C − (1 − POP) × (W − C)
   = (1 − |Δ|) × C − |Δ| × (W − C)
   = C − |Δ| × W
   ≥ 0  iff  C / W ≥ |Δ|.
```

So `|Δ_short|` is the breakeven C/W ratio. Trading at exactly that ratio is a coin-flip in expectation; we want a **margin** above it. `edge_buffer = 0.10` means "demand 10 % above breakeven C/W before firing."

This is a **strict** floor, not a soft preference. A candidate that fails this gate is dropped entirely, even if it has the highest absolute credit on the chain. The reasoning: chasing high-credit / high-delta spreads at thin margins is exactly how account blowups happen in retail credit-spread trading.

The floor is enforced at **three** layers — scanner, risk manager, executor — because the cost of a wrong fire is much higher than the cost of triple-validation. If you remove a layer, an upstream bug can leak a bad spread into production.

## 2. Mathematical Formula

```text
floor = |Δ_short| × (1 + edge_buffer)

required:  C / W ≥ floor

where
  Δ_short      ∈ [-1, 1]  — delta of the SHORT (sold) leg
  edge_buffer  ∈ [0, 1]   — margin over breakeven; default 0.10
  C            ∈ ℝ⁺       — net credit received
  W            ∈ ℝ⁺       — distance between short and long strikes
```

Default `edge_buffer = 0.10` is a global tuning choice; it's exposed on the preset (skill 13) as `PresetConfig.edge_buffer` so a user can ratchet it up to 0.20 for more conservative behaviour.

## 3. Reference Python Implementation

```python
# trading_agent/chain_scanner.py:160-162
def _cw_floor(short_delta: float, edge_buffer: float) -> float:
    """Required C/W floor = |Δ| × (1 + edge_buffer). |Δ| is breakeven C/W."""
    return abs(short_delta) * (1.0 + edge_buffer)
```

```python
# trading_agent/risk_manager.py:108-114
if self.delta_aware_floor and plan.legs:
    short_legs = [l for l in plan.legs if l.action == "sell"]
    short_max_delta = (max(abs(l.delta) for l in short_legs)
                       if short_legs else 0.0)
    cw_floor = short_max_delta * (1.0 + self.edge_buffer)
    floor_label = (f"|Δ|×(1+edge)={short_max_delta:.3f}"
                   f"×{1+self.edge_buffer:.2f}={cw_floor:.4f}")
```

```python
# trading_agent/executor.py:~216 (final guard before order submit)
cw_floor = short_max_delta * (1.0 + self.edge_buffer)
```

## 4. Edge Cases / Guardrails

- **Multi-leg short side (Iron Condor)** — `short_max_delta = max(abs(l.delta) for l in short_legs)`. We use the *worst* (largest absolute delta) leg as the floor input, not an average. An IC must clear the floor against its more-aggressive wing.
- **No short legs** — `short_max_delta = 0.0` → `floor = 0.0`. Any positive C/W passes. Safe because debit-only structures don't reach this gate in practice.
- **`edge_buffer = 0`** — degenerates to the breakeven check `C/W ≥ |Δ|`. Allowed but discouraged; the system warns in the log.
- **Triple enforcement** — same formula in `chain_scanner.py`, `risk_manager.py`, `executor.py`. The architectural invariant scanner (`scripts/checks/scan_invariant_check.py`) **fails CI** if a fourth implementation appears or if any of these three is removed. Don't refactor into a "shared helper" without updating the invariant check.
- **C/W ≥ 1** — credit ≥ width implies a debit, not a credit. The scanner's `_ev_per_dollar_risked` returns `None` in that case (skill 05); this floor check never sees it.
- **Negative credit** — defensive check elsewhere; this formula assumes `C > 0` and would otherwise produce a negative floor with weird semantics.

## 5. Cross-References

- [01 POP from short delta](01_pop_from_delta.md) — the math derivation depends on `POP ≈ 1 − |Δ|`.
- [05 EV per $ risked scoring](05_ev_per_dollar_risked.md) — uses the same C/W gate as a hard filter before computing EV.
- [13 Preset system & hot-reload](13_preset_system_hot_reload.md) — `edge_buffer` is set per-preset.

---

*Last verified against repo HEAD on 2026-05-02.*
