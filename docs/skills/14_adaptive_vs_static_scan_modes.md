# Adaptive vs static scan modes

> **One-line summary:** A single `scan_mode: Literal["static", "adaptive"]` preset field selects between two planning algorithms for vertical credit spreads. **Static** picks one (Δ, DTE, width) point from the preset's scalar fields and gates with `min_credit_ratio` — original behavior. **Adaptive** sweeps a `(DTE × Δ × width)` grid, scores each tuple by `EV/$risked`, demands `C/W ≥ |Δshort| × (1 + edge_buffer)`, and returns the highest-scoring candidate — or sits out if none clear the floor.
> **Source of truth:** [`trading_agent/strategy_presets.py:44-93`](../../trading_agent/strategy_presets.py); [`trading_agent/strategy.py:144-205, 302-355, 464-551`](../../trading_agent/strategy.py); [`trading_agent/chain_scanner.py:1-33`](../../trading_agent/chain_scanner.py).
> **Phase:** 1  •  **Group:** strategy
> **Depends on:** [01 POP from delta](01_pop_from_delta.md), [03 C/W floor](03_credit_to_width_floor.md), [05 EV per dollar risked](05_ev_per_dollar_risked.md), [13 Preset system & hot-reload](13_preset_system_hot_reload.md).
> **Consumed by:** `StrategyPlanner.plan` → `_plan_bull_put` / `_plan_bear_call` branch on `is_adaptive`; agent.py wires the same flag into `RiskManager` and `OrderExecutor` via `delta_aware_floor`.

---

## 1. Theory & Objective

The original planner ("static" mode) was a single-point picker: from the preset, grab one DTE, one max delta, one width policy, build that exact spread, then check if `C/W ≥ min_credit_ratio`. Simple, deterministic, easy to reason about, and the reference implementation against which everything else is validated.

The static picker has one structural weakness: it **commits before pricing**. If the chosen Δ-0.25 21-DTE point happens to price at C/W = 0.28 today (just below the 0.30 floor), the planner refuses the trade — even though Δ-0.30 25-DTE might price at C/W = 0.42 and be the *better* trade by every economic measure. Static mode can't see what it didn't price.

The "adaptive" mode fixes this by **searching the full preset grid** before committing. For every (DTE, target Δ, width%) tuple it picks the nearest weekly, finds the closest-Δ short contract, snaps the long strike, prices both legs off the NBBO mid, computes the actual `C/W` and `EV/$risked`, and ranks all positive-EV survivors by `annualized_score = EV × 365/DTE`. The winner is the highest-scoring candidate that *also* clears the breakeven-plus-edge floor; if none clear, the scanner returns `[]` and the agent journals a "no edge" skip.

The two modes share **everything that defines a trade**: same chain fetcher, same delta lookup, same strike-snapping math, same C/W formula, same POP approximation, same EV definition. They differ only in **how many points get priced**. This is by design — the architectural invariant scanner enforces a single C/W floor formula and a single scoring source so the modes can't drift.

The `delta_aware_floor` flag wired through `RiskManager` and `OrderExecutor` is the consequence: adaptive mode uses `|Δ| × (1 + edge_buffer)` as the floor (Δ-aware, varies per spread); static mode uses the scalar `min_credit_ratio`. Without this wire-through, the scanner could pick a candidate that the risk manager would then veto with the wrong floor — a planning/validation drift bug. The `agent.py` constructor passes both flags into all three components in the same place (`agent.py:191-224`) precisely so they can never get out of sync.

The decision to keep static as the default (`scan_mode: ScanMode = "static"` in the dataclass) is pragmatic: tests that predate the scanner shouldn't have to wire one in, and a fresh install that doesn't touch the dashboard gets the simpler, well-understood behavior.

## 2. Mathematical Formula

```text
ScanMode ∈ {"static", "adaptive"}

is_adaptive = (preset is not None) ∧ (preset.scan_mode == "adaptive")

────────────────────────────────────────────────────────────────────────
STATIC mode (single point)
────────────────────────────────────────────────────────────────────────
inputs:  preset.max_delta, preset.dte_vertical, preset.dte_window_days,
         preset.width_mode, preset.width_value, preset.min_credit_ratio

  expiration  = nearest weekly to preset.dte_vertical
                ∈ [dte_vertical - dte_window_days,
                   dte_vertical + dte_window_days]
  short_leg   = first contract with |Δ| ≤ max_delta in chain
  long_leg    = strike = short_strike ± snap(width_value)
  credit      = mid(short) − mid(long)
  width       = |short_strike − long_strike|

  ACCEPT  iff  credit / width ≥ min_credit_ratio
  RETURN  one SpreadPlan, or empty plan with rejection_reason

────────────────────────────────────────────────────────────────────────
ADAPTIVE mode (grid sweep)
────────────────────────────────────────────────────────────────────────
inputs:  preset.dte_grid, preset.delta_grid, preset.width_grid_pct,
         preset.edge_buffer, preset.min_pop

  candidates = []
  for dte_target in preset.dte_grid:
      expiration = nearest weekly to dte_target
      for target_delta in preset.delta_grid:
          for width_pct in preset.width_grid_pct:
              short = chain.closest_to(target_delta)
              long  = chain.snap(short.strike ± width_pct × spot)
              credit = price(short, long)
              cw     = credit / width
              pop    = 1 − |short.delta|                    # skill 01
              floor  = |short.delta| × (1 + edge_buffer)    # skill 03
              if pop < min_pop: continue
              if cw  < floor:  continue
              ev     = (POP × C/W − (1 − POP) × (1 − C/W)) / (1 − C/W)
              if ev <= 0: continue
              annualized = ev × (365 / dte_target)          # skill 05
              candidates.append( SpreadCandidate(... annualized) )

  candidates.sort(by=annualized, descending)
  RETURN  candidates[0]  ifany else []  →  agent journals "no edge"
```

## 3. Reference Python Implementation

```python
# trading_agent/strategy_presets.py:44-51 — the literal type
# Strategy planner mode:
#   "static"   — single (Δ, DTE, width) point from the preset's scalar fields,
#                C/W gated by ``min_credit_ratio``. Original behaviour.
#   "adaptive" — scan a grid of (DTE × Δ × width) tuples, score each by
#                ``EV_per_$risked = (POP×C/W − (1−POP)×(1−C/W)) / (1−C/W)``,
#                pick the highest-scoring candidate that clears the
#                breakeven-plus-edge_buffer floor, OR sit out if none do.
ScanMode = Literal["static", "adaptive"]
```

```python
# trading_agent/strategy.py:144-205 — lazy scanner construction + is_adaptive
# Adaptive scan-mode wiring. When ``preset`` is supplied and
# ``preset.scan_mode == 'adaptive'`` the planner routes vertical/IC builders
# through ChainScanner instead of the static single-point picker. ``preset``
# is None for legacy callers and tests that don't use the preset system.
preset: Optional[object] = None):
    self.data = data_provider
    self.max_delta = max_delta
    self.min_credit_ratio = min_credit_ratio
    self.preset = preset
    # Adaptive scanner is constructed lazily — only when the active preset
    # asks for it. Stays None in static mode so tests don't need to wire
    # a preset just to instantiate the planner.
    self._scanner: Optional[ChainScanner] = None
    if preset is not None and getattr(preset, "scan_mode", "static") == "adaptive":
        self._scanner = ChainScanner(
            data_provider=data_provider,
            preset=preset,
            dte_window_days=getattr(preset, "dte_window_days", 5),
        )
    # Last scan results — captured per cycle so the agent can persist them
    # to the journal alongside the picked plan.
    self.last_scan_candidates: List[SpreadCandidate] = []
    self.last_scan_side: Optional[str] = None
    self.last_scan_diagnostics: Optional[Dict] = None
    ...

@property
def is_adaptive(self) -> bool:
    """True iff the active preset asked for chain-scanner planning."""
    return self._scanner is not None
```

```python
# trading_agent/strategy.py:302-334 — branch point in the builders
def _plan_bull_put(self, ticker: str, analysis: RegimeAnalysis,
                   expiration: str) -> SpreadPlan:
    """Sell an OTM put, buy a further-OTM put."""
    if self.is_adaptive:
        return self._plan_via_scanner(ticker, "bull_put", analysis,
                                      fallback_expiration=expiration)

    contracts = self.data.fetch_option_chain(ticker, expiration, "put")
    if not contracts:
        return self._empty_plan(ticker, "Bull Put Spread", analysis,
                                 expiration, "No put contracts available")

    sold = self._find_sold_strike(contracts)
    if not sold:
        return self._empty_plan(ticker, "Bull Put Spread", analysis,
                                 expiration,
                                 f"No put with |delta| ≤ {self.max_delta}")

    bought = self._find_bought_strike(contracts, sold["strike"],
                                       direction="lower")
    if not bought:
        return self._empty_plan(ticker, "Bull Put Spread", analysis,
                                 expiration, "No suitable protective leg found")

    return self._assemble_plan(ticker, "Bull Put Spread", analysis,
                                expiration, sold, bought, "put")

def _plan_bear_call(self, ticker: str, analysis: RegimeAnalysis,
                    expiration: str) -> SpreadPlan:
    """Sell an OTM call, buy a further-OTM call."""
    if self.is_adaptive:
        return self._plan_via_scanner(ticker, "bear_call", analysis,
                                      fallback_expiration=expiration)
    ...  # static branch identical in shape to _plan_bull_put
```

```python
# trading_agent/strategy.py:464-551 — adaptive translator
def _plan_via_scanner(self, ticker: str, side: str,
                      analysis: RegimeAnalysis,
                      fallback_expiration: str) -> SpreadPlan:
    """
    Adaptive-mode planner. Routes through ChainScanner; converts the
    winning ``SpreadCandidate`` into a ``SpreadPlan`` whose legs and
    economics match what the scanner priced.

    On no-edge (empty candidate list) we return an empty plan with
    rejection_reason="No positive-EV candidate ..." so the agent's
    existing pipeline can journal it under the "skipped: no_edge"
    path without special-casing scanner output.
    """
    assert self._scanner is not None, "_plan_via_scanner needs adaptive preset"
    strategy_name = "Bull Put Spread" if side == "bull_put" else "Bear Call Spread"

    try:
        candidates = self._scanner.scan(ticker, side)
    except Exception as exc:
        logger.exception("[%s] Adaptive scan failed: %s", ticker, exc)
        return self._empty_plan(ticker, strategy_name, analysis,
                                 fallback_expiration,
                                 f"Adaptive scan crashed: {exc}")

    # Capture for the journal regardless of outcome — both the picks
    # AND the diagnostics so a zero-candidate cycle still tells a story.
    self.last_scan_candidates = list(candidates)
    self.last_scan_side = side
    scanner_diag = getattr(self._scanner, "last_diagnostics", None)
    self.last_scan_diagnostics = (
        scanner_diag.to_journal_dict() if scanner_diag is not None else None
    )

    if not candidates:
        reason = ("No positive-EV candidate found across DTE×Δ×width "
                  f"grid (edge_buffer={self.preset.edge_buffer:.0%}, "
                  f"min_pop={self.preset.min_pop:.0%})")
        logger.info("[%s] %s — sitting out", ticker, reason)
        return self._empty_plan(ticker, strategy_name, analysis,
                                 fallback_expiration, reason)

    best = candidates[0]
    opt_type = "put" if side == "bull_put" else "call"

    legs = [
        SpreadLeg(symbol=best.short_symbol, strike=best.short_strike,
                  action="sell", option_type=opt_type,
                  delta=best.short_delta, theta=0.0,
                  bid=best.short_bid, ask=best.short_ask,
                  mid=round((best.short_bid + best.short_ask) / 2, 4)),
        SpreadLeg(symbol=best.long_symbol, strike=best.long_strike,
                  action="buy", option_type=opt_type,
                  delta=best.short_delta * 0.4,  # rough — real Δ rides skew
                  theta=0.0,
                  bid=best.long_bid, ask=best.long_ask,
                  mid=round((best.long_bid + best.long_ask) / 2, 4)),
    ]
    max_loss = round((best.width - best.credit) * 100, 2)

    plan = SpreadPlan(
        ticker=ticker, strategy_name=strategy_name,
        regime=analysis.regime.value, legs=legs,
        spread_width=float(best.width), net_credit=float(best.credit),
        max_loss=max_loss,
        credit_to_width_ratio=round(best.cw_ratio, 4),
        expiration=best.expiration, reasoning=reasoning,
    )

    # Adaptive mode uses the scanner's own |Δ|×(1+edge_buffer) floor —
    # which the scanner already enforced — so we don't re-apply the
    # static min_credit_ratio gate here. The RiskManager (in adaptive
    # mode) will use the same delta-aware floor for its independent
    # check, keeping planning and validation consistent.
    return plan
```

```python
# trading_agent/agent.py:191-224 — wiring the floor flag through
self.risk_manager = RiskManager(
    max_risk_pct=max_risk_pct,
    min_credit_ratio=min_credit_ratio,
    max_delta=max_delta,
    ...,
    # Adaptive mode: replace the static C/W floor with a Δ-aware one —
    # same formula the scanner uses: |Δshort| × (1 + edge_buffer).
    delta_aware_floor=(self.preset.scan_mode == "adaptive"),
    edge_buffer=self.preset.edge_buffer,
)
self.executor: ExecutionPort = OrderExecutor(
    ...,
    # Adaptive mode: live-credit recheck + 1-tick haircut both use the
    # same Δ-aware floor RiskManager is enforcing, so a scanner-picked
    # plan can never be vetoed at execution time by a stale static
    # floor.  Mirrors the kwargs passed to RiskManager above.
    delta_aware_floor=(self.preset.scan_mode == "adaptive"),
    edge_buffer=self.preset.edge_buffer,
)
```

## 4. Edge Cases / Guardrails

- **`is_adaptive` is the single source of truth** — never check `preset.scan_mode == "adaptive"` directly inside `StrategyPlanner` methods; always use `self.is_adaptive`. The property handles the `preset is None` case (legacy callers / tests) by returning False without a getattr dance.
- **Lazy scanner construction** — `ChainScanner` is built only when needed. Tests that don't use the preset system can instantiate `StrategyPlanner` without paying scanner construction cost (which transitively imports `calendar_utils` and would fail without market hours mocking).
- **Mean-reversion and explicit IC stay static** — only `_plan_bull_put` and `_plan_bear_call` branch on `is_adaptive`. Mean-reversion strikes are dictated by which 3-σ band touched (timing-driven, not edge-driven). Iron Condors have their own neutral-strike logic. Routing them through the scanner would lose semantic meaning.
- **No re-application of `min_credit_ratio` in adaptive** — the scanner enforces `|Δ| × (1 + edge_buffer)` and the RiskManager uses the same Δ-aware floor (driven by `delta_aware_floor=True`). Re-applying the scalar `min_credit_ratio` would create a stricter floor than the scanner used → scanner picks pass strategy but fail risk → confusing "phantom rejection." The inline comment at `strategy.py:546-550` is load-bearing.
- **Scanner crash → empty plan, not crash** — `_plan_via_scanner` wraps `scanner.scan()` in try/except. A bug in the scanner produces an "Adaptive scan crashed: …" rejection_reason, journaled like any other skip. The agent loop never sees the exception. This was deliberate after early scanner bugs took down whole cycles.
- **Empty candidate list ≠ failure** — when no point clears the floor, `candidates == []` is the correct outcome ("the chain offers no positive-EV trade today"). The empty plan with `rejection_reason="No positive-EV candidate..."` is journaled under the same "skipped: no_edge" path the static planner uses for sub-floor C/W. Don't treat empty as an error.
- **`delta_aware_floor` MUST be wired to all three components** — `StrategyPlanner` (via the scanner), `RiskManager`, and `OrderExecutor` all need it. If you forget one, a scanner-picked plan can pass strategy → fail risk-manager → look like a planner bug. The triple-wire in `agent.py:191-224` is intentional. See [03 Credit/Width floor](03_credit_to_width_floor.md) for the broader triple-enforcement pattern.
- **Diagnostics captured per cycle** — `last_scan_candidates`, `last_scan_side`, `last_scan_diagnostics` are reset at the top of every `plan()` call. The agent's journal layer reads them after the call returns. Don't move the reset; cross-ticker bleed through these attrs would wrongly attribute candidates to the next ticker.
- **Rough long-leg delta** — `best.short_delta * 0.4` for the `SpreadLeg.delta` of the bought leg is a placeholder; the real long-leg delta rides the IV skew and isn't priced by the scanner. The number is good enough for the journal — no downstream code reads the bought-leg delta — but don't let it leak into PnL math.
- **`SPREAD_WIDTH = 5.0`, `TARGET_DTE = 35`, `DTE_RANGE = (28, 45)` are LEGACY** — class constants on `StrategyPlanner` only fire when no preset is supplied. Any new code path should pass a preset; the constants exist solely so older tests and the original demo scripts keep working.

## 5. Cross-References

- [13 Preset system & hot-reload](13_preset_system_hot_reload.md) — where `scan_mode` lives and how it gets to the planner.
- [03 Credit/Width floor](03_credit_to_width_floor.md) — the floor formula adaptive mode uses; explains the `delta_aware_floor` triple-wire.
- [05 EV per dollar risked](05_ev_per_dollar_risked.md) — the scanner's scoring function.
- [01 POP from delta](01_pop_from_delta.md) — the POP approximation feeding both the floor and the EV.
- [02 Strike snapping](02_strike_snapping.md) — used inside the scanner when deriving long-leg strikes from `width_pct × spot`.
- [04 Adaptive spread width](04_adaptive_spread_width.md) — the *static-mode* width logic; complementary to the adaptive grid.

---

*Last verified against repo HEAD on 2026-05-03.*
