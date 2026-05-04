# SDLC, glossary, and code conventions

> **One-line summary:** The canonical first-read for any new contributor — human or LLM. Defines the project glossary, the step-by-step SDLC for adding a new feature, the architectural invariants that CI enforces, the code-writing conventions every module follows, and the "when to read this" index for all 14 atomic skills.
> **Source of truth:** This document is itself the source of truth for project conventions; specific rules cite the file/line that demonstrates them.
> **Phase:** 1  •  **Group:** meta
> **Depends on:** nothing — start here.
> **Consumed by:** every contributor; every other skill cross-links back to the relevant section here.

---

## 1. Theory & Objective

The trading agent has two failure modes that hurt:

1. **Silent drift between modules** — a new feature touches `strategy.py` but not `risk_manager.py`, and the two start computing the same gate differently. Live trades that planning approved get vetoed at execution; backtested edge disappears in production.
2. **Re-derived knowledge** — a new contributor (or a fresh LLM session) re-discovers why we use Wilder smoothing, why POP is `1 − |Δ|`, why presets hot-reload, and re-implements something subtly wrong because the original reasoning lived in a commit message from six months ago.

This document targets both. It establishes the **vocabulary** so two contributors mean the same thing when they say "C/W floor" or "scan mode," the **process** so new features land with all the cross-cutting hooks wired, and the **conventions** so code shape stays predictable. The 14 atomic skills are the *what*; this file is the *how* and *why*.

The single most important rule: **single source of truth**. Every primitive that appears in more than one place must be defined once and imported, never re-implemented. CI enforces this for the C/W floor, the scoring functions, and the backtest decision seam (see §4 Architectural Invariants below). Think of every other invariant in this document the same way: if you find yourself writing the same thing twice, stop and refactor.

## 2. Glossary

Project-specific terms used across skills, code, and the journal. When a term has both a common-usage and a project-specific meaning, the project meaning wins inside this codebase.

| Term | Meaning |
|---|---|
| **POP** | Probability of Profit. Approximated as `1 − |Δshort|` for vertical credit spreads. See [01](01_pop_from_delta.md). |
| **C/W ratio** | Credit divided by spread width. The single economic gate for whether a credit spread is worth selling. See [03](03_credit_to_width_floor.md). |
| **C/W floor** | Minimum acceptable C/W ratio. In static mode this is the scalar `min_credit_ratio`; in adaptive mode it's `|Δshort| × (1 + edge_buffer)`. |
| **edge_buffer** | Required margin over breakeven C/W in adaptive mode. `0.10` means "demand 10 % over breakeven." Persisted in `STRATEGY_PRESET.json`. |
| **EV/$risked** | Expected value per dollar of capital at risk. `(POP × C/W − (1−POP) × (1−C/W)) / (1−C/W)`. The scanner's ranking metric. See [05](05_ev_per_dollar_risked.md). |
| **annualized score** | `EV/$risked × 365 / DTE`. Tie-breaker so short-DTE wins beat equal-EV long-DTE wins. |
| **DTE** | Days To Expiration of the option contract. Per-strategy targets live in the active preset. |
| **Regime** | One of `BULLISH`, `BEARISH`, `SIDEWAYS`, `MEAN_REVERSION` (plus a `NEUTRAL` alias). Output of `_determine_regime()`. See [11](11_six_regime_classifier.md). |
| **Anchor** | The benchmark a ticker is measured against for leadership Z-score. SPY for QQQ, XLK for AAPL, etc. See [07](07_anchor_map_for_leadership.md). |
| **Leadership Z-score** | Rolling Z-score of `(ticker_return − anchor_return)`. ≥ 1.5σ → bull put bias. See [08](08_leadership_zscore.md). |
| **VIX inhibitor** | Macro gate that demotes new bullish premium to Bear Call when 5-min VIX-change Z > +2σ. See [09](09_vix_zscore_inhibitor.md). |
| **Preset** | A `PresetConfig` dataclass instance. Bundles `max_delta`, per-strategy DTE, width policy, C/W floor, max-risk %. Three built-ins: Conservative / Balanced / Aggressive, plus Custom. See [13](13_preset_system_hot_reload.md). |
| **scan_mode** | `"static"` (single-point planner) or `"adaptive"` (grid scanner). Preset field. See [14](14_adaptive_vs_static_scan_modes.md). |
| **Wilder smoothing** | EMA with `alpha = 1 / window` (slower than standard EMA). Used for ADX. See [10](10_adx_wilder_smoothing.md). |
| **Sentinel pattern** | A `*_signal_available: bool` companion field that distinguishes "RPC failed → 0.0 default" from "real 0.0 reading." Used everywhere a missing macro overlay must not be silently treated as a real value. |
| **Cycle** | One full execution of `TradingAgent.run_cycle()`. Default cadence is 5 minutes during market hours. The unit at which presets reload. |
| **`STRATEGY_PRESET.json`** | Sentinel JSON file at the repo root. Dashboard writes; agent reads each cycle. Atomic temp+rename. See [13](13_preset_system_hot_reload.md). |
| **AGENT_RUNNING / DRY_RUN_MODE** | Sibling sentinel files at the repo root, same write-once-read-each-cycle pattern. |
| **Static mode** | Planning algorithm that picks one (Δ, DTE, width) point. C/W gated by `min_credit_ratio`. Original behaviour. |
| **Adaptive mode** | Planning algorithm that sweeps the preset's `(DTE × Δ × width)` grid and picks the best EV/$ candidate that clears the floor. See [14](14_adaptive_vs_static_scan_modes.md). |
| **Triple-enforcement** | The C/W floor is independently re-checked in `chain_scanner.py`, `risk_manager.py`, and `executor.py` using the same formula. Drift between the three is a CI failure. See [03](03_credit_to_width_floor.md). |
| **Strike grid** | The discrete set of strikes the exchange lists ($0.50, $1, $2.50, $5 increments depending on price). Snap math in [02](02_strike_snapping.md). |
| **Stale spread** | A quote where the inferred mid hasn't moved by more than `stale_spread_pct` (default 1 %) since the last cycle. Soft-pass with a journal flag. See [06](06_stale_spread_risk_gate.md). |
| **`decide()` seam** | `decision_engine.decide()` — the single function the backtester calls to evaluate a candidate. Live and backtest must both go through it. CI invariant #3. |
| **Phase II / Phase III** | Internal labels: Phase II = regime classification, Phase III = strategy planning. Search comments for these tags. |
| **Journal** | `signals.jsonl` + `signals.md` in `trade_journal/`. Append-only record of every cycle decision, including skips. The auditable artifact. |

## 3. SDLC for adding a new feature

A repeatable seven-step playbook. The order is load-bearing — skipping a step (especially #2 or #6) is how silent drift gets introduced.

### Step 1 — Write the skill before the code

Before writing any Python, draft a `docs/skills/NN_*.md` from `_template.md`. Filling in §2 (Mathematical Formula) and §4 (Edge Cases) **before** the implementation forces you to decide:

- What the inputs and outputs are.
- What edge cases the implementation must handle.
- Which existing skill this one depends on (and whether you're about to duplicate one).

If you can't write the math down, you don't understand the feature well enough to implement it. If you can't articulate the edge cases, you'll discover them in production.

### Step 2 — Identify cross-cutting hooks

For every primitive, ask: "where else does this need to be consistent?" The CI invariants in §4 codify the three known seams, but every new feature potentially adds more. Run through this checklist:

| If the feature touches… | Also wire it into… |
|---|---|
| C/W floor logic | `chain_scanner.py`, `risk_manager.py`, `executor.py` (CI-enforced) |
| Scoring math | Only `chain_scanner.py` and `decision_engine.py` may define `_score_candidate*` (CI-enforced) |
| Backtest evaluation | Must go through `decision_engine.decide()` (CI-enforced) |
| `RegimeAnalysis` dataclass | Daily path AND `multi_tf_regime._classify_intraday()` |
| Macro overlay (VIX-z, leadership-z, IV rank) | Wrap each in try/except in `_classify_intraday`; provide sentinel `*_signal_available` |
| New preset field | Add to `PresetConfig` dataclass; thread through `agent.py:169-204` to all three components |
| New rejection reason | Add to `REJECT_*` taxonomy in `chain_scanner.py:53-62`; update `_score_candidate_with_reason` |
| New strategy or regime label | Update `strategy.py` dispatch, `thesis_builder.py`, watchlist UI, all snapshot tests |

### Step 3 — Implement once, import everywhere

Single source of truth. If you're implementing a primitive that more than one module needs, define it in **one** module and import it. Never copy-paste. Examples already in the codebase:

- `_cw_floor()` in `chain_scanner.py` is imported by `risk_manager.py` and `executor.py`.
- `RegimeClassifier._determine_regime()` is called by both the daily path and `multi_tf_regime._classify_intraday()` (no shadow classifier).
- `_pop_from_delta()` is one helper feeding both static and adaptive code paths.

### Step 4 — Wire into the preset system, not into class constants

New tunables go into `PresetConfig`, not `StrategyPlanner` class attrs. Class constants (`SPREAD_WIDTH = 5.0`, `TARGET_DTE = 35`) are **legacy fallbacks** that exist only for tests and scripts that predate the preset system. Adding new ones perpetuates the legacy path.

When you add a `PresetConfig` field:

1. Add the field with a default (so existing `STRATEGY_PRESET.json` files still load).
2. Update `to_summary_line()` if it should appear in the dashboard status line.
3. Add to `_make_custom()` allow-list automatically (it reads `__dataclass_fields__`).
4. Surface it in the Streamlit Strategy-Profile panel (`live_monitor.py:849+`).
5. Thread it through `agent.py:169-204` to whichever component(s) need it.

### Step 5 — Test in three places

For any non-trivial feature:

1. **Unit test** in `tests/test_<module>.py`. Edge cases from your skill's §4.
2. **Integration test** in `tests/test_agent_integration.py` if the feature changes a cycle outcome.
3. **Verification harness** in `scripts/checks/` if the feature establishes a new invariant.

CI runs `pytest tests/` plus everything in `scripts/checks/`. The checks directory is for things `pytest` can't easily express (AST scans, journal-shape assertions, live-vs-backtest parity diffs).

### Step 6 — Run the architectural invariant scan

Before opening a PR:

```bash
python scripts/checks/scan_invariant_check.py
```

This is the canary for cross-module drift. If it fails, you've broken one of the three CI invariants — likely by adding a shadow `_score_candidate` somewhere it doesn't belong, or by removing the `decide()` call from the backtester. Don't suppress the failure; the scan exists because the failure mode it catches is silent and expensive.

### Step 7 — Update the manifest and skill index

- Add a row to `docs/skills/README.md` for the new skill.
- If the feature changes the cross-LLM handoff prompt, edit `PROJECT_MANIFEST.md`.
- Re-stamp the `*Last verified against repo HEAD on YYYY-MM-DD.*` footer of any skill you touched.

## 4. Architectural Invariants (CI-enforced)

These are asserted in `scripts/checks/scan_invariant_check.py`. Breaking one fails CI, and is caught by an AST walker rather than a runtime test — so it fires even if no test path exercises the broken code.

### Invariant 1 — Single C/W floor formula

The expression `|Δ| × (1 + edge_buffer)` appears identically in three modules:

- `trading_agent/chain_scanner.py` (planner-side; lets the scanner reject sub-floor candidates)
- `trading_agent/risk_manager.py` (independent re-check before the order goes out)
- `trading_agent/executor.py` (live-credit recheck + 1-tick haircut at fill time)

The triplication is **intentional defense in depth**. A bug in one is caught by the others. The CI invariant ensures all three stay textually-equivalent. See [03 Credit/Width floor](03_credit_to_width_floor.md) for the math, and `scripts/checks/scan_invariant_check.py:1-26` for the enforcer.

### Invariant 2 — Single source of scoring

The functions `_score_candidate`, `_score_candidate_with_reason`, and `_quote_credit` may only be **defined** inside `chain_scanner.py` and `decision_engine.py`. A definition anywhere else is a "shadow scorer" — by construction, it would let the backtester drift from live without any test catching it.

### Invariant 3 — Backtester wires through `decide()`

`streamlit/backtest_ui.py` must contain at least one call to `decide(` (the imported `decision_engine` entrypoint). If the call disappears the unified live-vs-backtest path is dead code and the backtester silently reverts to its homegrown σ-distance heuristic.

## 5. Code Conventions

Patterns to follow. None of these are CI-enforced; they're stylistic conventions that keep modules predictable.

### Frozen dataclasses for configuration

Configuration objects (`PresetConfig`, `AppConfig` sub-sections) are `@dataclass(frozen=True)`. Two reasons:

1. Accidental mutation in module A would silently change behaviour in module B; freezing surfaces the bug as `FrozenInstanceError` at the write site.
2. Frozen dataclasses are hashable, which lets us use `dataclasses.replace()` to derive variants without subclassing — see `_make_custom()` in `strategy_presets.py`.

To override at load time, use `replace(preset, edge_buffer=0.15)`, never assign to fields.

### Append-only dataclass fields

`RegimeAnalysis`, `SpreadPlan`, `SpreadCandidate`, `WatchlistRow` are all dataclasses that downstream code reads by attribute name. **Adding** a field with a default is safe; **renaming or removing** a field breaks every consumer (UI, tests, journal readers, snapshot files). When removing a field is genuinely needed, do it in a coordinated PR that updates every consumer at once.

### Sentinel pattern for missing data

When a field can be "absent" rather than "zero," pair it with a `*_signal_available: bool`:

```python
leadership_zscore: float = 0.0
leadership_signal_available: bool = False
```

Set `_signal_available = True` only inside the success branch (after the RPC returned a non-`None` result). Consumers check the sentinel before treating the value as data; UIs render `—` instead of `0.00`. This is how every macro overlay in `_classify_intraday` works (see [12](12_multi_timeframe_resolution.md)) and why the `last_bar_ts` field uses `Optional[datetime]` instead of `datetime.min`.

### Try/except per overlay, not per row

When populating multiple independent overlays on a row, wrap **each one** in its own try/except. A single failed RPC must not blank an otherwise-good row. The pattern in `multi_tf_regime._classify_intraday` is canonical: VIX-z fails → leadership-z still populates → IV rank still populates → row is useful.

### Atomic temp+rename for sentinel files

Any file that's read by another process (`STRATEGY_PRESET.json`, `AGENT_RUNNING`, journal files) is written via:

```python
tmp = fp.with_suffix(fp.suffix + ".tmp")
tmp.write_text(payload)
tmp.replace(fp)              # POSIX-atomic on the same filesystem
```

The temp must be in the **same directory** as the target so `replace()` is atomic. Never `/tmp/...` → final location across filesystems.

### `Optional[Foo]` over magic defaults

`None` means "not applicable." `0.0` and `""` mean "real zero / empty string." Don't conflate them. Constructor parameters that are genuinely optional default to `None` and the function picks the real default internally. The motivation: `dte_vertical: int = 21` in a constructor signature looks like a value, but `dte_vertical: Optional[int] = None` with `self._dte_vertical = dte_vertical or self.TARGET_DTE` inside makes the fallback chain explicit.

### Logging level discipline

- `logger.info(...)` — cycle-level events that a human auditor should see (preset loaded, plan picked, order placed, skip with reason).
- `logger.warning(...)` — recoverable problems that didn't kill the cycle (bad JSON in a preset file, RPC retry, fallback engaged).
- `logger.error(...)` — non-recoverable problems that aborted something (timeout, broker rejection, missing required config).
- `logger.debug(...)` — high-volume diagnostics (per-tick prices, per-candidate scores). Off by default in prod.
- `logger.exception(...)` — inside a handler for an unexpected exception; always with the underlying traceback.

### Module structure

Every module that participates in a cycle follows the same shape:

1. Module docstring explaining the role in the cycle.
2. Constants (UPPER_SNAKE).
3. Dataclasses (Inputs, Outputs).
4. Pure helper functions (`_cw_floor`, `_pop_from_delta`).
5. Class with a single public entry point (`plan`, `validate`, `scan`, `classify`).
6. `__all__` at the bottom listing the public surface.

Don't put anything mutable at module top level. Don't import from streamlit/ in any module other than streamlit/.

### Naming

- Public class methods: `lower_snake`. Private: `_lower_snake`. Sigils only via `_` prefix; no `__double_underscore` mangling.
- Constants: `UPPER_SNAKE`.
- Dataclasses: `PascalCase`, ending in `Plan`, `Analysis`, `Candidate`, `Config`, `Row`, `Diagnostics` per role.
- Booleans for "signal present": always `*_signal_available`.
- Booleans for "side gate": always `inhibit_*` or `is_*`.

## 6. How to add a new skill

(Mirrors `README.md`, repeated here so this doc is self-contained.)

1. Copy `docs/skills/_template.md` to `NN_short_kebab_name.md` (next free integer).
2. Fill in **all five sections**: Theory & Objective, Mathematical Formula, Reference Python Implementation, Edge Cases / Guardrails, Cross-References. Skipping §4 defeats the point.
3. Quote source verbatim — no paraphrasing. A reader should be able to grep the codebase and find your snippet.
4. Cite `file:line-line` in the **Source of truth** header. If you cite the file without lines, the citation is meaningless after the first refactor.
5. Add a row to `README.md`'s Phase 1 table. Update the "Reading order" if your skill is foundational.
6. If your skill establishes a new architectural invariant, add the AST check to `scripts/checks/scan_invariant_check.py`.

## 7. Skill Index — when to read each one

Read this list as "if you're about to touch X, you must understand Y first."

| If you're working on… | Read first |
|---|---|
| Anything that prices a vertical spread | [01 POP](01_pop_from_delta.md), [02 Strike snapping](02_strike_snapping.md), [03 C/W floor](03_credit_to_width_floor.md), [05 EV scoring](05_ev_per_dollar_risked.md) |
| Spread-width logic | [04 Adaptive spread width](04_adaptive_spread_width.md) |
| Risk validation | [03 C/W floor](03_credit_to_width_floor.md), [06 Stale-spread gate](06_stale_spread_risk_gate.md) |
| Anything reading a leadership signal | [07 Anchor map](07_anchor_map_for_leadership.md), [08 Leadership Z-score](08_leadership_zscore.md) |
| Macro / VIX gate logic | [09 VIX Z-score inhibitor](09_vix_zscore_inhibitor.md) |
| Indicator math (ADX, etc.) | [10 ADX with Wilder smoothing](10_adx_wilder_smoothing.md) |
| Anything that maps prices → regime | [11 Six-regime classifier](11_six_regime_classifier.md) |
| The watchlist or any intraday surface | [12 Multi-timeframe resolution](12_multi_timeframe_resolution.md) |
| Adding a tunable knob | [13 Preset system & hot-reload](13_preset_system_hot_reload.md) |
| Choosing between simple and grid-scan planning | [14 Adaptive vs static scan modes](14_adaptive_vs_static_scan_modes.md) |

## 8. Edge Cases / Guardrails

- **Don't bypass the SDLC for "small" features** — the silent-drift failure mode is itself silent. The discipline of writing the skill first, identifying cross-cutting hooks, and running the invariant scan is cheap; the cost of a Bull Put that planned with one floor and got vetoed at execution with another is hours of debugging.
- **Don't add to class-attribute constants** — `StrategyPlanner.SPREAD_WIDTH` etc. are kept for back-compat. New constants belong in `PresetConfig`. Adding to the class attrs perpetuates a pattern we want to deprecate.
- **Don't add a shadow scorer** — CI invariant 2 catches `_score_candidate*` defined outside `chain_scanner.py` / `decision_engine.py`. If you genuinely need a different scoring rule, put it in `decision_engine.py` behind a flag, not in a sibling module.
- **Don't sample-stdev a Z-score** — leadership-z and VIX-z both use **population** stdev (ddof=0). Mixing sample and population stdev between modules creates a 6 % bias on small windows that's invisible in unit tests but visible in production drift. See [08](08_leadership_zscore.md).
- **Don't `replace(0, pd.NA)` a float Series** — flips it to `object` dtype and breaks `.ewm().mean()`. Use `np.nan`. See [10](10_adx_wilder_smoothing.md).
- **Don't compute "1h VIX-z"** — VIX is a single instrument with one Z-score per fetch. Macro overlays are market-wide and the same value appears on every intraday row. See [12](12_multi_timeframe_resolution.md).
- **Don't move the sentinel assignment outside the try/except** — `*_signal_available = True` must live inside the success branch. Outside, a failed RPC silently sets True with a default value. See [12](12_multi_timeframe_resolution.md) §4.
- **Don't wire only two of the three C/W floor consumers** — all three (planner / risk / executor) get the same `delta_aware_floor` flag in `agent.py:191-224`. Skipping one creates a drift bug that looks like a flaky planner. See [14](14_adaptive_vs_static_scan_modes.md) §4.
- **Don't forget to update `to_summary_line()` when adding a preset field** — the dashboard status line is the human-visible smoke test that the field made it through. A field that doesn't appear in the summary line is a field nobody knows is active.
- **Don't change the reasoning-string format on a regime branch** — the watchlist UI and the LLM thesis builder both render `RegimeAnalysis.reasoning`. Reformatting breaks tooltips. If you must, update `tests/test_regime.py` snapshots in the same PR. See [11](11_six_regime_classifier.md) §4.
- **Don't tighten log levels in a refactor** — `logger.info` in the cycle path is the journal trail an operator reads to understand "what did the agent do in the last hour." Demoting to `debug` makes ops blind.

## 9. Cross-References

This file links to all 14 phase-1 skills:

- [01 POP from short delta](01_pop_from_delta.md)
- [02 Strike snapping to grid](02_strike_snapping.md)
- [03 Credit-to-Width floor](03_credit_to_width_floor.md)
- [04 Adaptive spread width](04_adaptive_spread_width.md)
- [05 EV per $ risked scoring](05_ev_per_dollar_risked.md)
- [06 Stale-spread risk gate](06_stale_spread_risk_gate.md)
- [07 Anchor map for leadership](07_anchor_map_for_leadership.md)
- [08 Leadership Z-score](08_leadership_zscore.md)
- [09 VIX Z-score inhibitor](09_vix_zscore_inhibitor.md)
- [10 ADX with Wilder smoothing](10_adx_wilder_smoothing.md)
- [11 Six-regime classifier](11_six_regime_classifier.md)
- [12 Multi-timeframe regime resolution](12_multi_timeframe_resolution.md)
- [13 Preset system & hot-reload](13_preset_system_hot_reload.md)
- [14 Adaptive vs static scan modes](14_adaptive_vs_static_scan_modes.md)

External documents:

- [`PROJECT_MANIFEST.md`](../../PROJECT_MANIFEST.md) — repo-level handoff prompt for cross-LLM context.
- [`README.md`](README.md) — skill-library index, reading order, "how to verify."
- [`scripts/checks/scan_invariant_check.py`](../../scripts/checks/scan_invariant_check.py) — the AST-level invariant enforcer described in §4.

---

*Last verified against repo HEAD on 2026-05-03.*
