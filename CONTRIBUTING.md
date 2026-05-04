# Contributing to trading-agent

Welcome. This file is the canonical entry point for anyone — human or LLM — adding a feature, fixing a bug, or refactoring inside this repo. It operationalises [`docs/skills/00_sdlc_and_conventions.md`](docs/skills/00_sdlc_and_conventions.md): if §0 is the *what* and *why*, this file is the *do this, in this order, every time*.

The single sentence summary: **read the skill, follow the SDLC, run the invariant scan, update the docs, open the PR.** Everything below expands on that sentence.

---

## 0. Before your first contribution

Spend 30 minutes on this exact path. Skipping it doesn't save time — it just shifts the cost from your prep into your reviewer's PR comments.

1. Read [`PROJECT_MANIFEST.md`](PROJECT_MANIFEST.md) — repo-level orientation.
2. Read [`docs/skills/00_sdlc_and_conventions.md`](docs/skills/00_sdlc_and_conventions.md) — glossary, SDLC, invariants, conventions. **Mandatory.**
3. Open [`docs/skills/README.md`](docs/skills/README.md) and use the "When to read each one" table to find the 1–3 atomic skills relevant to your change.
4. Read those skills end-to-end. Pay attention to §4 (Edge Cases / Guardrails) — that's where the production bugs live.

If you cannot say which skill files cover your change, you are not ready to write code. Stop and ask.

---

## 1. The seven-step SDLC

Repeated from skill 00 §3, with concrete commands. Follow these in order. Out-of-order = drift bugs.

### Step 1 — Write or update the skill file *first*

Before any Python:

- **New primitive:** copy `docs/skills/_template.md` to `NN_short_kebab_name.md` (next free integer). Fill in §1 Theory, §2 Math, §4 Edge Cases. §3 (Reference Python) goes in last; that's the implementation you'll write next.
- **Modifying existing logic:** open the skill that documents it. Update §3 to reflect the new code. If the math or edge cases change, update those too.

Why first: filling out §2 and §4 forces you to articulate inputs, outputs, edge cases, and dependencies. If you can't write the math, you don't understand the feature well enough to ship it.

### Step 2 — Identify cross-cutting hooks

Run through this table. **Every "yes" is a place you must update in the same PR.**

| If your change touches… | Also update… | Enforcement |
|---|---|---|
| C/W floor logic | `chain_scanner.py`, `risk_manager.py`, `executor.py` | CI invariant 1 |
| Scoring math | only `chain_scanner.py` and `decision_engine.py` may define `_score_candidate*` | CI invariant 2 |
| Backtest evaluation | `streamlit/backtest_ui.py` must call `decide(` | CI invariant 3 |
| `RegimeAnalysis` dataclass | daily path **and** `multi_tf_regime._classify_intraday()` | review |
| Macro overlay (VIX-z, lead-z, IV rank) | wrap each in try/except in `_classify_intraday`; provide `*_signal_available` sentinel | review |
| New `PresetConfig` field | add field with default; update `to_summary_line()`; thread through `agent.py:169-204` to all 3 components; expose in Streamlit panel | review |
| New rejection reason | add to `REJECT_*` taxonomy in `chain_scanner.py:53-62`; update `_score_candidate_with_reason` | review |
| New strategy or regime label | update `strategy.py` dispatch, `thesis_builder.py`, watchlist UI, snapshot tests | review |

If you find yourself unsure whether your change touches a hook, ask in the PR description. "I don't think this touches the C/W floor" is a fine line to write — it makes the reviewer check explicitly.

### Step 3 — Implement once, import everywhere

**Single source of truth.** If your primitive needs to be consistent in two places, define it in one and import it. Never copy-paste.

Reference patterns already in the codebase:

- `_cw_floor()` defined in `chain_scanner.py`, imported by `risk_manager.py` and `executor.py`.
- `RegimeClassifier._determine_regime()` called by both daily and intraday paths — no shadow classifier.
- `_pop_from_delta()` is one helper feeding both static and adaptive paths.

A second copy of any primitive is the bug, not a feature.

### Step 4 — Wire knobs into the preset, not into class constants

New tunables go into `PresetConfig`, not `StrategyPlanner` class attrs. Class constants (`SPREAD_WIDTH = 5.0`, `TARGET_DTE = 35`) are **legacy fallbacks** retained for tests and scripts that predate the preset system. Adding new ones perpetuates a pattern we want to deprecate.

When you add a `PresetConfig` field:

1. Add the field with a default value (so existing `STRATEGY_PRESET.json` files still load).
2. Update `PresetConfig.to_summary_line()` if it should appear on the dashboard status line.
3. Surface it in the Streamlit Strategy-Profile panel (`live_monitor.py:849+`).
4. Thread it through `agent.py:169-204` to whichever components need it.
5. Document it in [`docs/skills/13_preset_system_hot_reload.md`](docs/skills/13_preset_system_hot_reload.md).

### Step 5 — Test in three places

For any non-trivial feature:

1. **Unit test** in `tests/test_<module>.py`. Cover every edge case from your skill's §4.
2. **Integration test** in `tests/test_agent_integration.py` if the change can alter a cycle's outcome.
3. **Verification harness** in `scripts/checks/` if the feature establishes a new invariant.

Run locally:

```bash
pytest tests/                                          # full suite
pytest tests/test_<module>.py -v                       # focused
python scripts/checks/scan_invariant_check.py          # AST-level invariants
python scripts/checks/run_unified_backtest_check.py    # live↔backtest parity
```

If you added a verification harness, add it to `.github/workflows/ci.yml`.

### Step 6 — Run the architectural invariant scan

```bash
python scripts/checks/scan_invariant_check.py
```

This scan exists because the failure modes it catches are silent and expensive. **Do not suppress a failure.** If it complains, you've broken one of the three CI invariants — read the error, find the offending file, fix the textual divergence. Reading the scan source itself (`scripts/checks/scan_invariant_check.py`) is a reasonable next step if you don't recognise the failure.

### Step 7 — Update the manifest, skill index, and changelog

- Edit `docs/skills/README.md`'s table if you added a skill.
- Edit `PROJECT_MANIFEST.md` if your change affects the cross-LLM handoff prompt (DTE rules, preset semantics, invariants).
- Re-stamp the `*Last verified against repo HEAD on YYYY-MM-DD.*` footer of every skill you touched.

---

## 2. CI-enforced architectural invariants

Three invariants are asserted by `scripts/checks/scan_invariant_check.py`. Breaking one fails CI; the failure mode each prevents is silent and expensive.

### Invariant 1 — Single C/W floor formula

The expression `|Δ| × (1 + edge_buffer)` appears identically in `chain_scanner.py`, `risk_manager.py`, and `executor.py`. Triple-enforcement is intentional defense in depth — a bug in one is caught by the others. See [`docs/skills/03_credit_to_width_floor.md`](docs/skills/03_credit_to_width_floor.md).

### Invariant 2 — Single source of scoring

`_score_candidate`, `_score_candidate_with_reason`, and `_quote_credit` may only be **defined** inside `chain_scanner.py` and `decision_engine.py`. A definition anywhere else is a "shadow scorer" that lets the backtester drift from live by construction.

### Invariant 3 — Backtester wires through `decide()`

`streamlit/backtest_ui.py` must contain at least one call to `decide(`. If the call disappears the unified live↔backtest path is dead code and the backtester silently reverts to its homegrown σ-distance heuristic.

---

## 3. Code conventions

These are not CI-enforced — they're stylistic conventions that keep modules predictable. PRs that violate them will get review comments. Repeated violations across PRs will get a stricter linter.

### Frozen dataclasses for configuration

Configuration objects (`PresetConfig`, `AppConfig` sub-sections) are `@dataclass(frozen=True)`. To override at load time, use `dataclasses.replace(preset, edge_buffer=0.15)`, never assign to fields.

### Append-only dataclass fields

`RegimeAnalysis`, `SpreadPlan`, `SpreadCandidate`, `WatchlistRow` are read by attribute name across the codebase, journal, UI, and snapshot tests. **Adding** a field with a default is safe; **renaming or removing** breaks every consumer. Field removal needs a coordinated PR that updates every consumer at once.

### Sentinel pattern for missing data

When a field can be absent rather than zero, pair it with a `*_signal_available: bool`:

```python
leadership_zscore: float = 0.0
leadership_signal_available: bool = False
```

Set `_signal_available = True` only inside the success branch. A failed RPC must leave the sentinel `False`. Consumers (UI, journal, ML features) check the sentinel before treating the value as real.

### Try/except per overlay, not per row

When populating multiple independent macro overlays on a row, wrap **each one** in its own try/except. A single failed RPC must not blank an otherwise-good row. Canonical example: `multi_tf_regime._classify_intraday`.

### Atomic temp+rename for sentinel files

Files read by another process (`STRATEGY_PRESET.json`, `AGENT_RUNNING`, journal files) are written via:

```python
tmp = fp.with_suffix(fp.suffix + ".tmp")
tmp.write_text(payload)
tmp.replace(fp)              # POSIX-atomic on the same filesystem
```

The temp must be in the **same directory** as the target. Never `/tmp/...` → final location across filesystems.

### `Optional[Foo]` over magic defaults

`None` means "not applicable." `0.0` and `""` mean "real zero." Don't conflate them. Constructor parameters that are genuinely optional default to `None`; the function picks the real default internally.

### Logging level discipline

- `logger.info` — cycle-level events an auditor should see.
- `logger.warning` — recoverable problems (bad JSON, RPC retry, fallback engaged).
- `logger.error` — non-recoverable problems that aborted something.
- `logger.debug` — high-volume diagnostics, off by default in prod.
- `logger.exception` — only inside an exception handler; logs the traceback.

Don't demote `logger.info` to `logger.debug` in a refactor. The cycle-level info trail is what operators read to understand "what did the agent do in the last hour."

### Module structure

Every module participating in a cycle follows the same shape:

1. Module docstring.
2. Constants (UPPER_SNAKE).
3. Dataclasses.
4. Pure helper functions.
5. One class with one public entry point (`plan`, `validate`, `scan`, `classify`).
6. `__all__` listing the public surface.

Don't import from `streamlit/` in any non-streamlit module.

### Naming

- Public class methods: `lower_snake`. Private: `_lower_snake`.
- Constants: `UPPER_SNAKE`.
- Dataclasses: `PascalCase`, ending in role suffix (`Plan`, `Analysis`, `Candidate`, `Config`, `Row`, `Diagnostics`).
- Booleans for "signal present": always `*_signal_available`.
- Booleans for "side gate": always `inhibit_*` or `is_*`.

---

## 4. Common pitfalls (drawn from skill 00 §8)

These are real bugs that have shipped or nearly shipped. Each one has a corresponding "don't" in skill 00.

- **Skipping the SDLC for a "small" feature.** Silent drift is itself silent.
- **Adding a class-attr constant instead of a `PresetConfig` field.** Perpetuates the legacy path.
- **Defining a shadow `_score_candidate` outside the two allowed modules.** CI invariant 2 will catch it; don't suppress the failure.
- **Sample stdev (ddof=1) in a Z-score that's expected to use population stdev (ddof=0).** Creates a 6 % bias on small windows that's invisible in unit tests.
- **`series.replace(0, pd.NA)` on a float Series.** Coerces dtype to `object`; subsequent `.ewm().mean()` raises `DataError`. Use `np.nan`.
- **Inventing a "1h VIX-z."** VIX is one instrument with one Z-score per fetch. Macro overlays are market-wide.
- **Moving the sentinel assignment outside the try/except.** Sets `True` even when the RPC failed.
- **Wiring `delta_aware_floor` into 2 of 3 consumers.** Plans pass strategy → fail risk → look like planner bugs. All three components in `agent.py:191-224`.
- **Forgetting to update `to_summary_line()` for a new preset field.** Field is invisible on the dashboard; nobody knows it's active.
- **Reformatting a regime reasoning string without updating snapshot tests.** Breaks UI tooltips and the LLM thesis builder.

---

## 5. PR readiness checklist

Copy this into your PR description and tick before requesting review. The PR template (`.github/PULL_REQUEST_TEMPLATE.md`) injects it automatically.

```markdown
### Skill citation
- [ ] Listed every `docs/skills/NN_*.md` file relevant to this change.
- [ ] Updated §3 (Reference Python) of every skill whose code changed.
- [ ] Re-stamped the "Last verified" footer.

### Cross-cutting hooks (skill 00 §3 step 2)
- [ ] C/W floor — touched? If yes, all 3 of chain_scanner / risk_manager / executor updated.
- [ ] Scoring math — touched? If yes, definition stays inside chain_scanner.py / decision_engine.py.
- [ ] Backtest seam — backtest_ui.py still calls `decide(` (CI invariant 3).
- [ ] `RegimeAnalysis` schema — daily path AND `_classify_intraday()` updated.
- [ ] New `PresetConfig` field — defaulted, surfaced in `to_summary_line()` + Streamlit panel + threaded through `agent.py:169-204`.

### Conventions (skill 00 §5)
- [ ] No new class-attr constants — tunables in `PresetConfig`.
- [ ] No copy-pasted primitives — `Single source of truth` rule.
- [ ] `*_signal_available` sentinel for any field that can be "absent."
- [ ] Atomic temp+rename for any new sentinel file write.

### Tests
- [ ] Unit tests cover every §4 edge case from the relevant skill.
- [ ] Integration test if the cycle outcome can change.
- [ ] Verification harness in `scripts/checks/` if a new invariant.

### Verification
- [ ] `pytest tests/` passes locally.
- [ ] `python scripts/checks/scan_invariant_check.py` exits 0.
- [ ] Other relevant `scripts/checks/*.py` pass locally.
```

A PR that omits the checklist will be returned without review.

---

## 6. Reviewer's responsibilities

If you're reviewing a PR, your job is to verify the contributor followed §1 — not to follow it for them. Specifically:

1. **Skill files updated?** If the diff touches `trading_agent/foo.py` lines cited in `docs/skills/NN_*.md` §3, the skill must change in the same PR.
2. **Cross-cutting hooks honoured?** Cross-reference the §1 step 2 table with the diff's file list.
3. **Invariant scan output included?** PR description should show the green output, or CI should.
4. **Tests cover §4 of the skill?** If the skill lists 5 edge cases and the PR adds 3 unit tests, ask why.
5. **Convention compliance?** Frozen dataclasses, sentinel pattern, no new class-attr constants.

Reviewers should not approve a PR that skips the SDLC, even if the diff "looks fine." The cost of a drift bug isn't visible in the diff.

---

## 7. Where to ask

- **Skill clarification** — comment on the skill file or open a draft PR with a question.
- **Convention dispute** — propose an edit to skill 00 in a separate PR; this file then mirrors it.
- **"I think this should be a skill"** — open a draft PR with the skill stub; let the math/edge-cases review happen before any code.

---

*Last updated: 2026-05-03 against repo HEAD. Mirrors [`docs/skills/00_sdlc_and_conventions.md`](docs/skills/00_sdlc_and_conventions.md).*
