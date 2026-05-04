# Skill Library

Atomic, reusable concepts extracted from the trading agent. Each file is self-contained: theory → formula → reference Python → edge cases. A new collaborator (LLM or human) should be able to read any one skill in 5 minutes and reproduce the math without reading the rest of the codebase.

**Philosophy.** This library is **derived** from `trading_agent/` — every skill cites a `file:line` source of truth. If the source moves, the skill moves with it. Skills don't introduce new logic, they document what already exists in a form that's easier to reason about.

---

## Phase 1 (14 skills + meta, dependency-ordered)

Read top-to-bottom on a first pass. Each row's "Depends on" column lists the prerequisite skills. **Skill 00 is mandatory first reading** — it's the meta-skill covering glossary, SDLC, and conventions that every other skill assumes you've internalised.

| # | Skill | Group | Source of truth |
|---|---|---|---|
| 00 | [SDLC, glossary, and code conventions](00_sdlc_and_conventions.md) | meta | this folder + `scripts/checks/scan_invariant_check.py` |
| 01 | [POP from short delta](01_pop_from_delta.md) | strategy | `chain_scanner.py:110-112` |
| 02 | [Strike snapping to grid](02_strike_snapping.md) | strategy | `strategy.py:684`, `chain_scanner.py:496` |
| 03 | [Credit-to-Width floor](03_credit_to_width_floor.md) | risk | `chain_scanner.py:160-162`, `risk_manager.py:108-114` |
| 04 | [Adaptive spread width](04_adaptive_spread_width.md) | strategy | `strategy.py:653-685` |
| 05 | [EV per $ risked scoring](05_ev_per_dollar_risked.md) | strategy | `chain_scanner.py:165-207` |
| 06 | [Stale-spread risk gate](06_stale_spread_risk_gate.md) | risk | `risk_manager.py:50, 79, 166-190` |
| 07 | [Anchor map for leadership](07_anchor_map_for_leadership.md) | bias | `regime.py:36-110` |
| 08 | [Leadership Z-score](08_leadership_zscore.md) | bias | `market_data.py:1075, 1138-1176` |
| 09 | [VIX Z-score inhibitor](09_vix_zscore_inhibitor.md) | bias | `regime.py:117, 293-299`; `strategy.py:255-265` |
| 10 | [ADX with Wilder smoothing](10_adx_wilder_smoothing.md) | regime | `multi_tf_regime.py:367-423` |
| 11 | [Six-regime classifier](11_six_regime_classifier.md) | regime | `regime.py:209, 412-455` |
| 12 | [Multi-timeframe regime resolution](12_multi_timeframe_resolution.md) | regime | `multi_tf_regime.py:225-361` |
| 13 | [Preset system & hot-reload](13_preset_system_hot_reload.md) | architecture | `strategy_presets.py:54-295` |
| 14 | [Adaptive vs static scan modes](14_adaptive_vs_static_scan_modes.md) | architecture | `strategy.py:158-205, 302-334` |

## Phase 2 (planned, not yet written)

Hygiene and diagnostics. Useful but not edge-defining.

- `15_open_bar_skip.md` — Why we drop the first N bars after the open auction.
- `16_stale_data_age_detection.md` — `last_bar_ts` + 30-min wall-clock badge.
- `17_signal_availability_sentinel.md` — The `*_signal_available: bool` design pattern.
- `18_trend_conflict_detector.md` — 200-SMA slope vs short-term — diagnostic only.
- `19_bollinger_bandwidth_regime.md` — The 4 % SIDEWAYS rule, in isolation.
- `20_account_risk_pct_sizing.md` — Conservative 1 % / Balanced 2 % / Aggressive 3 %.
- `21_regime_to_strategy_routing.md` — The dispatch table from `README.md:82-89`.
- `22_width_aware_max_loss.md` — `(width − credit) × multiplier`.

---

## How to add a new skill

1. Copy `_template.md` to `NN_short_kebab_name.md` (next free number).
2. Fill in **all five sections**. Skipping §4 (edge cases) defeats the point.
3. Quote source verbatim — do not paraphrase. Future readers should be able to grep the code and find your snippet.
4. Add a row to the table above.
5. If your skill changes the inventory of phase 1 vs phase 2, update both lists.

## How to verify a skill is still accurate

For any skill `NN_*.md`:

1. Open the file linked in the **Source of truth** header.
2. Diff its contents against §3 of the skill file.
3. If they disagree, the source has drifted — update the skill or open a PR explaining why the divergence is intentional.

This is also a good first task for a new LLM joining the project: "pick one skill, verify the citation is still accurate."

## Reading order for a new contributor

If you only have 30 minutes:
- **Minute 0–5:** [`PROJECT_MANIFEST.md`](../../PROJECT_MANIFEST.md) at repo root.
- **Minute 5–10:** Skill 00 — glossary + SDLC + conventions. Sets vocabulary for everything else.
- **Minute 10–18:** Skills 01, 03, 04, 05 — the core spread-economics math.
- **Minute 18–25:** Skills 11, 12 — the regime classifier and how it composes across timeframes.
- **Minute 25–30:** Skill 13 — the preset system, which is the central control surface.

If you have a full afternoon, read 00 first, then all 14 in order.

---

*Last updated: 2026-05-03 against repo HEAD.*
