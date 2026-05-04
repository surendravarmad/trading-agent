<!--
  Read CONTRIBUTING.md and docs/skills/00_sdlc_and_conventions.md before filling
  this in. The checklist below is not optional — incomplete PRs are returned
  without review.

  If a section truly does not apply, write "N/A — <one-sentence reason>" so the
  reviewer can verify your reasoning rather than assume an oversight.
-->

## Summary

<!-- One paragraph: what changed, why. Cite the skill that documents the affected concept. -->

## Skills touched

<!-- List every docs/skills/NN_*.md file that documents code in this diff. -->
<!-- If you cannot point at a skill, you owe the project a new skill file before merging. -->

- `docs/skills/NN_*.md` — <what changed in §3 / §4>
- `docs/skills/NN_*.md` — <what changed>

## Cross-cutting hooks

Tick every box, or write "N/A — <reason>". This is the highest-signal section.

- [ ] **C/W floor logic** — untouched, OR all 3 of `chain_scanner.py` / `risk_manager.py` / `executor.py` updated. (CI invariant 1)
- [ ] **Scoring math** — `_score_candidate*` definitions remain inside `chain_scanner.py` and `decision_engine.py`. (CI invariant 2)
- [ ] **Backtest seam** — `streamlit/backtest_ui.py` still calls `decide(`. (CI invariant 3)
- [ ] **`RegimeAnalysis` schema** — untouched, OR daily path AND `multi_tf_regime._classify_intraday()` both updated.
- [ ] **Macro overlays (VIX-z, lead-z, IV rank)** — untouched, OR each is wrapped in its own try/except with a `*_signal_available` sentinel.
- [ ] **`PresetConfig` schema** — untouched, OR new field has a default, is surfaced in `to_summary_line()`, exposed in the Streamlit Strategy-Profile panel, and threaded through `agent.py:169-204` to every component that needs it.
- [ ] **Reject-reason taxonomy** — untouched, OR new `REJECT_*` constant added to `chain_scanner.py:53-62` AND `_score_candidate_with_reason` updated.
- [ ] **Strategy / regime label** — untouched, OR `strategy.py` dispatch + `thesis_builder.py` + watchlist UI + snapshot tests all updated.

## Conventions

- [ ] No new class-attribute constants — new tunables go into `PresetConfig`, not `StrategyPlanner.X = ...`.
- [ ] No copy-pasted primitives — single source of truth, importing across modules.
- [ ] Frozen dataclasses for new config; mutation via `dataclasses.replace()`.
- [ ] `*_signal_available` sentinel for every field that can be absent.
- [ ] Atomic temp+rename for any new file written from one process and read by another.
- [ ] `Optional[Foo]` over magic defaults where the field can mean "not applicable."
- [ ] Logging levels follow CONTRIBUTING §3 (`info` for cycle events, `warning` for recoverable, `error` for aborts, `exception` only inside handlers).

## Tests

- [ ] Unit tests added for every edge case listed in §4 of the relevant skill.
- [ ] Integration test added/updated if the change can alter a cycle outcome.
- [ ] New verification harness added to `scripts/checks/` if this PR establishes a new invariant.

```text
# Paste the green test output here, or link to the CI run:
$ pytest tests/
...

$ python scripts/checks/scan_invariant_check.py
[OK] Invariant 1: C/W floor formula present in chain_scanner.py, risk_manager.py, executor.py
[OK] Invariant 2: no shadow _score_candidate*
[OK] Invariant 3: backtest_ui.py calls decide(
```

## Documentation

- [ ] Skill file(s) updated to match the new code (Reference Python §3, edge cases §4).
- [ ] "Last verified against repo HEAD on YYYY-MM-DD" footer re-stamped on every touched skill.
- [ ] `docs/skills/README.md` index updated if a skill was added.
- [ ] `PROJECT_MANIFEST.md` updated if the cross-LLM handoff prompt changed.

## Risk and rollout

<!-- One paragraph: what's the worst-case behaviour if this is wrong in production? Is there a feature flag, preset toggle, or kill switch? -->

## Reviewer notes

<!-- Anything you want the reviewer to look at carefully. Surprising decisions, deferred work, follow-up issues. -->
