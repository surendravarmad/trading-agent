# CLAUDE.md — instructions for AI contributors

This file is for any LLM coding session (Claude Code, Cursor, an Agent SDK script, a copilot in another IDE) operating on this repo. **Read it first, every session.** Then follow the linked documents.

The project's value depends on a small set of architectural invariants and patterns. Drift is silent and expensive. Your job — the same as a human contributor's — is to follow the SDLC, respect the invariants, and update the documentation as you go.

---

## Read these in order, every session

1. [`PROJECT_MANIFEST.md`](PROJECT_MANIFEST.md) — repo orientation, current state, the cross-LLM handoff prompt.
2. [`docs/skills/00_sdlc_and_conventions.md`](docs/skills/00_sdlc_and_conventions.md) — glossary, SDLC, invariants, code conventions. **Mandatory.**
3. [`CONTRIBUTING.md`](CONTRIBUTING.md) — operationalises the SDLC into commands and a PR checklist.
4. The relevant atomic skill(s) under [`docs/skills/`](docs/skills/) — find them via the "When to read each one" table in `00_sdlc_and_conventions.md` §7.

If the user asks you to make a change and you cannot name the skill that documents the affected concept, **stop and ask** which skill applies. Don't guess.

---

## Hard rules — must never violate

These map 1:1 to the CI invariants in `scripts/checks/scan_invariant_check.py`. Violating any one fails CI, and the invariant scanner is an AST walker — there is no test path you can hide a violation behind.

1. **Single C/W floor formula.** The expression `|Δshort| × (1 + edge_buffer)` must appear identically in `trading_agent/chain_scanner.py`, `trading_agent/risk_manager.py`, and `trading_agent/executor.py`. If you change the formula, change all three in the same edit. See [`docs/skills/03_credit_to_width_floor.md`](docs/skills/03_credit_to_width_floor.md).
2. **Single source of scoring.** Functions named `_score_candidate`, `_score_candidate_with_reason`, or `_quote_credit` may only be **defined** in `trading_agent/chain_scanner.py` and `trading_agent/decision_engine.py`. Defining one elsewhere is a "shadow scorer" that lets the backtester drift from live. Import — don't redefine.
3. **Backtester wires through `decide()`.** `trading_agent/streamlit/backtest_ui.py` must contain at least one call to `decide(`. If you refactor that file, preserve the call. See [`docs/skills/14_adaptive_vs_static_scan_modes.md`](docs/skills/14_adaptive_vs_static_scan_modes.md) for context.

---

## Soft rules — strongly preferred patterns

Follow these unless the user explicitly tells you otherwise. They aren't CI-enforced, but the human reviewer will flag violations and ask you to fix them.

- **Single source of truth.** Define every primitive once and import. Never copy-paste — that *is* the bug.
- **Frozen dataclasses for config.** `@dataclass(frozen=True)`; mutate via `dataclasses.replace(obj, field=value)`, never assignment.
- **Append-only dataclass fields.** Adding a field with a default is safe. Renaming or removing a field is a coordinated change across the journal, UI, snapshot tests, and every consumer.
- **Sentinel pattern for missing data.** When a field can be absent (RPC failed) vs zero (real reading), pair it with a `*_signal_available: bool` and set the boolean only inside the success branch.
- **Try/except per overlay.** When populating multiple macro overlays, wrap each in its own try/except. One failed RPC must not blank the others.
- **Atomic temp+rename for sentinel files.** Use `tmp = fp.with_suffix(fp.suffix + ".tmp"); tmp.write_text(...); tmp.replace(fp)` — temp in same directory.
- **`Optional[Foo]` for genuinely-optional inputs.** Don't conflate `0.0` ("real zero") with `None` ("not applicable").
- **New tunables go in `PresetConfig`, not class constants.** Class attrs (`SPREAD_WIDTH = 5.0`, `TARGET_DTE = 35`) are legacy fallbacks for tests predating the preset system.
- **Logging discipline.** `info` for cycle events, `warning` for recoverable, `error` for aborts, `debug` for high-volume diagnostics, `exception` only inside handlers.

---

## Definition of done — for any non-trivial change

Before declaring "done," verify:

- [ ] The skill file(s) under `docs/skills/` documenting the affected concept have been updated. §3 (Reference Python) reflects the new code; §4 (Edge Cases) covers any new failure modes.
- [ ] The "Last verified against repo HEAD on YYYY-MM-DD" footer of every touched skill has been re-stamped to today's date (use `date -I`).
- [ ] If you added a `PresetConfig` field: defaulted, surfaced in `to_summary_line()`, exposed in the Streamlit Strategy-Profile panel (`live_monitor.py:849+`), threaded through `agent.py:169-204`.
- [ ] If you touched the C/W floor: all three of `chain_scanner.py`, `risk_manager.py`, `executor.py` carry the same formula.
- [ ] Unit tests cover every edge case listed in §4 of the relevant skill.
- [ ] `pytest tests/` passes locally.
- [ ] `python scripts/checks/scan_invariant_check.py` exits 0.
- [ ] `PROJECT_MANIFEST.md` updated if the cross-LLM handoff prompt changed.

If any box is unchecked, the change is incomplete — even if the user has not explicitly asked for it.

---

## Before you write any code

Ask yourself:

1. **Which skill file documents the concept I'm about to change?** If none, draft the skill first using `docs/skills/_template.md`.
2. **Which cross-cutting hooks does this change touch?** Walk the table in `CONTRIBUTING.md` §1 step 2.
3. **Am I about to define a primitive that already exists?** Grep before you write.
4. **Could the user have misunderstood my intent?** The code-conventions in this repo are dense; surface trade-offs before writing if uncertain.

---

## What to do when an instruction conflicts with these rules

If the user asks for something that would violate a hard rule (e.g. "just inline the floor formula in `risk_manager.py`, don't worry about the other two files"), **push back explicitly** before complying. Cite the invariant. Offer the compliant alternative.

If the user reaffirms the request after the explanation, proceed — but document the intentional divergence in the PR description and add a `# noqa: invariant-skipped — see PR #N` style comment so the next contributor (and the next LLM session) knows to look at the PR for context.

---

## Pointers

- Glossary, SDLC, invariants, conventions: [`docs/skills/00_sdlc_and_conventions.md`](docs/skills/00_sdlc_and_conventions.md)
- Operational checklist + commands: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- PR template: [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md)
- Skill index + reading order: [`docs/skills/README.md`](docs/skills/README.md)
- Architectural invariant scanner: [`scripts/checks/scan_invariant_check.py`](scripts/checks/scan_invariant_check.py)
- Repo-level orientation: [`PROJECT_MANIFEST.md`](PROJECT_MANIFEST.md)

---

*Last updated: 2026-05-03 against repo HEAD.*
