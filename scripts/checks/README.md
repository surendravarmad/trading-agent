# scripts/checks — drift-prevention smoke tests

These five checks back the live ↔ backtest unification. **All five must
pass before any change to scoring, pricing, or floor logic ships.**
They are designed to run in CI (see `.github/workflows/ci.yml`) and are
also runnable individually during local iteration.

| Script | What it asserts | Runtime |
|---|---|---|
| `scan_invariant_check.py` | AST: `\|Δ\|×(1+edge_buffer)` floor appears in `chain_scanner` / `risk_manager` / `executor`; no shadow `_score_candidate` / `_quote_credit` outside `chain_scanner` + `decision_engine`; `streamlit/backtest_ui.py` calls `decide(...)` | < 1 s |
| `run_scan_diagnostics_check.py` | `ChainScanner` + `decision_engine.decide()` integration | ~2 s |
| `run_unified_backtest_check.py` | `Backtester._build_alpaca_plan_via_decide()` smoke (synthetic chain) | ~3 s |
| `run_journal_split_check.py` | `JournalKB(run_mode=...)` writes to the right file; rejects unknown modes | < 1 s |
| `run_live_vs_backtest_parity_check.py` | End-to-end: same synthetic chain → identical strikes + matching credit (Δ ≤ $0.01) on both `ChainScanner.scan()` and `Backtester._build_alpaca_plan_via_decide()` | ~3 s |

## Running locally

From the repo root:

```bash
python3 scripts/checks/scan_invariant_check.py
python3 scripts/checks/run_scan_diagnostics_check.py
python3 scripts/checks/run_unified_backtest_check.py
python3 scripts/checks/run_journal_split_check.py
python3 scripts/checks/run_live_vs_backtest_parity_check.py
```

Or all five at once:

```bash
for f in scripts/checks/*.py; do python3 "$f" || { echo "FAIL: $f"; exit 1; }; done
```

## When a check fails

| Failing check | What it usually means |
|---|---|
| `scan_invariant_check` | Someone added a scoring helper outside `chain_scanner` / `decision_engine`, or the C/W floor formula moved on one side without the other. Grep for `cw_floor` and `_score_candidate` to find the drift. |
| `run_live_vs_backtest_parity_check` | `decision_engine.decide()` produces different strikes / credits when called from `ChainScanner.scan` vs `Backtester._build_alpaca_plan_via_decide`. Either the live path stopped delegating, or the backtest path is constructing `ChainSlice` differently. |
| `run_journal_split_check` | `JournalKB.run_mode` resolution broke — either `signals_live.jsonl` or `signals_backtest.jsonl` isn't being written, or the legacy `signals.jsonl` is being recreated. |
