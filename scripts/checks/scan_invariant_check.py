"""
AST integration check — invariants that prevent live↔backtest drift.

Three invariants are asserted in CI:

1. **C/W floor formula parity.**
   The expression ``|Δ|×(1+edge_buffer)`` appears identically in
   ``chain_scanner.py``, ``risk_manager.py``, and ``executor.py`` so a
   scanner-picked plan can never be rejected at planning or execution
   time by a stricter floor.

2. **Single source of scoring.**
   No module *outside* ``chain_scanner.py`` and ``decision_engine.py``
   may define a function named ``_score_candidate``, ``_score_candidate_with_reason``,
   or ``_quote_credit``. If one ever appears, the AST walker fails — that
   would be a shadow scorer that lets the backtester drift from live by
   construction.

3. **Backtester wires through ``decide()``.**
   ``streamlit/backtest_ui.py`` must contain at least one call to
   ``decide(`` (the imported decision_engine entrypoint). If the call
   disappears the unified path is dead code and the backtester is back
   on its homegrown σ-distance heuristic — that's drift.

Exits 0 if all invariants hold, 1 otherwise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "trading_agent"

EXPECTED_FILES = {
    "chain_scanner.py": "_cw_floor",          # helper function
    "risk_manager.py": "validate",            # contains Check 2
    "executor.py": "_recheck_live_economics", # live recheck path
}

# Names that may only be *defined* inside chain_scanner.py /
# decision_engine.py. Defining them anywhere else is a shadow scorer.
SCORING_PRIMITIVE_NAMES = (
    "_score_candidate",
    "_score_candidate_with_reason",
    "_quote_credit",
)
ALLOWED_SCORING_DEFINERS = ("chain_scanner.py", "decision_engine.py")

# File that must call decide() — the parity seam between live + backtest.
BACKTEST_FILE = "streamlit/backtest_ui.py"


def _has_floor_formula(tree: ast.AST) -> bool:
    """
    Search for: <something> * (1.0 + <edge_buffer attr or name>)
    where the LHS references an absolute-delta-like name (short_max_delta,
    short_delta, etc.). Returns True if at least one such expression exists.
    """
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
            continue
        # RHS must be (1.0 + edge_buffer) or (1 + edge_buffer)
        rhs = node.right
        if not (isinstance(rhs, ast.BinOp) and isinstance(rhs.op, ast.Add)):
            continue
        a, b = rhs.left, rhs.right
        ones = [x for x in (a, b) if isinstance(x, ast.Constant)
                and isinstance(x.value, (int, float)) and x.value == 1]
        if not ones:
            continue
        non_one = [x for x in (a, b) if x not in ones]
        if not non_one:
            continue
        edge = non_one[0]
        edge_id = (
            edge.attr if isinstance(edge, ast.Attribute) else
            edge.id if isinstance(edge, ast.Name) else
            None
        )
        if edge_id != "edge_buffer":
            continue
        # LHS — accept abs(...), name, attr
        lhs = node.left
        if isinstance(lhs, ast.Call) and isinstance(lhs.func, ast.Name) \
                and lhs.func.id == "abs":
            found = True
            break
        if isinstance(lhs, (ast.Name, ast.Attribute)):
            name = (lhs.attr if isinstance(lhs, ast.Attribute) else lhs.id)
            if "delta" in name.lower():
                found = True
                break
    return found


def _scoring_primitives_defined_in(tree: ast.AST) -> list[str]:
    """Return any scoring-primitive function names defined in *tree*."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in SCORING_PRIMITIVE_NAMES:
                out.append(node.name)
    return out


def _calls_decide(tree: ast.AST) -> bool:
    """True if *tree* contains at least one ``decide(...)`` call."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "decide":
                return True
            if isinstance(f, ast.Attribute) and f.attr == "decide":
                return True
    return False


def main() -> int:
    failures = []

    # ── Invariant 1: C/W floor formula parity ─────────────────────────────
    print("Invariant 1: C/W floor formula parity")
    for fname, marker in EXPECTED_FILES.items():
        path = ROOT / fname
        src = path.read_text()
        tree = ast.parse(src)
        if not _has_floor_formula(tree):
            failures.append(f"{fname}: missing |Δ|×(1+edge_buffer) formula")
        else:
            print(f"  OK   {fname}: |Δ|×(1+edge_buffer) found")

    # ── Invariant 2: single source of scoring ─────────────────────────────
    print("\nInvariant 2: scoring primitives only defined in chain_scanner / decision_engine")
    # Walk both trading_agent/ and tests/ — a "shadow scorer" in a test
    # file is just as dangerous because it'd give green tests on a
    # silently-divergent backtester.
    repo_root = ROOT.parent
    walk_dirs = [ROOT, repo_root / "tests"]
    for walk in walk_dirs:
        if not walk.exists():
            continue
        for py in walk.rglob("*.py"):
            try:
                rel = py.relative_to(ROOT).as_posix()
            except ValueError:
                rel = py.relative_to(repo_root).as_posix()
            # Skip the two allow-listed definers entirely.
            if rel in ALLOWED_SCORING_DEFINERS:
                continue
            try:
                tree = ast.parse(py.read_text())
            except SyntaxError as exc:
                failures.append(f"{rel}: parse error {exc!s}")
                continue
            defined = _scoring_primitives_defined_in(tree)
            if defined:
                failures.append(
                    f"{rel}: shadow scorer defines {defined!r} — must live in "
                    f"chain_scanner.py / decision_engine.py only"
                )
    if not any("shadow scorer" in f for f in failures):
        print("  OK   no shadow scorers found in any other module")

    # ── Invariant 3: backtester calls decide() ────────────────────────────
    print("\nInvariant 3: backtester wires through decision_engine.decide()")
    bt_path = ROOT / BACKTEST_FILE
    if not bt_path.exists():
        failures.append(f"{BACKTEST_FILE}: file not found")
    else:
        bt_tree = ast.parse(bt_path.read_text())
        if not _calls_decide(bt_tree):
            failures.append(
                f"{BACKTEST_FILE}: no decide() call found — unified path "
                f"is dead code; backtester would drift from live"
            )
        else:
            print(f"  OK   {BACKTEST_FILE}: decide() call wired in")

    if failures:
        print("\nFAIL — invariant broken:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nAll live↔backtest parity invariants hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
