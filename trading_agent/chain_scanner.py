"""
chain_scanner.py — adaptive credit-spread chain scanner.

Replaces the static "one (Δ, DTE, width) point" planning with a sweep over
a configurable grid. For every (DTE, target Δ-short, width) tuple we:

  1. pick the nearest weekly expiration to that DTE,
  2. fetch the relevant put/call chain,
  3. find the contract whose |Δ| is closest to target Δ-short,
  4. find a protective leg ``width_value × spot`` strikes away (snapped
     to the strike grid),
  5. price the spread off bid (sold) − ask (bought),
  6. score the spread by per-dollar-risked expected value, where

         POP        ≈ 1 − |Δshort|     (vertical credit spread approx)
         C/W        = credit / width
         EV/$risked = (POP × C/W − (1 − POP) × (1 − C/W)) / (1 − C/W)

  7. demand C/W ≥ |Δshort| × (1 + edge_buffer)
     (breakeven C/W is exactly |Δ|; the buffer is the requested edge).

Candidates that fail any hard filter (POP floor, edge floor, non-positive
credit, illiquid quotes) are dropped, never returned. The scanner returns
``[]`` when the chain offers no positive-EV trade — the agent treats this
as "skipped: no edge" and journals it instead of forcing a trade.

Annualized score = EV/$risked × (365 / DTE) is the final tie-breaker so
short-dated trades that win the same EV/$risked beat longer-dated ones
that take more time to harvest.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from trading_agent.calendar_utils import next_weekly_expiration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reject-reason taxonomy
# ---------------------------------------------------------------------------
# Stable string keys so the journal histogram is grep-able. Order roughly
# matches the order checks fire inside ``_score_candidate_with_reason``.
REJECT_NO_CHAIN              = "no_chain"
REJECT_NO_SHORT_CONTRACT     = "no_short_contract"
REJECT_NO_LONG_CONTRACT      = "no_long_contract"
REJECT_NON_POSITIVE_WIDTH    = "non_positive_width"
REJECT_DTE_NON_POSITIVE      = "dte_non_positive"
REJECT_POP_BELOW_MIN         = "pop_below_min"
REJECT_CREDIT_NON_POSITIVE   = "credit_non_positive"
REJECT_CREDIT_GE_WIDTH       = "credit_ge_width"
REJECT_CW_BELOW_FLOOR        = "cw_below_floor"
REJECT_EV_NON_POSITIVE       = "ev_non_positive"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpreadCandidate:
    """One scored candidate spread from the scanner."""

    side:                 str        # "bull_put" | "bear_call"
    expiration:           str
    dte:                  int
    short_strike:         float
    long_strike:          float
    short_delta:          float      # signed (puts negative, calls positive)
    short_symbol:         str
    long_symbol:          str
    short_bid:            float
    short_ask:            float
    long_bid:             float
    long_ask:             float
    credit:               float      # net credit ≥ 0
    width:                float
    cw_ratio:             float
    pop:                  float      # 1 − |Δshort|
    cw_floor:             float      # |Δ| × (1 + edge_buffer)
    ev_per_dollar_risked: float
    annualized_score:     float
    target_delta:         float      # the grid point that produced this row
    width_pct:            float      # the grid width fraction

    def to_journal_dict(self) -> Dict:
        d = asdict(self)
        # Round floats to keep JSON compact and human-readable.
        for k in ("short_delta", "credit", "width", "cw_ratio", "pop",
                  "cw_floor", "ev_per_dollar_risked", "annualized_score",
                  "target_delta", "width_pct",
                  "short_bid", "short_ask", "long_bid", "long_ask"):
            d[k] = round(float(d[k]), 4)
        return d


# ---------------------------------------------------------------------------
# Pure scoring helpers (unit-testable without market data)
# ---------------------------------------------------------------------------

def _pop_from_delta(short_delta: float) -> float:
    """POP ≈ 1 − |Δshort| for a vertical credit spread."""
    return max(0.0, 1.0 - abs(short_delta))


def _cw_floor(short_delta: float, edge_buffer: float) -> float:
    """Required C/W floor = |Δ| × (1 + edge_buffer). |Δ| is breakeven C/W."""
    return abs(short_delta) * (1.0 + edge_buffer)


def _ev_per_dollar_risked(credit: float, width: float,
                          short_delta: float) -> Optional[float]:
    """
    EV per dollar at risk for a credit spread.

    Returns None when the trade is structurally invalid (zero/negative
    credit, zero/negative width, or credit ≥ width which would imply a
    debit not a credit).
    """
    if width <= 0 or credit <= 0 or credit >= width:
        return None
    cw = credit / width
    pop = _pop_from_delta(short_delta)
    # gain_per_$1_width = cw   ;   loss_per_$1_width = (1 − cw)
    ev_per_width = pop * cw - (1.0 - pop) * (1.0 - cw)
    # Convert to per-$-risked basis: $-at-risk per $1 width = (1 − cw).
    return ev_per_width / (1.0 - cw)


def _score_candidate(credit: float, width: float, short_delta: float,
                     dte: int, edge_buffer: float, min_pop: float
                     ) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Score a candidate. Returns ``(cw, pop, cw_floor, ev_per_$risked,
    annualized_score)`` or ``None`` if the candidate fails any hard filter.
    """
    if dte <= 0:
        return None
    pop = _pop_from_delta(short_delta)
    if pop < min_pop:
        return None
    floor = _cw_floor(short_delta, edge_buffer)
    if width <= 0 or credit <= 0:
        return None
    cw = credit / width
    if cw < floor:
        return None
    ev = _ev_per_dollar_risked(credit, width, short_delta)
    if ev is None or ev <= 0:
        # ev > 0 is implied by cw > breakeven, but guard against rounding.
        return None
    annualized = ev * (365.0 / dte)
    return cw, pop, floor, ev, annualized


def _score_candidate_with_reason(
    credit: float, width: float, short_delta: float, dte: int,
    edge_buffer: float, min_pop: float,
) -> Dict[str, Any]:
    """
    Diagnostic sibling of ``_score_candidate``. Always returns a dict
    describing the outcome — never None — so the scanner can build a
    histogram of *why* candidates were rejected.

    Result keys
    -----------
    status     : "accepted" | "rejected"
    reason     : reject-reason key (when status == "rejected"); always
                 one of the ``REJECT_*`` constants above
    cw         : credit / width                (when computable)
    pop        : 1 − |Δshort|                  (when computable)
    cw_floor   : |Δ| × (1 + edge_buffer)       (when computable)
    ev         : EV per $-risked               (when computable)
    annualized : ev × (365/dte)                (only when accepted)

    The check order intentionally mirrors ``_score_candidate`` so accept
    decisions are bit-identical between the two functions.
    """
    if dte <= 0:
        return {"status": "rejected", "reason": REJECT_DTE_NON_POSITIVE}
    if width <= 0:
        return {"status": "rejected", "reason": REJECT_NON_POSITIVE_WIDTH}
    pop = _pop_from_delta(short_delta)
    floor = _cw_floor(short_delta, edge_buffer)
    if pop < min_pop:
        return {"status": "rejected", "reason": REJECT_POP_BELOW_MIN,
                "pop": pop, "cw_floor": floor}
    if credit <= 0:
        return {"status": "rejected", "reason": REJECT_CREDIT_NON_POSITIVE,
                "pop": pop, "cw_floor": floor}
    if credit >= width:
        return {"status": "rejected", "reason": REJECT_CREDIT_GE_WIDTH,
                "pop": pop, "cw_floor": floor, "cw": credit / width}
    cw = credit / width
    ev = _ev_per_dollar_risked(credit, width, short_delta)
    if cw < floor:
        return {"status": "rejected", "reason": REJECT_CW_BELOW_FLOOR,
                "pop": pop, "cw_floor": floor, "cw": cw, "ev": ev}
    if ev is None or ev <= 0:
        return {"status": "rejected", "reason": REJECT_EV_NON_POSITIVE,
                "pop": pop, "cw_floor": floor, "cw": cw, "ev": ev}
    annualized = ev * (365.0 / dte)
    return {"status": "accepted", "pop": pop, "cw_floor": floor,
            "cw": cw, "ev": ev, "annualized": annualized}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class ScanDiagnostics:
    """
    Per-scan summary attached to ``ChainScanner.last_diagnostics``. Fields
    are journal-safe (no SpreadCandidate refs, all primitives).

    Surfaced into ``signals.jsonl`` so a human reading the journal can
    answer *why* zero candidates passed instead of just *that* zero passed.
    """
    grid_points_total:    int                 # |DTE| × |Δ| × |w|
    expirations_resolved: int = 0             # weeklies actually picked
    grid_points_priced:   int = 0             # tuples that reached scoring
    rejects_by_reason:    Dict[str, int] = field(default_factory=dict)
    # Best near-miss: highest-EV candidate that *only* failed the C/W floor
    # gate. None when nothing came close enough to compute an EV. Stored as
    # plain dict (not SpreadCandidate) to keep the diagnostic decoupled
    # from the journal format.
    best_near_miss:       Optional[Dict[str, Any]] = None

    def record(self, reason: str, count: int = 1) -> None:
        """Increment the histogram bucket for ``reason``."""
        if count <= 0:
            return
        self.rejects_by_reason[reason] = (
            self.rejects_by_reason.get(reason, 0) + count
        )

    def to_journal_dict(self) -> Dict[str, Any]:
        return {
            "grid_points_total":    self.grid_points_total,
            "expirations_resolved": self.expirations_resolved,
            "grid_points_priced":   self.grid_points_priced,
            "rejects_by_reason":    dict(self.rejects_by_reason),
            "best_near_miss":       self.best_near_miss,
        }


# ---------------------------------------------------------------------------
# ChainScanner
# ---------------------------------------------------------------------------

class ChainScanner:
    """
    Sweeps the option chain across the preset's DTE × Δ × width grid and
    returns positive-EV candidates ranked by annualized risk-adjusted score.

    Inputs
    ------
    data_provider : MarketDataProvider
        Used for ``fetch_option_chain(ticker, expiration, type)``.
    preset : PresetConfig
        Source of grids (``dte_grid``, ``delta_grid``, ``width_grid_pct``)
        and floors (``edge_buffer``, ``min_pop``).
    dte_window_days : int
        Tolerance around each grid DTE when picking weekly expirations.
        Defaults to 5 (covers Mon–Fri of the target week).
    """

    # Inferred when the chain is too thin to compute a strike grid.
    DEFAULT_GRID_STEP = 1.0

    def __init__(self, data_provider, preset, dte_window_days: int = 5,
                 max_candidates: int = 10):
        self.data = data_provider
        self.preset = preset
        self.dte_window_days = dte_window_days
        self.max_candidates = max_candidates
        # Populated by every ``scan()`` call so callers (StrategyPlanner,
        # the agent journal write path) can surface *why* zero candidates
        # passed without inspecting log lines.
        self.last_diagnostics: Optional[ScanDiagnostics] = None

    # ------------------------------------------------------------------
    # Public scan API
    # ------------------------------------------------------------------

    def scan(self, ticker: str, side: str,
             today: Optional[date] = None) -> List[SpreadCandidate]:
        """
        Run the grid scan. Returns candidates sorted by annualized score
        descending; empty list if no point clears the floors.

        Parameters
        ----------
        ticker : str
            Underlying symbol.
        side : str
            "bull_put" or "bear_call".
        today : date, optional
            Pin the calendar (used in tests). Defaults to today's local date.
        """
        if side not in ("bull_put", "bear_call"):
            raise ValueError(f"Unsupported side {side!r}")
        if today is None:
            today = datetime.now().date()

        opt_type = "put" if side == "bull_put" else "call"
        candidates: List[SpreadCandidate] = []

        # Diagnostics scaffolding — sized off the preset grid so the user
        # can read "evaluated 12 of 16 grid points" right out of the journal.
        n_dte = len(list(self.preset.dte_grid))
        n_delta = len(list(self.preset.delta_grid))
        n_width = len(list(self.preset.width_grid_pct))
        diag = ScanDiagnostics(grid_points_total=n_dte * n_delta * n_width)
        # Track the highest-EV candidate that *only* failed the C/W floor —
        # the actionable near-miss the user wants to see when zero
        # candidates pass.
        best_near_miss_payload: Optional[Dict[str, Any]] = None

        # 1. Pick distinct expirations (grid items collapse onto the same
        #    weekly when DTEs are close together; dedup on the resolved
        #    expiration string so we don't fetch the same chain twice).
        seen_exps: Dict[str, int] = {}   # exp_str → dte
        for target_dte in self.preset.dte_grid:
            try:
                exp_date = next_weekly_expiration(
                    today=today,
                    target_dte=int(target_dte),
                    dte_min=max(1, int(target_dte) - self.dte_window_days),
                    dte_max=int(target_dte) + self.dte_window_days,
                )
            except Exception as exc:
                logger.warning("[%s] Expiration pick failed for DTE=%s: %s",
                               ticker, target_dte, exc)
                continue
            exp_str = exp_date.strftime("%Y-%m-%d")
            dte = (exp_date - today).days
            # Keep the smallest DTE if collisions; closer to the user's
            # asked grid-point will dominate ranking anyway.
            if exp_str not in seen_exps or dte < seen_exps[exp_str]:
                seen_exps[exp_str] = dte

        diag.expirations_resolved = len(seen_exps)
        if not seen_exps:
            logger.warning("[%s] Scanner found no expirations for grid %s",
                           ticker, list(self.preset.dte_grid))
            self.last_diagnostics = diag
            return []

        # 2. For each (expiration, target_delta, width), score one candidate.
        for exp_str, dte in seen_exps.items():
            chain = self.data.fetch_option_chain(ticker, exp_str, opt_type)
            if not chain:
                # All (Δ × width) tuples for this expiration are dead.
                diag.record(REJECT_NO_CHAIN, n_delta * n_width)
                logger.debug("[%s] No %s chain for %s — skipping",
                             ticker, opt_type, exp_str)
                continue

            spot_proxy = self._infer_spot_proxy(chain)
            grid_step = self._infer_grid_step(chain)

            for target_delta in self.preset.delta_grid:
                short_contract = self._find_short(chain, target_delta)
                if short_contract is None:
                    diag.record(REJECT_NO_SHORT_CONTRACT, n_width)
                    continue
                short_strike = float(short_contract["strike"])

                for width_pct in self.preset.width_grid_pct:
                    raw_width = width_pct * spot_proxy
                    width = self._snap_width_to_grid(raw_width, grid_step)
                    if width <= 0:
                        diag.record(REJECT_NON_POSITIVE_WIDTH)
                        continue
                    long_strike = (short_strike - width if side == "bull_put"
                                   else short_strike + width)
                    long_contract = self._find_strike(chain, long_strike)
                    if long_contract is None:
                        diag.record(REJECT_NO_LONG_CONTRACT)
                        continue
                    actual_width = abs(short_strike - float(long_contract["strike"]))
                    if actual_width <= 0:
                        diag.record(REJECT_NON_POSITIVE_WIDTH)
                        continue

                    credit = round(
                        float(short_contract["bid"]) - float(long_contract["ask"]),
                        2,
                    )
                    short_delta = float(short_contract["delta"])

                    diag.grid_points_priced += 1
                    result = _score_candidate_with_reason(
                        credit=credit,
                        width=actual_width,
                        short_delta=short_delta,
                        dte=dte,
                        edge_buffer=self.preset.edge_buffer,
                        min_pop=self.preset.min_pop,
                    )
                    if result["status"] == "rejected":
                        reason = result["reason"]
                        diag.record(reason)
                        # Surface the closest the chain came to passing —
                        # only candidates that failed *only* the C/W floor
                        # qualify (those have full pop/cw/ev populated and
                        # are within reach of a tighter edge_buffer dial).
                        if reason == REJECT_CW_BELOW_FLOOR:
                            cand_payload = {
                                "expiration":   exp_str,
                                "dte":          dte,
                                "short_strike": short_strike,
                                "long_strike":  float(long_contract["strike"]),
                                "short_delta":  round(short_delta, 4),
                                "credit":       credit,
                                "width":        round(actual_width, 4),
                                "cw_ratio":     round(result.get("cw") or 0.0, 4),
                                "cw_floor":     round(result.get("cw_floor") or 0.0, 4),
                                "pop":          round(result.get("pop") or 0.0, 4),
                                "ev":           round(result.get("ev") or 0.0, 4),
                                "target_delta": float(target_delta),
                                "width_pct":    float(width_pct),
                            }
                            cur_ev = (best_near_miss_payload or {}).get("ev", -1e9)
                            if cand_payload["ev"] > cur_ev:
                                best_near_miss_payload = cand_payload
                        continue

                    candidates.append(SpreadCandidate(
                        side=side,
                        expiration=exp_str,
                        dte=dte,
                        short_strike=short_strike,
                        long_strike=float(long_contract["strike"]),
                        short_delta=short_delta,
                        short_symbol=str(short_contract.get("symbol", "")),
                        long_symbol=str(long_contract.get("symbol", "")),
                        short_bid=float(short_contract["bid"]),
                        short_ask=float(short_contract["ask"]),
                        long_bid=float(long_contract["bid"]),
                        long_ask=float(long_contract["ask"]),
                        credit=credit,
                        width=actual_width,
                        cw_ratio=result["cw"],
                        pop=result["pop"],
                        cw_floor=result["cw_floor"],
                        ev_per_dollar_risked=result["ev"],
                        annualized_score=result["annualized"],
                        target_delta=float(target_delta),
                        width_pct=float(width_pct),
                    ))

        # 3. Rank: annualized score desc, then absolute credit desc as tiebreak.
        candidates.sort(
            key=lambda c: (c.annualized_score, c.credit),
            reverse=True,
        )

        diag.best_near_miss = best_near_miss_payload
        self.last_diagnostics = diag

        if not candidates:
            # Promote the top reject reason into the log line so an operator
            # tailing trading_agent.log doesn't have to open the journal.
            top_reason = max(diag.rejects_by_reason.items(),
                             key=lambda kv: kv[1], default=(None, 0))
            logger.info(
                "[%s] Scanner found 0 positive-EV candidates "
                "(%d expirations, %d/%d grid points priced; "
                "top reject=%s×%d; near-miss EV=%.4f)",
                ticker, len(seen_exps),
                diag.grid_points_priced, diag.grid_points_total,
                top_reason[0], top_reason[1],
                (best_near_miss_payload or {}).get("ev", 0.0),
            )
        else:
            top = candidates[0]
            logger.info(
                "[%s] Scanner top pick: %s %sd Δ=%.3f w=$%.2f "
                "credit=$%.2f C/W=%.4f (floor %.4f) EV/$=%.4f ann=%.4f "
                "(%d total candidates)",
                ticker, top.side, top.dte, top.short_delta, top.width,
                top.credit, top.cw_ratio, top.cw_floor, top.ev_per_dollar_risked,
                top.annualized_score, len(candidates),
            )
        return candidates[: self.max_candidates]

    # ------------------------------------------------------------------
    # Chain inspection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_grid_step(chain: List[Dict]) -> float:
        """Modal gap between consecutive sorted strikes."""
        strikes = sorted({float(c["strike"]) for c in chain if c.get("strike")})
        if len(strikes) < 3:
            return ChainScanner.DEFAULT_GRID_STEP
        gaps: Dict[float, int] = {}
        for a, b in zip(strikes, strikes[1:]):
            gap = round(b - a, 2)
            if gap > 0:
                gaps[gap] = gaps.get(gap, 0) + 1
        if not gaps:
            return ChainScanner.DEFAULT_GRID_STEP
        return max(gaps, key=lambda g: gaps[g])

    @staticmethod
    def _infer_spot_proxy(chain: List[Dict]) -> float:
        """
        Spot proxy: the strike of the contract whose |Δ| is closest to 0.50.
        ATM strike ≈ spot for OPRA-listed equities/ETFs.
        Falls back to the median strike if no Δs are present.
        """
        atm = [c for c in chain if c.get("delta") not in (None, 0)
               and c.get("strike")]
        if atm:
            atm.sort(key=lambda c: abs(abs(float(c["delta"])) - 0.50))
            return float(atm[0]["strike"])
        strikes = sorted(float(c["strike"]) for c in chain if c.get("strike"))
        if not strikes:
            return 0.0
        return strikes[len(strikes) // 2]

    @staticmethod
    def _snap_width_to_grid(raw_width: float, grid_step: float) -> float:
        """Round-up to the nearest strike-grid multiple (≥ 1 step)."""
        if grid_step <= 0:
            return raw_width
        steps = max(1, int(round(raw_width / grid_step + 0.4999)))
        return round(steps * grid_step, 4)

    @staticmethod
    def _find_short(chain: List[Dict], target_delta: float) -> Optional[Dict]:
        """
        Pick the contract whose |Δ| is closest to target_delta and has a
        priceable mid (> 0). Filters out contracts whose bid is zero
        (typical for far-OTM strikes that would price to 0 credit).
        """
        candidates = [c for c in chain
                      if c.get("delta") not in (None, 0)
                      and float(c.get("bid", 0) or 0) > 0
                      and float(c.get("strike", 0) or 0) > 0]
        if not candidates:
            return None
        target = abs(target_delta)
        candidates.sort(key=lambda c: abs(abs(float(c["delta"])) - target))
        return candidates[0]

    @staticmethod
    def _find_strike(chain: List[Dict], target_strike: float) -> Optional[Dict]:
        """Find contract whose strike is closest to target_strike."""
        candidates = [c for c in chain if float(c.get("strike", 0) or 0) > 0]
        if not candidates:
            return None
        candidates.sort(key=lambda c: abs(float(c["strike"]) - target_strike))
        return candidates[0]


__all__ = [
    "SpreadCandidate",
    "ScanDiagnostics",
    "ChainScanner",
    "_pop_from_delta",
    "_cw_floor",
    "_ev_per_dollar_risked",
    "_score_candidate",
    "_score_candidate_with_reason",
    # Reject-reason taxonomy (stable journal keys).
    "REJECT_NO_CHAIN",
    "REJECT_NO_SHORT_CONTRACT",
    "REJECT_NO_LONG_CONTRACT",
    "REJECT_NON_POSITIVE_WIDTH",
    "REJECT_DTE_NON_POSITIVE",
    "REJECT_POP_BELOW_MIN",
    "REJECT_CREDIT_NON_POSITIVE",
    "REJECT_CREDIT_GE_WIDTH",
    "REJECT_CW_BELOW_FLOOR",
    "REJECT_EV_NON_POSITIVE",
]
