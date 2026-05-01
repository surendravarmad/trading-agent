"""
Phase III — PLAN
Selects optimal spread type and strikes based on the detected
market regime and option chain Greeks.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from trading_agent.regime import Regime, RegimeAnalysis
from trading_agent.market_data import MarketDataProvider
from trading_agent.calendar_utils import next_weekly_expiration
from trading_agent.chain_scanner import ChainScanner, SpreadCandidate

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Strategy data structures
# ------------------------------------------------------------------

@dataclass
class SpreadLeg:
    """A single option leg."""
    symbol: str
    strike: float
    action: str      # "sell" or "buy"
    option_type: str  # "put" or "call"
    delta: float
    theta: float
    bid: float
    ask: float
    mid: float


@dataclass
class SpreadPlan:
    """Complete trade plan for a credit spread."""
    ticker: str
    strategy_name: str   # "Bull Put Spread", "Bear Call Spread", "Iron Condor"
    regime: str
    legs: List[SpreadLeg]
    spread_width: float
    net_credit: float
    max_loss: float
    credit_to_width_ratio: float
    expiration: str
    reasoning: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    valid: bool = True
    rejection_reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "strategy": self.strategy_name,
            "regime": self.regime,
            "legs": [
                {
                    "symbol": l.symbol, "strike": l.strike,
                    "action": l.action, "type": l.option_type,
                    "delta": l.delta, "bid": l.bid, "ask": l.ask,
                }
                for l in self.legs
            ],
            "spread_width": self.spread_width,
            "net_credit": self.net_credit,
            "max_loss": self.max_loss,
            "credit_to_width_ratio": self.credit_to_width_ratio,
            "expiration": self.expiration,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
            "valid": self.valid,
            "rejection_reason": self.rejection_reason,
        }


# ------------------------------------------------------------------
# Strategy selector / planner
# ------------------------------------------------------------------

class StrategyPlanner:
    """
    Maps regime → strategy and picks strikes from the option chain.

    Regime            Strategy
    ──────────────    ───────────────────────────────────────────
    MEAN_REVERSION    Mean Reversion Spread (highest priority)
    BULLISH           Bull Put Spread
    BULLISH + RS_Z    Bull Put Spread  (Z-scored leadership bias)
    BEARISH           Bear Call Spread
    SIDEWAYS          Iron Condor
    SIDEWAYS + RS_Z   Bull Put Spread  (Z-scored leadership bias)

    Inter-market gate (Item 3 of the ETF macro patch)
    ─────────────────────────────────────────────────
    When ``analysis.inter_market_inhibit_bullish`` is True (VIX 5-min
    z-score > +2.0 σ) we suppress new bullish-premium openings:
      * BULLISH      → demoted to Bear Call Spread
      * SIDEWAYS     → demoted to Bear Call Spread (no put wing)
      * MEAN_REVERSION (lower band) → falls through unchanged
    """

    # ETF-only RS gate: trigger Bull-Put bias when the ticker leads its
    # configured anchor by ≥ 1.5 σ on the rolling intraday distribution.
    # Replaces the prior flat 0.1 % threshold which:
    #   - never fired for SPY (SPY-vs-SPY = 0.0)
    #   - misfired during low-vol drift (sub-noise flickers tripped it)
    # 1.5 σ ≈ 13th percentile two-tailed — a real leadership move, not
    # routine noise, and still loose enough to actually fire several
    # times per session in a normal regime.
    RS_ZSCORE_THRESHOLD = 1.5

    # Legacy class-level defaults — kept for back-compat with callers that
    # don't pass overrides (and for tests that read them as class attrs).
    # The Strategy-Profile preset system overrides these at __init__ time
    # via the ``preset`` keyword; see ``strategy_presets.PresetConfig`` and
    # ``trading_agent/strategy_presets.py`` for the active values.
    SPREAD_WIDTH = 5.0
    TARGET_DTE = 35
    DTE_RANGE = (28, 45)
    MIN_DELTA = 0.15            # floor — below this is too far OTM (low credit)

    # Strategy "kind" labels used by _pick_expiration to select per-strategy DTE.
    KIND_VERTICAL       = "vertical"        # Bull Put / Bear Call
    KIND_IRON_CONDOR    = "iron_condor"
    KIND_MEAN_REVERSION = "mean_reversion"

    def __init__(self, data_provider: MarketDataProvider,
                 max_delta: float = 0.20,   # ceiling for short-leg delta (~80% POP)
                 min_credit_ratio: float = 0.33,
                 *,
                 # Per-strategy DTE targets (None → use TARGET_DTE class default).
                 dte_vertical: Optional[int] = None,
                 dte_iron_condor: Optional[int] = None,
                 dte_mean_reversion: Optional[int] = None,
                 dte_window_days: Optional[int] = None,
                 # Width policy overrides — when both are None, fall back to the
                 # legacy formula: max(SPREAD_WIDTH, 3*grid, 2.5% × spot).
                 width_mode: Optional[str] = None,           # "pct_of_spot" | "fixed_dollar"
                 width_value: Optional[float] = None,
                 # Adaptive scan-mode wiring. When ``preset`` is supplied and
                 # ``preset.scan_mode == 'adaptive'`` the planner routes
                 # vertical/IC builders through ChainScanner instead of the
                 # static single-point picker. ``preset`` is None for legacy
                 # callers and tests that don't use the preset system.
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
        # to the journal alongside the picked plan. Cleared at the start of
        # every plan() call.
        self.last_scan_candidates: List[SpreadCandidate] = []
        self.last_scan_side: Optional[str] = None
        # Per-scan diagnostics (reject-reason histogram, best near-miss,
        # grid coverage). Mirrors ChainScanner.last_diagnostics; ``None``
        # in static mode or when no scan has run this cycle. Surfaced into
        # signals.jsonl so the user can answer "why did the scanner pass?"
        # straight from the journal.
        self.last_scan_diagnostics: Optional[Dict] = None

        # Per-strategy DTE targets and search window.
        self._dte_vertical       = dte_vertical       if dte_vertical       is not None else self.TARGET_DTE
        self._dte_iron_condor    = dte_iron_condor    if dte_iron_condor    is not None else self.TARGET_DTE
        self._dte_mean_reversion = dte_mean_reversion if dte_mean_reversion is not None else max(7, self.TARGET_DTE - 14)
        # Default ± window if caller didn't supply one. Width=7 keeps the picker
        # within the same calendar-month band.
        default_window = max(1, (self.DTE_RANGE[1] - self.DTE_RANGE[0]) // 2)
        self._dte_window = dte_window_days if dte_window_days is not None else default_window
        # Track whether *any* DTE override was explicitly supplied. When all
        # are None we preserve the exact legacy DTE_RANGE behaviour for the
        # vertical kind, which is what the test suite (and existing live
        # callers without preset support) rely on.
        self._dte_overridden = any(
            v is not None for v in (
                dte_vertical, dte_iron_condor, dte_mean_reversion, dte_window_days,
            )
        )

        # Width policy. None/None preserves the legacy adaptive formula.
        self._width_mode  = width_mode
        self._width_value = width_value

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @property
    def is_adaptive(self) -> bool:
        """True iff the active preset asked for chain-scanner planning."""
        return self._scanner is not None

    def plan(self, ticker: str, analysis: RegimeAnalysis) -> SpreadPlan:
        """
        Build the best credit-spread plan for *ticker* given regime *analysis*.

        Priority order:
          1. Mean Reversion — 3-std BB touch overrides everything
          2. Inter-market inhibit — VIX z-score > +2σ demotes Bull-Put / IC
             to Bear Call (Item 3 of the ETF macro patch)
          3. Z-scored leadership bias — leadership_zscore > 1.5 σ → Bull Put
             (Items 1 & 2 of the ETF macro patch)
          4. Normal regime → Bull Put / Bear Call / Iron Condor

        Each branch picks its own expiration with the per-strategy DTE
        from the active preset (verticals are typically shorter-dated
        than Iron Condors; mean-reversion shortest of all).

        When the active preset's ``scan_mode`` is ``"adaptive"`` the
        vertical and Iron-Condor branches delegate to ``ChainScanner``,
        which sweeps the (DTE × Δ × width) grid and returns the highest
        EV-per-$-risked candidate, or ``None`` (→ skip) when no point
        clears the breakeven-plus-edge floor. Mean-reversion and
        explicit IC trades remain on the static path because their
        timing/strike logic is regime-driven, not edge-driven.
        """
        # Reset scan state at the start of every plan call so leftover
        # candidates from the previous ticker can never bleed into this
        # one's journal.
        self.last_scan_candidates = []
        self.last_scan_side = None
        self.last_scan_diagnostics = None
        # --- Priority 1: Mean Reversion (3-std BB touch) ---
        # Mean reversion intentionally bypasses the inter-market gate —
        # a 3-std band touch IS a fear-spike condition, and the side
        # of the trade is dictated by the touch direction, so the gate
        # is redundant here.
        if analysis.regime == Regime.MEAN_REVERSION:
            expiration = self._pick_expiration(self.KIND_MEAN_REVERSION)
            logger.info("[%s] Planning Mean-Reversion, expiration %s",
                        ticker, expiration)
            return self._plan_mean_reversion(ticker, analysis, expiration)

        # --- Priority 2: Inter-market inhibit (VIX gate) ---
        # When the macro fear gate fires, refuse to open new bullish
        # premium (Bull Put, Iron Condor put-wing).  Falls through to
        # Bear Call — still a credit spread, but oriented to profit
        # from the downside continuation that the VIX spike implies.
        # If the regime is already Bearish, the gate is a no-op (we'd
        # have picked Bear Call anyway).
        inter_market_inhibit = getattr(
            analysis, "inter_market_inhibit_bullish", False)
        if inter_market_inhibit and analysis.regime in (
                Regime.BULLISH, Regime.SIDEWAYS):
            expiration = self._pick_expiration(self.KIND_VERTICAL)
            logger.info(
                "[%s] VIX inter-market inhibit (z=%.2f σ) → demoting "
                "%s to Bear Call Spread, expiration %s",
                ticker, getattr(analysis, "vix_zscore", 0.0),
                analysis.regime.value, expiration)
            return self._plan_bear_call(ticker, analysis, expiration)

        # --- Priority 3: Z-scored leadership bias ---
        leadership_z = getattr(analysis, "leadership_zscore", 0.0)
        leadership_anchor = getattr(analysis, "leadership_anchor", "")
        rs_outperforming = (leadership_anchor
                            and leadership_z > self.RS_ZSCORE_THRESHOLD)
        if rs_outperforming and analysis.regime in (
                Regime.BULLISH, Regime.SIDEWAYS):
            expiration = self._pick_expiration(self.KIND_VERTICAL)
            logger.info(
                "[%s] Z-scored leadership bias (vs %s, z=%.2f σ) → "
                "Bull Put Spread, expiration %s",
                ticker, leadership_anchor, leadership_z, expiration)
            return self._plan_bull_put(ticker, analysis, expiration)

        # --- Priority 4: Normal regime mapping ---
        if analysis.regime == Regime.BULLISH:
            expiration = self._pick_expiration(self.KIND_VERTICAL)
            logger.info("[%s] Planning Bull Put Spread, expiration %s",
                        ticker, expiration)
            return self._plan_bull_put(ticker, analysis, expiration)
        elif analysis.regime == Regime.BEARISH:
            expiration = self._pick_expiration(self.KIND_VERTICAL)
            logger.info("[%s] Planning Bear Call Spread, expiration %s",
                        ticker, expiration)
            return self._plan_bear_call(ticker, analysis, expiration)
        else:
            expiration = self._pick_expiration(self.KIND_IRON_CONDOR)
            logger.info("[%s] Planning Iron Condor, expiration %s",
                        ticker, expiration)
            return self._plan_iron_condor(ticker, analysis, expiration)

    # ------------------------------------------------------------------
    # Individual strategy builders
    # ------------------------------------------------------------------

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

        contracts = self.data.fetch_option_chain(ticker, expiration, "call")
        if not contracts:
            return self._empty_plan(ticker, "Bear Call Spread", analysis,
                                     expiration, "No call contracts available")

        sold = self._find_sold_strike(contracts)
        if not sold:
            return self._empty_plan(ticker, "Bear Call Spread", analysis,
                                     expiration,
                                     f"No call with |delta| ≤ {self.max_delta}")

        bought = self._find_bought_strike(contracts, sold["strike"],
                                           direction="higher")
        if not bought:
            return self._empty_plan(ticker, "Bear Call Spread", analysis,
                                     expiration, "No suitable protective leg found")

        return self._assemble_plan(ticker, "Bear Call Spread", analysis,
                                    expiration, sold, bought, "call")

    def _plan_mean_reversion(self, ticker: str, analysis: RegimeAnalysis,
                             expiration: str) -> SpreadPlan:
        """
        When price touches a 3-std Bollinger Band, sell a spread that
        profits from the expected reversion back toward the mean.

        Upper band touch → sell Bear Call Spread above current price
        Lower band touch → sell Bull Put Spread below current price
        """
        direction = getattr(analysis, "mean_reversion_direction", "")
        logger.info("[%s] Mean Reversion signal (%s band) — planning spread",
                    ticker, direction)

        if direction == "upper":
            # Price extended to upside → expect reversion down → Bear Call Spread
            result = self._plan_bear_call(ticker, analysis, expiration)
            result.strategy_name = "Mean Reversion Spread"
            result.reasoning = (
                f"Mean Reversion: price ({analysis.current_price:.2f}) touched "
                f"upper 3-std Bollinger Band. "
                f"Selling Bear Call Spread expecting reversion toward mean. "
                + result.reasoning
            )
            return result
        else:
            # Price extended to downside → expect reversion up → Bull Put Spread
            result = self._plan_bull_put(ticker, analysis, expiration)
            result.strategy_name = "Mean Reversion Spread"
            result.reasoning = (
                f"Mean Reversion: price ({analysis.current_price:.2f}) touched "
                f"lower 3-std Bollinger Band. "
                f"Selling Bull Put Spread expecting reversion toward mean. "
                + result.reasoning
            )
            return result

    def _plan_iron_condor(self, ticker: str, analysis: RegimeAnalysis,
                          expiration: str) -> SpreadPlan:
        """Combine a bull put spread and a bear call spread."""
        put_contracts = self.data.fetch_option_chain(ticker, expiration, "put")
        call_contracts = self.data.fetch_option_chain(ticker, expiration, "call")

        if not put_contracts or not call_contracts:
            return self._empty_plan(ticker, "Iron Condor", analysis,
                                     expiration, "Option chain unavailable")

        sold_put = self._find_sold_strike(put_contracts)
        sold_call = self._find_sold_strike(call_contracts)

        if not sold_put or not sold_call:
            return self._empty_plan(ticker, "Iron Condor", analysis,
                                     expiration,
                                     f"Cannot find strikes with |delta| ≤ {self.max_delta}")

        bought_put = self._find_bought_strike(put_contracts,
                                               sold_put["strike"], "lower")
        bought_call = self._find_bought_strike(call_contracts,
                                                sold_call["strike"], "higher")

        if not bought_put or not bought_call:
            return self._empty_plan(ticker, "Iron Condor", analysis,
                                     expiration, "Protective legs not found")

        # Build legs
        legs = [
            self._make_leg(sold_put, "sell", "put"),
            self._make_leg(bought_put, "buy", "put"),
            self._make_leg(sold_call, "sell", "call"),
            self._make_leg(bought_call, "buy", "call"),
        ]

        # Credit = sell premiums – buy premiums (using bid/ask for realism)
        credit_put_side = sold_put["bid"] - bought_put["ask"]
        credit_call_side = sold_call["bid"] - bought_call["ask"]
        net_credit = round(credit_put_side + credit_call_side, 2)

        # Max loss is the wider side minus total credit
        width_put = abs(sold_put["strike"] - bought_put["strike"])
        width_call = abs(sold_call["strike"] - bought_call["strike"])
        wider = max(width_put, width_call)
        max_loss = round((wider - net_credit) * 100, 2)
        ratio = round(net_credit / wider, 4) if wider else 0

        reasoning = (f"Iron Condor for sideways market. "
                     f"Put side: {sold_put['strike']}/{bought_put['strike']}, "
                     f"Call side: {sold_call['strike']}/{bought_call['strike']}. "
                     f"Net credit ${net_credit}, Max loss ${max_loss}.")

        plan = SpreadPlan(
            ticker=ticker, strategy_name="Iron Condor",
            regime=analysis.regime.value, legs=legs,
            spread_width=wider, net_credit=net_credit,
            max_loss=max_loss, credit_to_width_ratio=ratio,
            expiration=expiration, reasoning=reasoning,
        )

        # Validate credit ratio
        if ratio < self.min_credit_ratio:
            plan.valid = False
            plan.rejection_reason = (
                f"Credit-to-width ratio {ratio:.4f} < minimum {self.min_credit_ratio}")

        return plan

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

        ``fallback_expiration`` is used only for the empty-plan when the
        scanner produced nothing — gives the journal a sensible expiry
        stamp instead of an empty string.
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

        reasoning = (
            f"{strategy_name} (adaptive). {ticker} {analysis.regime.value} "
            f"regime; scanner picked {best.dte}-DTE Δ-{abs(best.short_delta):.3f}, "
            f"width=${best.width:.2f}, credit=${best.credit:.2f}, "
            f"C/W={best.cw_ratio:.4f} (floor {best.cw_floor:.4f}), "
            f"POP={best.pop:.0%}, EV/$risked={best.ev_per_dollar_risked:+.4f} "
            f"(annualized {best.annualized_score:+.3f}). "
            f"{len(candidates)} candidate(s) cleared the floor."
        )

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

    def _pick_expiration(self, kind: str = "vertical") -> str:
        """
        Choose the weekly expiration nearest to the target DTE for *kind*.

        ``kind`` is one of ``vertical`` (Bull Put / Bear Call), ``iron_condor``
        (4-leg neutral), or ``mean_reversion`` (3-σ snapback). The active
        Strategy-Profile preset (Conservative / Balanced / Aggressive /
        Custom) supplies the per-kind DTE; absent overrides we fall back
        to the class-level ``TARGET_DTE`` so legacy callers keep working.

        Uses NYSE calendar (pandas_market_calendars) so holiday-Fridays
        (e.g. Good Friday) correctly resolve to Thursday expiration where
        the weekly options actually list. Uses local date (not UTC) to
        match the cron-run trading day.
        """
        if kind == self.KIND_IRON_CONDOR:
            target = self._dte_iron_condor
        elif kind == self.KIND_MEAN_REVERSION:
            target = self._dte_mean_reversion
        else:
            target = self._dte_vertical

        # No preset override + vertical kind → preserve the exact legacy
        # DTE_RANGE behaviour (target=35, range=(28,45)) so existing tests
        # and pre-preset callers see identical picks. Once any DTE override
        # is supplied we switch to the window-based range, which is also
        # used unconditionally for IC and MR (no legacy callers for those).
        if not self._dte_overridden and kind == self.KIND_VERTICAL:
            dte_min, dte_max = self.DTE_RANGE
        else:
            dte_min = max(1, target - self._dte_window)
            dte_max = target + self._dte_window

        today = datetime.now().date()
        candidate = next_weekly_expiration(
            today=today,
            target_dte=target,
            dte_min=dte_min,
            dte_max=dte_max,
        )
        dte = (candidate - today).days
        logger.debug("Expiration selected: %s (%d DTE, kind=%s)",
                     candidate, dte, kind)
        return candidate.strftime("%Y-%m-%d")

    def _find_sold_strike(self, contracts: List[Dict]) -> Optional[Dict]:
        """
        Pick the short-leg contract targeting the 0.20–0.25 delta window
        (MIN_DELTA to max_delta).

        Priority:
          1. Contracts in the sweet-spot [MIN_DELTA, max_delta] — pick the
             one closest to max_delta (highest premium, still within POP target)
          2. Fallback: any contract with |delta| ≤ max_delta (chain may be sparse)
        """
        # Primary: within the target delta window
        sweet_spot = [
            c for c in contracts
            if self.MIN_DELTA <= abs(c["delta"]) <= self.max_delta and c["mid"] > 0
        ]
        if sweet_spot:
            sweet_spot.sort(key=lambda c: abs(c["delta"]), reverse=True)
            return sweet_spot[0]

        # Fallback: any valid OTM delta (sparse chains)
        fallback = [
            c for c in contracts
            if 0 < abs(c["delta"]) <= self.max_delta and c["mid"] > 0
        ]
        if not fallback:
            return None
        fallback.sort(key=lambda c: abs(c["delta"]), reverse=True)
        logger.debug("Delta sweet-spot empty — using fallback delta %.3f",
                     abs(fallback[0]["delta"]))
        return fallback[0]

    @staticmethod
    def _strike_grid_step(contracts: List[Dict]) -> float:
        """
        Infer the strike-grid step size from the chain by taking the
        modal gap between consecutive sorted strikes.

        SPY/QQQ trade on a $5 grid for far-dated expirations and a $1
        grid near the money; this returns whichever is dominant.  For
        IWM and most equities the grid is $1 or $2.50.  Returns 1.0 as
        a conservative floor if the chain is too thin to infer a grid.
        """
        strikes = sorted({float(c["strike"]) for c in contracts if c.get("strike")})
        if len(strikes) < 3:
            return 1.0
        gaps: Dict[float, int] = {}
        for a, b in zip(strikes, strikes[1:]):
            gap = round(b - a, 2)
            if gap > 0:
                gaps[gap] = gaps.get(gap, 0) + 1
        if not gaps:
            return 1.0
        # Modal gap (most common spacing).
        return max(gaps, key=lambda g: gaps[g])

    def _pick_spread_width(self, contracts: List[Dict],
                           sold_strike: float) -> float:
        """
        Compute the spread width.

        Two paths:
          * **Preset override** — when the active preset specifies a
            ``width_mode`` and ``width_value`` (set in __init__), that
            policy takes precedence. ``pct_of_spot`` uses ``width_value
            × sold_strike``; ``fixed_dollar`` uses ``width_value`` raw.
            Either is then snapped UP to the strike grid.
          * **Legacy adaptive formula** — when no override is supplied,
            take ``max(SPREAD_WIDTH, 3 × strike_grid_step, 2.5% × spot
            proxy)`` and snap UP to the grid. This is the original
            behavior and remains the back-compat default.

        The sold-leg strike is the spot proxy (within ~2 σ of spot for a
        0.20-delta short put — close enough for width sizing).
        """
        grid = self._strike_grid_step(contracts)
        spot_proxy = sold_strike

        if self._width_mode == "pct_of_spot" and self._width_value is not None:
            candidate = max(grid, self._width_value * spot_proxy)
        elif self._width_mode == "fixed_dollar" and self._width_value is not None:
            candidate = max(grid, float(self._width_value))
        else:
            # Legacy adaptive width.
            candidate = max(self.SPREAD_WIDTH, 3 * grid, 0.025 * spot_proxy)

        # Snap UP to the strike grid so a real strike sits at this distance.
        snapped = grid * max(1, int(round(candidate / grid + 0.4999)))
        return float(snapped)

    def _find_bought_strike(self, contracts: List[Dict],
                            sold_strike: float,
                            direction: str) -> Optional[Dict]:
        """
        Find the protective leg an adaptive width away in the right
        direction.  Width is computed by :meth:`_pick_spread_width`
        based on the chain's strike grid and the sold-leg strike.
        """
        width = self._pick_spread_width(contracts, sold_strike)
        target = (sold_strike - width if direction == "lower"
                  else sold_strike + width)

        candidates = sorted(contracts, key=lambda c: abs(c["strike"] - target))
        for c in candidates:
            if direction == "lower" and c["strike"] < sold_strike:
                return c
            if direction == "higher" and c["strike"] > sold_strike:
                return c
        return None

    def _assemble_plan(self, ticker, name, analysis, expiration,
                       sold, bought, opt_type) -> SpreadPlan:
        """Build a two-leg spread plan and validate credit ratio."""
        legs = [
            self._make_leg(sold, "sell", opt_type),
            self._make_leg(bought, "buy", opt_type),
        ]
        width = abs(sold["strike"] - bought["strike"])
        # Use bid for sold (what buyer pays), ask for bought (what seller wants)
        # This gives more realistic pricing for better execution
        net_credit = round(sold["bid"] - bought["ask"], 2)
        max_loss = round((width - net_credit) * 100, 2)
        ratio = round(net_credit / width, 4) if width else 0

        reasoning = (f"{name} on {ticker} ({analysis.regime.value} regime). "
                     f"Sold {sold['strike']} (Δ={sold['delta']:.3f}), "
                     f"Bought {bought['strike']}. "
                     f"Credit ${net_credit}, Width ${width}, Max loss ${max_loss}.")

        plan = SpreadPlan(
            ticker=ticker, strategy_name=name,
            regime=analysis.regime.value, legs=legs,
            spread_width=width, net_credit=net_credit,
            max_loss=max_loss, credit_to_width_ratio=ratio,
            expiration=expiration, reasoning=reasoning,
        )

        if ratio < self.min_credit_ratio:
            plan.valid = False
            plan.rejection_reason = (
                f"Credit-to-width ratio {ratio:.4f} < minimum {self.min_credit_ratio}")

        return plan

    @staticmethod
    def _make_leg(contract: Dict, action: str, opt_type: str) -> SpreadLeg:
        return SpreadLeg(
            symbol=contract["symbol"],
            strike=contract["strike"],
            action=action,
            option_type=opt_type,
            delta=contract["delta"],
            theta=contract.get("theta", 0),
            bid=contract["bid"],
            ask=contract["ask"],
            mid=contract["mid"],
        )

    def _empty_plan(self, ticker, name, analysis, expiration, reason) -> SpreadPlan:
        return SpreadPlan(
            ticker=ticker, strategy_name=name,
            regime=analysis.regime.value, legs=[],
            spread_width=0, net_credit=0, max_loss=0,
            credit_to_width_ratio=0, expiration=expiration,
            reasoning=reason, valid=False, rejection_reason=reason,
        )
