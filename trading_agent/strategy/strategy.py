"""
Phase III — PLAN
Selects optimal spread type and strikes based on the detected
market regime and option chain Greeks.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from trading_agent.strategy.regime import Regime, RegimeAnalysis
from trading_agent.market.market_data import MarketDataProvider
from trading_agent.market.calendar_utils import next_weekly_expiration
from trading_agent.config.loader import StrategyRules

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

    # Legacy fixed width — retained as a floor; the active width is now
    # computed per-underlying by ``_pick_spread_width`` below.  At spot=$200
    # this floor matches the legacy behavior; at spot=$700 (SPY/QQQ) the
    # adaptive width takes over and produces ~$15-20 wide spreads, which
    # is what's needed to clear the 1/3-of-width credit target on a
    # 30-40 DTE 0.20-delta short put.
    SPREAD_WIDTH = 5.0
    # Theta capture is concentrated in the 25-40 DTE band; the prior
    # 45 DTE / (35,50) range was too far out, leaving credits thin and
    # forcing the credit/width gate into perpetual fail mode.
    TARGET_DTE = 35
    DTE_RANGE = (28, 45)
    # Delta targeting window for the short leg: maximises POP while capping risk.
    # Lowered from 0.20 to 0.15 to preserve a meaningful sweet-spot range when
    # max_delta is 0.20 (default). With MIN_DELTA == max_delta the band would
    # collapse to a single point and every trade would land in the fallback branch.
    MIN_DELTA = 0.15            # floor — below this is too far OTM (low credit)
    # max_delta passed via __init__ (default 0.20 — matches README design, ~80% POP)

    def __init__(self, data_provider: MarketDataProvider,
                 max_delta: float = 0.20,   # ceiling for short-leg delta (~80% POP)
                 min_credit_ratio: float = 0.33,
                 rules: "StrategyRules | None" = None):
        self.data = data_provider
        self.max_delta = max_delta
        self.min_credit_ratio = min_credit_ratio
        r = rules or StrategyRules()
        self.RS_ZSCORE_THRESHOLD = r.rs_zscore_threshold
        self.SPREAD_WIDTH = r.spread_width_floor
        self.TARGET_DTE = r.target_dte
        self.DTE_RANGE = r.dte_range
        self.MIN_DELTA = r.min_delta

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

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
        """
        expiration = self._pick_expiration()
        logger.info("[%s] Planning %s strategy, expiration %s",
                     ticker, analysis.regime.value, expiration)

        # --- Priority 1: Mean Reversion (3-std BB touch) ---
        # Mean reversion intentionally bypasses the inter-market gate —
        # a 3-std band touch IS a fear-spike condition, and the side
        # of the trade is dictated by the touch direction, so the gate
        # is redundant here.
        if analysis.regime == Regime.MEAN_REVERSION:
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
            logger.info(
                "[%s] VIX inter-market inhibit (z=%.2f σ) → demoting "
                "%s to Bear Call Spread",
                ticker, getattr(analysis, "vix_zscore", 0.0),
                analysis.regime.value)
            return self._plan_bear_call(ticker, analysis, expiration)

        # --- Priority 3: Z-scored leadership bias ---
        leadership_z = getattr(analysis, "leadership_zscore", 0.0)
        leadership_anchor = getattr(analysis, "leadership_anchor", "")
        rs_outperforming = (leadership_anchor
                            and leadership_z > self.RS_ZSCORE_THRESHOLD)
        if rs_outperforming and analysis.regime in (
                Regime.BULLISH, Regime.SIDEWAYS):
            logger.info(
                "[%s] Z-scored leadership bias (vs %s, z=%.2f σ) → Bull Put Spread",
                ticker, leadership_anchor, leadership_z)
            return self._plan_bull_put(ticker, analysis, expiration)

        # --- Priority 4: Normal regime mapping ---
        if analysis.regime == Regime.BULLISH:
            return self._plan_bull_put(ticker, analysis, expiration)
        elif analysis.regime == Regime.BEARISH:
            return self._plan_bear_call(ticker, analysis, expiration)
        else:
            return self._plan_iron_condor(ticker, analysis, expiration)

    # ------------------------------------------------------------------
    # Individual strategy builders
    # ------------------------------------------------------------------

    def _plan_bull_put(self, ticker: str, analysis: RegimeAnalysis,
                       expiration: str) -> SpreadPlan:
        """Sell an OTM put, buy a further-OTM put."""
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

    def _pick_expiration(self) -> str:
        """
        Choose the weekly expiration nearest to TARGET_DTE (45), biasing
        toward the UPPER end of DTE_RANGE to reduce gamma risk.

        Uses NYSE calendar (pandas_market_calendars) so that holiday-Fridays
        (e.g. Good Friday) correctly resolve to Thursday expiration, which
        is where the weekly options actually list.

        Uses local date (not UTC) to match the cron-run trading day.
        """
        today = datetime.now().date()
        candidate = next_weekly_expiration(
            today=today,
            target_dte=self.TARGET_DTE,
            dte_min=self.DTE_RANGE[0],
            dte_max=self.DTE_RANGE[1],
        )
        dte = (candidate - today).days
        logger.debug("Expiration selected: %s (%d DTE)", candidate, dte)
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
        Compute an adaptive spread width that scales with spot, the
        underlying's strike grid, and the legacy floor.

        Three constraints, take the maximum:
          1. ``SPREAD_WIDTH`` (legacy floor — never go narrower than this)
          2. ``3 × strike_grid_step``  (span at least 3 strikes so the
             two legs aren't priced almost identically)
          3. ``2.5% × spot``  (roughly the move size that makes the
             1/3-of-width credit math work at 25-40 DTE / 0.20 delta)

        The result is then snapped UP to the strike grid so a contract
        actually exists at that distance.
        """
        grid = self._strike_grid_step(contracts)
        # Use the sold-leg strike as the spot proxy — it's the closest
        # we have without re-piping the underlying price down here, and
        # for a 0.20-delta short the strike is within ~2 σ of spot.
        spot_proxy = sold_strike
        # Hold-strikes-far-apart constraint
        candidate = max(self.SPREAD_WIDTH, 3 * grid, 0.025 * spot_proxy)
        # Snap UP to the grid so a real strike sits at this distance
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
