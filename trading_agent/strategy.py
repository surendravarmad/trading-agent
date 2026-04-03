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
    BULLISH + RS      Bull Put Spread  (relative strength bias)
    BEARISH           Bear Call Spread
    SIDEWAYS          Iron Condor
    SIDEWAYS + RS     Bull Put Spread  (relative strength bias)
    """

    # Minimum relative strength differential (vs SPY or QQQ) to trigger bias
    RS_THRESHOLD = 0.001   # 0.1% outperformance in 5-min window

    SPREAD_WIDTH = 5.0          # standardised width — controls buying-power usage
    TARGET_DTE = 45             # bias toward upper DTE range to reduce gamma risk
    DTE_RANGE = (35, 50)        # tightened window; upper bound slows gamma impact
    # Delta targeting window for the short leg: maximises POP while capping risk
    MIN_DELTA = 0.20            # floor — below this is too far OTM (low credit)
    # max_delta passed via __init__ (default 0.25)

    def __init__(self, data_provider: MarketDataProvider,
                 max_delta: float = 0.25,   # ceiling for short-leg delta
                 min_credit_ratio: float = 0.33):
        self.data = data_provider
        self.max_delta = max_delta
        self.min_credit_ratio = min_credit_ratio

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def plan(self, ticker: str, analysis: RegimeAnalysis) -> SpreadPlan:
        """
        Build the best credit-spread plan for *ticker* given regime *analysis*.

        Priority order:
          1. Mean Reversion — 3-std BB touch overrides everything
          2. Relative Strength bias — outperforming SPY/QQQ → Bull Put Spread
          3. Normal regime → Bull Put / Bear Call / Iron Condor
        """
        expiration = self._pick_expiration()
        logger.info("[%s] Planning %s strategy, expiration %s",
                     ticker, analysis.regime.value, expiration)

        # --- Priority 1: Mean Reversion (3-std BB touch) ---
        if analysis.regime == Regime.MEAN_REVERSION:
            return self._plan_mean_reversion(ticker, analysis, expiration)

        # --- Priority 2: Relative Strength bias ---
        rs_spy = getattr(analysis, "relative_strength_vs_spy", 0.0)
        rs_qqq = getattr(analysis, "relative_strength_vs_qqq", 0.0)
        rs_outperforming = (rs_spy > self.RS_THRESHOLD
                            or rs_qqq > self.RS_THRESHOLD)
        if rs_outperforming and analysis.regime in (Regime.BULLISH, Regime.SIDEWAYS):
            logger.info(
                "[%s] Relative strength bias → Bull Put Spread "
                "(RS_SPY=%.4f, RS_QQQ=%.4f)",
                ticker, rs_spy, rs_qqq)
            return self._plan_bull_put(ticker, analysis, expiration)

        # --- Priority 3: Normal regime mapping ---
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
        Choose the Friday nearest to TARGET_DTE (45), biasing toward the
        UPPER end of DTE_RANGE to reduce gamma risk near expiration.

        When both adjacent Fridays are within the allowed range the further
        one (higher DTE) is preferred — this keeps theta decay slower and
        gives more time for the trade to work.

        Uses local date (not UTC) to match the cron-run trading day.
        """
        today = datetime.now().date()
        target = today + timedelta(days=self.TARGET_DTE)

        days_to_next = (4 - target.weekday()) % 7
        next_friday = target + timedelta(days=days_to_next)
        prev_friday = next_friday - timedelta(days=7)

        min_expiry = today + timedelta(days=self.DTE_RANGE[0])
        max_expiry = today + timedelta(days=self.DTE_RANGE[1])

        next_ok = min_expiry <= next_friday <= max_expiry
        prev_ok = min_expiry <= prev_friday <= max_expiry

        if next_ok and prev_ok:
            # Both valid — pick the further Friday (higher DTE = less gamma risk)
            candidate = next_friday
        elif next_ok:
            candidate = next_friday
        elif prev_ok:
            candidate = prev_friday
        else:
            # Neither adjacent Friday fits; clamp to the closest valid Friday
            candidate = next_friday if days_to_next <= 3 else prev_friday
            if candidate > max_expiry:
                candidate -= timedelta(days=7)
            elif candidate < min_expiry:
                candidate += timedelta(days=7)

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

    def _find_bought_strike(self, contracts: List[Dict],
                            sold_strike: float,
                            direction: str) -> Optional[Dict]:
        """
        Find the protective leg ~SPREAD_WIDTH away in the right direction.
        """
        target = (sold_strike - self.SPREAD_WIDTH if direction == "lower"
                  else sold_strike + self.SPREAD_WIDTH)

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
