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

    Regime          Strategy
    ────────────    ─────────────────
    BULLISH         Bull Put Spread
    BEARISH         Bear Call Spread
    SIDEWAYS        Iron Condor
    """

    SPREAD_WIDTH = 5.0          # dollars between sold / bought strikes
    TARGET_DTE = 44             # days-to-expiration target (adjusted for available options)
    DTE_RANGE = (21, 45)        # acceptable DTE window

    def __init__(self, data_provider: MarketDataProvider,
                 max_delta: float = 0.20,
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
        """
        expiration = self._pick_expiration()
        logger.info("[%s] Planning %s strategy, expiration %s",
                     ticker, analysis.regime.value, expiration)

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
        """Choose the nearest Friday within our DTE window."""
        today = datetime.utcnow().date()
        target = today + timedelta(days=self.TARGET_DTE)
        # Roll to the next Friday
        days_until_friday = (4 - target.weekday()) % 7
        expiry = target + timedelta(days=days_until_friday)
        return expiry.strftime("%Y-%m-%d")

    def _find_sold_strike(self, contracts: List[Dict]) -> Optional[Dict]:
        """
        From the chain, pick the contract whose |delta| is closest to
        (but ≤) self.max_delta.
        """
        candidates = [
            c for c in contracts
            if 0 < abs(c["delta"]) <= self.max_delta and c["mid"] > 0
        ]
        if not candidates:
            return None
        # Closest delta to the ceiling (highest premium while staying OTM)
        candidates.sort(key=lambda c: abs(c["delta"]), reverse=True)
        return candidates[0]

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
