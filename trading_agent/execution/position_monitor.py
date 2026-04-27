"""
Position Monitor
=================
Fetches open positions from Alpaca, computes unrealized P&L against
the original trade plan, and generates exit signals based on:

  1. HARD_STOP:        Spread value ≥ 3× initial credit (immediate, no debounce)
  2. PROFIT_TARGET:    50% of max credit captured
  3. STRIKE_PROXIMITY: Underlying within 1% of any short strike (immediate)
  4. DTE_SAFETY:       Last trading day before expiry after 15:30 ET (immediate)
  5. REGIME_SHIFT:     Current regime contradicts the position's strategy

Signals marked "immediate" bypass the 3-cycle debounce in agent.py and
trigger a market-order close without waiting for confirmation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

import requests

from trading_agent.market.calendar_utils import is_last_trading_day_before
from trading_agent.strategy.regime import Regime
from trading_agent.config.loader import PositionMonitorRules

logger = logging.getLogger(__name__)


class ExitSignal(Enum):
    HOLD = "hold"
    STOP_LOSS = "stop_loss"          # legacy alias kept for compatibility
    HARD_STOP = "hard_stop"          # spread value ≥ 3× credit (immediate)
    PROFIT_TARGET = "profit_target"  # 50% of credit captured
    REGIME_SHIFT = "regime_shift"    # regime no longer matches strategy
    STRIKE_PROXIMITY = "strike_proximity"  # underlying within 1% of short strike
    DTE_SAFETY = "dte_safety"        # Thursday before expiry ≥ 15:30 ET
    EXPIRED = "expired"


# Signals that bypass the 3-cycle debounce — close immediately
IMMEDIATE_EXIT_SIGNALS = {
    ExitSignal.HARD_STOP,
    ExitSignal.STRIKE_PROXIMITY,
    ExitSignal.DTE_SAFETY,
}

# Which regime each strategy is compatible with
STRATEGY_REGIME_MAP = {
    "Bull Put Spread":      Regime.BULLISH,
    "Bear Call Spread":     Regime.BEARISH,
    "Iron Condor":          Regime.SIDEWAYS,
    "Mean Reversion Spread": None,   # direction-neutral; never regime-shift closed
}


@dataclass
class PositionSnapshot:
    """A single option leg position from Alpaca."""
    symbol: str
    qty: int
    side: str               # "long" or "short"
    avg_entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_plpc: float
    asset_class: str         # "us_option" for options


@dataclass
class SpreadPosition:
    """
    Aggregated view of a credit spread (2 or 4 legs) linked back to
    the original trade plan.
    """
    underlying: str
    strategy_name: str
    legs: List[PositionSnapshot]
    original_credit: float        # net credit received at entry (per share)
    max_loss: float               # defined max loss from the trade plan ($)
    spread_width: float
    net_unrealized_pl: float      # sum of all legs' unrealized P&L
    expiration: str = ""          # option expiration date YYYY-MM-DD
    short_strikes: List[float] = field(default_factory=list)  # short-leg strikes
    exit_signal: ExitSignal = ExitSignal.HOLD
    exit_reason: str = ""


class PositionMonitor:
    """
    Monitors open option positions and generates exit signals.

    Parameters
    ----------
    profit_target_pct : float
        Close when unrealized profit ≥ this fraction of the initial credit
        collected.  Default 0.50 (50% profit target — capital retainment).
    hard_stop_multiplier : float
        Close immediately when the spread has lost this multiple of the
        original credit.  Default 3.0 (hard stop at 3× credit).
    strike_proximity_pct : float
        Close immediately when underlying price is within this fraction of
        any short strike.  Default 0.01 (1%).
    """

    def __init__(self, api_key: str, secret_key: str,
                 base_url: str = "https://paper-api.alpaca.markets/v2",
                 stop_loss_pct: float = 0.50,    # kept for legacy compat
                 profit_target_pct: float = 0.50,  # 50% profit taker
                 hard_stop_multiplier: float = 3.0,
                 strike_proximity_pct: float = 0.01,
                 rules: "PositionMonitorRules | None" = None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        r = rules or PositionMonitorRules()
        self.stop_loss_pct = stop_loss_pct
        self.profit_target_pct = r.profit_target_pct if rules else profit_target_pct
        self.hard_stop_multiplier = r.hard_stop_multiplier if rules else hard_stop_multiplier
        self.strike_proximity_pct = r.strike_proximity_pct if rules else strike_proximity_pct
        self._dte_safety_hour = r.dte_safety_hour
        self._dte_safety_minute = r.dte_safety_minute

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # Fetch positions from Alpaca
    # ------------------------------------------------------------------

    def fetch_open_positions(self) -> List[PositionSnapshot]:
        """GET /v2/positions — filters to us_option only."""
        url = f"{self.base_url}/positions"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=10)
            resp.raise_for_status()
            positions_data = resp.json()

            positions = []
            for p in positions_data:
                snap = PositionSnapshot(
                    symbol=p.get("symbol", ""),
                    qty=int(p.get("qty", 0)),
                    side=p.get("side", ""),
                    avg_entry_price=float(p.get("avg_entry_price", 0)),
                    current_price=float(p.get("current_price", 0)),
                    market_value=float(p.get("market_value", 0)),
                    cost_basis=float(p.get("cost_basis", 0)),
                    unrealized_pl=float(p.get("unrealized_pl", 0)),
                    unrealized_plpc=float(p.get("unrealized_plpc", 0)),
                    asset_class=p.get("asset_class", ""),
                )
                positions.append(snap)

            option_positions = [p for p in positions if p.asset_class == "us_option"]
            logger.info("Fetched %d total positions, %d are options",
                        len(positions), len(option_positions))
            return option_positions

        except requests.RequestException as exc:
            logger.error("Failed to fetch positions: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Group legs into spread positions
    # ------------------------------------------------------------------

    def group_into_spreads(self, positions: List[PositionSnapshot],
                           trade_plans: List[Dict]) -> List[SpreadPosition]:
        """
        Match open option positions to their original trade plans using
        the option symbols in each plan's legs.
        """
        spreads = []

        for plan in trade_plans:
            tp = plan.get("trade_plan", {})
            plan_legs = tp.get("legs", [])
            plan_symbols = {leg["symbol"] for leg in plan_legs}
            if not plan_symbols:
                continue

            matched_legs = [p for p in positions if p.symbol in plan_symbols]
            if not matched_legs:
                continue

            # Extract short-strike prices from the plan for proximity checks
            short_strikes = [
                leg["strike"] for leg in plan_legs
                if leg.get("action") == "sell" and "strike" in leg
            ]

            net_pl = sum(leg.unrealized_pl for leg in matched_legs)

            spread = SpreadPosition(
                underlying=tp.get("ticker", ""),
                strategy_name=tp.get("strategy", ""),
                legs=matched_legs,
                original_credit=tp.get("net_credit", 0),
                max_loss=tp.get("max_loss", 0),
                spread_width=tp.get("spread_width", 0),
                net_unrealized_pl=net_pl,
                expiration=tp.get("expiration", ""),
                short_strikes=short_strikes,
            )
            spreads.append(spread)

        logger.info("Grouped positions into %d spread(s)", len(spreads))
        return spreads

    # ------------------------------------------------------------------
    # Evaluate exit signals
    # ------------------------------------------------------------------

    def evaluate(self, spreads: List[SpreadPosition],
                 current_regimes: Dict[str, Regime],
                 underlying_prices: Optional[Dict[str, float]] = None
                 ) -> List[SpreadPosition]:
        """
        Check each spread against all exit rules and assign exit_signal.

        Parameters
        ----------
        underlying_prices : dict mapping ticker → current price, used for
            the strike-proximity guard.
        """
        prices = underlying_prices or {}

        for spread in spreads:
            signal, reason = self._check_exit(
                spread, current_regimes, prices.get(spread.underlying, 0.0))
            spread.exit_signal = signal
            spread.exit_reason = reason

            if signal != ExitSignal.HOLD:
                immediate = signal in IMMEDIATE_EXIT_SIGNALS
                logger.warning(
                    "[%s] EXIT SIGNAL: %s%s — %s | P&L=$%.2f",
                    spread.underlying, signal.value,
                    " (IMMEDIATE)" if immediate else " (debounce)",
                    reason, spread.net_unrealized_pl)
            else:
                logger.info(
                    "[%s] HOLD — P&L=$%.2f (credit=$%.2f, max_loss=$%.2f)",
                    spread.underlying, spread.net_unrealized_pl,
                    spread.original_credit, spread.max_loss)

        return spreads

    def _check_exit(self, spread: SpreadPosition,
                    current_regimes: Dict[str, Regime],
                    underlying_price: float = 0.0):
        """Return (ExitSignal, reason) for a single spread."""

        credit_value = spread.original_credit * 100   # per-contract dollar value

        # --- 1. Hard stop: spread has lost 3× the initial credit (IMMEDIATE) ---
        hard_stop_threshold = credit_value * self.hard_stop_multiplier
        loss = -spread.net_unrealized_pl   # positive when losing
        if loss >= hard_stop_threshold > 0:
            return (
                ExitSignal.HARD_STOP,
                f"Loss ${loss:.2f} ≥ {self.hard_stop_multiplier:.0f}× credit "
                f"${credit_value:.2f} (threshold=${hard_stop_threshold:.2f})"
            )

        # --- 2. Legacy stop-loss: loss ≥ 50% of defined max-loss ---
        loss_threshold = spread.max_loss * self.stop_loss_pct
        if loss >= loss_threshold > 0:
            return (
                ExitSignal.STOP_LOSS,
                f"Loss ${loss:.2f} ≥ {self.stop_loss_pct*100:.0f}% of "
                f"max loss ${spread.max_loss:.2f}"
            )

        # --- 3. Profit target: 50% of credit captured ---
        profit_threshold = credit_value * self.profit_target_pct
        if spread.net_unrealized_pl >= profit_threshold > 0:
            return (
                ExitSignal.PROFIT_TARGET,
                f"Profit ${spread.net_unrealized_pl:.2f} ≥ "
                f"{self.profit_target_pct*100:.0f}% of credit "
                f"${credit_value:.2f}"
            )

        # --- 4. Strike proximity guard (IMMEDIATE) ---
        if underlying_price > 0 and spread.short_strikes:
            for strike in spread.short_strikes:
                proximity = abs(underlying_price - strike) / strike
                if proximity <= self.strike_proximity_pct:
                    return (
                        ExitSignal.STRIKE_PROXIMITY,
                        f"Underlying ${underlying_price:.2f} is within "
                        f"{proximity*100:.2f}% of short strike ${strike:.0f} "
                        f"— closing to prevent ITM assignment"
                    )

        # --- 5. DTE safety: liquidate by 15:30 ET on Thursday before expiry ---
        dte_signal = self._check_dte_safety(spread.expiration)
        if dte_signal:
            return (ExitSignal.DTE_SAFETY, dte_signal)

        # --- 6. Regime shift ---
        ticker = spread.underlying
        if ticker in current_regimes:
            current_regime = current_regimes[ticker]
            expected_regime = STRATEGY_REGIME_MAP.get(spread.strategy_name)
            if expected_regime and current_regime != expected_regime:
                return (
                    ExitSignal.REGIME_SHIFT,
                    f"Regime shifted to {current_regime.value} but holding "
                    f"{spread.strategy_name} (expects {expected_regime.value})"
                )

        return (ExitSignal.HOLD, "")

    @staticmethod
    def _check_dte_safety(expiration: str) -> str:
        """
        Return a non-empty reason string if the DTE safety rule triggers.

        Rule: if today is the **last NYSE trading day strictly before
        expiration** AND current time is ≥ 15:30 ET, return a warning.

        We avoid carrying an option into its final day of life to prevent
        last-day gamma explosion and assignment risk. Using the NYSE
        calendar (pandas_market_calendars) correctly handles holiday
        weeks — e.g. when Good Friday closes the market, the last trading
        day before a Friday expiration is Thursday; when a Wednesday
        expiration week lands (unusual), the rule fires on Tuesday.
        """
        if not expiration:
            return ""
        try:
            exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
            # Convert to ET for the time check
            now_utc = datetime.now(timezone.utc)
            # ET = UTC-4 (EDT) or UTC-5 (EST); use UTC-4 (market hours)
            now_et_hour = (now_utc.hour - 4) % 24
            now_et_minute = now_utc.minute
            today = now_utc.date()

            after_cutoff = (now_et_hour > self._dte_safety_hour or
                            (now_et_hour == self._dte_safety_hour and now_et_minute >= self._dte_safety_minute))
            last_day = is_last_trading_day_before(today, exp_date)

            if last_day and after_cutoff:
                return (
                    f"DTE safety: expiration {expiration} is the next "
                    f"trading day. Liquidating by 15:30 ET to avoid "
                    f"last-day gamma risk."
                )
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, spreads: List[SpreadPosition]) -> Dict:
        total_pl = sum(s.net_unrealized_pl for s in spreads)
        signals: Dict[str, int] = {}
        for s in spreads:
            sig = s.exit_signal.value
            signals[sig] = signals.get(sig, 0) + 1

        return {
            "total_spreads": len(spreads),
            "total_unrealized_pl": round(total_pl, 2),
            "signals": signals,
            "positions": [
                {
                    "underlying": s.underlying,
                    "strategy": s.strategy_name,
                    "pl": round(s.net_unrealized_pl, 2),
                    "signal": s.exit_signal.value,
                    "reason": s.exit_reason,
                    "expiration": s.expiration,
                    "short_strikes": s.short_strikes,
                }
                for s in spreads
            ],
        }
