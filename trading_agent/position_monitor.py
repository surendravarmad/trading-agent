"""
Position Monitor
=================
Fetches open positions from Alpaca, computes unrealized P&L against
the original trade plan, and generates exit signals based on:

  1. STOP-LOSS:    Close when unrealized loss ≥ 50% of max defined loss
  2. PROFIT-TARGET: Close when unrealized profit ≥ 75% of max credit received
  3. REGIME-SHIFT:  Close when current regime contradicts the position's strategy
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import requests

from trading_agent.regime import Regime

logger = logging.getLogger(__name__)


class ExitSignal(Enum):
    HOLD = "hold"
    STOP_LOSS = "stop_loss"
    PROFIT_TARGET = "profit_target"
    REGIME_SHIFT = "regime_shift"
    EXPIRED = "expired"


# Which regime each strategy is compatible with
STRATEGY_REGIME_MAP = {
    "Bull Put Spread": Regime.BULLISH,
    "Bear Call Spread": Regime.BEARISH,
    "Iron Condor": Regime.SIDEWAYS,
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
    Aggregated view of a credit spread (2 or 4 legs grouped by underlying).
    This links Alpaca positions back to the original trade plan.
    """
    underlying: str
    strategy_name: str
    legs: List[PositionSnapshot]
    original_credit: float       # net credit received at entry
    max_loss: float              # defined max loss from the trade plan
    spread_width: float
    net_unrealized_pl: float     # sum of all legs' unrealized P&L
    exit_signal: ExitSignal = ExitSignal.HOLD
    exit_reason: str = ""


class PositionMonitor:
    """
    Monitors open option positions against risk thresholds and
    regime changes, producing exit signals when action is needed.
    """

    def __init__(self, api_key: str, secret_key: str,
                 base_url: str = "https://paper-api.alpaca.markets/v2",
                 stop_loss_pct: float = 0.50,
                 profit_target_pct: float = 0.75):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.stop_loss_pct = stop_loss_pct          # 50% of max loss
        self.profit_target_pct = profit_target_pct    # 75% of credit

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
        """
        GET /v2/positions — returns all open positions.
        Filters to options positions only (asset_class == 'us_option').
        """
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

        *trade_plans* is a list of dicts loaded from the trade_plan JSON files.
        """
        spreads = []

        for plan in trade_plans:
            tp = plan.get("trade_plan", {})
            plan_symbols = {leg["symbol"] for leg in tp.get("legs", [])}
            if not plan_symbols:
                continue

            # Find matching positions
            matched_legs = [p for p in positions if p.symbol in plan_symbols]
            if not matched_legs:
                continue

            net_pl = sum(leg.unrealized_pl for leg in matched_legs)

            spread = SpreadPosition(
                underlying=tp.get("ticker", ""),
                strategy_name=tp.get("strategy", ""),
                legs=matched_legs,
                original_credit=tp.get("net_credit", 0),
                max_loss=tp.get("max_loss", 0),
                spread_width=tp.get("spread_width", 0),
                net_unrealized_pl=net_pl,
            )
            spreads.append(spread)

        logger.info("Grouped positions into %d spread(s)", len(spreads))
        return spreads

    # ------------------------------------------------------------------
    # Evaluate exit signals
    # ------------------------------------------------------------------

    def evaluate(self, spreads: List[SpreadPosition],
                 current_regimes: Dict[str, Regime]) -> List[SpreadPosition]:
        """
        Check each spread position against exit rules:
          1. Stop-loss:     unrealized loss ≥ 50% of max_loss
          2. Profit-target: unrealized profit ≥ 75% of original credit × 100
          3. Regime-shift:  current regime contradicts the strategy
        """
        for spread in spreads:
            signal, reason = self._check_exit(spread, current_regimes)
            spread.exit_signal = signal
            spread.exit_reason = reason

            if signal != ExitSignal.HOLD:
                logger.warning(
                    "[%s] EXIT SIGNAL: %s — %s | P&L=$%.2f",
                    spread.underlying, signal.value, reason,
                    spread.net_unrealized_pl)
            else:
                logger.info(
                    "[%s] HOLD — P&L=$%.2f (credit=$%.2f, max_loss=$%.2f)",
                    spread.underlying, spread.net_unrealized_pl,
                    spread.original_credit, spread.max_loss)

        return spreads

    def _check_exit(self, spread: SpreadPosition,
                    current_regimes: Dict[str, Regime]):
        """Return (ExitSignal, reason) for a single spread."""

        # --- 1. Stop-loss: unrealized loss ≥ 50% of max_loss ---
        # unrealized_pl is negative when losing
        loss_threshold = spread.max_loss * self.stop_loss_pct
        if spread.net_unrealized_pl < 0 and abs(spread.net_unrealized_pl) >= loss_threshold:
            return (
                ExitSignal.STOP_LOSS,
                f"Unrealized loss ${abs(spread.net_unrealized_pl):.2f} "
                f"≥ {self.stop_loss_pct*100:.0f}% of max loss ${spread.max_loss:.2f} "
                f"(threshold=${loss_threshold:.2f})"
            )

        # --- 2. Profit-target: captured ≥ 75% of credit ---
        # When a credit spread is winning, unrealized P&L is positive
        # (the options we sold are decaying, net position value shrinks)
        credit_value = spread.original_credit * 100  # per-contract
        profit_threshold = credit_value * self.profit_target_pct
        if spread.net_unrealized_pl > 0 and spread.net_unrealized_pl >= profit_threshold:
            return (
                ExitSignal.PROFIT_TARGET,
                f"Unrealized profit ${spread.net_unrealized_pl:.2f} "
                f"≥ {self.profit_target_pct*100:.0f}% of credit "
                f"${credit_value:.2f} (threshold=${profit_threshold:.2f})"
            )

        # --- 3. Regime-shift: strategy no longer matches regime ---
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

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, spreads: List[SpreadPosition]) -> Dict:
        """Generate a summary dict for logging."""
        total_pl = sum(s.net_unrealized_pl for s in spreads)
        signals = {}
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
                }
                for s in spreads
            ],
        }
