"""
Phase IV — ACT
Submits validated credit-spread orders to Alpaca's paper trading API.
Also handles dry-run logging and trade-plan persistence.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import requests

from trading_agent.strategy import SpreadPlan
from trading_agent.risk_manager import RiskVerdict

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Fires multi-leg option orders to Alpaca Paper API, or writes
    them to a JSON plan file in dry-run mode.
    """

    def __init__(self, api_key: str, secret_key: str,
                 base_url: str = "https://paper-api.alpaca.markets/v2",
                 trade_plan_dir: str = "trade_plans",
                 dry_run: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.trade_plan_dir = trade_plan_dir
        self.dry_run = dry_run
        os.makedirs(self.trade_plan_dir, exist_ok=True)

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, verdict: RiskVerdict) -> Dict:
        """
        Execute a trade plan that has passed risk checks.
        Returns a result dict with order details or dry-run log.
        """
        plan = verdict.plan

        # Save the plan regardless of mode
        plan_path = self._save_plan(plan, verdict)

        if not verdict.approved:
            logger.warning("[%s] Trade REJECTED by risk manager — skipping.",
                           plan.ticker)
            return {
                "status": "rejected",
                "reason": verdict.summary,
                "plan_file": plan_path,
            }

        if self.dry_run:
            logger.info("[%s] DRY RUN — trade plan written to %s",
                        plan.ticker, plan_path)
            return {
                "status": "dry_run",
                "plan_file": plan_path,
                "plan": plan.to_dict(),
            }

        # Live paper execution
        return self._submit_order(plan, plan_path)

    # ------------------------------------------------------------------
    # Alpaca order submission
    # ------------------------------------------------------------------

    def _submit_order(self, plan: SpreadPlan, plan_path: str) -> Dict:
        """
        Submit a multi-leg option order to Alpaca.
        Uses POST /v2/orders with order_class='mleg'.

        Alpaca mleg payload format (from API docs):
        - Top-level: type, time_in_force, order_class, qty, limit_price
        - Legs array: symbol, ratio_qty (string), side, position_intent
        - limit_price: string, NEGATIVE for credit, POSITIVE for debit
        - No top-level 'side' field for mleg orders
        """
        legs_payload = []
        for leg in plan.legs:
            # Map our action to Alpaca's position_intent
            if leg.action == "sell":
                position_intent = "sell_to_open"
                side = "sell"
            else:
                position_intent = "buy_to_open"
                side = "buy"

            legs_payload.append({
                "symbol": leg.symbol,
                "ratio_qty": "1",           # string, not int
                "side": side,
                "position_intent": position_intent,
            })

        # Alpaca sign convention:
        #   Credit received → negative limit_price
        #   Debit paid      → positive limit_price
        # Our plan.net_credit is always positive for credits, so negate it.
        limit_price_value = -abs(plan.net_credit)

        order_payload = {
            "type": "limit",
            "time_in_force": "day",
            "order_class": "mleg",
            "qty": "1",                             # top-level quantity (string)
            "limit_price": str(limit_price_value),   # string, negative = credit
            "legs": legs_payload,
        }

        logger.info("[%s] Submitting %s order to Alpaca: %s",
                     plan.ticker, plan.strategy_name,
                     json.dumps(order_payload, indent=2))

        try:
            resp = requests.post(
                f"{self.base_url}/orders",
                headers=self._headers(),
                json=order_payload,
                timeout=15,
            )

            # Capture response body BEFORE raise_for_status so we always
            # have the Alpaca error detail regardless of status code.
            resp_body = None
            try:
                resp_body = resp.json()
            except Exception:
                resp_body = resp.text

            resp.raise_for_status()

            order_id = resp_body.get("id", "unknown") if isinstance(resp_body, dict) else "unknown"
            logger.info("[%s] Order SUBMITTED — ID: %s", plan.ticker, order_id)

            result = {
                "status": "submitted",
                "order_id": order_id,
                "plan_file": plan_path,
                "alpaca_response": resp_body,
            }
            self._append_to_plan(plan_path, {"order_result": result})
            return result

        except requests.RequestException as exc:
            # Build a detailed error message including Alpaca's response
            error_msg = str(exc)
            if resp_body is not None:
                logger.error("[%s] Alpaca response body: %s", plan.ticker, resp_body)
                error_msg += f" | Alpaca detail: {resp_body}"
            elif hasattr(exc, 'response') and exc.response is not None:
                try:
                    detail = exc.response.json()
                    logger.error("[%s] Alpaca response body: %s", plan.ticker, detail)
                    error_msg += f" | Alpaca detail: {detail}"
                except Exception:
                    raw = exc.response.text
                    logger.error("[%s] Alpaca raw response: %s", plan.ticker, raw)
                    error_msg += f" | Alpaca raw: {raw}"

            logger.error("[%s] Order FAILED: %s", plan.ticker, error_msg)
            result = {
                "status": "error",
                "error": error_msg,
                "plan_file": plan_path,
            }
            self._append_to_plan(plan_path, {"order_error": result})
            return result

    # ------------------------------------------------------------------
    # Close positions
    # ------------------------------------------------------------------

    def close_spread(self, spread) -> Dict:
        """
        Close an open credit spread by submitting a closing mleg order.
        Each leg's position_intent is reversed:
          sell_to_open → buy_to_close
          buy_to_open  → sell_to_close

        Uses DELETE /v2/positions/{symbol} for each leg individually,
        which is the Alpaca-recommended approach for closing option positions.
        """
        results = []
        for leg in spread.legs:
            result = self._close_single_leg(leg.symbol)
            results.append(result)

        all_ok = all(r.get("status") == "closed" for r in results)
        summary = {
            "action": "close_spread",
            "underlying": spread.underlying,
            "strategy": spread.strategy_name,
            "signal": spread.exit_signal.value,
            "reason": spread.exit_reason,
            "leg_results": results,
            "all_closed": all_ok,
        }

        if all_ok:
            logger.info("[%s] Spread CLOSED successfully (%s)",
                        spread.underlying, spread.exit_signal.value)
        else:
            logger.warning("[%s] Spread close PARTIAL — some legs failed",
                           spread.underlying)

        return summary

    def _close_single_leg(self, symbol: str) -> Dict:
        """
        DELETE /v2/positions/{symbol} — close a single option position.
        Alpaca handles the direction (buy_to_close / sell_to_close)
        automatically based on whether the position is long or short.
        """
        url = f"{self.base_url}/positions/{symbol}"
        try:
            resp = requests.delete(url, headers=self._headers(), timeout=15)

            resp_body = None
            try:
                resp_body = resp.json()
            except Exception:
                resp_body = resp.text

            resp.raise_for_status()
            logger.info("Closed position: %s", symbol)
            return {"status": "closed", "symbol": symbol, "response": resp_body}

        except requests.RequestException as exc:
            error_msg = str(exc)
            if resp_body is not None:
                logger.error("Close %s response: %s", symbol, resp_body)
                error_msg += f" | Detail: {resp_body}"
            logger.error("Failed to close position %s: %s", symbol, error_msg)
            return {"status": "error", "symbol": symbol, "error": error_msg}

    # ------------------------------------------------------------------
    # Plan file management
    # ------------------------------------------------------------------

    def _save_plan(self, plan: SpreadPlan, verdict: RiskVerdict) -> str:
        """Persist the full trade plan + risk verdict to JSON."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"trade_plan_{plan.ticker}_{timestamp}.json"
        filepath = os.path.join(self.trade_plan_dir, filename)

        payload = {
            "trade_plan": plan.to_dict(),
            "risk_verdict": {
                "approved": verdict.approved,
                "account_balance": verdict.account_balance,
                "max_allowed_loss": verdict.max_allowed_loss,
                "checks_passed": verdict.checks_passed,
                "checks_failed": verdict.checks_failed,
                "summary": verdict.summary,
            },
            "mode": "dry_run" if self.dry_run else "live",
        }

        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)

        logger.info("Trade plan saved to %s", filepath)
        return filepath

    def _append_to_plan(self, filepath: str, data: Dict):
        """Append additional data (order result) to an existing plan file."""
        try:
            with open(filepath, "r") as f:
                existing = json.load(f)
            existing.update(data)
            with open(filepath, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as exc:
            logger.error("Failed to update plan file %s: %s", filepath, exc)
