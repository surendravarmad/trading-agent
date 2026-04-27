"""
Phase IV — ACT
Submits validated credit-spread orders to Alpaca's paper trading API.
Also handles dry-run logging and trade-plan persistence.

Single-file persistence model
------------------------------
Each ticker gets ONE persistent file:  trade_plans/{TICKER}.json

The file contains a "state_history" array so every run is preserved
without cluttering the directory.  Only the last MAX_HISTORY entries
are kept.  Old timestamped files (trade_plan_{TICKER}_{TS}.json) are
left untouched for backward compatibility with the position monitor.

File structure::

    {
      "ticker":        "AAPL",
      "created":       "2026-04-01T15:00:00+00:00",
      "last_updated":  "2026-04-01T15:58:00+00:00",
      "state_history": [
        {
          "run_id":       "20260401_155800",
          "timestamp":    "2026-04-01T15:58:00+00:00",
          "trade_plan":   { ... },
          "risk_verdict": { ... },
          "mode":         "dry_run",
          "order_result": { ... }   // appended after submission
        },
        ...
      ]
    }
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import requests

from trading_agent.strategy.strategy import SpreadPlan
from trading_agent.strategy.risk_manager import RiskVerdict
from trading_agent.config.loader import ExecutionRules

if TYPE_CHECKING:
    from trading_agent.market.market_data import MarketDataProvider

logger = logging.getLogger(__name__)

# Module-level default seeded from trading_rules.yaml; importable by name.
_exec_rules = ExecutionRules()
MAX_HISTORY = _exec_rules.max_history


class OrderExecutor:
    """
    Fires multi-leg option orders to Alpaca Paper API, or writes
    them to a per-ticker JSON plan file in dry-run mode.
    """

    # Class-level fallback for backward compatibility
    PRICE_DRIFT_WARN_PCT = 0.10   # 10 %

    def __init__(self, api_key: str, secret_key: str,
                 base_url: str = "https://paper-api.alpaca.markets/v2",
                 trade_plan_dir: str = "trade_plans",
                 dry_run: bool = True,
                 data_provider: Optional["MarketDataProvider"] = None,
                 max_risk_pct: float = 0.02,
                 min_credit_ratio: float = 0.33,
                 rules: "ExecutionRules | None" = None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.trade_plan_dir = trade_plan_dir
        self.dry_run = dry_run
        self.data_provider = data_provider   # used for live quote refresh
        # Same ceilings the RiskManager enforces as guardrails #2 and #4.
        # Sizing AND the live-credit re-check share these so planning-time
        # validation and execution-time validation never drift apart.
        self.max_risk_pct = max_risk_pct
        self.min_credit_ratio = min_credit_ratio
        r = rules or ExecutionRules()
        self.PRICE_DRIFT_WARN_PCT = r.price_drift_warn_pct
        self._max_history = r.max_history
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

        # Save the plan (always) — returns (filepath, run_id)
        plan_path, run_id = self._save_plan(plan, verdict)

        if not verdict.approved:
            logger.warning("[%s] Trade REJECTED by risk manager — skipping.",
                           plan.ticker)
            return {
                "status": "rejected",
                "reason": verdict.summary,
                "plan_file": plan_path,
                "run_id": run_id,
            }

        if self.dry_run:
            logger.info("[%s] DRY RUN — trade plan written to %s",
                        plan.ticker, plan_path)
            return {
                "status": "dry_run",
                "plan_file": plan_path,
                "run_id": run_id,
                "plan": plan.to_dict(),
            }

        # Live paper execution
        return self._submit_order(plan, plan_path, run_id, verdict.account_balance)

    # ------------------------------------------------------------------
    # Alpaca order submission
    # ------------------------------------------------------------------

    def _calculate_qty(self, plan: SpreadPlan, account_balance: float,
                       live_credit: Optional[float] = None) -> int:
        """
        Size contracts so the position's total max loss stays within the
        same budget the RiskManager validated against (``max_risk_pct ×
        equity`` — guardrail #4).

        ::

            credit                 = live_credit if provided else plan.net_credit
            max_loss_per_contract  = (spread_width − credit) × 100
            qty                    = floor(max_risk_pct × equity
                                           / max_loss_per_contract)

        Passing ``live_credit`` ensures sizing reflects the refreshed
        bid/ask at submission time rather than the stale planning-time
        credit — see ``_submit_order``.

        Returns **0** when no integer quantity fits inside the budget (e.g.
        a single contract's max loss alone exceeds the ceiling, or inputs
        are non-positive).  The caller MUST treat 0 as "abort submission"
        — never silently floor to 1, which would otherwise bypass the
        guardrail.
        """
        credit = live_credit if live_credit is not None else plan.net_credit
        max_loss_per_contract = (plan.spread_width - credit) * 100
        if max_loss_per_contract <= 0 or account_balance <= 0:
            return 0
        max_risk_dollars = account_balance * self.max_risk_pct
        return int(max_risk_dollars // max_loss_per_contract)

    def _recheck_live_economics(self, plan: SpreadPlan,
                                live_credit: float,
                                account_balance: float) -> Tuple[bool, str]:
        """
        Re-validate the two economics-bearing guardrails against live bid/ask.

        Between planning (Phase III) and execution (Phase VI) the net credit
        can drift materially with the book.  Only the market-independent
        checks (market-open, account-type, buying-power, underlying
        liquidity) stay valid; ``credit_to_width_ratio`` and ``max_loss``
        must be recomputed from ``live_credit`` and re-validated against
        the same thresholds the RiskManager uses:

          * guardrail #2  — credit / width  ≥  ``min_credit_ratio``
          * guardrail #4  — max_loss (per contract) ≤ ``max_risk_pct × equity``

        Returns
        -------
        (ok, reason)
            ``ok`` is True when both checks pass; ``reason`` is empty on
            success, else a short human-readable "live_credit_risk: ..."
            string describing the first failure.
        """
        width = plan.spread_width
        if width <= 0:
            return (False,
                    f"live_credit_risk: spread_width {width} is non-positive")

        live_ratio = live_credit / width
        if live_ratio < self.min_credit_ratio:
            return (False,
                    f"live_credit_risk: credit/width {live_ratio:.4f} < "
                    f"{self.min_credit_ratio} "
                    f"(live_credit=${live_credit:.2f}, width=${width:.2f}, "
                    f"planning ratio was {plan.credit_to_width_ratio:.4f})")

        live_max_loss = (width - live_credit) * 100
        max_allowed = account_balance * self.max_risk_pct
        if live_max_loss > max_allowed:
            return (False,
                    f"live_credit_risk: max_loss ${live_max_loss:.2f} > "
                    f"{self.max_risk_pct*100:.0f}% × ${account_balance:,.2f} "
                    f"(=${max_allowed:.2f}) "
                    f"(live_credit=${live_credit:.2f}, "
                    f"planning max_loss was ${plan.max_loss:.2f})")

        return (True, "")

    def _submit_order(self, plan: SpreadPlan, plan_path: str,
                      run_id: str, account_balance: float = 0.0) -> Dict:
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
            if leg.action == "sell":
                position_intent = "sell_to_open"
                side = "sell"
            else:
                position_intent = "buy_to_open"
                side = "buy"

            legs_payload.append({
                "symbol": leg.symbol,
                "ratio_qty": "1",
                "side": side,
                "position_intent": position_intent,
            })

        # Refresh bid/ask from live market right before sending the order.
        # The option chain was fetched during Phase III (planning); by now
        # seconds-to-minutes may have passed.  Use fresh quotes so the
        # limit_price reflects what the market is actually offering.
        live_credit = self._refresh_limit_price(plan)
        if live_credit is None:
            # Quote fetch failed — fall back to the planned credit and warn
            logger.warning(
                "[%s] Could not refresh live quotes — using planned "
                "credit $%.2f as limit_price (may not fill)",
                plan.ticker, plan.net_credit)
            live_credit = plan.net_credit
        else:
            drift = abs(live_credit - plan.net_credit)
            drift_pct = drift / plan.net_credit if plan.net_credit else 0
            if drift_pct > self.PRICE_DRIFT_WARN_PCT:
                logger.warning(
                    "[%s] Credit drifted %.1f%% since planning "
                    "(plan=$%.2f → live=$%.2f)",
                    plan.ticker, drift_pct * 100,
                    plan.net_credit, live_credit)
            else:
                logger.info("[%s] Live credit $%.2f (plan was $%.2f)",
                            plan.ticker, live_credit, plan.net_credit)

        # Re-validate the economics-bearing guardrails against LIVE credit.
        # The RiskManager approved this plan at planning time using
        # plan.net_credit; if the bid/ask has drifted materially the
        # credit-to-width or max-loss checks may no longer hold.  The
        # other guardrails (market-open, paper, buying-power, underlying
        # liquidity) are environment-dependent and haven't changed.
        ok, recheck_reason = self._recheck_live_economics(
            plan, live_credit, account_balance)
        if not ok:
            logger.error("[%s] Live-credit risk recheck FAILED — %s. "
                         "Aborting order submission.",
                         plan.ticker, recheck_reason)
            result = {
                "status": "rejected",
                "reason": recheck_reason,
                "plan_file": plan_path,
                "run_id": run_id,
            }
            self._append_to_plan(plan_path, run_id, {"order_result": result})
            return result

        # Alpaca sign convention: credit → negative limit_price
        limit_price_value = -abs(live_credit)

        # Size off the LIVE credit so qty reflects the economics we're
        # actually submitting, not the stale planning-time credit.
        qty = self._calculate_qty(plan, account_balance, live_credit=live_credit)
        if qty < 1:
            max_loss_per_contract = (plan.spread_width - live_credit) * 100
            max_risk_dollars = account_balance * self.max_risk_pct
            reason = (
                f"qty=0: max_loss_per_contract ${max_loss_per_contract:.2f} "
                f"> sizing budget ${max_risk_dollars:.2f} "
                f"({self.max_risk_pct*100:.0f}% × ${account_balance:,.2f}) "
                f"(live_credit=${live_credit:.2f})"
            )
            logger.error(
                "[%s] Position sizing produced qty=0 — %s. Aborting order "
                "submission rather than silently flooring to 1 contract.",
                plan.ticker, reason)
            result = {
                "status": "rejected",
                "reason": reason,
                "plan_file": plan_path,
                "run_id": run_id,
            }
            self._append_to_plan(plan_path, run_id, {"order_result": result})
            return result

        logger.info(
            "[%s] Position size: %d contract(s) "
            "(max_risk_pct=%.0f%%, equity=$%.2f)",
            plan.ticker, qty, self.max_risk_pct * 100, account_balance)

        order_payload = {
            "type": "limit",
            "time_in_force": "day",
            "order_class": "mleg",
            "qty": str(qty),
            "limit_price": str(limit_price_value),
            "legs": legs_payload,
        }

        logger.info("[%s] Submitting %s order to Alpaca: %s",
                     plan.ticker, plan.strategy_name,
                     json.dumps(order_payload, indent=2))

        resp_body = None
        try:
            resp = requests.post(
                f"{self.base_url}/orders",
                headers=self._headers(),
                json=order_payload,
                timeout=15,
            )

            try:
                resp_body = resp.json()
            except Exception:
                resp_body = resp.text

            resp.raise_for_status()

            order_id = (resp_body.get("id", "unknown")
                        if isinstance(resp_body, dict) else "unknown")
            logger.info("[%s] Order SUBMITTED — ID: %s", plan.ticker, order_id)

            result = {
                "status": "submitted",
                "order_id": order_id,
                "plan_file": plan_path,
                "run_id": run_id,
                "alpaca_response": resp_body,
            }
            self._append_to_plan(plan_path, run_id, {"order_result": result})
            return result

        except requests.RequestException as exc:
            error_msg = str(exc)
            if resp_body is not None:
                logger.error("[%s] Alpaca response body: %s", plan.ticker, resp_body)
                error_msg += f" | Alpaca detail: {resp_body}"
            elif hasattr(exc, "response") and exc.response is not None:
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
                "run_id": run_id,
            }
            self._append_to_plan(plan_path, run_id, {"order_error": result})
            return result

    def _refresh_limit_price(self, plan: SpreadPlan) -> Optional[float]:
        """
        Fetch live bid/ask for the plan's leg symbols and recalculate
        net credit using the same bid-for-sold / ask-for-bought convention
        used during planning.

        Returns the fresh net credit, or None if the fetch fails.
        """
        if self.data_provider is None:
            return None

        symbols = [leg.symbol for leg in plan.legs]
        quotes = self.data_provider.fetch_option_quotes(symbols)
        if not quotes:
            return None

        # Identify sold vs bought legs and recalculate credit
        total_credit = 0.0
        for leg in plan.legs:
            q = quotes.get(leg.symbol)
            if q is None:
                logger.warning("[%s] No live quote for %s — aborting refresh",
                               plan.ticker, leg.symbol)
                return None
            if leg.action == "sell":
                total_credit += q["bid"]   # receive the bid
            else:
                total_credit -= q["ask"]   # pay the ask

        return round(total_credit, 2)

    # ------------------------------------------------------------------
    # Close positions
    # ------------------------------------------------------------------

    def close_spread(self, spread) -> Dict:
        """
        Close an open credit spread.
        Uses DELETE /v2/positions/{symbol} for each leg individually.
        """
        results = []
        for leg in spread.legs:
            results.append(self._close_single_leg(leg.symbol))

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
        """DELETE /v2/positions/{symbol} — close a single option leg."""
        url = f"{self.base_url}/positions/{symbol}"
        resp_body = None
        try:
            resp = requests.delete(url, headers=self._headers(), timeout=15)
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
    # Plan file management  — single persistent file per ticker
    # ------------------------------------------------------------------

    def _save_plan(self, plan: SpreadPlan,
                   verdict: RiskVerdict) -> tuple[str, str]:
        """
        Persist the trade plan + risk verdict to a single per-ticker file.

        Returns
        -------
        (filepath, run_id)
        """
        now = datetime.now(timezone.utc)
        run_id = now.strftime("%Y%m%d_%H%M%S")
        ts = now.isoformat()

        filepath = os.path.join(self.trade_plan_dir,
                                f"trade_plan_{plan.ticker}.json")

        # Load existing file or initialise fresh
        try:
            with open(filepath) as fh:
                persistent = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            persistent = {
                "ticker": plan.ticker,
                "created": ts,
                "state_history": [],
            }

        entry = {
            "run_id": run_id,
            "timestamp": ts,
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

        persistent["last_updated"] = ts
        persistent["state_history"].append(entry)

        # Trim to keep only the most recent _max_history runs
        if len(persistent["state_history"]) > self._max_history:
            persistent["state_history"] = (
                persistent["state_history"][-self._max_history:]
            )

        with open(filepath, "w") as fh:
            json.dump(persistent, fh, indent=2)

        logger.info("Trade plan saved to %s (run_id=%s, history=%d)",
                    filepath, run_id, len(persistent["state_history"]))

        # Auto-generate companion HTML report
        try:
            from trading_agent.trade_plan_report import generate_report
            html_path = generate_report(filepath)
            logger.debug("HTML report updated: %s", html_path)
        except Exception as exc:
            logger.debug("HTML report generation skipped: %s", exc)

        return filepath, run_id

    def _append_to_plan(self, filepath: str, run_id: str, data: Dict) -> None:
        """
        Merge *data* into the state_history entry matching *run_id*.
        Falls back to updating the last entry if run_id is not found.
        """
        try:
            with open(filepath) as fh:
                persistent = json.load(fh)

            history = persistent.get("state_history", [])
            target = next(
                (e for e in reversed(history) if e.get("run_id") == run_id),
                history[-1] if history else None,
            )
            if target is not None:
                target.update(data)

            persistent["last_updated"] = datetime.now(timezone.utc).isoformat()

            with open(filepath, "w") as fh:
                json.dump(persistent, fh, indent=2)

        except Exception as exc:
            logger.error("Failed to update plan file %s: %s", filepath, exc)
