"""Tests for the order executor — dry-run and plan persistence."""

import json
import os
import pytest

from trading_agent.executor import OrderExecutor
from trading_agent.risk_manager import RiskVerdict
from trading_agent.strategy import SpreadPlan, SpreadLeg


def _make_verdict(approved=True, plan=None) -> RiskVerdict:
    if plan is None:
        plan = SpreadPlan(
            ticker="SPY", strategy_name="Bull Put Spread", regime="bullish",
            legs=[
                SpreadLeg("SPY250425P00480000", 480.0, "sell", "put",
                          -0.15, -0.05, 1.20, 1.40, 1.30),
                SpreadLeg("SPY250425P00475000", 475.0, "buy", "put",
                          -0.10, -0.03, 0.80, 1.00, 0.90),
            ],
            spread_width=5.0, net_credit=0.40, max_loss=460.0,
            credit_to_width_ratio=0.08, expiration="2025-04-25",
            reasoning="Test", valid=approved,
        )
    return RiskVerdict(
        approved=approved, plan=plan, account_balance=100_000,
        max_allowed_loss=2_000,
        checks_passed=["check1"] if approved else [],
        checks_failed=[] if approved else ["failed_check"],
        summary="APPROVED" if approved else "REJECTED",
    )


class TestDryRun:
    def test_dry_run_writes_plan_file(self, tmp_path):
        executor = OrderExecutor(
            api_key="test", secret_key="test",
            trade_plan_dir=str(tmp_path), dry_run=True)
        verdict = _make_verdict(approved=True)
        result = executor.execute(verdict)

        assert result["status"] == "dry_run"
        assert os.path.exists(result["plan_file"])

        with open(result["plan_file"]) as f:
            data = json.load(f)
        assert data["trade_plan"]["ticker"] == "SPY"
        assert data["mode"] == "dry_run"

    def test_rejected_trade_not_executed(self, tmp_path):
        executor = OrderExecutor(
            api_key="test", secret_key="test",
            trade_plan_dir=str(tmp_path), dry_run=True)
        verdict = _make_verdict(approved=False)
        result = executor.execute(verdict)
        assert result["status"] == "rejected"


class TestPlanPersistence:
    def test_plan_file_contains_risk_verdict(self, tmp_path):
        executor = OrderExecutor(
            api_key="test", secret_key="test",
            trade_plan_dir=str(tmp_path), dry_run=True)
        verdict = _make_verdict(approved=True)
        result = executor.execute(verdict)

        with open(result["plan_file"]) as f:
            data = json.load(f)

        assert "risk_verdict" in data
        assert data["risk_verdict"]["approved"] is True
        assert data["risk_verdict"]["account_balance"] == 100_000
