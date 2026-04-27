"""Strategy package — re-exports for backward compatibility.

The old flat import `from trading_agent.strategy import StrategyPlanner`
continues to work because this __init__.py re-exports from strategy.strategy.
"""
from trading_agent.strategy.strategy import (
    StrategyPlanner, SpreadPlan, SpreadLeg,
)
from trading_agent.strategy.regime import (
    Regime, RegimeAnalysis, RegimeClassifier,
    LEADERSHIP_ANCHORS, VIX_INHIBIT_ZSCORE,
)
from trading_agent.strategy.risk_manager import RiskManager, RiskVerdict

__all__ = [
    "StrategyPlanner", "SpreadPlan", "SpreadLeg",
    "Regime", "RegimeAnalysis", "RegimeClassifier",
    "LEADERSHIP_ANCHORS", "VIX_INHIBIT_ZSCORE",
    "RiskManager", "RiskVerdict",
]
