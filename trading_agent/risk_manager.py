# Backward-compatibility shim — real implementation has moved to the subpackage.
# Delete this file after all callers have been updated to use the new import path.
from trading_agent.strategy.risk_manager import *  # noqa: F401,F403
from trading_agent.strategy.risk_manager import RiskManager, RiskVerdict
