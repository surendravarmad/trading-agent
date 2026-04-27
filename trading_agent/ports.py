# Backward-compatibility shim — real implementation has moved to the subpackage.
# Delete this file after all callers have been updated to use the new import path.
from trading_agent.core.ports import *  # noqa: F401,F403
from trading_agent.core.ports import MarketDataPort, ExecutionPort, PositionsPort, OrdersPort, AccountPort
