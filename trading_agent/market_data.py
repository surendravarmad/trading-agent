# Backward-compatibility shim — real implementation has moved to the subpackage.
# Delete this file after all callers have been updated to use the new import path.
from trading_agent.market.market_data import *  # noqa: F401,F403
from trading_agent.market.market_data import MarketDataProvider, InsufficientDataError, PRICE_HISTORY_TTL, SNAPSHOT_TTL, OPTION_CHAIN_TTL, INTRADAY_RETURN_TTL
