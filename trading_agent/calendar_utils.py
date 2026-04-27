# Backward-compatibility shim — real implementation has moved to trading_agent/market/calendar_utils.py.
from trading_agent.market.calendar_utils import *  # noqa: F401,F403
from trading_agent.market.calendar_utils import next_weekly_expiration, is_last_trading_day_before, is_trading_day
