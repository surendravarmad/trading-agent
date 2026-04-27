# Backward-compatibility shim — real implementation has moved to the subpackage.
# Delete this file after all callers have been updated to use the new import path.
from trading_agent.intelligence.fine_tuning import *  # noqa: F401,F403
