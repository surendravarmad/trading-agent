# Backward-compatibility shim — real implementation has moved to the subpackage.
# Delete this file after all callers have been updated to use the new import path.
from trading_agent.core.stage_plan import *  # noqa: F401,F403
from trading_agent.core.stage_plan import _StagePlanMixin
