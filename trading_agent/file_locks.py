# Backward-compatibility shim — real implementation has moved to the subpackage.
# Delete this file after all callers have been updated to use the new import path.
from trading_agent.utils.file_locks import *  # noqa: F401,F403
from trading_agent.utils.file_locks import locked_append_json, atomic_write_json
