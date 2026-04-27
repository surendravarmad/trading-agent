# Backward-compatibility shim — real implementation has moved to trading_agent/utils/shutdown.py.
# Delete this file after all callers have been updated to use the new import path.
from trading_agent.utils.shutdown import *  # noqa: F401,F403
from trading_agent.utils.shutdown import (
    shutdown_requested, reset_shutdown_flag, install_signal_handlers,
    graceful_exit, hard_exit,
)
