"""
Logging configuration — sets up file + console handlers with
a consistent format across all modules.
"""

import logging
import os
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """
    Configure the root logger with both console and rotating file output.
    Returns the root logger.
    """
    os.makedirs(log_dir, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)
    timestamp = datetime.utcnow().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"trading_agent_{timestamp}.log")

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers on re-init
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)

    root.info("Logging initialised — level=%s, file=%s", log_level, log_file)
    return root
