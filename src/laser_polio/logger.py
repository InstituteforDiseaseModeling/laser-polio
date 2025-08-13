"""
Logging configuration and utilities for laser-polio.

This module provides colored logging functionality and custom log levels
for the laser-polio simulation framework.
"""

import logging
import os
from datetime import datetime
from typing import ClassVar

import numpy as np
import pytz

__all__ = ["VALID", "ColorFormatter", "LogColors", "fmt", "logger", "valid"]


class LogColors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BROWN = "\033[38;5;94m"  # Approximate brown using 256-color mode
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"


# Add a custom log level for validation messages
VALID = 15
logging.addLevelName(VALID, "VALID")


class ColorFormatter(logging.Formatter):
    """Custom formatter that adds colors to log level names and logger names."""

    LEVEL_COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: LogColors.BROWN,
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.MAGENTA,
        VALID: LogColors.BLUE,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)
        record.levelname = f"{color}{record.levelname}{LogColors.RESET}"
        record.name = f"{color}{record.name}{LogColors.RESET}"
        return super().format(record)


def valid(self, message, *args, **kwargs):
    """Add a 'valid' method to the Logger class for validation messages."""
    if self.isEnabledFor(VALID):
        self._log(VALID, message, args, **kwargs)


# Add the valid method to the Logger class
logging.Logger.valid = valid

# Set up the main logger
logger = logging.getLogger("laser-polio")
logger.propagate = False  # Prevents double/multiple logging

# Add console handler with color formatting
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColorFormatter("[%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(console_handler)

# Configure file logging with timestamped log files
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
local_tz = pytz.timezone("America/Los_Angeles")  # Replace with your local timezone
timestamp = datetime.now(local_tz).strftime("%Y%m%d-%H%M%S")
log_file = os.path.join(log_dir, f"simulation_log-{timestamp}.txt")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    filemode="w",  # Overwrite each time you run; use "a" to append
)

# Also create a module-specific logger


def fmt(arr, precision=2):
    """Format NumPy arrays as single-line strings with no wrapping."""
    return np.array2string(
        np.asarray(arr),  # Ensures even scalars/lists work
        separator=" ",
        threshold=np.inf,
        max_line_width=np.inf,
        precision=precision,
    )
