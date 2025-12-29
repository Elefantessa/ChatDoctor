"""
Logging utilities for ChatDoctor.
Provides configured loggers with optional rich console support.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from rich.logging import RichHandler
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_rich: bool = True,
    name: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging for ChatDoctor.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path to write logs.
        use_rich: Use rich console handler.
        name: Logger name. If None, configures root logger.

    Returns:
        Configured logger.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Remove existing handlers
    logger.handlers.clear()

    # Format string
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # Console handler
    if use_rich and RICH_AVAILABLE:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
        )
        console_handler.setLevel(log_level)
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))

    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
