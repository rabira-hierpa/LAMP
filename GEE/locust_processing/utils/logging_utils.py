"""
Logging utilities for the locust processing package.
"""

import logging
import sys
from typing import Optional


def setup_logging(log_file: str = 'locust_export.log') -> logging.Logger:
    """
    Configure logging to both console and file.

    Args:
        log_file: Path to the log file

    Returns:
        Configured logger instance
    """
    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers if any
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger


def get_logger() -> logging.Logger:
    """
    Get the configured logger. If not already configured, returns the root logger.

    Returns:
        Logger instance
    """
    return logging.getLogger()
