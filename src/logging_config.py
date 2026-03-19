"""
Logging Configuration
=====================
Configures both console and file-based logging for the framework.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """Configure framework-wide logging with console and file handlers.

    Parameters
    ----------
    level : str
        Logging level: DEBUG, INFO, WARNING, ERROR.
    log_dir : str
        Directory for log files.
    log_to_file : bool
        Enable file logging.
    log_to_console : bool
        Enable console logging.

    Returns
    -------
    logging.Logger – the root logger for the framework.
    """
    root_logger = logging.getLogger("pcb")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f"pcb_{timestamp}.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info("File logging enabled → %s", log_path / f"pcb_{timestamp}.log")

    return root_logger
