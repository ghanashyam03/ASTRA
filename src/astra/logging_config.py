"""Logging configuration for ASTRA."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Setup stdout logging and silence heavy dependencies."""
    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)
    # Silence noisy libraries
    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
