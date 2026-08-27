"""Run-scoped standard-library logging."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time


def configure_run_logger(experiment_id: str, log_path: str | Path) -> logging.Logger:
    """Create an INFO-level logger writing to both console and a run log."""

    destination = Path(log_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"windblade.run.{experiment_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)sZ | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(destination, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def close_run_logger(logger: logging.Logger) -> None:
    """Flush and detach every handler owned by a run logger."""

    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
