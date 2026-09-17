"""Loguru sinks: coloured console output plus a timestamped file under logs/<command>/."""

import sys
import time
from pathlib import Path

from loguru import logger


def configure(command, label, root="logs"):
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}",
        level="INFO",
    )
    path = Path(root) / command / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{label}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(path, level="INFO")
    return path
