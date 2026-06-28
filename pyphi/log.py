# log.py
"""Utilities for logging and progress bars."""

from __future__ import annotations

import logging
from pathlib import Path

from tqdm import tqdm

_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s %(processName)s: %(message)s"


class TqdmHandler(logging.StreamHandler):
    """Logging handler that writes through ``tqdm`` in order to not break
    progress bars.
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream, end=self.terminator)
            self.flush()
        except Exception:  # pylint: disable=broad-except
            self.handleError(record)


def enable_logging(level: str = "INFO", file: str | Path | None = None) -> None:
    """Route PyPhi's logs to the console or a file.

    With no arguments, ``INFO``-and-above messages from the ``pyphi`` logger
    are written to stderr through a progress-bar-safe handler. Pass ``file`` to
    write to that path instead. Calling this again replaces the handler a
    previous call installed (handlers do not stack).

    Args:
        level: A standard level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
            ``"ERROR"``, ``"CRITICAL"``). An unknown name raises ``ValueError``.
        file: A path to log to. If ``None``, logs go to stderr.
    """
    logger = logging.getLogger("pyphi")
    for installed in list(logger.handlers):
        if not isinstance(installed, logging.NullHandler):
            logger.removeHandler(installed)
            installed.close()
    handler: logging.Handler = (
        TqdmHandler() if file is None else logging.FileHandler(str(file))
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
