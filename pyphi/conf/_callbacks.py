"""Field-change callbacks for the layered config.

Logging is reconfigured whenever any of the three log fields change, and a
warning is emitted if ``distinction_phi_normalization`` is changed after
initial load (since it would invalidate cached MICE on existing systems).

The ``_loaded`` flag suppresses warnings during default-state setup; it
flips to ``True`` after any ``pyphi_config.yml`` auto-load completes (or
immediately after construction if no user config is present).
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Any
from warnings import warn

from pyphi.warnings import PyPhiWarning


class _LoadedFlag:
    """One-shot flag; flips ``True`` after import-time config bootstrap."""

    value: bool = False


_loaded = _LoadedFlag()


def mark_loaded() -> None:
    """Flip the loaded flag; called once after import-time config bootstrap."""
    _loaded.value = True


def is_loaded() -> bool:
    return _loaded.value


def configure_logging(
    log_file: str | Path,
    log_file_level: str | None,
    log_stdout_level: str | None,
) -> None:
    """Reconfigure PyPhi logging based on the current configuration."""
    handlers: dict[str, dict[str, Any]] = {
        "file": {
            "level": log_file_level,
            "filename": str(log_file),
            "class": "logging.FileHandler",
            "formatter": "standard",
        },
        "stdout": {
            "level": log_stdout_level,
            "class": "pyphi.log.TqdmHandler",
            "formatter": "standard",
        },
    }
    root_handlers = (["file"] if log_file_level else []) + (
        ["stdout"] if log_stdout_level else []
    )
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(name)s] %(levelname)s "
                    "%(processName)s: %(message)s"
                }
            },
            "handlers": handlers,
            "root": {
                "level": "DEBUG",
                "handlers": root_handlers,
            },
        }
    )


def warn_distinction_phi_normalization_change() -> None:
    """Warn that cached MICE on existing systems will not reflect the new setting."""
    if not _loaded.value:
        return
    warn(
        """
IMPORTANT: Changes to `distinction_phi_normalization` will not be reflected in
new MICE computations for existing System objects if the MICE have been
previously computed, since they are cached.

Make sure to call `system.clear_caches()` before re-computing MICE with
the new setting.
        """,
        category=PyPhiWarning,
        stacklevel=4,
    )
