"""Field-change callbacks for the layered config.

A warning is emitted if ``distinction_phi_normalization`` is changed after
initial load (since it would invalidate cached MICE on existing systems).

The ``_loaded`` flag suppresses warnings during default-state setup; it
flips to ``True`` after any ``pyphi_config.yml`` auto-load completes (or
immediately after construction if no user config is present).
"""

from __future__ import annotations

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


def warn_distinction_phi_normalization_change(old: Any, new: Any) -> None:
    """Warn that cached MICE on existing systems will not reflect the new setting."""
    if not _loaded.value:
        return
    warn(
        f"""
IMPORTANT: `distinction_phi_normalization` changed: {old!r} -> {new!r}.

The change will not be reflected in new MICE computations for existing
System objects if the MICE have been previously computed, since they are
cached. (A scoped `config.override` warns once when it applies the change
and once more when it restores the previous value on exit; the second
warning concerns Systems computed inside the block.)

Make sure to call `system.clear_caches()` before re-computing MICE with
the new setting.
        """,
        category=PyPhiWarning,
        stacklevel=4,
    )
