"""Provenance record attached to top-level result objects.

A :class:`Provenance` captures how, when, and by what code a result was
computed: the pyphi version and source revision, a timestamp and wall-clock
duration, the RNG seed when one was used, and the Python / numpy / scipy
versions and platform. It is a sibling to :class:`pyphi.conf.ConfigSnapshot`
(which records the configuration), so a saved result is self-describing.
"""

from __future__ import annotations

import functools
import importlib.metadata
import platform as _platform
import subprocess
from dataclasses import dataclass
from dataclasses import replace
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy

_PACKAGE_ROOT = Path(__file__).resolve().parent


@functools.cache
def _git_info() -> tuple[str | None, bool | None]:
    """Return ``(commit_sha, is_dirty)`` for the package's working tree.

    Returns ``(None, None)`` when git is unavailable or the package is not
    inside a working tree (e.g. an installed wheel). Cached: the subprocess
    runs at most once per process.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PACKAGE_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_PACKAGE_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None, None
    return sha, bool(status.strip())


@dataclass(frozen=True)
class Provenance:
    """Immutable record of how, when, and by what code a result was computed."""

    pyphi_version: str
    git_sha: str | None
    git_dirty: bool | None
    timestamp: str
    python_version: str
    numpy_version: str
    scipy_version: str
    platform: str
    wall_time: float | None = None
    seed: int | None = None
    note: str | None = None

    @classmethod
    def capture(
        cls, *, wall_time: float | None = None, seed: int | None = None
    ) -> Provenance:
        """Capture the current environment into a :class:`Provenance`.

        ``wall_time`` (seconds) is supplied by the compute entry point; ``seed``
        is supplied only by code paths that consumed an RNG. Both default to
        ``None`` for deterministic, directly-constructed results.
        """
        sha, dirty = _git_info()
        return cls(
            pyphi_version=importlib.metadata.version("pyphi"),
            git_sha=sha,
            git_dirty=dirty,
            timestamp=datetime.now(UTC).isoformat(),
            python_version=_platform.python_version(),
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
            platform=f"{_platform.system()}/{_platform.machine()}",
            wall_time=wall_time,
            seed=seed,
        )

    def with_wall_time(self, wall_time: float) -> Provenance:
        """Return a copy with ``wall_time`` set (the record is frozen)."""
        return replace(self, wall_time=wall_time)

    def display_rows(self) -> list[tuple[str, str]]:
        """Return ``(label, value)`` pairs for the display layer."""
        git = "n/a"
        if self.git_sha is not None:
            git = self.git_sha[:12] + (" (dirty)" if self.git_dirty else "")
        rows = [
            ("pyphi", self.pyphi_version),
            ("git", git),
            ("Computed", self.timestamp),
            (
                "Wall time",
                "n/a" if self.wall_time is None else f"{self.wall_time:.3g} s",
            ),
            ("Python", self.python_version),
            ("numpy", self.numpy_version),
            ("scipy", self.scipy_version),
            ("Platform", self.platform),
        ]
        if self.seed is not None:
            rows.append(("Seed", str(self.seed)))
        if self.note is not None:
            rows.append(("Note", self.note))
        return rows

    def to_json(self) -> dict[str, Any]:
        return {
            "pyphi_version": self.pyphi_version,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "timestamp": self.timestamp,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "scipy_version": self.scipy_version,
            "platform": self.platform,
            "wall_time": self.wall_time,
            "seed": self.seed,
            "note": self.note,
        }

    @classmethod
    def from_json(cls, dct: dict[str, Any]) -> Provenance:
        return cls(**dct)


def _set_provenance(result: Any, prov: Provenance) -> None:
    """Assign ``prov`` to ``result.provenance``, working around frozen results."""
    try:
        result.provenance = prov
    except (AttributeError, TypeError):
        object.__setattr__(result, "provenance", prov)


def stamp_wall_time(result: Any, elapsed: float) -> Any:
    """Set ``elapsed`` seconds on ``result.provenance`` if it has one.

    Returns ``result``. A no-op when the result carries no provenance, so it
    is safe to call on any value returned from a compute entry point. The
    provenance record is frozen, so a copy with ``wall_time`` set replaces it.
    """
    prov = getattr(result, "provenance", None)
    if prov is None:
        return result
    _set_provenance(result, prov.with_wall_time(elapsed))
    return result


class HasProvenance:
    """Mixin for result types that carry a :class:`Provenance` record.

    Provides :meth:`with_provenance` so a user can record their own context
    (a free-form ``note``, the ``seed`` they controlled) on a computed result.
    """

    provenance: Provenance | None

    def with_provenance(self, **fields: Any) -> HasProvenance:
        """Update this result's provenance record in place and return ``self``.

        ``fields`` are merged into the existing record, e.g.
        ``result.with_provenance(note="run 1", seed=42)``. Unknown field names
        raise :class:`TypeError`. Provenance is metadata, not part of the
        result's value, so the update never affects equality, diffs, or stored
        goldens; updating in place (rather than copying the whole result) keeps
        that explicit.
        """
        prov = self.provenance or Provenance.capture()
        _set_provenance(self, replace(prov, **fields))
        return self


__all__ = ["HasProvenance", "Provenance", "stamp_wall_time"]
