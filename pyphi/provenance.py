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
import json
import platform as _platform
import re
import subprocess
from collections.abc import Mapping
from dataclasses import asdict
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
        ownership = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "__init__.py"],
            cwd=_PACKAGE_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if ownership.returncode != 0:
            # The discovered repository does not track the package (e.g. an
            # installed wheel inside another project's git tree); its commit
            # says nothing about the pyphi code version.
            return None, None
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
    estimator: dict[str, Any] | None = None
    """Structured record of how an estimated input was produced (data regime,
    estimator model, prior, sample counts). ``None`` for results computed from
    an exactly specified substrate."""

    @classmethod
    def capture(
        cls,
        *,
        wall_time: float | None = None,
        seed: int | None = None,
        estimator: dict[str, Any] | None = None,
    ) -> Provenance:
        """Capture the current environment into a :class:`Provenance`.

        ``wall_time`` (seconds) is supplied by the compute entry point; ``seed``
        is supplied only by code paths that consumed an RNG; ``estimator`` is
        supplied only by estimation entry points. All default to ``None`` for
        deterministic, directly-constructed results.
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
            estimator=estimator,
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
        if self.estimator is not None:
            rows.append(
                (
                    "Estimator",
                    ", ".join(
                        f"{key}={self.estimator.get(key)}"
                        for key in ("regime", "model", "n_transitions")
                    ),
                )
            )
        return rows


def _stem_value(value: Any) -> str:
    """Format a parameter value for use in a filename stem."""
    return re.sub(r"[^A-Za-z0-9_+-]", "-", str(value).replace(".", "p"))


def format_stem(
    name: str,
    params: Mapping[str, Any] | None = None,
    run_label: str | None = None,
) -> str:
    """Build a filename stem encoding a script's parameters.

    Joins ``name``, one ``{key}{value}`` segment per ``params`` entry (in
    insertion order), and ``run_label`` when given, with underscores.
    Values and the run label are formatted with ``str()``; ``.`` becomes
    ``p`` (so ``0.7`` → ``0p7`` and the filename keeps a single suffix)
    and any character outside ``[A-Za-z0-9_+-]`` becomes ``-``. ``name``
    is used verbatim.

    Examples
    --------
    >>> format_stem("study", {"seed": 42, "noise": 0.7}, "pilot")
    'study_seed42_noise0p7_pilot'
    """
    parts = [name]
    for key, value in (params or {}).items():
        parts.append(f"{key}{_stem_value(value)}")
    if run_label:
        parts.append(_stem_value(run_label))
    return "_".join(parts)


def unique_path(directory: Path | str, stem: str, suffix: str) -> Path:
    """Return a non-clobbering path: ``stem+suffix``, else ``stem_v2+suffix``, ...

    Creates ``directory`` (with parents) if it does not exist. Never
    returns a path that already exists, so earlier outputs are never
    overwritten.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}{suffix}"
    version = 2
    while path.exists():
        path = directory / f"{stem}_v{version}{suffix}"
        version += 1
    return path


def _json_default(obj: Any) -> Any:
    """``json.dumps`` fallback for numpy values and paths."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _capture_metadata(
    params: Mapping[str, Any] | None,
    seed: int | None,
    note: str | None,
) -> Provenance:
    """Capture a :class:`Provenance`, resolving the seed from ``params``."""
    if seed is None and params is not None and "seed" in params:
        seed = int(params["seed"])
    prov = Provenance.capture(seed=seed)
    if note is not None:
        prov = replace(prov, note=note)
    return prov


def save_json(
    data: Any,
    directory: Path | str,
    name: str,
    *,
    params: Mapping[str, Any] | None = None,
    run_label: str | None = None,
    seed: int | None = None,
    note: str | None = None,
) -> Path:
    """Write ``data`` to a self-describing, non-clobbering JSON file.

    The file holds the envelope ``{"provenance": ..., "params": ...,
    "data": ...}``. The filename encodes ``params`` and ``run_label``
    (see :func:`format_stem`); an existing file is never overwritten (a
    ``_v2``/``_v3`` suffix is added instead). The provenance record
    stores the seed from ``seed`` or, when omitted, from
    ``params["seed"]``. numpy scalars and arrays in ``data`` are
    converted to JSON-native values.

    Returns the written path.
    """
    prov = _capture_metadata(params, seed, note)
    path = unique_path(directory, format_stem(name, params, run_label), ".json")
    envelope = {
        "provenance": asdict(prov),
        "params": dict(params or {}),
        "data": data,
    }
    path.write_text(json.dumps(envelope, indent=2, default=_json_default))
    return path


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


__all__ = [
    "HasProvenance",
    "Provenance",
    "format_stem",
    "save_json",
    "stamp_wall_time",
    "unique_path",
]
