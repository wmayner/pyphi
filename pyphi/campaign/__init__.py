"""Distribute PyPhi computations across an HTCondor pool as batch campaigns.

A campaign is a self-contained directory of serialized task files that
independent condor jobs execute via ``python -m pyphi.campaign run``; results
are collected from per-task output files. :func:`prepare` writes the
directory, the user submits the generated submit file with ``condor_submit``,
and :func:`status` / :func:`collect` operate purely on the directory's
files — a task is done exactly when its output file exists and loads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "CampaignTask",
    "CampaignTaskOutput",
    "CellOutput",
]


@dataclass(frozen=True)
class CampaignTask:
    """One condor job's worth of work: the cells it owns and their context.

    ``config_overrides`` is the preparing session's configuration in override
    form (JSON-compatible); the runner installs it beneath each cell's
    formalism preset. ``compute_ref`` is a ``"module:qualname"`` reference
    used when the computation is a callable rather than ``"sia"``/``"ces"``.
    """

    task_id: int
    kind: str
    compute: str | None
    compute_ref: str | None
    config_overrides: dict[str, Any]
    cells: tuple[tuple[Any, str, tuple[int, ...], tuple[int, ...]], ...]
    skip_uncomputable: bool


@dataclass(frozen=True)
class CellOutput:
    """One cell's outcome: ``ok`` (with the result), ``skipped``, or ``error``."""

    status: str
    result: Any | None
    traceback: str | None


@dataclass(frozen=True)
class CampaignTaskOutput:
    """A task's per-cell outcomes, aligned 1:1 with the task's cells."""

    task_id: int
    pyphi_version: str
    entries: tuple[CellOutput, ...]
