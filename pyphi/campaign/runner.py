"""Execute one campaign task file and write its output document.

The runner is a fixed entry point (``python -m pyphi.campaign run``) that
behaves identically inside the campaign's container, in a local shell, and
under test: it loads the task, loads the substrates it references, installs
the shipped configuration beneath each cell's formalism preset, runs the
cells in order, and atomically writes one output document holding a per-cell
outcome. The process exit code is nonzero when any cell errored, so
scheduler logs reflect failures, but the output document is written in every
case.
"""

from __future__ import annotations

import importlib.metadata
import traceback as _traceback
from pathlib import Path
from typing import Any

from pyphi import serialize
from pyphi.campaign import CampaignTask
from pyphi.campaign import CampaignTaskOutput
from pyphi.campaign import CellOutput
from pyphi.campaign import _resolve_compute_ref
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.sweep import _run_cell
from pyphi.sweep import _Skipped

__all__ = ["run_task"]


def _load_substrates(task: CampaignTask, substrates_dir: Path) -> dict:
    labels = {cell[0] for cell in task.cells}
    return {
        label: serialize.load(substrates_dir / f"substrate-{label}.json.gz")
        for label in labels
    }


def _write_output(output: CampaignTaskOutput, outputs_dir: Path) -> None:
    final = outputs_dir / f"task-{output.task_id:04d}.json.gz"
    if final.exists():
        n = 1
        while (
            attempt := final.with_name(f"task-{output.task_id:04d}.attempt-{n}.json.gz")
        ).exists():
            n += 1
        final.rename(attempt)
    # The temporary name keeps the .json.gz suffixes so format inference is
    # unchanged; Path.replace makes the final path appear atomically.
    tmp = final.with_name(f".tmp-{final.name}")
    serialize.save(output, tmp)
    tmp.replace(final)


def run_task(
    task_path: Any,
    substrates_dir: Any = "substrates",
    outputs_dir: Any = ".",
) -> int:
    """Run one task file; return 0 if every cell is ok or skipped, else 1.

    Parameters
    ----------
    task_path
        Path to a serialized campaign task.
    substrates_dir
        Directory holding the campaign's serialized substrates.
    outputs_dir
        Directory to write ``task-<id>.json.gz`` into (atomically; a
        pre-existing output is preserved under an ``attempt-<n>`` name).
    """
    task = serialize.load(task_path)
    substrates = _load_substrates(task, Path(substrates_dir))
    compute = (
        task.compute
        if task.compute is not None
        else _resolve_compute_ref(task.compute_ref)
    )
    entries: list[CellOutput] = []
    failed = False
    for label, formalism, subset, state in task.cells:
        overrides = {
            **task.config_overrides,
            **presets.by_name[formalism],
            "parallel": False,
            "progress_bars": False,
        }
        try:
            with config.override(**overrides):
                result = _run_cell(
                    (label, subset, state),
                    substrates=substrates,
                    compute=compute,
                    skip=task.skip_uncomputable,
                )
            if isinstance(result, _Skipped):
                entries.append(CellOutput(status="skipped", result=None, traceback=None))
            else:
                entries.append(CellOutput(status="ok", result=result, traceback=None))
        except Exception:
            entries.append(
                CellOutput(
                    status="error", result=None, traceback=_traceback.format_exc()
                )
            )
            failed = True
    output = CampaignTaskOutput(
        task_id=task.task_id,
        pyphi_version=importlib.metadata.version("pyphi"),
        entries=tuple(entries),
    )
    _write_output(output, Path(outputs_dir))
    return 1 if failed else 0
