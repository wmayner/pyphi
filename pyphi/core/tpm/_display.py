"""Rich display for TPM types: a state-by-node probability grid plus a labeled
xarray export.

Shared by :class:`pyphi.core.tpm.factored.FactoredTPM` and
:class:`pyphi.core.tpm.joint_distribution.JointTPM` so both render the same
state-by-node card.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from math import prod
from typing import Any

from pyphi.conf import config
from pyphi.display import Description
from pyphi.display import Row
from pyphi.display import Section
from pyphi.display import Table
from pyphi.utils import all_states


def require_xarray() -> Any:
    """Import and return the ``xarray`` module, or raise a helpful error."""
    try:
        import xarray as xr
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise ImportError(
            "to_xarray() requires the optional 'xarray' dependency; "
            "install with `pip install pyphi[xarray]`."
        ) from e
    return xr


def _state_label(state: tuple[int, ...]) -> str:
    return "(" + ",".join(map(str, state)) + ")"


def state_by_node_description(
    *,
    title: str,
    compact: str,
    unit_labels: Sequence[str],
    state_axis_sizes: Sequence[int],
    prob_on_for_state: Callable[[tuple[int, ...]], Sequence[float]],
) -> Description:
    """Build a card showing the state-by-node conditional as a matrix grid.

    Rows are current states (little-endian order), columns are units (labeled by
    ``unit_labels``), and each cell is ``P(unit "on" | current state)``. The
    grid is capped at ``config.infrastructure.repr_max_table_rows`` rows (large
    state spaces scroll and show an overflow indicator).
    """
    n_units = len(unit_labels)
    total = prod(state_axis_sizes) if state_axis_sizes else 1
    cap = config.infrastructure.repr_max_table_rows
    headers = ("state", *unit_labels)
    rows: list[tuple[Any, ...]] = []
    for i, state in enumerate(all_states(tuple(state_axis_sizes))):
        if i >= cap:
            break
        rows.append((_state_label(state), *(float(p) for p in prob_on_for_state(state))))
    overflow = max(0, total - len(rows))
    grid = Table(headers=headers, rows=tuple(rows), grid=True, overflow=overflow)
    return Description(
        title=title,
        subtitle=f"{n_units} units · {total} states",
        sections=(
            Section(rows=(Row("Units", n_units), Row("States", total))),
            Section(label="P(next unit on | current state)", body=(grid,)),
        ),
        compact=compact,
    )


def distribution_grid_description(
    *,
    title: str,
    compact: str,
    unit_labels: Sequence[str],
    alphabet_sizes: Sequence[int],
    dist_for_state: Callable[[tuple[int, ...]], Sequence[Sequence[float]]],
) -> Description:
    """Build a card showing the full per-unit next-state distribution as a grid.

    Rows are current states (little-endian order); columns are ``(unit,
    next-state)`` pairs labeled ``"{unit_label}={state}"``; each cell is
    ``P(unit = next-state | current state)``. Used for non-binary TPMs, where a
    single "on" probability per unit does not apply. Capped at
    ``config.infrastructure.repr_max_table_rows`` rows.
    """
    n_units = len(alphabet_sizes)
    total = prod(alphabet_sizes) if alphabet_sizes else 1
    cap = config.infrastructure.repr_max_table_rows
    headers = (
        "state",
        *(
            f"{unit_labels[unit]}={state}"
            for unit in range(n_units)
            for state in range(alphabet_sizes[unit])
        ),
    )
    rows: list[tuple[Any, ...]] = []
    for i, state in enumerate(all_states(tuple(alphabet_sizes))):
        if i >= cap:
            break
        flat = [float(p) for dist in dist_for_state(state) for p in dist]
        rows.append((_state_label(state), *flat))
    overflow = max(0, total - len(rows))
    grid = Table(headers=headers, rows=tuple(rows), grid=True, overflow=overflow)
    return Description(
        title=title,
        subtitle=f"{n_units} units · {total} states",
        sections=(
            Section(
                rows=(
                    Row("Units", n_units),
                    Row("Alphabet sizes", tuple(alphabet_sizes)),
                )
            ),
            Section(label="P(next unit = state | current state)", body=(grid,)),
        ),
        compact=compact,
    )
