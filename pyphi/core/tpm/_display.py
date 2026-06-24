"""Matrix-grid builders for TPM and substrate display, plus the xarray export.

The grid :class:`~pyphi.display.Table` builders here are shared: a TPM renders
its own state-by-node card from them, and a :class:`~pyphi.substrate.Substrate`
embeds the same grids (connectivity + TPM) in its card.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

from pyphi.conf import config
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


def state_by_node_grid(
    *,
    unit_labels: Sequence[str],
    state_axis_sizes: Sequence[int],
    prob_on_for_state: Callable[[tuple[int, ...]], Sequence[float]],
) -> Table:
    """A state-by-node matrix grid: rows = current state (little-endian), columns
    = units (labeled by ``unit_labels``), cell = ``P(unit "on" | state)``.

    Capped at ``config.infrastructure.repr_max_table_rows`` rows.
    """
    cap = config.infrastructure.repr_max_table_rows
    total = 1
    for size in state_axis_sizes:
        total *= size
    headers = ("state", *unit_labels)
    rows: list[tuple[Any, ...]] = []
    for i, state in enumerate(all_states(tuple(state_axis_sizes))):
        if i >= cap:
            break
        rows.append((_state_label(state), *(float(p) for p in prob_on_for_state(state))))
    overflow = max(0, total - len(rows))
    return Table(headers=headers, rows=tuple(rows), grid=True, overflow=overflow)


def distribution_grid(
    *,
    unit_labels: Sequence[str],
    alphabet_sizes: Sequence[int],
    dist_for_state: Callable[[tuple[int, ...]], Sequence[Sequence[float]]],
) -> Table:
    """A full per-unit next-state distribution grid: rows = current state, columns
    = ``(unit, next-state)`` pairs labeled ``"{unit_label}={state}"``, cell =
    ``P(unit = next-state | state)``.

    Used for non-binary TPMs, where a single "on" probability per unit does not
    apply. Capped at ``config.infrastructure.repr_max_table_rows`` rows.
    """
    cap = config.infrastructure.repr_max_table_rows
    n_units = len(alphabet_sizes)
    total = 1
    for size in alphabet_sizes:
        total *= size
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
    return Table(headers=headers, rows=tuple(rows), grid=True, overflow=overflow)


def connectivity_grid(unit_labels: Sequence[str], cm: Any) -> Table:
    """An adjacency-matrix grid: rows = from-unit, columns = to-unit, cell =
    ``1`` for a connection and ``·`` for none.
    """
    labels = [str(label) for label in unit_labels]
    headers = ("", *labels)
    rows = tuple(
        (labels[i], *("1" if cm[i][j] else "·" for j in range(len(labels))))
        for i in range(len(labels))
    )
    return Table(headers=headers, rows=rows, grid=True)
