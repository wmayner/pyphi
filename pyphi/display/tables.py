"""Helpers for building capped collection tables for display."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from itertools import islice
from typing import Any

from pyphi.conf import config
from pyphi.display.description import Table


def capped_table(
    headers: tuple[str, ...],
    items: Iterable[Any],
    row: Callable[[Any], tuple[Any, ...]],
    total: int,
    cap: int | None = None,
    header_tones: tuple[str | None, ...] = (),
    cell_tones: Callable[[Any], tuple[str | None, ...]] | None = None,
) -> Table:
    """Build a :class:`Table` from the first ``cap`` items, recording overflow.

    ``total`` is the full collection size; ``cap`` defaults to
    ``config.infrastructure.repr_max_table_rows``. Only ``cap`` items are
    materialized, so a huge collection (e.g. millions of relations) is not
    fully realized just to display a handful of rows. ``header_tones`` optionally
    colors individual column headers; ``cell_tones`` optionally returns a
    per-column tone tuple for each item, coloring body cells (HTML only).
    """
    if cap is None:
        cap = config.infrastructure.repr_max_table_rows
    materialized = list(islice(items, cap))
    rows = tuple(row(item) for item in materialized)
    overflow = max(0, total - len(rows))
    row_tones = tuple(cell_tones(item) for item in materialized) if cell_tones else ()
    return Table(
        headers=headers,
        rows=rows,
        overflow=overflow,
        header_tones=header_tones,
        row_tones=row_tones,
    )
