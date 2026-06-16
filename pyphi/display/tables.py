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
) -> Table:
    """Build a :class:`Table` from the first ``cap`` items, recording overflow.

    ``total`` is the full collection size; ``cap`` defaults to
    ``config.infrastructure.repr_max_table_rows``. Only ``cap`` items are
    materialized, so a huge collection (e.g. millions of relations) is not
    fully realized just to display a handful of rows.
    """
    if cap is None:
        cap = config.infrastructure.repr_max_table_rows
    rows = tuple(row(item) for item in islice(items, cap))
    overflow = max(0, total - len(rows))
    return Table(headers=headers, rows=rows, overflow=overflow)
