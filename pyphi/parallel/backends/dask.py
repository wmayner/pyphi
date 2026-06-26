"""Skeleton DaskScheduler.

Stub implementation that documents the Protocol shape against
``dask.distributed`` without depending on the import. Cluster deployment
fills this in as a separate follow-up project; the Protocol is the contract
that unblocks it.

The import of ``dask.distributed`` is deferred until ``map_reduce`` is
actually called (which raises NotImplementedError). Until then, importing
this module is free.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from typing import Any


class DaskScheduler:
    """Stub scheduler placeholder for cluster deployments."""

    @property
    def supports_shared_state(self) -> bool:
        return False

    def map_reduce(
        self,
        fn: Callable[..., Any],
        items: Iterable[Any],
        *more_items: Iterable[Any],
        reducer: Callable[[Iterable[Any]], Any] = list,
        config_snapshot: Any | None = None,
        chunking: Any = None,
        progress: Any = None,
        shortcircuit: Any = None,
        ordered: bool = False,
        map_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del (
            fn,
            items,
            more_items,
            reducer,
            config_snapshot,
            chunking,
            progress,
            shortcircuit,
            ordered,
            map_kwargs,
        )
        raise NotImplementedError(
            "DaskScheduler is a stub; fill in for cluster deployments. "
            "Full Dask/cluster support is a planned follow-up."
        )
