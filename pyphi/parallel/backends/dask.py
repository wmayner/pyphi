"""Distributed-cluster scheduler stub.

Conforms to the :class:`~pyphi.parallel.scheduler.Scheduler` Protocol so that
``config.parallel_backend = "dask"`` resolves to a concrete object, but the
map-reduce implementation is not yet provided. The module has no dependency on
``dask.distributed``; importing it is free, and calling
:meth:`DaskScheduler.map_reduce` raises :exc:`NotImplementedError`.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from typing import Any


class DaskScheduler:
    """Scheduler Protocol stub for distributed-cluster execution.

    Reports ``supports_shared_state = False``; :meth:`map_reduce` raises
    :exc:`NotImplementedError`.
    """

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
