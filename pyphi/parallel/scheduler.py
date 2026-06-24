"""Scheduler Protocol and policy types for the parallelization layer.

The Protocol abstracts process / thread / dask backends behind a single
``map_reduce`` entry point. Policies bundle the parameters that today live as
loose kwargs on ``MapReduce.__init__`` so backends share a stable surface.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Protocol
from typing import TypeVar
from typing import runtime_checkable

R = TypeVar("R")
T = TypeVar("T")


def _never_short_circuit(_result: Any) -> bool:
    return False


@dataclass(frozen=True)
class ChunkingPolicy:
    """Controls how items are batched for a worker.

    ``chunksize=None`` selects cost-sampling at the scheduler. Provide a value
    to bypass sampling.
    """

    chunksize: int | None = None
    sequential_threshold: int = 1
    size_func: Callable[[Any], float] | None = None
    target_seconds: float = 1.0


@dataclass(frozen=True)
class ProgressPolicy:
    enabled: bool = False
    desc: str = ""
    total: int | None = None


@dataclass(frozen=True)
class ShortcircuitPolicy:
    func: Callable[[Any], bool] = field(default=_never_short_circuit)
    callback: Callable[[Iterable[Any]], None] | None = None


@runtime_checkable
class Scheduler(Protocol):
    """Backend-agnostic map-reduce dispatcher."""

    def map_reduce(
        self,
        fn: Callable[..., R],
        items: Iterable[Any],
        *more_items: Iterable[Any],
        reducer: Callable[[Iterable[R]], T] = list,  # type: ignore[assignment]
        config_snapshot: Any | None = None,
        chunking: ChunkingPolicy | None = None,
        progress: ProgressPolicy | None = None,
        shortcircuit: ShortcircuitPolicy | None = None,
        ordered: bool = False,
        map_kwargs: dict[str, Any] | None = None,
    ) -> T: ...

    @property
    def supports_shared_state(self) -> bool: ...


def default_scheduler(backend: str | None = None) -> Scheduler:
    """Return the scheduler for ``backend`` (or ``config.parallel_backend``).

    ``"auto"`` resolves to ``LocalThreadScheduler`` on free-threaded runtimes
    and ``LocalProcessScheduler`` otherwise.
    """
    import sys

    from pyphi.conf import config

    if backend is None:
        backend = config.infrastructure.parallel_backend
    if backend == "auto":
        gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
        if not gil_enabled:
            from pyphi.parallel.backends.local_thread import LocalThreadScheduler

            return LocalThreadScheduler()
        from pyphi.parallel.backends.local_process import LocalProcessScheduler

        return LocalProcessScheduler()
    if backend in ("local", "process"):
        from pyphi.parallel.backends.local_process import LocalProcessScheduler

        return LocalProcessScheduler()
    if backend == "thread":
        from pyphi.parallel.backends.local_thread import LocalThreadScheduler

        return LocalThreadScheduler()
    if backend == "dask":
        from pyphi.parallel.backends.dask import DaskScheduler

        return DaskScheduler()
    raise ValueError(f"unknown parallel_backend: {backend!r}")
