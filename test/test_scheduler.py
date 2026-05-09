"""Tests for the Scheduler Protocol, policies, and resolver."""

from __future__ import annotations

from pyphi.parallel.scheduler import ChunkingPolicy
from pyphi.parallel.scheduler import ProgressPolicy
from pyphi.parallel.scheduler import Scheduler
from pyphi.parallel.scheduler import ShortcircuitPolicy
from pyphi.parallel.scheduler import default_scheduler


def test_scheduler_protocol_is_runtime_checkable():
    class _Stub:
        def map_reduce(self, fn, items, *_more_items, **_kwargs):
            return list(map(fn, items))

        @property
        def supports_shared_state(self) -> bool:
            return False

    assert isinstance(_Stub(), Scheduler)


def test_chunking_policy_defaults_are_none():
    p = ChunkingPolicy()
    assert p.chunksize is None
    assert p.sequential_threshold == 1
    assert p.size_func is None


def test_progress_policy_default_is_off():
    p = ProgressPolicy()
    assert p.enabled is False
    assert p.desc == ""
    assert p.total is None


def test_shortcircuit_policy_default_never_short_circuits():
    p = ShortcircuitPolicy()
    assert p.func(0) is False
    assert p.func("anything") is False
    assert p.callback is None


def test_default_scheduler_is_callable():
    """default_scheduler() is the resolver entry point."""
    # Just verify it exists and is callable; we don't instantiate the
    # concrete schedulers until later phases.
    assert callable(default_scheduler)


def test_parallel_backend_accepts_new_values():
    """The legacy 'local' alias plus 'process'/'thread'/'dask'/'auto'."""
    from pyphi.conf import config

    for value in ("local", "process", "thread", "dask", "auto"):
        with config.override(parallel_backend=value):
            assert config.infrastructure.parallel_backend == value


def _square(x):
    """Top-level function for cloudpickle serialization."""
    return x * x


def _read_precision(_x):
    """Worker-side function reading config to verify snapshot delivery."""
    from pyphi.conf import config

    return config.numerics.precision


def test_local_process_scheduler_implements_protocol():
    from pyphi.parallel.backends.local_process import LocalProcessScheduler

    s = LocalProcessScheduler()
    assert isinstance(s, Scheduler)
    assert s.supports_shared_state is False


def test_local_process_scheduler_basic_map_reduce():
    from pyphi.parallel.backends.local_process import LocalProcessScheduler

    s = LocalProcessScheduler()
    result = s.map_reduce(_square, [1, 2, 3, 4, 5], reducer=sum)
    assert result == 1 + 4 + 9 + 16 + 25


def test_local_process_scheduler_propagates_config_override():
    """Workers see config changes captured at map_reduce dispatch time."""
    from pyphi.conf import config
    from pyphi.parallel.backends.local_process import LocalProcessScheduler

    s = LocalProcessScheduler()

    with config.override(precision=11):
        results = s.map_reduce(_read_precision, [1, 2, 3], reducer=list)

    assert all(r == 11 for r in results)


def test_local_thread_scheduler_implements_protocol():
    from pyphi.parallel.backends.local_thread import LocalThreadScheduler

    s = LocalThreadScheduler()
    assert isinstance(s, Scheduler)
    assert s.supports_shared_state is True


def test_local_thread_scheduler_basic_map_reduce():
    from pyphi.parallel.backends.local_thread import LocalThreadScheduler

    s = LocalThreadScheduler()
    result = s.map_reduce(lambda x: x + 1, [10, 20, 30], reducer=sum)
    assert result == 11 + 21 + 31


def test_local_thread_scheduler_does_not_apply_snapshot():
    """Threads share parent's globals; apply must be a no-op (no overwrite)."""
    from pyphi.conf import config
    from pyphi.parallel.backends.local_thread import LocalThreadScheduler

    s = LocalThreadScheduler()
    with config.override(precision=11):
        parent_view = config.numerics.precision

        def read_precision(_):
            return config.numerics.precision

        worker_views = s.map_reduce(read_precision, [1, 2, 3], reducer=list)

    assert parent_view == 11
    assert worker_views == [11, 11, 11]


def test_dask_scheduler_skeleton_lazy_import():
    """Importing the dask backend must not load dask.distributed."""
    import sys

    sys.modules.pop("dask.distributed", None)

    from pyphi.parallel.backends import dask as _dask_module  # noqa: F401

    assert "dask.distributed" not in sys.modules


def test_dask_scheduler_raises_not_implemented():
    import pytest

    from pyphi.parallel.backends.dask import DaskScheduler

    s = DaskScheduler()
    assert isinstance(s, Scheduler)
    assert s.supports_shared_state is False
    with pytest.raises(NotImplementedError, match=r"DaskScheduler is a stub"):
        s.map_reduce(lambda x: x, [1, 2, 3])
