"""Tests for the Scheduler Protocol, policies, and resolver."""

from __future__ import annotations

import pytest

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
    from pyphi.parallel.backends.dask import DaskScheduler

    s = DaskScheduler()
    assert isinstance(s, Scheduler)
    assert s.supports_shared_state is False
    with pytest.raises(NotImplementedError, match=r"DaskScheduler is a stub"):
        s.map_reduce(lambda x: x, [1, 2, 3])


# ============================================================================
# Snapshot-apply dedup mechanism
# ============================================================================
#
# These tests pin the worker-side dedup hook in
# :mod:`pyphi.parallel.backends.local_process` rather than the public outcome
# already covered by ``test_local_process_scheduler_propagates_config_override``.
# A regression where a worker re-applies an unchanged snapshot on every task
# would silently produce correct results but hammer the global config —
# directly testing the dedup keeps that regression visible.


def _patch_install_snapshot(config_obj):
    """Wrap config.install_snapshot to count calls without mutating instance.

    ``_GlobalConfig.__setattr__`` rejects unknown attribute names, so
    ``mock.patch.object(config, ...)`` fails on cleanup. Patching the class
    method instead is what works.
    """

    from typing import ClassVar

    class _Counter:
        calls: ClassVar[list] = []

    original = type(config_obj).install_snapshot

    def counting(self, snapshot):
        _Counter.calls.append(snapshot)
        return original(self, snapshot)

    type(config_obj).install_snapshot = counting
    return _Counter, original


def _restore_install_snapshot(config_obj, original):
    type(config_obj).install_snapshot = original


def test_apply_snapshot_dedup_skips_repeated_identical_snapshot():
    from pyphi.conf import config
    from pyphi.parallel.backends import local_process

    # Reset module state to simulate a fresh worker process.
    local_process._LAST_APPLIED_SNAPSHOT_HASH = None
    local_process._PARENT_PID = None

    snap = config.snapshot()
    counter, original = _patch_install_snapshot(config)
    try:
        local_process._apply_snapshot_if_changed(snap)
        local_process._apply_snapshot_if_changed(snap)
        local_process._apply_snapshot_if_changed(snap)
    finally:
        _restore_install_snapshot(config, original)

    assert len(counter.calls) == 1


def test_apply_snapshot_dedup_reapplies_when_snapshot_changes():
    from pyphi.conf import config
    from pyphi.parallel.backends import local_process

    local_process._LAST_APPLIED_SNAPSHOT_HASH = None
    local_process._PARENT_PID = None

    snap1 = config.snapshot()
    with config.override(precision=11):
        snap2 = config.snapshot()

    counter, original = _patch_install_snapshot(config)
    try:
        local_process._apply_snapshot_if_changed(snap1)
        local_process._apply_snapshot_if_changed(snap2)  # different
        local_process._apply_snapshot_if_changed(snap1)  # back to snap1
    finally:
        _restore_install_snapshot(config, original)

    assert len(counter.calls) == 3


def test_apply_snapshot_skips_when_running_in_parent_pid():
    """Threads share parent globals; the apply hook short-circuits there."""
    import os

    from pyphi.conf import config
    from pyphi.parallel.backends import local_process

    local_process._LAST_APPLIED_SNAPSHOT_HASH = None
    # Mark the test process as the parent — exactly what LocalThreadScheduler
    # does before dispatching.
    local_process._PARENT_PID = os.getpid()

    snap = config.snapshot()
    counter, original = _patch_install_snapshot(config)
    try:
        local_process._apply_snapshot_if_changed(snap)
        local_process._apply_snapshot_if_changed(snap)
    finally:
        _restore_install_snapshot(config, original)
        local_process._PARENT_PID = None

    assert len(counter.calls) == 0


# ============================================================================
# map_reduce backend selection
# ============================================================================
#
# ``map_reduce`` resolves its ``backend=`` through :func:`default_scheduler`,
# so ``"thread"`` runs on the thread scheduler and ``"dask"`` reaches the
# Dask stub (which raises ``NotImplementedError``). These tests pin that
# the backend argument is actually honored.


def _double(x):
    """Top-level function for cloudpickle serialization."""
    return x * 2


@pytest.mark.parametrize("backend", ["auto", "local", "process", "thread"])
def test_map_reduce_executes_on_all_local_backends(backend):
    """map_reduce routes to the selected local backend and computes correctly."""
    from pyphi.parallel import map_reduce

    out = map_reduce(
        _double, [1, 2, 3, 4, 5], backend=backend, sequential_threshold=1, chunksize=2
    )
    assert sorted(out) == [2, 4, 6, 8, 10]


def test_map_reduce_rejects_unknown_backend():
    from pyphi.parallel import map_reduce

    with pytest.raises(ValueError, match=r"unknown parallel_backend"):
        map_reduce(_double, [1, 2, 3], backend="invalid")


def test_map_reduce_dask_backend_is_not_implemented():
    """backend='dask' now actually routes to the DaskScheduler stub."""
    from pyphi.parallel import map_reduce

    with pytest.raises(NotImplementedError):
        map_reduce(
            _double, [1, 2, 3, 4, 5], backend="dask", sequential_threshold=1, chunksize=2
        )


def _increment(x):
    return x + 1


def _identity(x):
    return x


def test_map_reduce_sequential_matches_builtin():
    from pyphi.parallel import map_reduce

    out = map_reduce(_double, [1, 2, 3], parallel=False)
    assert sorted(out) == [2, 4, 6]


def test_map_reduce_reduce_func_min_with_kwargs():
    from pyphi.parallel import map_reduce

    # empty input + min(default=...) must return the default, mirroring AC usage
    out = map_reduce(
        _identity, [], reduce_func=min, reduce_kwargs={"default": 99}, parallel=False
    )
    assert out == 99


def test_map_reduce_parallel_equals_sequential():
    import pyphi
    from pyphi.parallel import map_reduce

    items = list(range(50))
    with pyphi.config.override(parallel=True):
        par = map_reduce(_square, items, chunksize=8)
    seq = map_reduce(_square, items, parallel=False)
    assert sorted(par) == sorted(seq)


def test_map_reduce_backend_thread_routes_to_thread_scheduler():
    from pyphi.parallel import map_reduce

    out = map_reduce(
        _increment, [1, 2, 3], backend="thread", sequential_threshold=1, chunksize=1
    )
    assert sorted(out) == [2, 3, 4]


def test_map_reduce_invokes_shortcircuit_callback_when_sequential():
    """The shortcircuit callback fires even on the sequential fallback path."""
    from pyphi.parallel import map_reduce

    seen = []
    out = map_reduce(
        _identity,
        [1, 2, 3, 4, 5],
        parallel=False,
        shortcircuit_func=lambda r: r == 3,
        shortcircuit_callback=lambda *_: seen.append("stopped"),
    )
    assert 3 in out
    assert seen == ["stopped"]
