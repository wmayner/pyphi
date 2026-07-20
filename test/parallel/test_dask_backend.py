"""Tests for the Dask distributed-cluster scheduler."""

from __future__ import annotations

import pytest

from pyphi.conf import config
from pyphi.parallel.backends.dask import DaskScheduler
from pyphi.parallel.scheduler import ChunkingPolicy
from pyphi.parallel.scheduler import ProgressPolicy
from pyphi.parallel.scheduler import Scheduler
from pyphi.parallel.scheduler import ShortcircuitPolicy

distributed = pytest.importorskip("distributed")


def _plus_one(x):
    """Top-level function for serialization."""
    return x + 1


def _identity(x):
    """Top-level function for serialization."""
    return x


def _read_precision(_x):
    from pyphi.conf import config as worker_config

    return worker_config.numerics.precision


def test_implements_protocol():
    s = DaskScheduler()
    assert isinstance(s, Scheduler)
    assert s.supports_shared_state is False


def test_requires_active_client():
    # Must be defined before any test that uses the module-scoped
    # ``dask_client`` fixture: once that fixture instantiates, its client
    # stays current for the rest of the module.
    with pytest.raises(RuntimeError, match=r"distributed\.Client"):
        DaskScheduler().map_reduce(_plus_one, [1, 2, 3])


def test_basic_map_reduce(dask_client):
    result = DaskScheduler().map_reduce(
        _plus_one,
        [1, 2, 3, 4],
        reducer=sum,
        chunking=ChunkingPolicy(chunksize=1, sequential_threshold=1),
    )
    assert result == 2 + 3 + 4 + 5


def test_ordered_returns_submission_order(dask_client):
    result = DaskScheduler().map_reduce(
        _identity,
        [3, 1, 2],
        reducer=list,
        chunking=ChunkingPolicy(chunksize=1, sequential_threshold=1),
        ordered=True,
    )
    assert result == [3, 1, 2]


def test_snapshot_propagation(dask_client):
    with config.override(precision=11):
        result = DaskScheduler().map_reduce(
            _read_precision,
            [1, 2, 3],
            reducer=list,
            chunking=ChunkingPolicy(chunksize=1, sequential_threshold=1),
        )
    assert result == [11, 11, 11]


def test_shortcircuit_collects_deterministic_prefix(dask_client):
    fired = []
    result = DaskScheduler().map_reduce(
        _identity,
        [1, 2, 3, 4, 5],
        reducer=list,
        chunking=ChunkingPolicy(chunksize=1, sequential_threshold=1),
        shortcircuit=ShortcircuitPolicy(
            func=lambda r: r == 2, callback=lambda *_: fired.append(True)
        ),
    )
    assert result == [1, 2]
    assert fired == [True]


def test_empty_items(dask_client):
    assert DaskScheduler().map_reduce(_plus_one, [], reducer=list) == []


def test_progress(dask_client, monkeypatch):
    bars = []

    class RecordingBar:
        def __init__(self, total=None, desc=""):
            self.total = total
            self.desc = desc
            self.updates = 0
            self.closed = False
            bars.append(self)

        def update(self, n=1):
            self.updates += n

        def close(self):
            self.closed = True

    monkeypatch.setattr("pyphi.parallel.backends.dask.LocalProgressBar", RecordingBar)
    DaskScheduler().map_reduce(
        _plus_one,
        [1, 2, 3, 4],
        reducer=list,
        chunking=ChunkingPolicy(chunksize=1, sequential_threshold=1),
        progress=ProgressPolicy(enabled=True, desc="cells"),
    )
    (bar,) = bars
    assert bar.total == 4
    assert bar.desc == "cells"
    assert bar.updates == 4
    assert bar.closed


def _nested_dispatch(_x):
    from pyphi.parallel.backends.dask import DaskScheduler

    def double(y):
        return y * 2

    return DaskScheduler().map_reduce(double, [1, 2, 3], reducer=list)


def test_nested_dispatch_runs_in_task(dask_client):
    """map_reduce reached from inside a worker task runs in-task, not by
    submitting back to the cluster (which can deadlock an occupied pool)."""
    fut = dask_client.submit(_nested_dispatch, 0, pure=False)
    assert fut.result() == [2, 4, 6]
