"""Determinism guards for the parallel layer.

Parallel evaluation must yield the same results as sequential evaluation:
short-circuit truncation must happen at the same submission-order prefix on
every backend, tie selection must not depend on worker completion order, and
a worker exception must not leave orphaned work running in the executors.
"""

from __future__ import annotations

import time

import pytest

from pyphi.parallel import false
from pyphi.parallel import map_reduce
from pyphi.parallel.scheduler import ShortcircuitPolicy


def test_shortcircuit_policy_active():
    assert not ShortcircuitPolicy().active
    assert not ShortcircuitPolicy(func=false).active
    assert ShortcircuitPolicy(func=lambda r: r == 0).active


def test_thread_backend_sub_threshold_honors_shortcircuit():
    calls = []
    collected = []

    def record(x):
        calls.append(x)
        return x

    result = map_reduce(
        record,
        [3, 0, 2],
        parallel=True,
        backend="thread",
        sequential_threshold=10,
        shortcircuit_func=lambda r: r == 0,
        shortcircuit_callback=collected.append,
        progress=False,
    )
    assert result == [3, 0]
    assert calls == [3, 0]
    assert collected == [[3, 0]]


def test_thread_backend_shortcircuit_collects_submission_order_prefix():
    def slow_identity(delay, value):
        time.sleep(delay)
        return value

    delays = [0.5, 0.4, 0.3, 0.2, 0.1]
    values = [1, 1, 0, 1, 0]
    result = map_reduce(
        slow_identity,
        delays,
        values,
        parallel=True,
        backend="thread",
        sequential_threshold=1,
        shortcircuit_func=lambda r: r == 0,
        progress=False,
    )
    assert result == [1, 1, 0]


def _boom_or_sleep(x):
    if x == 0:
        raise ValueError("boom")
    time.sleep(0.05)
    return x


def test_worker_exception_cancels_pending_process_chunks():
    from pyphi.parallel.backends.local_process import LocalMapReduce

    # loky marks a future RUNNING as soon as it enters its prefetch queue
    # (2 * workers + 1 deep), so use enough single-item chunks that many
    # futures are still pending when the exception surfaces.
    mr = LocalMapReduce(
        map_func=_boom_or_sleep,
        iterables=(list(range(512)),),
        reduce_func=list,
        reduce_kwargs={},
        chunksize=1,
        progress=False,
        total=512,
    )
    with pytest.raises(ValueError, match="boom"):
        mr.run()
    assert any(future.cancelled() for future in mr._futures)


def test_worker_exception_cancels_pending_thread_futures():
    calls = []

    def boom_or_sleep(x):
        calls.append(x)
        if x == 0:
            raise ValueError("boom")
        time.sleep(0.3)
        return x

    with pytest.raises(ValueError, match="boom"):
        map_reduce(
            boom_or_sleep,
            list(range(32)),
            parallel=True,
            backend="thread",
            sequential_threshold=1,
            progress=False,
        )
    assert len(calls) < 32
