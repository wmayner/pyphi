"""Determinism guards for the parallel layer.

Parallel evaluation must yield the same results as sequential evaluation:
short-circuit truncation must happen at the same submission-order prefix on
every backend, tie selection must not depend on worker completion order, and
a worker exception must not leave orphaned work running in the executors.
"""

from __future__ import annotations

import time

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
