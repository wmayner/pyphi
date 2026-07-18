"""Determinism guards for the parallel layer.

Parallel evaluation must yield the same results as sequential evaluation:
short-circuit truncation must happen at the same submission-order prefix on
every backend, tie selection must not depend on worker completion order, and
a worker exception must not leave orphaned work running in the executors.
"""

from __future__ import annotations

import time

import pytest

from pyphi import Direction
from pyphi import examples
from pyphi.conf import config
from pyphi.conf import presets
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


def test_find_mice_tied_purview_winner_independent_of_result_order(monkeypatch):
    from pyphi.formalism import queries

    system = examples.iit4_2023_fig6a_system()
    with config.override(**presets.iit4_2026):
        baseline = queries.find_mice(system, Direction.CAUSE, (0,), parallel=False)

        real_map_reduce = queries.map_reduce

        def reversed_map_reduce(fn, items, **kwargs):
            results = real_map_reduce(
                fn, items, **{**kwargs, "parallel": False, "progress": False}
            )
            return list(reversed(list(results)))

        monkeypatch.setattr(queries, "map_reduce", reversed_map_reduce)
        adversarial = queries.find_mice(system, Direction.CAUSE, (0,))

    assert len(baseline.purview_ties) >= 2, "case must be a genuine phi tie"
    assert adversarial.purview == baseline.purview
    assert {m.purview for m in adversarial.purview_ties} == {
        m.purview for m in baseline.purview_ties
    }


def test_state_mip_map_reduce_collects_in_input_order(monkeypatch):
    import pyphi.formalism.iit4.formalism as iit4_formalism
    from pyphi import System
    from pyphi.formalism import queries

    captured = []
    real_map_reduce = iit4_formalism.map_reduce

    def capturing_map_reduce(fn, items, *more_items, **kwargs):
        captured.append(kwargs)
        return real_map_reduce(fn, items, *more_items, **kwargs)

    monkeypatch.setattr(iit4_formalism, "map_reduce", capturing_map_reduce)
    with config.override(**presets.iit4_2026):
        system = System(examples.basic_substrate(), (1, 0, 0))
        queries.find_mip(system, Direction.CAUSE, (0,), (0, 1))

    assert captured, "the state-MIP path should invoke map_reduce"
    assert all(kwargs.get("ordered") is True for kwargs in captured)


def _identity(value):
    return value


def _is_zero(value):
    return value == 0


def test_shortcircuit_callback_args_honored_on_parallel_paths():
    """The caller's ``shortcircuit_callback_args`` reaches the callback on the
    thread and process parallel paths, not only the sequential path."""
    for backend in ("thread", "local"):
        received = []
        with config.override(parallel=True, parallel_backend=backend):
            map_reduce(
                _identity,
                list(range(32)),
                reduce_func=list,
                shortcircuit_func=_is_zero,
                shortcircuit_callback=received.append,
                shortcircuit_callback_args="sentinel",
                parallel=True,
                sequential_threshold=1,
                chunksize=1,
                progress=False,
            )
        assert received == ["sentinel"], backend


def test_size_func_with_shortcircuit_rejected():
    with pytest.raises(ValueError, match="short-circuit"):
        map_reduce(
            _identity,
            [1, 2, 3],
            reduce_func=list,
            size_func=lambda item: item,
            shortcircuit_func=bool,
            parallel=False,
            progress=False,
        )
