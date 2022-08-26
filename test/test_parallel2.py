import time
from datetime import timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, sentinel

import pytest
import ray
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis_utils import (
    anything,
    anything_comparable,
    anything_pickleable_and_hashable,
    iterable_or_list,
)

from pyphi.compute import parallel


@given(st.lists(iterable_or_list(anything())))
def test_try_lens(iterables):
    expected = min(
        map(len, filter(lambda iterable: hasattr(iterable, "__len__"), iterables)),
        default=None,
    )
    assert parallel._try_lens(*iterables) == expected


@given(anything())
def test_noshortcircuit(x):
    assert x != parallel._NoShortCircuit()
    assert not parallel._NoShortCircuit()


def shortcircuit_tester(func, list_and_index, ordered=True, shortcircuit_value=None):
    items, idx = list_and_index

    expected = list(items)
    actual = list(func(items))

    if ordered:
        assert expected == actual
    else:
        assert set(expected) == set(actual)

    if items and shortcircuit_value is None:
        idx = items.index(items[idx])
        shortcircuit_value = items[idx]
    else:
        items.insert(idx, shortcircuit_value)

    expected = list(items)

    if shortcircuit_value is not None:
        actual = list(func(items, shortcircuit_value=shortcircuit_value))
        if ordered:
            assert expected[: idx + 1] == actual
        else:
            assert shortcircuit_value in actual

        mock = Mock()
        actual = list(
            func(
                items,
                shortcircuit_value=shortcircuit_value,
                shortcircuit_callback=mock,
            )
        )
        # TODO(4.0) call not detected when parallel; used SharedMock or similar
        # mock.assert_called()
        if ordered:
            assert expected[: idx + 1] == actual
        else:
            assert shortcircuit_value in actual


@given(
    list_and_index=list_and_index(anything_comparable()),
)
def test_shortcircuit(list_and_index):
    shortcircuit_tester(parallel.shortcircuit, list_and_index)


@ray.remote
def remote_sleep(x, t=0.1):
    for _ in range(int(x)):
        time.sleep(t)
    return x


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=timedelta(seconds=10),
)
@given(args=st.lists(st.integers(min_value=0, max_value=1), max_size=2))
@pytest.mark.filterwarnings("ignore:.*:pytest.PytestUnraisableExceptionWarning")
def test_as_completed(ray_context, args):
    args = sorted(args, reverse=True)
    expected = sorted(args)
    actual = list(parallel.as_completed([remote_sleep.remote(i) for i in args]))
    assert expected == actual


# TODO hangs; maybe just wait for Ray PR to be merged
# @pytest.mark.filterwarnings("ignore:.*:pytest.PytestUnraisableExceptionWarning")
# def test_cancel_all(ray_context):
#     tasks = [remote_sleep.remote(i) for i in [100] * 10]
#     parallel.cancel_all(tasks)
#     with pytest.raises(
#         (ray.exceptions.TaskCancelledError, ray.exceptions.RayTaskError)
#     ):
#         ray.get(tasks[0])


@given(st.lists(everything_except(Decimal)))
def test_get_local(items):
    with patch("pyphi.compute.parallel.cancel_all") as mock:
        expected = list(items)
        actual = list(parallel.get(items))
        mock.assert_not_called()
        assert expected == actual


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    st.lists(st.integers()),
)
@pytest.mark.filterwarnings("ignore:.*:pytest.PytestUnraisableExceptionWarning")
def test_get_parallel(ray_context, expected):
    @ray.remote
    def f(x):
        return x

    refs = [f.remote(x) for x in expected]
    assert set(expected) == set(parallel.get(refs, parallel=True))


def test_map_with_no_args():
    with pytest.raises(ValueError):
        list(parallel.map(lambda x: x))


@pytest.mark.filterwarnings("ignore:.*:pytest.PytestUnraisableExceptionWarning")
def test_map_with_iterator_no_chunksize(ray_context, func):
    with pytest.raises(ValueError):
        parallel.map(func, iter([1, 2, 3]), parallel=True, chunksize=None)


def arglists(elements):
    return st.lists(teed(iterable_or_list(elements), n=2), min_size=1).map(
        lambda _: list(zip(*_))
    )


@pytest.fixture
def func():
    def func(*args):
        if args:
            return args[0]
        return args

    return func


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    arglists=arglists(anything()),
)
def test_map_sequential(
    func,
    arglists,
):
    expected = list(map(func, *arglists[0]))
    actual = list(parallel._map_sequential(func, *arglists[1]))
    assert expected == actual


@pytest.mark.filterwarnings("ignore:.*:pytest.PytestUnraisableExceptionWarning")
def test_map_with_lambda(ray_context):
    expected = set([1, 2, 3])
    actual = set(parallel.map(lambda x: x, expected, parallel=True))
    assert expected == actual


def test_map_with_iterators_and_empty_args(func):
    assert [] == parallel.map(func, iter([]), parallel=True, chunksize=100)


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
@given(
    arglists=arglists(anything_pickleable_and_hashable()),
    _parallel=st.booleans(),
    max_size=st.integers() | st.none(),
    max_depth=st.integers() | st.none(),
    branch_factor=st.integers(),
    chunksize=st.integers(),
    sequential_threshold=st.integers(),
    inflight_limit=st.integers(),
)
@pytest.mark.filterwarnings("ignore:.*:pytest.PytestUnraisableExceptionWarning")
def test_map_with_iterators(
    ray_context_local,
    func,
    arglists,
    _parallel,
    max_size,
    max_depth,
    branch_factor,
    chunksize,
    sequential_threshold,
    inflight_limit,
):
    expected = set(map(func, *arglists[0]))
    actual = set(
        parallel.map(
            func,
            *arglists[1],
            max_size=max_size,
            max_depth=max_depth,
            branch_factor=branch_factor,
            chunksize=chunksize,
            sequential_threshold=sequential_threshold,
            inflight_limit=inflight_limit,
            parallel=_parallel,
        )
    )
    assert expected == actual


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
@given(
    list_and_index=list_and_index(anything_pickleable_and_hashable()),
    _parallel=st.booleans(),
    max_size=st.integers() | st.none(),
    max_depth=st.integers() | st.none(),
    branch_factor=st.integers(),
    chunksize=st.integers() | st.none(),
    sequential_threshold=st.integers(),
    inflight_limit=st.integers(),
)
@pytest.mark.filterwarnings("ignore:.*:pytest.PytestUnraisableExceptionWarning")
def test_map_with_shortcircuit(
    ray_context_local,
    func,
    list_and_index,
    max_size,
    max_depth,
    branch_factor,
    _parallel,
    chunksize,
    sequential_threshold,
    inflight_limit,
):
    def _func(items, **kwargs):
        return parallel.map(
            func,
            items,
            **kwargs,
            max_size=max_size,
            max_depth=max_depth,
            branch_factor=branch_factor,
            chunksize=chunksize,
            sequential_threshold=sequential_threshold,
            inflight_limit=inflight_limit,
            parallel=_parallel,
        )

    shortcircuit_tester(
        _func,
        list_and_index,
        shortcircuit_value=sentinel.shortcircuit,
        ordered=(not _parallel),
    )


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
@given(
    arglists=arglists(st.integers()),
    max_size=st.integers() | st.none(),
    max_depth=st.integers() | st.none(),
    branch_factor=st.integers(),
    chunksize=st.integers(),
    _parallel=st.booleans(),
    sequential_threshold=st.integers(),
    inflight_limit=st.integers(),
)
@pytest.mark.filterwarnings("ignore:.*:pytest.PytestUnraisableExceptionWarning")
def test_map_reduce(
    ray_context_local,
    func,
    arglists,
    max_size,
    max_depth,
    branch_factor,
    _parallel,
    chunksize,
    sequential_threshold,
    inflight_limit,
):
    def reduce_func(x):
        return max(x, default=None)

    expected = reduce_func(map(func, *arglists[0]))
    actual = parallel.map_reduce(
        func,
        reduce_func,
        *arglists[1],
        max_size=max_size,
        max_depth=max_depth,
        branch_factor=branch_factor,
        chunksize=chunksize,
        sequential_threshold=sequential_threshold,
        inflight_limit=inflight_limit,
        parallel=_parallel,
    )
    assert expected == actual
