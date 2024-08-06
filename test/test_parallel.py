import time
from datetime import timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest
import ray
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

from pyphi import parallel

from .hypothesis_utils import (
    anything,
    anything_comparable,
    anything_pickleable_and_hashable,
    everything_except,
    iterable_or_list,
    list_and_index,
    teed,
)


def shortcircuit_tester(func, list_and_index, ordered=True):
    items, idx = list_and_index

    # No shortcircuiting
    expected = list(items)
    actual = list(func(items))
    if ordered:
        assert expected == actual
    else:
        assert set(expected) == set(actual)

    # Skip if list is empty
    if not items:
        return

    # Get first index of item and define shortcircuit func as checking for that
    # item
    idx = items.index(items[idx])
    shortcircuit_func = lambda x: x == items[idx]

    # With shortcircuiting
    expected = list(items)
    actual = list(func(items, shortcircuit_func=shortcircuit_func))
    if ordered:
        assert expected[: idx + 1] == actual

        # Check callback was called
        # TODO(4.0) call not detected when parallel; used SharedMock or similar
        mock = Mock()
        actual = list(
            func(
                items,
                shortcircuit_func=shortcircuit_func,
                shortcircuit_callback=mock,
            )
        )
        mock.assert_called()
        assert expected[: idx + 1] == actual
    else:
        assert items[idx] in actual


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
    with patch("pyphi.parallel.cancel_all") as mock:
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
def test_get_remote(ray_context, expected):
    @ray.remote
    def f(x):
        return x

    refs = [f.remote(x) for x in expected]
    assert set(expected) == set(parallel.get(refs, remote=True))


def test_map_repr():
    mr = parallel.MapReduce(lambda x: x, [1, 2, 3])
    str(mr)
    repr(mr)
    print(mr)


def test_map_with_no_args():
    with pytest.raises(TypeError):
        list(parallel.MapReduce(lambda x: x))


@pytest.mark.filterwarnings("ignore:.*:pytest.PytestUnraisableExceptionWarning")
def test_map_with_iterator_no_chunksize(ray_context, func):
    with pytest.raises(ValueError):
        parallel.MapReduce(func, iter([1, 2, 3]), parallel=True, chunksize=None)


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
    args=arglists(anything()),
)
def test_map_sequential(
    func,
    args,
):
    iterables1, iterables2 = args
    expected = list(map(func, *iterables1))
    actual = list(parallel.MapReduce(func, *iterables2, parallel=False).run())
    assert expected == actual


@pytest.mark.filterwarnings("ignore:.*:pytest.PytestUnraisableExceptionWarning")
def test_map_with_lambda(ray_context):
    expected = set([1, 2, 3])
    actual = set(parallel.MapReduce(lambda x: x, expected, parallel=True).run())
    assert expected == actual


@pytest.mark.filterwarnings("ignore:.*:pytest.PytestUnraisableExceptionWarning")
def test_map_with_iterators_and_empty_args(func):
    assert [] == parallel.MapReduce(func, iter([]), parallel=True, chunksize=100).run()


@composite
def map_reduce_kwargs_common(draw):
    return dict(
        chunksize=draw(st.integers(min_value=1)),
        sequential_threshold=draw(st.integers(min_value=1)),
        max_depth=draw(st.integers(min_value=1) | st.none()),
        branch_factor=draw(st.integers(min_value=2)),
        inflight_limit=draw(st.integers(min_value=1)),
        ordered=draw(st.booleans()),
    )


@composite
def map_reduce_kwargs_iterators(draw):
    return {
        **draw(map_reduce_kwargs_common()),
        **dict(
            max_size=None,
            max_leaves=None,
            total=None,
        ),
    }


@composite
def map_reduce_kwargs_sequences(draw):
    return {
        **draw(map_reduce_kwargs_common()),
        **dict(
            max_size=draw(st.integers(min_value=1)),
            max_leaves=draw(st.integers(min_value=1)),
            total=None,
        ),
    }


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
@given(
    args=arglists(anything_pickleable_and_hashable()),
    kwargs=map_reduce_kwargs_iterators(),
)
@pytest.mark.filterwarnings("ignore:.*:pytest.PytestUnraisableExceptionWarning")
def test_map_with_iterators(
    ray_context,
    func,
    args,
    kwargs,
):
    iterables1, iterables2 = args
    expected = list(map(func, *iterables1))
    actual = parallel.MapReduce(
        func,
        *iterables2,
        parallel=True,
        **kwargs,
    ).run()
    if kwargs["ordered"]:
        assert expected == actual
    else:
        assert set(expected) == set(actual)


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
@given(
    list_and_index=list_and_index(anything_pickleable_and_hashable()),
    kwargs=map_reduce_kwargs_sequences(),
    _parallel=st.booleans() | st.none(),
)
@pytest.mark.filterwarnings("ignore:.*:pytest.PytestUnraisableExceptionWarning")
def test_map_with_shortcircuit(
    ray_context,
    func,
    list_and_index,
    kwargs,
    _parallel,
):
    def _func(items, **additional_kwargs):
        return parallel.MapReduce(
            func,
            items,
            **kwargs,
            **additional_kwargs,
        ).run()

    shortcircuit_tester(
        _func,
        list_and_index,
        ordered=(not _parallel),
    )


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
@given(
    args=arglists(st.integers()),
    kwargs=map_reduce_kwargs_iterators(),
    _parallel=st.booleans() | st.none(),
)
@pytest.mark.filterwarnings("ignore:.*:pytest.PytestUnraisableExceptionWarning")
def test_map_reduce(
    ray_context,
    func,
    args,
    kwargs,
    _parallel,
):
    iterables1, iterables2 = args

    def reduce_func(x, some_kwarg=None):
        assert some_kwarg is not None
        return max(x, default=None)

    expected = reduce_func(map(func, *iterables1), some_kwarg=1)
    actual = parallel.MapReduce(
        func,
        *iterables2,
        reduce_func=reduce_func,
        reduce_kwargs=dict(some_kwarg=1),
        **kwargs,
        parallel=_parallel,
    ).run()
    assert expected == actual


# TODO(4.0) unit tests for tree.py
