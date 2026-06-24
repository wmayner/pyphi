"""Tests for the parallel computation module."""

from decimal import Decimal
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

from pyphi import parallel

from .hypothesis_utils import anything
from .hypothesis_utils import anything_comparable
from .hypothesis_utils import anything_pickleable_and_hashable
from .hypothesis_utils import everything_except
from .hypothesis_utils import iterable_or_list
from .hypothesis_utils import list_and_index
from .hypothesis_utils import teed


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

    def shortcircuit_func(x):
        return x == items[idx]

    # With shortcircuiting
    expected = list(items)
    actual = list(func(items, shortcircuit_func=shortcircuit_func))
    if ordered:
        assert expected[: idx + 1] == actual

        # Check callback was called
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


@given(st.lists(everything_except(Decimal)))
def test_get_local(items):
    with patch("pyphi.parallel.cancel_all") as mock:
        expected = list(items)
        actual = list(parallel.get(items))
        mock.assert_not_called()
        assert expected == actual


def test_parallel_exception_handling():
    """Test that exceptions in parallel computation are properly propagated."""

    def raise_error(x):
        raise Exception("I don't wanna!")

    with pytest.raises(Exception, match=r"I don't wanna!"):
        parallel.map_reduce(raise_error, [1], parallel=True, chunksize=1)


def test_map_with_no_args():
    with pytest.raises(TypeError):
        parallel.map_reduce(lambda x: x)


def test_map_with_iterator_no_chunksize():
    # An unknown-length iterable with no explicit chunksize is now handled by
    # cost-sampling rather than raising; it runs and returns all results.
    result = parallel.map_reduce(
        lambda x: x, iter([1, 2, 3]), parallel=True, chunksize=None
    )
    assert sorted(result) == [1, 2, 3]


def arglists(elements):
    return st.lists(teed(iterable_or_list(elements), n=2), min_size=1).map(
        lambda _: list(zip(*_, strict=False))
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
    actual = list(parallel.map_reduce(func, *iterables2, parallel=False))
    assert expected == actual


def _identity(x):
    """Top-level identity function for pickling compatibility."""
    return x


def _get_first(*args):
    """Top-level function that returns first arg (or empty tuple)."""
    if args:
        return args[0]
    return args


def test_map_with_function_parallel():
    """Test parallel execution with a picklable function.

    Note: Lambda functions cannot be used with parallel=True because
    ProcessPoolExecutor requires picklable functions. Use top-level
    function definitions instead.
    """
    expected = {1, 2, 3}
    actual = set(parallel.map_reduce(_identity, expected, parallel=True, chunksize=1))
    assert expected == actual


def test_map_with_iterators_and_empty_args():
    result = parallel.map_reduce(lambda x: x, iter([]), parallel=True, chunksize=100)
    assert result == []


@composite
def map_reduce_kwargs_common(draw):
    return {
        "chunksize": draw(st.integers(min_value=1, max_value=8192)),
        "sequential_threshold": draw(st.integers(min_value=1, max_value=2048)),
        "ordered": draw(st.booleans()),
    }


@composite
def map_reduce_kwargs_iterators(draw):
    return {
        **draw(map_reduce_kwargs_common()),
        "total": None,
    }


@composite
def map_reduce_kwargs_sequences(draw):
    return {
        **draw(map_reduce_kwargs_common()),
        "total": None,
    }


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
@given(
    args=arglists(anything_pickleable_and_hashable()),
    kwargs=map_reduce_kwargs_iterators(),
)
@pytest.mark.slow
def test_map_with_iterators_parallel(
    func,
    args,
    kwargs,
):
    iterables1, iterables2 = args
    expected = list(map(func, *iterables1))
    actual = parallel.map_reduce(
        func,
        *iterables2,
        parallel=True,
        **kwargs,
    )
    if kwargs["ordered"]:
        assert expected == actual
    else:
        assert set(expected) == set(actual)


@settings(
    deadline=None,
)
@given(
    list_and_index=list_and_index(anything_pickleable_and_hashable()),
    kwargs=map_reduce_kwargs_sequences(),
)
def test_map_with_shortcircuit(
    list_and_index,
    kwargs,
):
    def _func(items, **additional_kwargs):
        return parallel.map_reduce(
            _get_first,
            items,
            **kwargs,
            **additional_kwargs,
        )

    shortcircuit_tester(
        _func,
        list_and_index,
        ordered=kwargs["ordered"],
    )


def _max_reduce(x, some_kwarg=None):
    """Top-level reduce function."""
    assert some_kwarg is not None
    return max(x, default=None)


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
@given(
    args=arglists(st.integers()),
    kwargs=map_reduce_kwargs_iterators(),
    _parallel=st.booleans() | st.none(),
)
def test_map_reduce(
    args,
    kwargs,
    _parallel,
):
    iterables1, iterables2 = args

    expected = _max_reduce(map(_get_first, *iterables1), some_kwarg=1)
    actual = parallel.map_reduce(
        _get_first,
        *iterables2,
        reduce_func=_max_reduce,
        reduce_kwargs={"some_kwarg": 1},
        **kwargs,
        parallel=_parallel,
    )
    assert expected == actual


# Tests for the local backend specifically
# ========================================


def _double(x):
    """Top-level double function for pickling compatibility."""
    return x * 2


def test_local_backend_basic():
    """Test basic parallel execution with local backend."""
    result = parallel.map_reduce(
        _double,
        [1, 2, 3, 4, 5],
        parallel=True,
        chunksize=2,
    )
    assert set(result) == {2, 4, 6, 8, 10}


def test_local_backend_with_reduce():
    """Test parallel execution with custom reduce function."""
    result = parallel.map_reduce(
        _double,
        [1, 2, 3, 4, 5],
        reduce_func=sum,
        parallel=True,
        chunksize=2,
    )
    assert result == 30  # 2+4+6+8+10


def test_local_backend_sequential_fallback():
    """Test that small workloads run sequentially.

    Note: Lambda can be used here because sequential mode doesn't
    use multiprocessing.
    """
    # With sequential_threshold high enough, should run sequentially
    result = parallel.map_reduce(
        lambda x: x * 2,
        [1, 2, 3],
        parallel=True,
        sequential_threshold=100,  # Higher than len(items)
        chunksize=2,
    )
    assert set(result) == {2, 4, 6}


def test_backend_selection():
    """Test backend auto-detection, explicit selection, and rejection."""
    assert sorted(
        parallel.map_reduce(_identity, [1, 2, 3], backend="auto", chunksize=1)
    ) == [1, 2, 3]
    assert sorted(
        parallel.map_reduce(_identity, [1, 2, 3], backend="local", chunksize=1)
    ) == [1, 2, 3]

    with pytest.raises(ValueError, match="unknown parallel_backend"):
        parallel.map_reduce(_identity, [1, 2, 3], backend="invalid", chunksize=1)


def test_cancel_all_with_futures():
    """Test cancel_all function with concurrent.futures.Future objects."""
    from concurrent.futures import Future

    # Create some mock futures
    futures = [Future() for _ in range(3)]
    result = parallel.cancel_all(futures)
    assert len(result) == 3


# Tests for get_num_processes
# ============================


class TestGetNumProcesses:
    """Tests for get_num_processes edge cases."""

    def test_zero_workers_raises_error(self):
        """``parallel_workers=0`` raises ValueError."""
        from pyphi import config

        with (
            config.override(parallel_workers=0),
            pytest.raises(ValueError, match="may not be 0"),
        ):
            parallel.get_num_processes()

    def test_negative_workers_calculates_correctly(self):
        """``parallel_workers=-1`` means all CPUs, ``-2`` means all but one."""
        import multiprocessing

        from pyphi import config

        cpu_count = multiprocessing.cpu_count()
        with config.override(parallel_workers=-1):
            # -1 means cpu_count + (-1) + 1 = cpu_count
            assert parallel.get_num_processes() == cpu_count

        with config.override(parallel_workers=-2):
            # -2 means cpu_count + (-2) + 1 = cpu_count - 1
            assert parallel.get_num_processes() == cpu_count - 1

    def test_negative_workers_too_negative_raises(self):
        """``parallel_workers`` too negative raises ValueError."""
        import multiprocessing

        from pyphi import config

        cpu_count = multiprocessing.cpu_count()
        # e.g., if cpu_count=8, parallel_workers=-9 would give 0 or negative
        too_negative = -(cpu_count + 1)
        with (
            config.override(parallel_workers=too_negative),
            pytest.raises(ValueError, match="too negative"),
        ):
            parallel.get_num_processes()

    def test_workers_exceeds_available_returns_available(self):
        """When requesting more workers than CPUs, returns CPU count."""
        import multiprocessing

        from pyphi import config

        cpu_count = multiprocessing.cpu_count()
        with config.override(parallel_workers=cpu_count + 10):
            result = parallel.get_num_processes()
            assert result == cpu_count

    def test_positive_workers_returns_value(self):
        """Positive ``parallel_workers`` returns that value."""
        from pyphi import config

        with config.override(parallel_workers=2):
            assert parallel.get_num_processes() == 2
