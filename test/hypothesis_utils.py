import pickle
from functools import partial
from itertools import tee as _tee
from typing import Hashable

from hypothesis import assume
from hypothesis import strategies as st
from hypothesis.strategies import composite


class PrettyIter:
    """An iterator that displays its contents."""

    def __init__(self, values):
        self._values, _repr = _tee(values, 2)
        self._repr = list(_repr)
        self._iter = iter(self._values)

    def __iter__(self):
        return self._iter

    def __next__(self):
        return next(self._iter)

    def __repr__(self):
        return f"iter({self._repr!r})"


def everything_except(*excluded_types):
    return (
        st.from_type(type)
        .flatmap(st.from_type)
        .filter(lambda x: not isinstance(x, excluded_types))
    )


def anything():
    return everything_except()


@composite
def anything_comparable(draw):
    example = draw(anything())
    try:
        assume(example == example)
    except:
        assume(False)
    return example


@composite
def anything_pickleable(draw):
    example = draw(anything())
    try:
        assume(example == pickle.loads(pickle.dumps(example)))
    except:
        assume(False)
    return example


@composite
def anything_pickleable_and_hashable(draw):
    example = draw(anything())
    try:
        assume(
            isinstance(example, Hashable)
            and example == pickle.loads(pickle.dumps(example))
        )
    except:
        assume(False)
    return example


@composite
def list_and_index(draw, elements):
    l = draw(st.lists(elements))
    n = len(l)
    index = draw(st.integers(min_value=0, max_value=(n - 1) if n else 0))
    return (l, index)


def iterable_or_list(elements):
    return st.iterables(elements) | st.iterables(elements)


def tee(iterable, n=2):
    return tuple(map(PrettyIter, _tee(iterable, n)))


def teed(strategy, n=2):
    return strategy.map(partial(tee, n=n))
