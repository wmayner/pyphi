from itertools import chain

import pytest

from pyphi import combinatorics, utils

pair_indices_answers = [
    (
        (4,),
        dict(),
        [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 2),
            (2, 3),
            (3, 3),
        ],
    ),
    (
        (4,),
        dict(k=1),
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        ],
    ),
    (
        (4, 2),
        dict(),
        [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 1),
            (1, 2),
            (1, 3),
        ],
    ),
]


@pytest.mark.parametrize("args, kwargs, answer", pair_indices_answers)
def test_pair_indices(args, kwargs, answer):
    assert list(combinatorics.pair_indices(*args, **kwargs)) == answer


@pytest.mark.parametrize("args, kwargs, answer", pair_indices_answers)
def test_pairs(args, kwargs, answer):
    if len(args) == 1:
        args = [list(range(n)) for n in args]
        assert list(combinatorics.pairs(*args, **kwargs)) == answer


@pytest.fixture
def sets():
    return [
        frozenset(x)
        for x in [
            [5],
            [5, 6],
            [8, 9],
            [7, 8, 9],
            [6, 7],
            [5, 9],
            [7, 9],
            [5, 6, 7, 9],
        ]
    ]


@pytest.fixture
def nonempty_intersection_answer_by_order(sets):
    # Find answer with naive algorithm
    answer = [
        frozenset(sets.index(x) for x in combination)
        for combination in utils.powerset(sets, nonempty=True, max_size=None)
        if len(combination) >= 2 and frozenset.intersection(*combination)
    ]
    return {i: set(x for x in answer if len(x) == i) for i in set(map(len, answer))}


@pytest.fixture
def nonempty_intersection_answer(nonempty_intersection_answer_by_order):
    return list(chain.from_iterable(nonempty_intersection_answer_by_order.values()))


size_args = [(0, None), (2, None), (4, None), (999, None), (0, 4), (3, 4), (0, 999)]


@pytest.mark.parametrize("min_size, max_size", size_args)
def test_combinations_with_nonempty_intersection(
    sets, nonempty_intersection_answer_by_order, min_size, max_size
):
    result = combinatorics.combinations_with_nonempty_intersection_by_order(
        sets, min_size=min_size, max_size=max_size
    )
    if max_size is None:
        max_size = max(nonempty_intersection_answer_by_order)
    answer = {
        k: v
        for k, v in nonempty_intersection_answer_by_order.items()
        if min_size <= k <= max_size
    }
    assert answer == result


@pytest.mark.parametrize("min_size, max_size", size_args)
def test_explicit_combinations_with_nonempty_intersection(
    sets, nonempty_intersection_answer, min_size, max_size
):
    result = list(
        combinatorics.combinations_with_nonempty_intersection(
            sets, min_size=min_size, max_size=max_size
        )
    )
    if max_size is None:
        max_size = max(map(len, nonempty_intersection_answer))
    answer = [
        combination
        for combination in nonempty_intersection_answer
        if min_size <= len(combination) <= max_size
    ]
    assert answer == result
