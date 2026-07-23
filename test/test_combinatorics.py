import itertools
import math
import random
from itertools import chain

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyphi import combinatorics
from pyphi import utils

pair_indices_answers = [
    (
        (4,),
        {},
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
        {"k": 1},
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
        {},
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
    return {i: {x for x in answer if len(x) == i} for i in set(map(len, answer))}


@pytest.fixture
def nonempty_intersection_answer(nonempty_intersection_answer_by_order):
    return list(chain.from_iterable(nonempty_intersection_answer_by_order.values()))


size_args = [(0, None), (2, None), (4, None), (999, None), (0, 4), (3, 4), (0, 999)]


@pytest.mark.parametrize("min_size, max_size", size_args)
def test_combinations_with_nonempty_intersection(
    sets, nonempty_intersection_answer_by_order, min_size, max_size
):
    result = set(
        combinatorics.combinations_with_nonempty_intersection(
            sets, min_size=min_size, max_size=max_size
        )
    )
    if max_size is None:
        max_size = max(nonempty_intersection_answer_by_order)
    answer = {
        combination
        for k, v in nonempty_intersection_answer_by_order.items()
        if min_size <= k <= max_size
        for combination in v
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
    # The enumerator yields unique combinations; order is not part of its
    # contract (the lazy DFS yields depth-first, not grouped by size).
    assert set(answer) == set(result)
    assert len(answer) == len(result)


def _bruteforce_nonempty_intersection(sets, min_size, max_size):
    n = len(sets)
    upper = n if max_size is None else min(max_size, n)
    expected = set()
    for size in range(max(2, min_size), upper + 1):
        for combo in itertools.combinations(range(n), size):
            inter = sets[combo[0]]
            for i in combo[1:]:
                inter = inter & sets[i]
            if inter:
                expected.add(frozenset(combo))
    return expected


@given(
    sets=st.lists(
        st.frozensets(st.integers(min_value=0, max_value=5), max_size=4),
        min_size=0,
        max_size=8,
    ),
    min_size=st.integers(min_value=0, max_value=5),
    max_size=st.integers(min_value=0, max_value=6),
)
def test_combinations_with_nonempty_intersection_matches_bruteforce(
    sets, min_size, max_size
):
    result = list(
        combinatorics.combinations_with_nonempty_intersection(
            sets, min_size=min_size, max_size=max_size
        )
    )
    expected = _bruteforce_nonempty_intersection(sets, min_size, max_size)
    # Yields each combination exactly once, and exactly the oracle's set.
    assert set(result) == expected
    assert len(result) == len(expected)


def _brute_force_min_over_size(values):
    total = 0.0
    for size in range(2, len(values) + 1):
        for subset in itertools.combinations(values, size):
            total += min(subset) / size
    return total


@pytest.mark.parametrize(
    "values",
    [
        [],
        [3.0],
        [1.0, 2.0],
        [1.0, 2.0, 3.0],
        [3.0, 1.0, 2.0],
        [2.0, 2.0, 2.0],
        [0.5, 1.5, 0.25, 4.0, 4.0, 0.1],
    ],
)
def test_sum_of_minimum_over_size_matches_brute_force(values):
    assert combinatorics.sum_of_minimum_over_size_among_subsets(values) == pytest.approx(
        _brute_force_min_over_size(values)
    )


def test_sum_of_minimum_over_size_small_inputs_are_zero():
    assert combinatorics.sum_of_minimum_over_size_among_subsets([]) == 0.0
    assert combinatorics.sum_of_minimum_over_size_among_subsets([7.0]) == 0.0


def test_sum_of_minimum_over_size_known_value():
    assert combinatorics.sum_of_minimum_over_size_among_subsets(
        [1.0, 2.0, 3.0]
    ) == pytest.approx(0.5 + 0.5 + 1.0 + 1.0 / 3.0)


def _brute_force_min_of_size(values, size):
    return math.fsum(min(subset) for subset in itertools.combinations(values, size))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("size", [1, 2, 3, 5])
def test_sum_of_minimum_of_size_among_subsets(seed, size):
    rng = random.Random(seed)
    values = [rng.uniform(0.0, 2.0) for _ in range(7)]
    assert combinatorics.sum_of_minimum_of_size_among_subsets(
        values, size
    ) == pytest.approx(_brute_force_min_of_size(values, size))


def test_sum_of_minimum_of_size_out_of_range():
    values = [1.0, 2.0]
    assert combinatorics.sum_of_minimum_of_size_among_subsets(values, 0) == 0.0
    assert combinatorics.sum_of_minimum_of_size_among_subsets(values, 3) == 0.0


def _random_set_family(rng, num_sets=5, universe=6):
    return [
        frozenset(i for i in range(universe) if rng.random() < 0.5)
        for _ in range(num_sets)
    ]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_intersection_closure_matches_brute_force(seed):
    rng = random.Random(seed)
    sets = _random_set_family(rng)
    base = [s for s in sets if s]
    expected = set()
    for r in range(1, len(base) + 1):
        for family in itertools.combinations(base, r):
            intersection = frozenset.intersection(*family)
            if intersection:
                expected.add(intersection)
    assert combinatorics.intersection_closure(sets) == expected


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_exact_intersection_counts_match_brute_force(seed):
    rng = random.Random(seed)
    sets = _random_set_family(rng)
    expected = {}
    for r in range(2, len(sets) + 1):
        for indices in itertools.combinations(range(len(sets)), r):
            intersection = frozenset.intersection(*(sets[i] for i in indices))
            if intersection:
                expected[intersection] = expected.get(intersection, 0) + 1
    assert combinatorics.exact_intersection_counts(sets) == expected


def test_sum_of_minimum_among_subsets_no_int64_overflow():
    """The 2**k subset weight overflowed int64 for k>62, silently corrupting
    Σφ_r once an atom was shared by >63 distinctions. The fix stays exact for
    small groups and saturates to +inf (a valid ceiling) at extreme scale."""
    import math

    from pyphi.combinatorics import sum_of_minimum_among_subsets as f

    def ref(values):  # exact via Python big ints
        vs = sorted(values)
        n = len(vs)
        return float(sum(vs[i] * (2 ** (n - 1 - i) - 1) for i in range(n)))

    for n in (2, 10, 40, 63):
        v = [0.1 * (i + 1) for i in range(n)]
        assert math.isclose(f(v), ref(v), rel_tol=1e-9)
    v70 = [0.1] * 70  # old int64 path wrapped here
    assert math.isclose(f(v70), ref(v70), rel_tol=1e-9) and f(v70) > 1e18
    assert f([0.1] * 1100) == math.inf  # saturates, no raise
    assert f([0.0] * 1100 + [0.2]) == 0.0  # zeros give 0, not 0*inf=nan


def test_sum_of_minimum_among_subsets_boundary_sum_saturates():
    """At n=1024 every count fits float64 but their sum does not; the total
    saturates to inf instead of escalating an overflow warning."""
    import numpy as np

    assert combinatorics.sum_of_minimum_among_subsets(np.ones(1024)) == math.inf


def test_sum_of_minimum_over_size_among_subsets_saturates_to_inf():
    """Raised OverflowError once a group exceeded 1023 values."""
    import numpy as np

    values = np.ones(1100)
    assert combinatorics.sum_of_minimum_over_size_among_subsets(values) == math.inf


def test_sum_of_minimum_over_size_among_subsets_zero_values_contribute_nothing():
    """Zero values carry the overflowing coefficients but contribute 0, so
    the total reduces to the nonzero tail's."""
    import numpy as np

    values = np.concatenate([np.zeros(300), np.ones(1000)])
    result = combinatorics.sum_of_minimum_over_size_among_subsets(values)
    assert math.isfinite(result)
    assert result == pytest.approx(
        combinatorics.sum_of_minimum_over_size_among_subsets(np.ones(1000)),
        rel=1e-12,
    )
