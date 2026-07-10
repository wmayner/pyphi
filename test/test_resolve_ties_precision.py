"""Precision-awareness of the resolve_ties selection engine.

Every selection must (a) treat sub-tolerance float differences as ties
and (b) return the same winner set regardless of candidate order.
"""

import itertools
from dataclasses import dataclass
from dataclasses import field

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from pyphi import resolve_ties


@dataclass(frozen=True)
class FakePartition:
    key: bytes

    def lex_key(self):
        return self.key


@dataclass(frozen=True)
class FakeMIP:
    phi: float
    normalized_phi: float
    partition: FakePartition
    purview: tuple = field(default=(0,))


NOISE = 5.6e-16  # the observed mirror-isomorphic candidate gap


def _mips(*specs):
    """Build FakeMIPs from (phi, normalized_phi, lexbyte) triples."""
    return [
        FakeMIP(phi=p, normalized_phi=n, partition=FakePartition(bytes([b])))
        for p, n, b in specs
    ]


class TestResolveClustering:
    def test_noise_tied_candidates_both_survive_default_strategy(self):
        # Default mip_tie_resolution = ["NORMALIZED_PHI", "NEGATIVE_PHI"]:
        # sub-tolerance gaps on both keys must not drop either candidate.
        a, b = _mips((0.3, 0.15, 1), (0.3 + NOISE, 0.15 + NOISE, 2))
        survivors = list(resolve_ties.partitions([a, b]))
        assert set(survivors) == {a, b}

    def test_genuine_difference_still_selects(self):
        a, b = _mips((0.3, 0.15, 1), (0.4, 0.2, 2))
        survivors = list(resolve_ties.partitions([a, b]))
        assert survivors == [a]  # min normalized_phi

    def test_lexicographic_tolerance_per_component(self):
        # Component 1 tied within tolerance -> decision falls to component 2,
        # even though exact comparison of component 1 would differ.
        a, b = _mips((0.5, 0.15 + NOISE, 1), (0.2, 0.15, 2))
        # operation=min over NORMALIZED_PHI: tied -> NEGATIVE_PHI: min(-phi)
        # = max phi -> a (phi 0.5) wins.
        survivors = list(resolve_ties.partitions([a, b]))
        assert survivors == [a]

    def test_permutation_invariance_exhaustive(self):
        mips = _mips(
            (0.3, 0.15, 1),
            (0.3 + NOISE, 0.15 + NOISE, 2),
            (0.3 - NOISE, 0.15 - NOISE, 3),
            (0.7, 0.4, 4),
        )
        results = {
            frozenset(resolve_ties.partitions(list(perm)))
            for perm in itertools.permutations(mips)
        }
        assert len(results) == 1

    def test_integer_keys_compare_exactly(self):
        # PURVIEW_SIZE is an int key; exact comparison, no tolerance.
        a = FakeMIP(0.3, 0.15, FakePartition(b"\x01"), purview=(0,))
        b = FakeMIP(0.3, 0.15, FakePartition(b"\x02"), purview=(0, 1))
        survivors = list(resolve_ties.resolve([a, b], ["PURVIEW_SIZE"], operation=max))
        assert survivors == [b]


class TestCascadeClustering:
    def test_apply_level_clusters_float_keys(self):
        a, b = _mips((0.3, 0.15, 1), (0.3 + NOISE, 0.15, 2))
        level = resolve_ties.CascadeLevel(
            postulate="Integration", op="argmax", key=lambda m: m.phi
        )
        assert set(resolve_ties._apply_level([a, b], level)) == {a, b}

    def test_apply_level_exact_for_bytes(self):
        a, b = _mips((0.3, 0.15, 1), (0.3, 0.15, 2))
        level = resolve_ties.CascadeLevel(
            postulate="Determinism",
            op="argmin",
            key=lambda m: m.partition.lex_key(),
        )
        assert resolve_ties._apply_level([a, b], level) == (a,)


@settings(max_examples=200, deadline=None)
@given(
    base=st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
    n_twins=st.integers(min_value=2, max_value=5),
    n_others=st.integers(min_value=0, max_value=4),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_property_twins_always_coselected(base, n_twins, n_others, seed):
    """Candidates perturbed by sub-tolerance noise are always co-selected,
    under every input permutation (sampled)."""
    import random

    rng = random.Random(seed)
    twins = [
        FakeMIP(
            phi=base + i * NOISE,
            normalized_phi=(base + i * NOISE) / 2,
            partition=FakePartition(bytes([i])),
        )
        for i in range(n_twins)
    ]
    others = [
        FakeMIP(
            phi=base + 1.0 + j,
            normalized_phi=(base + 1.0 + j) / 2,
            partition=FakePartition(bytes([100 + j])),
        )
        for j in range(n_others)
    ]
    pool = twins + others
    rng.shuffle(pool)
    survivors = set(resolve_ties.partitions(pool))
    assert survivors == set(twins)  # twins are the min tier, all co-selected
