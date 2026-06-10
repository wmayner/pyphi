"""Tests for pyphi.formalism.iit4.bounds (Zaeemzadeh & Tononi 2024).

Validation batteries per the design spec:
(a) independent recomputation of the formulas (brute force, linprog);
(b) property tests against the real pipeline = domain-whitelist evidence;
(d) Bound III end-to-end measure parity via the construction TPM.
Battery (c), reference goldens, lives in test_bounds_reference_golden.py.
"""

import dataclasses
import itertools

import pytest

from pyphi import config
from pyphi.conf import presets
from pyphi.formalism.iit4 import bounds
from pyphi.models.partitions import JointPartition
from pyphi.models.partitions import Part


class TestUpperBound:
    def test_float_protocol(self):
        bound = bounds.UpperBound(
            value=6, certified=True, assumptions=("binary units",), citation="Eq 6"
        )
        assert float(bound) == 6.0
        assert isinstance(float(bound), float)

    def test_frozen(self):
        bound = bounds.UpperBound(
            value=6, certified=True, assumptions=(), citation="Eq 6"
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            bound.value = 7  # pyright: ignore[reportAttributeAccessIssue]

    def test_integral_values_stay_int(self):
        bound = bounds.UpperBound(
            value=6, certified=True, assumptions=(), citation="Eq 6"
        )
        assert isinstance(bound.value, int)


class TestCounts:
    @pytest.mark.parametrize("n,expected", [(1, 1), (2, 3), (3, 7), (10, 1023)])
    def test_number_of_possible_distinctions(self, n, expected):
        assert bounds.number_of_possible_distinctions(n) == expected

    def test_number_of_possible_distinctions_of_order(self):
        assert bounds.number_of_possible_distinctions_of_order(4, 2) == 6
        total = sum(
            bounds.number_of_possible_distinctions_of_order(4, k) for k in range(1, 5)
        )
        assert total == bounds.number_of_possible_distinctions(4)

    @pytest.mark.parametrize("n,expected", [(1, 1), (2, 7), (3, 127)])
    def test_number_of_possible_relations(self, n, expected):
        # 2 ** (2 ** n - 1) - 1: nonempty subsets of candidate distinctions.
        assert bounds.number_of_possible_relations(n) == expected

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="positive"):
            bounds.number_of_possible_distinctions(0)
        with pytest.raises(ValueError, match="order"):
            bounds.number_of_possible_distinctions_of_order(3, 4)

    @staticmethod
    def _brute_force_face_counts(n):
        """Enumerate size->count of purview-slot subsets by exact overlap.

        Slots: every nonempty subset of units appears as exactly one cause
        purview and one effect purview (unique purviews). A candidate
        relation face is a subset of at least two slots with nonempty
        common overlap.
        """
        subsets = [
            frozenset(combo)
            for r in range(1, n + 1)
            for combo in itertools.combinations(range(n), r)
        ]
        slots = [*subsets, *subsets]  # cause slot + effect slot per purview
        counts = {}
        for r in range(2, len(slots) + 1):
            for combo in itertools.combinations(range(len(slots)), r):
                overlap = frozenset.intersection(*(slots[i] for i in combo))
                if overlap:
                    counts[len(overlap)] = counts.get(len(overlap), 0) + 1
        return counts

    @pytest.mark.parametrize("n", [2, 3])
    def test_relation_faces_with_unique_purviews_match_brute_force(self, n):
        expected = self._brute_force_face_counts(n)
        for k in range(1, n + 1):
            actual = (
                bounds.number_of_possible_relation_faces_with_unique_purviews_of_order(
                    n, k
                )
            )
            assert actual == expected.get(k, 0), f"n={n}, k={k}"
        assert bounds.number_of_possible_relation_faces_with_unique_purviews(n) == sum(
            expected.values()
        )

    def test_relation_faces_hand_values_n2(self):
        assert (
            bounds.number_of_possible_relation_faces_with_unique_purviews_of_order(2, 1)
            == 20
        )
        assert (
            bounds.number_of_possible_relation_faces_with_unique_purviews_of_order(2, 2)
            == 1
        )


class TestDomainGuard:
    def test_default_config_is_in_domain(self):
        # Default: IIT_4_0_2023 + GENERALIZED_INTRINSIC_DIFFERENCE.
        bound = bounds.distinction_phi_upper_bound((0, 1), (0, 1, 2))
        assert bound.value == 6

    def test_iit3_raises(self):
        with (
            config.override(**presets.iit3),
            pytest.raises(ValueError, match="Zaeemzadeh"),
        ):
            bounds.distinction_phi_upper_bound((0,), (0,))

    def test_unsupported_mechanism_measure_raises(self):
        with (
            config.override(mechanism_phi_measure="EMD"),
            pytest.raises(ValueError, match="EMD"),
        ):
            bounds.distinction_phi_upper_bound((0,), (0,))

    def test_system_guard_checks_partition_scheme(self):
        with (
            config.override(system_partition_scheme="DIRECTED_BIPARTITION"),
            pytest.raises(ValueError, match="partition scheme"),
        ):
            bounds.system_phi_upper_bound(3)

    def test_counts_are_measure_free(self):
        with config.override(**presets.iit3):
            assert bounds.number_of_possible_distinctions(3) == 7


class TestObjectBounds:
    def test_distinction_phi_upper_bound(self):
        bound = bounds.distinction_phi_upper_bound((0, 1), (1, 2, 3))
        assert bound.value == 6
        assert bound.certified
        assert bound.citation == "Theorem 1"
        assert "binary units" in bound.assumptions

    def test_distinction_phi_upper_bound_empty_raises(self):
        with pytest.raises(ValueError, match="nonempty"):
            bounds.distinction_phi_upper_bound((), (0,))
        with pytest.raises(ValueError, match="nonempty"):
            bounds.distinction_phi_upper_bound((0,), ())

    def test_partition_phi_upper_bound(self):
        # Bipartition of a 2-mechanism over itself severs 2 connections.
        partition = JointPartition(Part((0,), (0,)), Part((1,), (1,)))
        bound = bounds.partition_phi_upper_bound(partition)
        assert bound.value == partition.num_connections_cut() == 2
        assert bound.certified
        assert bound.citation == "Lemma 2"

    def test_relation_phi_upper_bound(self):
        bound = bounds.relation_phi_upper_bound([0.5, 2.0, 1.25])
        assert bound.value == 0.5
        assert bound.certified

    def test_relation_phi_upper_bound_empty_raises(self):
        with pytest.raises(ValueError, match="nonempty"):
            bounds.relation_phi_upper_bound([])

    def test_system_phi_upper_bound(self):
        bound = bounds.system_phi_upper_bound(4)
        assert bound.value == 12
        assert bound.certified
        assert bound.citation == "Table 2"
        assert any("self-connections" in a for a in bound.assumptions)


class TestSumPhiDistinctions:
    @pytest.mark.parametrize("n", range(1, 13))
    def test_bound_i_matches_brute_force(self, n):
        # Eq 6: every mechanism at phi = |M| * n (purview = whole system).
        brute = sum(
            len(mechanism) * n
            for r in range(1, n + 1)
            for mechanism in itertools.combinations(range(n), r)
        )
        bound = bounds.sum_phi_distinctions_upper_bound(n, bound="I")
        assert bound.value == brute == n * n * 2 ** (n - 1)
        assert bound.certified
        assert bound.citation == "Eq 6"
        assert isinstance(bound.value, int)

    @pytest.mark.parametrize("n", range(1, 13))
    def test_bound_ii_matches_brute_force(self, n):
        # Eq 7: every mechanism at phi = |M| ** 2 (purview = mechanism).
        brute = sum(
            len(mechanism) ** 2
            for r in range(1, n + 1)
            for mechanism in itertools.combinations(range(n), r)
        )
        bound = bounds.sum_phi_distinctions_upper_bound(n, bound="II")
        assert bound.value == brute == n * (n + 1) * 2**n // 4
        assert not bound.certified
        assert any("unique purviews" in a for a in bound.assumptions)
        assert bound.citation == "Eq 7"

    def test_phi_e_star_endpoints(self):
        # K = 1: a single self-copy unit; severing the self-connection
        # halves the probability: phi = 1. K = N: complete partition fully
        # marginalizes every unit: phi = N ** 2.
        for n in range(1, 8):
            assert bounds._phi_e_star(n, 1) == pytest.approx(1.0)
            assert bounds._phi_e_star(n, n) == pytest.approx(float(n * n))

    def test_phi_e_star_hand_value(self):
        # N=3, K=2: MIP is the non-self-cutting bipartition;
        # phi = -2 * log2(3/4). Verified against the 2.0 pipeline.
        assert bounds._phi_e_star(3, 2) == pytest.approx(0.8300749985576875, abs=1e-12)

    def test_phi_e_star_below_theorem_1(self):
        # Theorem 3: for 1 < K < N the construction cannot achieve K ** 2.
        for n in range(3, 9):
            for k in range(2, n):
                assert bounds._phi_e_star(n, k) < k * k

    def test_bound_iii_hand_values(self):
        assert bounds.sum_phi_distinctions_upper_bound(2, bound="III").value == (
            pytest.approx(6.0)
        )
        assert bounds.sum_phi_distinctions_upper_bound(3, bound="III").value == (
            pytest.approx(12 + 3 * 0.8300749985576875)
        )

    def test_bound_iii_certificate(self):
        bound = bounds.sum_phi_distinctions_upper_bound(4, bound="III")
        assert not bound.certified
        assert any("conjecture" in a for a in bound.assumptions)
        assert bound.citation == "Sec 2.1.3"

    @pytest.mark.parametrize("n", range(2, 9))
    def test_bound_ordering(self, n):
        # Fig 3: Bound III <= Bound II <= Bound I (equality at n = 2).
        bound_i = float(bounds.sum_phi_distinctions_upper_bound(n, bound="I"))
        bound_ii = float(bounds.sum_phi_distinctions_upper_bound(n, bound="II"))
        bound_iii = float(bounds.sum_phi_distinctions_upper_bound(n, bound="III"))
        assert bound_iii <= bound_ii + 1e-9
        assert bound_ii <= bound_i

    def test_invalid_bound_id_raises(self):
        with pytest.raises(ValueError, match="bound"):
            bounds.sum_phi_distinctions_upper_bound(3, bound="IV")
