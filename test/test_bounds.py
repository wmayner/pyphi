"""Tests for pyphi.formalism.iit4.bounds (Zaeemzadeh & Tononi 2024).

Validation batteries per the design spec:
(a) independent recomputation of the formulas (brute force, linprog);
(b) property tests against the real pipeline = domain-whitelist evidence;
(d) Bound III end-to-end measure parity via the construction TPM.
Battery (c), reference goldens, lives in test_bounds_reference_golden.py.
"""

import dataclasses
import itertools
import math

import numpy as np
import pytest
import scipy.optimize

from pyphi import config
from pyphi.conf import presets
from pyphi.examples import EXAMPLES
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


def _naive_subset_min_sum(values):
    """Brute force: sum of the minimum over all subsets of size >= 2."""
    values = list(values)
    total = 0.0
    for r in range(2, len(values) + 1):
        for combo in itertools.combinations(range(len(values)), r):
            total += min(values[i] for i in combo)
    return total


def _weighted_sorted_sum(values):
    """Eq 11 inner sum, expanded: ascending sort, i-th smallest element
    (0-based) is the minimum of 2**(R - 1 - i) - 1 subsets."""
    values = sorted(values)
    count = len(values)
    return sum(v * (2 ** (count - 1 - i) - 1) for i, v in enumerate(values))


def _table3_bound_i_nonself(n):
    """Verbatim Table 3 Bound I sum-of-relation-phi formula (no self term)."""
    total = 0
    for k in range(1, n + 1):
        exponent = sum(math.comb(n, i) for i in range(k, n + 1))
        group = math.comb(n, k)
        total += k * (2**exponent - 2 ** (exponent - group) - group)
    return n * total


def _table3_bound_ii_nonself(n):
    """Verbatim Table 3 Bound II sum-of-relation-phi formula (no self term)."""
    total = 0
    for k in range(1, n + 1):
        exponent = sum(math.comb(n - 1, i) for i in range(k - 1, n))
        group = math.comb(n - 1, k - 1)
        total += k * (2**exponent - 2 ** (exponent - group) - group)
    return n * total


class TestGroupedSubsetMinSum:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_matches_brute_force_bound_i_profile(self, n):
        expanded = [k for k in range(1, n + 1) for _ in range(math.comb(n, k))]
        grouped = bounds._grouped_subset_min_sum(
            [(k, math.comb(n, k)) for k in range(1, n + 1)]
        )
        assert grouped == _naive_subset_min_sum(expanded)
        assert grouped == _weighted_sorted_sum(expanded)

    @pytest.mark.parametrize("n", range(2, 9))
    def test_matches_weighted_sum_float_profile(self, n):
        # Bound III ratios are floats; agreement within float tolerance.
        ratios = [bounds._phi_e_star(n, k) / k for k in range(1, n + 1)]
        expanded = [
            ratios[k - 1] for k in range(1, n + 1) for _ in range(math.comb(n, k))
        ]
        grouped = bounds._grouped_subset_min_sum(
            [(ratios[k - 1], math.comb(n, k)) for k in range(1, n + 1)]
        )
        assert grouped == pytest.approx(_weighted_sorted_sum(expanded), rel=1e-12)


class TestSumPhiRelations:
    @pytest.mark.parametrize("n", range(1, 11))
    def test_bound_i_matches_table3_verbatim(self, n):
        bound = bounds.sum_phi_relations_upper_bound(n, bound="I")
        self_term = n * n * 2 ** (n - 1)
        assert bound.value == _table3_bound_i_nonself(n) + self_term
        assert isinstance(bound.value, int)
        assert not bound.certified

    @pytest.mark.parametrize("n", range(1, 11))
    def test_bound_ii_matches_table3_verbatim(self, n):
        bound = bounds.sum_phi_relations_upper_bound(n, bound="II")
        self_term = n * (n + 1) * 2**n // 4
        assert bound.value == _table3_bound_ii_nonself(n) + self_term

    def test_hand_values_n2(self):
        assert bounds.sum_phi_relations_upper_bound(2, bound="I").value == 16
        assert bounds.sum_phi_relations_upper_bound(2, bound="II").value == 8
        assert bounds.sum_phi_relations_upper_bound(2, bound="III").value == (
            pytest.approx(14.0)
        )
        general = bounds.sum_phi_relations_upper_bound(2, bound="GENERAL")
        assert float(general) == pytest.approx(88 / 3)
        assert general.certified

    def test_profile_bounds_are_conditional(self):
        for bound_id in ("I", "II", "III"):
            bound = bounds.sum_phi_relations_upper_bound(3, bound=bound_id)
            assert not bound.certified
            assert any("profile" in a for a in bound.assumptions)

    @pytest.mark.parametrize("n", range(2, 9))
    def test_general_dominates_profile_bound_i(self, n):
        # Eq 16 uses the LP maximum, which dominates any specific profile.
        general = float(bounds.sum_phi_relations_upper_bound(n, bound="GENERAL"))
        profile = float(bounds.sum_phi_relations_upper_bound(n, bound="I"))
        assert general >= profile

    def test_lp_closed_form_matches_linprog(self):
        # Eq 14: max sum(y_i (2**(R - i) - 1)) over ascending y >= 0 with
        # sum(y) <= S equals S ((2**R - 1) / R - 1).
        rng = np.random.default_rng(20260610)
        for _ in range(20):
            num_relata = int(rng.integers(2, 9))
            budget = float(rng.uniform(0.5, 50.0))
            coeffs = np.array(
                [2.0 ** (num_relata - i) - 1 for i in range(1, num_relata + 1)]
            )
            constraints = np.zeros((num_relata, num_relata))
            constraints[0] = 1.0  # budget row
            for i in range(1, num_relata):
                constraints[i, i - 1] = 1.0  # y_{i-1} <= y_i  (ascending)
                constraints[i, i] = -1.0
            limits = np.zeros(num_relata)
            limits[0] = budget
            result = scipy.optimize.linprog(
                c=-coeffs,
                A_ub=constraints,
                b_ub=limits,
                bounds=[(0, None)] * num_relata,
            )
            assert result.success
            expected = budget * ((2.0**num_relata - 1) / num_relata - 1)
            assert -result.fun == pytest.approx(expected, rel=1e-9)


class TestBigPhi:
    def test_general_is_certified(self):
        bound = bounds.big_phi_upper_bound(3, bound="GENERAL")
        assert bound.certified
        expected = float(bounds.sum_phi_distinctions_upper_bound(3, bound="I")) + (
            float(bounds.sum_phi_relations_upper_bound(3, bound="GENERAL"))
        )
        assert float(bound) == pytest.approx(expected)

    def test_profile_bounds_are_conditional(self):
        for bound_id in ("I", "II", "III"):
            assert not bounds.big_phi_upper_bound(3, bound=bound_id).certified


class TestConstructionHelpers:
    def test_construction_tpm_n3_k2(self):
        # Hand-checked rows (little-endian state order): a unit turns OFF
        # with probability 1 iff it is OFF and at least one other is OFF.
        tpm = bounds._construction_tpm(3, 2)
        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_array_equal(tpm, expected)

    def test_candidate_partition_count(self):
        for n in range(2, 8):
            for k in range(1, n):
                assert len(list(bounds._candidate_partitions(n, k))) == k // 2 + 1
            assert len(list(bounds._candidate_partitions(n, n))) == 1

    def test_candidate_partitions_sever_expected_connections(self):
        partitions = list(bounds._candidate_partitions(5, 4))
        # Bipartitions (1, 3), (2, 2): sever 2 j (k - j); special cut: k.
        assert sorted(p.num_connections_cut() for p in partitions) == [4, 6, 8]


class TestReport:
    def test_report_by_size(self):
        result = bounds.report(n=3)
        assert float(result["system_phi"]) == 6
        assert result["sum_phi_distinctions:I"].value == 36
        assert result["sum_phi_distinctions:II"].value == 24
        assert float(result["sum_phi_distinctions:III"]) == pytest.approx(
            12 + 3 * 0.8300749985576875
        )
        assert result["sum_phi_relations:GENERAL"].certified
        assert result["big_phi:GENERAL"].certified
        assert result["number_of_possible_distinctions"] == 7
        assert result["number_of_possible_relations"] == 127

    def test_report_requires_exactly_one_input(self):
        with pytest.raises(ValueError, match="exactly one"):
            bounds.report()
        with pytest.raises(ValueError, match="exactly one"):
            bounds.report(n=3, substrate=object())  # pyright: ignore[reportArgumentType]

    def test_report_from_substrate(self):
        substrate = EXAMPLES["substrate"]["basic"]()
        result = bounds.report(substrate=substrate)
        assert float(result["system_phi"]) == 6  # 3 binary units

    def test_report_rejects_nonbinary_substrate(self):
        class FakeTPM:
            alphabet_sizes = (2, 3)

        class FakeSubstrate:
            factored_tpm = FakeTPM()
            size = 2

        with pytest.raises(ValueError, match="binary"):
            bounds.report(substrate=FakeSubstrate())  # pyright: ignore[reportArgumentType]
