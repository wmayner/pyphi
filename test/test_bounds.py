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
import warnings

import numpy as np
import pytest
import scipy.optimize
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from pyphi import config
from pyphi.conf import presets
from pyphi.conf.formalism import IITConfig
from pyphi.direction import Direction
from pyphi.examples import EXAMPLES
from pyphi.exceptions import StateUnreachableError
from pyphi.formalism import iit4 as new_big_phi
from pyphi.formalism.iit4 import bounds
from pyphi.measures.distribution import resolve_mechanism_measure
from pyphi.measures.distribution import resolve_system_measure
from pyphi.models.partitions import JointPartition
from pyphi.models.partitions import Part
from pyphi.substrate import Substrate
from pyphi.system import System
from pyphi.utils import all_states

from .hypothesis_utils import small_system


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


# The whitelist admission evidence: under every (version, measure)
# combination shipped in the domain frozensets, structures computed by the
# real pipeline never exceed the certified bounds. A combination whose
# tests are not green here must be removed from the domain.
DOMAIN_CONFIGS = {
    "iit4_2023": presets.iit4_2023,
    "iit4_2026": presets.iit4_2026,  # system measure: INTRINSIC_INFORMATION
    "iit4_2026_gid_system": {"iit": IITConfig(version="IIT_4_0_2026")},
}

PROPERTY_SETTINGS = settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
        HealthCheck.data_too_large,
    ],
)

TOL = 1e-9


def _ces(system):
    return new_big_phi.ces(
        system,
        system_measure=resolve_system_measure(config.formalism.iit.system_phi_measure),
        specification_measure=resolve_mechanism_measure(
            config.formalism.iit.specification_measure
        ),
    )


def _assert_certified_bounds_hold(system):
    n = len(system.node_indices)
    ces = _ces(system)
    sum_phi_d = 0.0
    for distinction in ces.distinctions:
        phi = float(distinction.phi)
        sum_phi_d += phi
        for side in (distinction.cause, distinction.effect):
            side_phi = float(side.phi)
            # Theorem 1.
            theorem_1 = bounds.distinction_phi_upper_bound(
                distinction.mechanism, side.purview
            )
            assert side_phi <= float(theorem_1) + TOL
            # Lemma 2: phi is bounded by the connections the MIP severed.
            lemma_2 = bounds.partition_phi_upper_bound(side.partition)
            assert side_phi <= float(lemma_2) + TOL
    # Eq 6.
    assert sum_phi_d <= float(bounds.sum_phi_distinctions_upper_bound(n, bound="I")) + (
        TOL
    )
    # Relation bound. Partially structural in 2.0: Relation.phi is
    # |overlap| * min(phi_d / |purview union|), which is min-based by
    # construction; this is a consistency check, not independent evidence.
    sum_phi_r = 0.0
    for relation in ces.relations:  # pyright: ignore[reportGeneralTypeIssues]
        sum_phi_r += float(relation.phi)
        relata_phis = [float(distinction.phi) for distinction in relation]
        assert (
            float(relation.phi)
            <= float(bounds.relation_phi_upper_bound(relata_phis)) + TOL
        )
    # Eq 6 + Eq 16.
    assert (
        sum_phi_d + sum_phi_r
        <= float(bounds.big_phi_upper_bound(n, bound="GENERAL")) + TOL
    )
    return ces


class TestCertifiedBoundsAgainstPipeline:
    @pytest.mark.parametrize("config_name", sorted(DOMAIN_CONFIGS))
    @pytest.mark.parametrize("example_name", ["basic", "xor", "grid3"])
    def test_examples(self, config_name, example_name):
        system = EXAMPLES["system"][example_name]()
        with config.override(**DOMAIN_CONFIGS[config_name]):
            _assert_certified_bounds_hold(system)

    @pytest.mark.parametrize("config_name", sorted(DOMAIN_CONFIGS))
    def test_system_phi_bound_on_examples(self, config_name):
        system = EXAMPLES["system"]["basic"]()
        n = len(system.node_indices)
        with config.override(**DOMAIN_CONFIGS[config_name]):
            sia = new_big_phi.sia(
                system,
                system_measure=resolve_system_measure(
                    config.formalism.iit.system_phi_measure
                ),
                specification_measure=resolve_mechanism_measure(
                    config.formalism.iit.specification_measure
                ),
            )
            assert float(sia.phi) <= float(bounds.system_phi_upper_bound(n)) + TOL

    @pytest.mark.parametrize("config_name", sorted(DOMAIN_CONFIGS))
    @PROPERTY_SETTINGS
    @given(data=st.data())
    def test_random_systems(self, config_name, data):
        with config.override(
            **DOMAIN_CONFIGS[config_name], validate_system_states=False
        ):
            system = data.draw(small_system(min_size=2, max_size=3))
            n = len(system.node_indices)
            sum_phi_d = 0.0
            for distinction in system.distinctions():
                phi = float(distinction.phi)
                sum_phi_d += phi
                for side in (distinction.cause, distinction.effect):
                    theorem_1 = bounds.distinction_phi_upper_bound(
                        distinction.mechanism, side.purview
                    )
                    assert float(side.phi) <= float(theorem_1) + TOL
                    lemma_2 = bounds.partition_phi_upper_bound(side.partition)
                    assert float(side.phi) <= float(lemma_2) + TOL
            assert (
                sum_phi_d
                <= float(bounds.sum_phi_distinctions_upper_bound(n, bound="I")) + TOL
            )


class TestConjectureProbes:
    """Non-gating probes of the conditional/conjectured bounds.

    A genuine violation of Bound III on a real system would be a finding
    about the conjecture (its generality is an open question in the
    paper), not a test bug: report it, do not fail.
    """

    @staticmethod
    def _sum_phi_d_first_reachable_state(substrate, n):
        """Sum of distinction phi at the first state computable both ways.

        Mirrors the reachable-state scan in the paper's experiment code;
        returns None if no state of the random TPM supports a full
        cause-and-effect analysis.
        """
        for state in all_states(n):
            try:
                with config.override(validate_system_states=False):
                    system = System(substrate, state=state, node_indices=(0, 1, 2))
                    return sum(float(d.phi) for d in system.distinctions())
            except StateUnreachableError:
                continue
        return None

    def test_random_deterministic_systems(self):
        rng = np.random.default_rng(20260610)
        n = 3
        bound_values = {
            bound_id: float(bounds.sum_phi_distinctions_upper_bound(n, bound=bound_id))
            for bound_id in ("I", "II", "III")
        }
        violations = {"II": [], "III": []}
        analyzed = 0
        for trial in range(20):
            tpm = (rng.random((2**n, n)) > rng.random()).astype(float)
            substrate = Substrate(tpm, cm=np.ones((n, n)))
            sum_phi_d = self._sum_phi_d_first_reachable_state(substrate, n)
            if sum_phi_d is None:
                continue
            analyzed += 1
            # Certified: gating.
            assert sum_phi_d <= bound_values["I"] + TOL
            for bound_id in ("II", "III"):
                if sum_phi_d > bound_values[bound_id] + TOL:
                    violations[bound_id].append((trial, sum_phi_d))
        assert analyzed >= 10, f"only {analyzed}/20 random systems were analyzable"
        for bound_id, found in violations.items():
            if found:
                warnings.warn(
                    f"Bound {bound_id} exceeded by {len(found)}/20 random "
                    f"deterministic 3-unit systems: {found}. The bound is "
                    f"not certified; this is a data point about its "
                    f"domain, not a bug.",
                    stacklevel=1,
                )


class TestConstructionParity:
    """Battery (d): the closed-form Bound III machinery against the real
    pipeline on the construction TPM. This is the strongest check that
    2.0's measure semantics (binary GID, NUM_CONNECTIONS_CUT
    normalization) match the paper's intrinsic-difference setup."""

    @staticmethod
    def _construction_system(n, k):
        tpm = bounds._construction_tpm(n, k)
        substrate = Substrate(tpm, cm=np.ones((n, n)))
        return System(substrate, state=(0,) * n, node_indices=tuple(range(n)))

    @pytest.mark.parametrize("n,k", [(n, k) for n in (2, 3, 4) for k in range(1, n + 1)])
    def test_phi_e_star_matches_pipeline_on_candidates(self, n, k):
        system = self._construction_system(n, k)
        mechanism = tuple(range(k))
        mip = system.find_mip(
            Direction.EFFECT,
            mechanism,
            mechanism,
            partitions=list(bounds._candidate_partitions(n, k)),
        )
        assert float(mip.phi) == pytest.approx(bounds._phi_e_star(n, k), abs=1e-10)

    @pytest.mark.parametrize("n,k", [(3, 2), (4, 2), (4, 3)])
    def test_candidates_contain_the_global_mip(self, n, k):
        # The S3 narrowing argument: the MIP over ALL mechanism partitions
        # equals the MIP over the k // 2 + 1 candidates.
        system = self._construction_system(n, k)
        mechanism = tuple(range(k))
        full_mip = system.find_mip(Direction.EFFECT, mechanism, mechanism)
        assert float(full_mip.phi) == pytest.approx(bounds._phi_e_star(n, k), abs=1e-10)

    @pytest.mark.parametrize("n,k_star", [(3, 2), (3, 3)])
    def test_achieved_sum_phi_e_below_bound_iii(self, n, k_star):
        # The full structure of the construction must respect Bound III
        # (Fig 2: the bound is tight at the best k_star).
        system = self._construction_system(n, k_star)
        bound_iii = float(bounds.sum_phi_distinctions_upper_bound(n, bound="III"))
        achieved = 0.0
        with config.override(validate_system_states=False):
            for distinction in system.distinctions():
                achieved += float(distinction.effect.phi)
        assert achieved <= bound_iii + 1e-9
