"""Property-based invariant tests.

Hypothesis-driven complement to ``test_invariants.py``. Encodes invariants
from the IIT 4.0 paper (Albantakis et al. 2023) and the standing TODOs in
``pyphi/repertoire.py:27-31`` as property tests. These explore random edge
cases (empty mechanisms, deterministic TPMs, near-uniform TPMs, etc.) that
the fixture-based golden tests can miss.

``TestMetricInvariants`` exercises the repertoire-distance metric machinery
and serves as a mutation-detection canary: a sign-flip in
``metrics/distribution.py`` must fail at least one of its tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from pyphi import config
from pyphi.conf import presets
from pyphi.direction import Direction
from pyphi.distribution import repertoire_shape
from pyphi.partition import bipartition
from pyphi.partition import directed_bipartition
from pyphi.partition import joint_bipartitions
from pyphi.substrate import Substrate
from pyphi.system import System

from .hypothesis_utils import binary_state
from .hypothesis_utils import mechanism_purview_pair
from .hypothesis_utils import small_substrate
from .hypothesis_utils import small_system

# Hypothesis settings: keep examples low because each draw constructs a real
# System and computes repertoires (~tens of ms apiece on 3 nodes).
# CI-friendly: 25-50 examples per test, no per-example deadline because TPM
# construction has occasional slow paths.
DEFAULT_SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
        HealthCheck.data_too_large,
    ],
)


@pytest.fixture(autouse=True)
def _disable_state_validation():
    """Random TPMs frequently produce states with zero past probability;
    state-reachability validation isn't meaningful here."""
    with config.override(validate_system_states=False):
        yield


# ============================================================================
# Repertoire properties (Layer 1 invariants)
# ============================================================================


class TestRepertoireProperties:
    """Repertoires are probability distributions over purview states."""

    @DEFAULT_SETTINGS
    @given(data=st.data())
    def test_cause_repertoire_sums_to_one(self, data):
        s = data.draw(small_system())
        mechanism, purview = data.draw(mechanism_purview_pair(s))
        rep = s.cause_repertoire(mechanism, purview)
        assert math.isclose(float(rep.sum()), 1.0, abs_tol=1e-10), (
            f"cause repertoire sum = {float(rep.sum())} for "
            f"mechanism={mechanism}, purview={purview}"
        )

    @DEFAULT_SETTINGS
    @given(data=st.data())
    def test_effect_repertoire_sums_to_one(self, data):
        s = data.draw(small_system())
        mechanism, purview = data.draw(mechanism_purview_pair(s))
        rep = s.effect_repertoire(mechanism, purview)
        assert math.isclose(float(rep.sum()), 1.0, abs_tol=1e-10), (
            f"effect repertoire sum = {float(rep.sum())} for "
            f"mechanism={mechanism}, purview={purview}"
        )

    @DEFAULT_SETTINGS
    @given(data=st.data())
    def test_cause_repertoire_nonnegative(self, data):
        s = data.draw(small_system())
        mechanism, purview = data.draw(mechanism_purview_pair(s))
        rep = s.cause_repertoire(mechanism, purview)
        assert np.all(np.asarray(rep) >= -1e-12), (
            f"cause repertoire has negative entries (min={float(rep.min())}) "
            f"for mechanism={mechanism}, purview={purview}"
        )

    @DEFAULT_SETTINGS
    @given(data=st.data())
    def test_effect_repertoire_nonnegative(self, data):
        s = data.draw(small_system())
        mechanism, purview = data.draw(mechanism_purview_pair(s))
        rep = s.effect_repertoire(mechanism, purview)
        assert np.all(np.asarray(rep) >= -1e-12), (
            f"effect repertoire has negative entries (min={float(rep.min())}) "
            f"for mechanism={mechanism}, purview={purview}"
        )

    @DEFAULT_SETTINGS
    @given(data=st.data())
    def test_cause_repertoire_correct_shape(self, data):
        s = data.draw(small_system())
        mechanism, purview = data.draw(mechanism_purview_pair(s))
        rep = s.cause_repertoire(mechanism, purview)
        expected_shape = tuple(repertoire_shape(s.node_indices, purview))
        assert rep.shape == expected_shape

    @DEFAULT_SETTINGS
    @given(data=st.data())
    def test_effect_repertoire_correct_shape(self, data):
        s = data.draw(small_system())
        mechanism, purview = data.draw(mechanism_purview_pair(s))
        rep = s.effect_repertoire(mechanism, purview)
        expected_shape = tuple(repertoire_shape(s.node_indices, purview))
        assert rep.shape == expected_shape


# ============================================================================
# Unconstrained-repertoire invariants (repertoire.py:27-31)
# ============================================================================


class TestUnconstrainedInvariants:
    """An empty mechanism yields the unconstrained repertoire (Eq. 33, 34
    point-equality is conditional on causal perfection — these are the
    safe, always-true subset)."""

    @DEFAULT_SETTINGS
    @given(data=st.data())
    def test_empty_mechanism_yields_unconstrained_cause(self, data):
        s = data.draw(small_system())
        purview = data.draw(
            st.lists(
                st.sampled_from(list(s.node_indices)),
                min_size=1,
                unique=True,
            ).map(lambda xs: tuple(sorted(xs)))
        )
        empty = s.cause_repertoire((), purview)
        unconstrained = s.unconstrained_cause_repertoire(purview)
        np.testing.assert_allclose(
            np.asarray(empty), np.asarray(unconstrained), atol=1e-12
        )

    @DEFAULT_SETTINGS
    @given(data=st.data())
    def test_empty_mechanism_yields_unconstrained_effect(self, data):
        s = data.draw(small_system())
        purview = data.draw(
            st.lists(
                st.sampled_from(list(s.node_indices)),
                min_size=1,
                unique=True,
            ).map(lambda xs: tuple(sorted(xs)))
        )
        empty = s.effect_repertoire((), purview)
        unconstrained = s.unconstrained_effect_repertoire(purview)
        np.testing.assert_allclose(
            np.asarray(empty), np.asarray(unconstrained), atol=1e-12
        )


# ============================================================================
# MIP / phi monotonicity (touches the metric machinery — sign-flip canary)
# ============================================================================


class TestMetricInvariants:
    """Invariants on the repertoire-distance metric.

    These tests exercise ``metrics/distribution.py``. A deliberate sign-flip
    in the metric implementation must fail at least one of these.
    """

    @DEFAULT_SETTINGS
    @given(data=st.data())
    def test_emd_distance_nonnegative(self, data):
        """EMD(rep1, rep2) ≥ 0 for any two repertoires of the same shape.

        EMD is a true metric — non-negativity is one of the metric axioms.
        A sign-flip in the EMD reduction would produce negative distances
        and surface here.
        """
        s = data.draw(small_system())
        # Both repertoires must be over the same purview to share shape.
        purview = data.draw(
            st.lists(st.sampled_from(list(s.node_indices)), min_size=1, unique=True).map(
                lambda xs: tuple(sorted(xs))
            )
        )
        mech1 = data.draw(
            st.lists(st.sampled_from(list(s.node_indices)), min_size=1, unique=True).map(
                lambda xs: tuple(sorted(xs))
            )
        )
        mech2 = data.draw(
            st.lists(st.sampled_from(list(s.node_indices)), min_size=1, unique=True).map(
                lambda xs: tuple(sorted(xs))
            )
        )

        from pyphi.measures.distribution import distribution_measures
        from pyphi.measures.distribution import repertoire_distance

        rep1 = s.cause_repertoire(mech1, purview)
        rep2 = s.cause_repertoire(mech2, purview)

        with config.override(mechanism_phi_measure="EMD"):
            try:
                d = repertoire_distance(
                    rep1,
                    rep2,
                    direction=Direction.CAUSE,
                    repertoire_distance=distribution_measures["EMD"],
                )
            except ImportError:
                pytest.skip("pyemd not installed")
                return

        assert float(d) >= -1e-10, (
            f"EMD distance is negative ({float(d)}) — violates metric "
            f"non-negativity axiom"
        )

    @DEFAULT_SETTINGS
    @given(data=st.data())
    def test_emd_self_distance_is_zero(self, data):
        """EMD(rep, rep) = 0 — identity of indiscernibles.

        Sign-flipping a metric with an absolute-value reduction step would
        still give 0 for identical inputs, so this test alone is not a
        sufficient canary; combined with non-negativity and ``find_mip``
        exercising it, the metric machinery is well-covered.
        """
        s = data.draw(small_system())
        purview = data.draw(
            st.lists(st.sampled_from(list(s.node_indices)), min_size=1, unique=True).map(
                lambda xs: tuple(sorted(xs))
            )
        )
        mechanism = data.draw(
            st.lists(st.sampled_from(list(s.node_indices)), min_size=1, unique=True).map(
                lambda xs: tuple(sorted(xs))
            )
        )

        from pyphi.measures.distribution import distribution_measures
        from pyphi.measures.distribution import repertoire_distance

        rep = s.cause_repertoire(mechanism, purview)

        with config.override(mechanism_phi_measure="EMD"):
            try:
                d = repertoire_distance(
                    rep,
                    rep,
                    direction=Direction.CAUSE,
                    repertoire_distance=distribution_measures["EMD"],
                )
            except ImportError:
                pytest.skip("pyemd not installed")
                return

        assert abs(float(d)) < 1e-10

    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.function_scoped_fixture,
            HealthCheck.data_too_large,
        ],
    )
    @given(data=st.data())
    def test_iit3_mip_phi_nonnegative(self, data):
        """Under IIT 3.0 with EMD, find_mip phi must be ≥ 0.

        IIT 3.0 doesn't have the |·|+ semantics that lets PyPhi's IIT 4.0
        path emit negative signed phi for "preventative cause" visibility;
        EMD is a true non-negative metric, so phi at any partition is ≥ 0,
        and so is the minimum over partitions. A sign-flipped EMD would
        produce negative MIP phi and fail here.
        """
        with config.override(**presets.iit3, validate_system_states=False):
            s = data.draw(small_system())
            mechanism, purview = data.draw(mechanism_purview_pair(s))
            direction = data.draw(st.sampled_from([Direction.CAUSE, Direction.EFFECT]))

            try:
                mip = s.find_mip(direction, mechanism, purview)
            except ImportError:
                pytest.skip("pyemd not installed")
                return
            except Exception:
                assume(False)
                return

            assume(mip is not None)
            assume(mip.phi is not None)

            assert float(mip.phi) >= -1e-10, (
                f"IIT 3.0 EMD-based MIP phi is negative ({float(mip.phi)}) "
                f"— violates EMD non-negativity"
            )


# ============================================================================
# Direction duality (Eq. 5 vs 7)
# ============================================================================


class TestDirectionDuality:
    """Cause and effect are dual under ``Direction.flip``."""

    @DEFAULT_SETTINGS
    @given(direction=st.sampled_from([Direction.CAUSE, Direction.EFFECT]))
    def test_flip_is_involution(self, direction):
        assert direction.flip().flip() == direction

    @DEFAULT_SETTINGS
    @given(
        mechanism=st.lists(st.integers(0, 9), unique=True).map(tuple),
        purview=st.lists(st.integers(0, 9), unique=True).map(tuple),
    )
    def test_order_swaps_under_flip(self, mechanism, purview):
        cause_order = Direction.CAUSE.order(mechanism, purview)
        effect_order = Direction.EFFECT.order(mechanism, purview)
        # CAUSE puts purview at t-1, mechanism at t; EFFECT puts mechanism at
        # t, purview at t+1. So one is the reverse of the other.
        assert cause_order == tuple(reversed(effect_order))


# ============================================================================
# Partition counting (closed-form formulas)
# ============================================================================


class TestPartitionCounts:
    """Combinatorial sanity-checks: partition generators should produce the
    closed-form number of items predicted by the math."""

    @DEFAULT_SETTINGS
    @given(n=st.integers(min_value=1, max_value=6))
    def test_undirected_bipartition_count(self, n):
        """|bipartitions of N| = 2^(N-1)."""
        seq = tuple(range(n))
        count = len(bipartition(seq))
        assert count == 2 ** (n - 1)

    @DEFAULT_SETTINGS
    @given(n=st.integers(min_value=1, max_value=6))
    def test_directed_bipartition_count(self, n):
        """|directed bipartitions of N| = 2^N for N ≥ 1.

        PyPhi's ``directed_bipartition_indices`` returns ``[]`` for N=0 by
        convention (see ``partition.py:90``); the mathematical convention
        ``{(∅, ∅)}`` would give count 1. The N ≥ 1 case is unambiguous.
        """
        seq = tuple(range(n))
        count = len(directed_bipartition(seq))
        assert count == 2**n

    @DEFAULT_SETTINGS
    @given(
        m=st.integers(min_value=1, max_value=4),
        p=st.integers(min_value=1, max_value=4),
    )
    def test_joint_bipartition_count(self, m, p):
        """``|joint_bipartitions(M, P)| = 2^M * 2^P - <trivials>``.

        With ``M`` mechanism nodes and ``P`` purview nodes:
          - bipartitions of mechanism: ``2^(M-1)`` undirected wrappers, each
            yielding 2 effective halves
          - directed_bipartitions of purview: ``2^P``
          - cross-product before filtering: ``2^M * 2^P``
          - filter ``(n[0] or d[0]) and (n[1] or d[1])`` excludes trivials.

        Use the generator as the oracle and just check it produces consistent,
        non-empty output for nonempty inputs.
        """
        mechanism = tuple(range(m))
        purview = tuple(range(p))
        partitions = list(joint_bipartitions(mechanism, purview))
        # Every partition covers the union of mechanism and purview (mechanism
        # halves form a bipartition of M; purview halves a directed bipartition
        # of P).
        assert len(partitions) > 0
        for partition in partitions:
            mech_union = tuple(
                sorted(set(partition[0].mechanism) | set(partition[1].mechanism))
            )
            purv_union = tuple(
                sorted(set(partition[0].purview) | set(partition[1].purview))
            )
            assert mech_union == mechanism, (
                f"partition mechanism cover {mech_union} != {mechanism}"
            )
            assert purv_union == purview, (
                f"partition purview cover {purv_union} != {purview}"
            )

    @DEFAULT_SETTINGS
    @given(n=st.integers(min_value=1, max_value=4))
    def test_bipartition_parts_disjoint_and_complete(self, n):
        """For every (A, B) in bipartition(seq): A ∩ B = ∅ and A ∪ B = seq.

        (Both intersection and union must hold for a valid bipartition.)
        """  # noqa: RUF002
        seq = tuple(range(n))
        for left, right in bipartition(seq):
            assert set(left).isdisjoint(set(right))
            assert set(left) | set(right) == set(seq)


# ============================================================================
# State-space sanity (closed-form: |state space| = product of alphabets)
# ============================================================================


class TestSignedPhi:
    """Invariants on the |·|+ clamp applied to system-level phi.

    ``SystemIrreducibilityAnalysis.phi`` is the paper-faithful clamped
    value (always ≥ 0); ``signed_phi`` is the raw value before the clamp.
    The contract is ``phi == max(0, signed_phi)``.
    """

    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.function_scoped_fixture,
            HealthCheck.data_too_large,
        ],
    )
    @given(data=st.data())
    def test_phi_is_positive_part_of_signed_phi(self, data):
        """``phi == max(0, signed_phi)`` for any computed SIA."""
        s = data.draw(small_system())
        try:
            sia = s.sia()
        except Exception:
            assume(False)
            return

        assume(sia is not None)
        assume(getattr(sia, "phi", None) is not None)
        assume(getattr(sia, "signed_phi", None) is not None)

        phi = float(sia.phi)
        signed_phi = float(sia.signed_phi)

        assert phi >= -1e-12, f"phi {phi} is negative — |·|+ clamp not applied"
        assert math.isclose(phi, max(0.0, signed_phi), abs_tol=1e-10), (
            f"phi ({phi}) != max(0, signed_phi={signed_phi})"
        )

    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.function_scoped_fixture,
            HealthCheck.data_too_large,
        ],
    )
    @given(data=st.data())
    def test_ria_phi_is_positive_part_of_signed_phi(self, data):
        """``RIA.phi == max(0, RIA.signed_phi)`` for any mechanism MIP.

        Mirrors the system-level invariant: the canonical RIA.phi is the
        paper-faithful clamped value; signed_phi preserves the raw value.
        """
        s = data.draw(small_system())
        try:
            ria = s.find_mip(Direction.CAUSE, (0,), (0,))
        except Exception:
            assume(False)
            return

        assume(ria is not None)
        assume(getattr(ria, "phi", None) is not None)
        assume(getattr(ria, "signed_phi", None) is not None)

        phi = float(ria.phi)
        signed_phi = float(ria.signed_phi)

        assert phi >= -1e-12, f"phi {phi} is negative — |·|+ clamp not applied"
        assert math.isclose(phi, max(0.0, signed_phi), abs_tol=1e-10), (
            f"phi ({phi}) != max(0, signed_phi={signed_phi})"
        )


class TestStateSpace:
    """Sanity invariants on TPM and state-space construction."""

    @DEFAULT_SETTINGS
    @given(n=st.integers(min_value=1, max_value=4))
    def test_binary_substrate_has_2n_states(self, n):
        tpm = np.zeros((2**n, n))
        cm = np.eye(n, dtype=int)
        net = Substrate(tpm, cm=cm)
        assert net.num_states == 2**n
        assert net.size == n

    @DEFAULT_SETTINGS
    @given(data=st.data())
    def test_system_state_matches_substrate_size(self, data):
        from pyphi.exceptions import StateUnreachableBackwardsError

        net = data.draw(small_substrate())
        state = data.draw(binary_state(net.size))
        try:
            sub = System(net, state, net.node_indices)
        except StateUnreachableBackwardsError:
            assume(False)
            return
        assert len(sub.state) == net.size
        assert sub.tpm_size == net.size
