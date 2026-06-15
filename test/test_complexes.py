import pytest

from pyphi import System
from pyphi import config
from pyphi import examples
from pyphi.conf import presets
from pyphi.formalism import iit3
from pyphi.formalism.iit4 import ces
from pyphi.measures.distribution import resolve_mechanism_measure
from pyphi.measures.distribution import resolve_system_measure
from pyphi.substrate import possible_complexes

from .conftest import IIT_3_CONFIG


def test_possible_complexes(s):
    assert list(possible_complexes(s.substrate, s.state)) == [
        System.from_substrate(s.substrate, s.state, (0, 1, 2)),
        System.from_substrate(s.substrate, s.state, (1, 2)),
        System.from_substrate(s.substrate, s.state, (0, 2)),
        System.from_substrate(s.substrate, s.state, (0, 1)),
        System.from_substrate(s.substrate, s.state, (1,)),
    ]


# IIT 3.0 Regression Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The tests below were originally written for IIT 3.0 and validate that
# the library produces consistent results under that configuration.
# The IIT_3_CONFIG is defined in conftest.py and shared across test files.


class TestComplexesIIT30:
    """Regression tests for substrate complexes with IIT 3.0 configuration.

    These tests were originally written for IIT 3.0 and validate that
    the library produces consistent results under that configuration.
    """

    @pytest.fixture(autouse=True)
    def _apply_iit30_config(self):
        with IIT_3_CONFIG:
            yield

    def test_irreducible_sias_standard(self, s):
        """Test ``irreducible_sias`` for standard substrate (IIT 3.0).

        Under IIT 3.0 with ``DIRECTED_BI`` system partitions, the standard
        ``s`` fixture has exactly three irreducible candidate systems
        (|big_phi| > 0). Verify their phi values, node indices, and
        ordering (iteration in :func:`possible_complexes` order, not
        phi-sorted).
        """
        sias = s.substrate.irreducible_sias(s.state)
        assert len(sias) == 3
        nodes_and_phis = [(c.node_indices, float(c.phi)) for c in sias]
        expected = [
            ((0, 1, 2), 2.3125),
            ((1, 2), 1.0),
            ((0, 2), 0.5),
        ]
        for (got_nodes, got_phi), (exp_nodes, exp_phi) in zip(
            nodes_and_phis, expected, strict=True
        ):
            assert got_nodes == exp_nodes
            assert got_phi == pytest.approx(exp_phi, rel=1e-6)

    def test_complexes_standard(self, s):
        """Test ``complexes`` (non-overlapping maxima) for standard substrate.

        Greedy condensation over the three irreducible systems
        ``[(0,1,2):2.3125, (1,2):1.0, (0,2):0.5]`` accepts ``(0,1,2)`` first
        (highest phi), then rejects both overlapping candidates, leaving
        a single complex — the full substrate.
        """
        cx = s.substrate.complexes(s.state)
        assert len(cx) == 1
        assert cx[0].node_indices == (0, 1, 2)
        assert float(cx[0].phi) == pytest.approx(2.3125, rel=1e-6)

    def test_all_sias_standard(self, s):
        """Test ``all_sias`` for standard substrate (IIT 3.0).

        Iterates over ``possible_complexes`` (not all ``2**n - 1`` subsets),
        so for the standard ``s`` fixture it returns 5 systems with phi
        values ``[0.0, 0.0, 0.5, 1.0, 2.3125]`` — exactly three of which
        are irreducible.
        """
        sias = s.substrate.all_sias(s.state)
        assert len(sias) == 5
        phis = sorted(float(c.phi) for c in sias)
        assert phis == pytest.approx([0.0, 0.0, 0.5, 1.0, 2.3125], rel=1e-6)
        assert sum(1 for phi in phis if phi > 0) == 3

    def test_maximal_complex(self, s):
        """Test ``maximal_complex`` for standard substrate (IIT 3.0)."""
        major = s.substrate.maximal_complex(s.state)
        assert float(major.phi) == pytest.approx(2.3125, rel=1e-6)
        assert major.node_indices == (0, 1, 2)

    @pytest.mark.slow
    def test_all_complexes_parallelization(self, s):
        """Parallel and serial ``all_sias`` must produce identical results."""
        with config.override(parallel=False, progress_bars=False):
            serial = s.substrate.all_sias(s.state)
        with config.override(parallel=True, progress_bars=False):
            parallel = s.substrate.all_sias(s.state)
        assert sorted(serial, key=lambda x: x.phi) == sorted(
            parallel, key=lambda x: x.phi
        )

    def test_complexes_are_complex_objects_iit3(self, s):
        from pyphi.models.complex import Complex

        cx = s.substrate.complexes(s.state)
        assert isinstance(cx, tuple)
        assert len(cx) == 1
        assert isinstance(cx[0], Complex)
        assert cx[0].is_maximal is True
        assert cx[0].node_indices == (0, 1, 2)

    def test_complexes_excluded_iit3(self, s):
        # The single complex (0,1,2) excludes the overlapping lower-phi
        # irreducible candidates (1,2):1.0 and (0,2):0.5.
        cx = s.substrate.complexes(s.state)
        assert {e.node_indices for e in cx[0].excluded} == {(1, 2), (0, 2)}

    def test_maximal_complex_null_object_iit3(self, s):
        from pyphi.models.complex import Complex

        mc = s.substrate.maximal_complex(s.state, candidates=[])
        assert isinstance(mc, Complex)
        assert bool(mc) is False
        assert mc.node_indices == ()


class TestSiaCesConsistencyIIT30:
    """``iit3.sia(s).ces`` must equal ``iit3.ces(s)`` for every canonical
    substrate.

    ``iit3.sia`` builds the unpartitioned CES internally and attaches it to
    the returned ``SystemIrreducibilityAnalysis``. That CES must be the same
    object the standalone ``iit3.ces`` would compute for the same system.
    A divergence indicates the SIA's internal CES-building path is using a
    different mechanism set, partition scheme, measure, or parallel dispatch
    than the public ``ces()`` function.
    """

    @pytest.mark.parametrize(
        ("substrate_factory", "state"),
        [
            (examples.basic_substrate, (1, 0, 0)),
            (examples.xor_substrate, (0, 0, 0)),
            (examples.rule110_substrate, (1, 0, 1)),
            (examples.grid3_substrate, (1, 0, 0)),
        ],
        ids=["basic", "xor", "rule110", "grid3"],
    )
    def test_sia_ces_matches_standalone_ces(self, substrate_factory, state):
        """iit3.sia and iit3.ces produce the same SIA phi for the same system."""
        with config.override(**presets.iit3, progress_bars=False):
            substrate = substrate_factory()
            system = System.from_substrate(substrate, state, substrate.node_indices)
            ces_sia_phi = iit3.ces(system).sia.phi
            direct_sia_phi = iit3.sia(system).phi
        assert ces_sia_phi == direct_sia_phi, (
            f"iit3.ces(s).sia.phi diverged from iit3.sia(s).phi for "
            f"{substrate_factory.__name__} {state}: {ces_sia_phi} vs {direct_sia_phi}"
        )


# IIT 4.0 Golden Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# These tests validate the current IIT 4.0 behavior using new_big_phi.ces().


class TestCauseEffectStructureIIT40:
    """Golden tests for IIT 4.0 phi_structure computation.

    These tests validate the phi_structure function under IIT 4.0 defaults.
    """

    def _measure_kwargs(self):
        return {
            "system_measure": resolve_system_measure(
                config.formalism.iit.system_phi_measure
            ),
            "specification_measure": resolve_mechanism_measure(
                config.formalism.iit.specification_measure
            ),
        }

    def test_phi_structure_basic(self, s):
        """Golden test: phi_structure for basic system (IIT 4.0)."""
        result = ces(s, **self._measure_kwargs())

        # Golden values computed with IIT 4.0 defaults
        assert result.big_phi == pytest.approx(1.0, rel=1e-6)
        assert len(result.distinctions) == 2
        assert len(result.relations) == 0

    def test_phi_structure_system_creation(self):
        """Golden test: phi_structure with explicit system (IIT 4.0).

        Constructs the standard system explicitly (rather than via the
        ``s`` fixture) and verifies phi_structure returns the same big_phi
        and distinction count as :meth:`test_phi_structure_basic`.
        """
        substrate = examples.basic_substrate()
        state = (1, 0, 0)
        system = System(substrate, state)

        result = ces(system, **self._measure_kwargs())

        assert result.big_phi == pytest.approx(1.0, rel=1e-6)
        assert len(result.distinctions) == 2
        assert len(result.relations) == 0


class TestSubstrateMethodsIIT40:
    """Substrate-level method coverage under IIT 4.0.

    Exercises the unified substrate-method API (``all_sias``,
    ``irreducible_sias``, ``complexes``, ``maximal_complex``) which
    previously had no IIT 4.0 condensation entry point.
    """

    @staticmethod
    def _indices(sia):
        from pyphi.substrate import _sia_node_indices

        return _sia_node_indices(sia)

    def test_substrate_complexes_invariants(self, s):
        """``substrate.complexes(state)`` returns non-overlapping local maxima.

        Asserts the structural invariants every complexes list must
        satisfy under either formalism: positive phi, descending order,
        and pairwise-disjoint node sets.
        """
        cx = s.substrate.complexes(s.state)
        # Phi values positive and non-increasing.
        for c in cx:
            assert float(c.phi) > 0
        phis = [float(c.phi) for c in cx]
        assert phis == sorted(phis, reverse=True)
        # Pairwise-disjoint node indices.
        seen: set[int] = set()
        for c in cx:
            indices = self._indices(c)
            assert indices is not None
            nodes = set(indices)
            assert nodes.isdisjoint(seen)
            seen.update(nodes)

    def test_substrate_maximal_complex_matches_complexes_head(self, s):
        """``maximal_complex`` agrees with the head of ``complexes``."""
        cx = s.substrate.complexes(s.state)
        maximal = s.substrate.maximal_complex(s.state)
        if cx:
            assert self._indices(maximal) == self._indices(cx[0])
            assert float(maximal.phi) == pytest.approx(float(cx[0].phi), rel=1e-9)

    def test_dual_and_xor_substrate_exclusion_cascade(self):
        """``dual_and_xor`` in state ``(1,0,1,0)`` exercises the IIT 4.0
        substrate-exclusion cascade across two φ_s tiers with disjoint
        complexes plus overlap-filtering.

        ``irreducible_sias`` returns four candidates: two 2-node systems
        at ``φ_s=2.0`` over the disjoint pairs ``(0,1)`` and ``(2,3)``,
        and two single-node systems at ``φ_s=1.0`` over ``(1,)`` and
        ``(3,)``. The cascade:

        - Accepts both 2-node systems at the top tier (disjoint, no
          Composition escalation needed).
        - Filters out both single-node systems at the lower tier
          because their units overlap the already-accepted complexes.

        Asserts the algorithm yields exactly the two 2-node complexes.
        """
        from test.example_substrates import dual_and_xor_substrate

        substrate = dual_and_xor_substrate()
        state = (1, 0, 1, 0)
        cx = substrate.complexes(state)
        assert len(cx) == 2
        node_index_sets = {tuple(sorted(c.node_indices)) for c in cx}
        assert node_index_sets == {(0, 1), (2, 3)}
        for c in cx:
            assert float(c.phi) == pytest.approx(2.0, rel=1e-6)

    def test_substrate_all_sias_superset_of_irreducible(self, s):
        """``all_sias`` is a superset of ``irreducible_sias``."""
        all_ = s.substrate.all_sias(s.state)
        irr = s.substrate.irreducible_sias(s.state)

        def _key(c):
            indices = self._indices(c)
            assert indices is not None
            return tuple(indices)

        all_nodes = {_key(c) for c in all_}
        irr_nodes = {_key(c) for c in irr}
        assert irr_nodes.issubset(all_nodes)
        # Irreducible ones have phi > 0.
        for c in irr:
            assert float(c.phi) > 0


def test_iit3_ces_returns_cause_effect_structure(s):
    """iit3.ces returns a CauseEffectStructure with sia, distinctions, and
    NullRelations.
    """
    from pyphi.models.ces import CauseEffectStructure
    from pyphi.relations import NullRelations

    with config.override(**presets.iit3):
        result = iit3.ces(s)

    assert isinstance(result, CauseEffectStructure)
    assert result.sia is not None
    assert isinstance(result.relations, NullRelations)
    assert result.relations.num_relations() == 0
    assert result.distinctions is not None


def test_iit3_sia_no_longer_carries_unpartitioned_ces(s):
    """IIT 3.0 SIA stores partitioned_distinctions but not ces or substrate.

    The unpartitioned distinctions live on the wrapping CauseEffectStructure.
    """
    with config.override(**presets.iit3):
        sia = iit3.sia(s)

    assert not hasattr(sia, "ces"), (
        "sia.ces removed; access via iit3.ces(s).distinctions"
    )
    assert not hasattr(sia, "substrate"), (
        "sia.substrate removed; callers hold it externally"
    )
    assert not hasattr(sia, "partitioned_ces"), "renamed to partitioned_distinctions"
    assert hasattr(sia, "partitioned_distinctions"), "compute receipt of the MIP"
    assert sia.partitioned_distinctions is not None


class TestComplexWrapperIIT40:
    """B16: complexes() returns Complex objects under IIT 4.0."""

    def test_complexes_are_complex_objects(self, s):
        from pyphi.models.complex import Complex

        cx = s.substrate.complexes(s.state)
        assert isinstance(cx, tuple)
        assert all(isinstance(c, Complex) for c in cx)

    def test_exactly_one_is_maximal(self, s):
        cx = s.substrate.complexes(s.state)
        assert sum(1 for c in cx if c.is_maximal) == 1
        assert cx[0].is_maximal is True

    def test_dual_and_xor_excluded_records(self):
        from test.example_substrates import dual_and_xor_substrate

        substrate = dual_and_xor_substrate()
        cx = substrate.complexes((1, 0, 1, 0))
        assert {tuple(sorted(c.node_indices)) for c in cx} == {(0, 1), (2, 3)}
        by_units = {tuple(sorted(c.node_indices)): c for c in cx}
        # The single-node candidates (1,) and (3,) are excluded by the
        # 2-node complexes they overlap.
        assert {e.node_indices for e in by_units[(0, 1)].excluded} == {(1,)}
        assert {e.node_indices for e in by_units[(2, 3)].excluded} == {(3,)}


class TestMaximalComplexWrapperIIT40:
    """B16: maximal_complex() returns a Complex (null-object when empty)."""

    def test_maximal_complex_is_complex(self, s):
        from pyphi.models.complex import Complex

        mc = s.substrate.maximal_complex(s.state)
        assert isinstance(mc, Complex)
        assert mc.is_maximal is True
        assert mc.node_indices == s.substrate.complexes(s.state)[0].node_indices

    def test_maximal_complex_null_object(self, s):
        from pyphi.models.complex import Complex

        # Forcing an empty candidate set yields no complexes.
        mc = s.substrate.maximal_complex(s.state, candidates=[])
        assert isinstance(mc, Complex)
        assert bool(mc) is False
        assert mc.node_indices == ()
        assert float(mc.phi) == 0.0
        assert mc.is_maximal is True
        assert mc.excluded == ()
