import pytest

from pyphi import System
from pyphi import config
from pyphi import examples
from pyphi.formalism.iit4 import ces
from pyphi.metrics.distribution import resolve_mechanism_measure
from pyphi.metrics.distribution import resolve_system_measure
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
            ((0, 1, 2), 0.5),
            ((1, 2), 2.0),
            ((0, 2), 1.0),
        ]
        for (got_nodes, got_phi), (exp_nodes, exp_phi) in zip(
            nodes_and_phis, expected, strict=True
        ):
            assert got_nodes == exp_nodes
            assert got_phi == pytest.approx(exp_phi, rel=1e-6)

    def test_complexes_standard(self, s):
        """Test ``complexes`` (non-overlapping maxima) for standard substrate.

        Greedy condensation over the three irreducible systems
        ``[(0,1,2):0.5, (1,2):2.0, (0,2):1.0]`` accepts ``(1,2)`` first
        (highest phi), then rejects both overlapping candidates, leaving
        a single complex.
        """
        cx = s.substrate.complexes(s.state)
        assert len(cx) == 1
        assert cx[0].node_indices == (1, 2)
        assert float(cx[0].phi) == pytest.approx(2.0, rel=1e-6)

    def test_all_sias_standard(self, s):
        """Test ``all_sias`` for standard substrate (IIT 3.0).

        Iterates over ``possible_complexes`` (not all ``2**n - 1`` subsets),
        so for the standard ``s`` fixture it returns 5 systems with phi
        values ``[0.0, 0.0, 0.5, 1.0, 2.0]`` — exactly three of which are
        irreducible.
        """
        sias = s.substrate.all_sias(s.state)
        assert len(sias) == 5
        phis = sorted(float(c.phi) for c in sias)
        assert phis == pytest.approx([0.0, 0.0, 0.5, 1.0, 2.0], rel=1e-6)
        assert sum(1 for phi in phis if phi > 0) == 3

    def test_maximal_complex(self, s):
        """Test ``maximal_complex`` for standard substrate (IIT 3.0)."""
        major = s.substrate.maximal_complex(s.state)
        assert float(major.phi) == pytest.approx(2.0, rel=1e-6)
        assert major.node_indices == (1, 2)

    @pytest.mark.slow
    @pytest.mark.outdated
    def test_all_complexes_parallelization(self, s):
        """Test that parallel and serial computation give same results (IIT 3.0).

        Known-broken on develop with the comment "TODO fix this horribly
        outdated mess that never worked in the first place :P". The failure
        is order-dependent: runs green in isolation but fails when run after
        other tests in the same session (likely cache/state leak from an
        earlier test). Marked ``outdated`` to match develop's posture — it
        should not run under the normal ``--slow`` lane until the underlying
        flake is diagnosed. Opt in with ``--outdated --slow`` for targeted
        investigation.
        """
        with config.override(parallel=False, progress_bars=False):
            serial = s.substrate.all_sias(s.state)
        with config.override(parallel=True, progress_bars=False):
            parallel = s.substrate.all_sias(s.state)
        assert sorted(serial, key=lambda x: x.phi) == sorted(
            parallel, key=lambda x: x.phi
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
