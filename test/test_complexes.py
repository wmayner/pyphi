import pytest

from pyphi import System
from pyphi import config
from pyphi import examples
from pyphi.formalism import iit3
from pyphi.formalism.iit4 import phi_structure
from pyphi.metrics.distribution import resolve_mechanism_metric
from pyphi.metrics.distribution import resolve_system_metric
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

    def test_complexes_standard(self, s):
        """Test complexes computation for standard substrate (IIT 3.0).

        Under IIT 3.0 with ``DIRECTED_BI`` system partitions, the standard
        ``s`` fixture has exactly three irreducible complexes. Verify the
        full set: phi values, node indices, and ordering.
        """
        complexes = list(iit3.complexes(s.substrate, s.state))
        assert len(complexes) == 3
        # Verify each complex's phi and node_indices
        nodes_and_phis = [(c.system.node_indices, float(c.phi)) for c in complexes]
        # complexes() iterates in possible_complexes order, not phi-sorted
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

    def test_all_complexes_standard(self, s):
        """Test all_complexes computation for standard substrate (IIT 3.0).

        ``all_complexes`` iterates over ``possible_complexes`` (not all
        ``2**n - 1`` subsets), so for the standard ``s`` fixture it returns 5
        systems with phi values ``[0.0, 0.0, 0.5, 1.0, 2.0]`` — exactly
        three of which are irreducible (matching
        :meth:`test_complexes_standard`).
        """
        complexes = list(iit3.all_complexes(s.substrate, s.state))
        assert len(complexes) == 5
        phis = sorted(float(c.phi) for c in complexes)
        assert phis == pytest.approx([0.0, 0.0, 0.5, 1.0, 2.0], rel=1e-6)
        assert sum(1 for phi in phis if phi > 0) == 3

    def test_major_complex(self, s):
        """Test major_complex computation for standard substrate (IIT 3.0)."""
        major = iit3.major_complex(s.substrate, s.state)
        assert float(major.phi) == pytest.approx(2.0, rel=1e-6)
        assert major.system.node_indices == (1, 2)

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
            serial = list(iit3.all_complexes(s.substrate, s.state))
        with config.override(parallel=True, progress_bars=False):
            parallel = list(iit3.all_complexes(s.substrate, s.state))
        assert sorted(serial, key=lambda x: x.phi) == sorted(
            parallel, key=lambda x: x.phi
        )


# IIT 4.0 Golden Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# These tests validate the current IIT 4.0 behavior using new_big_phi.phi_structure().


class TestCauseEffectStructureIIT40:
    """Golden tests for IIT 4.0 phi_structure computation.

    These tests validate the phi_structure function under IIT 4.0 defaults.
    """

    def _metric_kwargs(self):
        return {
            "system_metric": resolve_system_metric(
                config.formalism.iit.system_phi_measure
            ),
            "specification_metric": resolve_mechanism_metric(
                config.formalism.iit.specification_measure
            ),
        }

    def test_phi_structure_basic(self, s):
        """Golden test: phi_structure for basic system (IIT 4.0)."""
        result = phi_structure(s, **self._metric_kwargs())

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

        result = phi_structure(system, **self._metric_kwargs())

        assert result.big_phi == pytest.approx(1.0, rel=1e-6)
        assert len(result.distinctions) == 2
        assert len(result.relations) == 0
