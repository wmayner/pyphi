import numpy as np

from pyphi import examples
from pyphi.conf import config
from pyphi.conf import presets

# Unconstrained cause repertoire tests
# ====================================

# Matlab default substrate
# ~~~~~~~~~~~~~~~~~~~~~~


def test_unconstrained_cause_repertoire_matlab_0(s):
    # Purview {m0}
    purview = (0,)
    assert np.array_equal(
        s.unconstrained_cause_repertoire(
            purview,
        ),
        np.array([[[0.5]], [[0.5]]]),
    )


def test_unconstrained_cause_repertoire_matlab_1(s):
    purview = (0, 1)
    assert np.array_equal(
        s.unconstrained_cause_repertoire(purview),
        np.array([[[0.25], [0.25]], [[0.25], [0.25]]]),
    )


def test_unconstrained_cause_repertoire_matlab_2(s):
    purview = (0, 1, 2)
    # fmt: off
    answer = np.array([
        [[0.125, 0.125],
         [0.125, 0.125]],
        [[0.125, 0.125],
         [0.125, 0.125]],
    ])
    # fmt: on
    assert np.array_equal(s.unconstrained_cause_repertoire(purview), answer)


# Unconstrained effect repertoire tests
# =====================================

# Matlab default substrate
# ~~~~~~~~~~~~~~~~~~~~~~


def test_unconstrained_effect_repertoire_matlab_0(s):
    purview = (0,)
    assert np.array_equal(
        s.unconstrained_effect_repertoire(purview), np.array([[[0.25]], [[0.75]]])
    )


def test_unconstrained_effect_repertoire_matlab_1(s):
    purview = (0, 1)
    assert np.array_equal(
        s.unconstrained_effect_repertoire(purview),
        np.array([[[0.125], [0.125]], [[0.375], [0.375]]]),
    )


def test_unconstrained_effect_repertoire_matlab_2(s):
    purview = (0, 1, 2)
    # fmt: off
    answer = np.array([
        [[0.0625, 0.0625],
         [0.0625, 0.0625]],
        [[0.1875, 0.1875],
         [0.1875, 0.1875]],
    ])
    # fmt: on
    assert np.array_equal(s.unconstrained_effect_repertoire(purview), answer)


# Shape contract on subsystems with external (background) nodes
# =============================================================
#
# A repertoire carries one axis per *substrate* node (background nodes are
# unitary axes), regardless of how many nodes are in the system. The
# unconstrained (empty-mechanism) cause repertoire must honor this like every
# other repertoire, so that products of per-part repertoires (e.g. in
# ``partitioned_repertoire``) broadcast over matching axes.


def test_unconstrained_cause_repertoire_full_ndim_on_subsystem(subsys_n0n2):
    # subsys_n0n2 spans nodes (0, 2) of a 3-node substrate; node 1 is external.
    assert subsys_n0n2.external_indices == (1,)
    repertoire = subsys_n0n2.unconstrained_cause_repertoire((0,))
    # One axis per substrate node: (2, 1, 1), not the system-only (2, 1).
    assert repertoire.shape == (2, 1, 1)


def test_unconstrained_cause_and_effect_same_ndim_on_subsystem(subsys_n0n2):
    # Cause and effect unconstrained repertoires over the same purview must
    # agree in dimensionality (both span the full substrate).
    for purview in [(0,), (2,), (0, 2)]:
        cause = subsys_n0n2.unconstrained_cause_repertoire(purview)
        effect = subsys_n0n2.unconstrained_effect_repertoire(purview)
        assert cause.ndim == effect.ndim == len(subsys_n0n2.substrate.node_indices)
        assert cause.shape == effect.shape


# IIT 3.0 complexes over a fully-connected substrate
# ==================================================
#
# Dense connectivity generates mechanism partitions whose empty-mechanism
# cause part has a non-trailing purview; with the wrong repertoire ndim their
# product collapses and the repertoire-distance shape check raises. This
# exercises the candidate subsystems with external nodes end to end.


def test_iit3_maximal_complex_fully_connected_substrate():
    system = examples.rule110_system()
    with config.override(**presets.iit3):
        complex_ = system.substrate.maximal_complex(system.state)
        assert tuple(complex_.node_indices) == (0, 1, 2)
        assert abs(float(complex_.phi) - 1.357083) < 1e-4
