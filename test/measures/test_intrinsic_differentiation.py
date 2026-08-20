from pyphi import Direction
from pyphi import config
from pyphi.examples import EXAMPLES
from pyphi.measures.distribution import resolve_mechanism_measure


def test_intrinsic_information():
    with config.override(specification_measure="INTRINSIC_SPECIFICATION"):
        system = EXAMPLES["system"]["differentiation_micro_1"]()
        mechanism = (0, 1)
        result = system.intrinsic_information(
            Direction.CAUSE,
            mechanism,
            mechanism,
            specification_measure=resolve_mechanism_measure(
                config.formalism.iit.specification_measure
            ),
        )
        assert result.state == (1, 1)
        # Exact computed value (bitwise).
        assert result.intrinsic_information == 1.8857840667050532


def test_composite_intrinsic_information_state_none_is_per_state():
    """``state=None`` yields the per-state vector, matching per-state calls.

    The differentiation operand of the composite must be the per-state
    surprisal (Mayner et al. 2026, Eq. 4: i_diff is defined per cause/effect
    state), not one global minimum broadcast over all states.
    """
    import numpy as np

    from pyphi.measures import distribution as d

    forward = np.array([0.5, 0.4, 0.05, 0.05])
    unconstrained = np.array([0.25, 0.01, 0.37, 0.37])
    selectivity = np.ones(4)

    array = np.asarray(
        d.intrinsic_information(forward, unconstrained, selectivity, state=None)
    ).squeeze()
    per_state = [
        float(d.intrinsic_information(forward, unconstrained, selectivity, state=(i,)))
        for i in range(4)
    ]
    assert np.allclose(array, per_state)


def test_composite_cause_side_uses_bayes_posterior():
    """The differentiation operand is the cause repertoire (Eq. 11 posterior),
    not the unnormalized forward likelihoods.

    On the cause side the forward repertoire sums to Z != 1; taking the
    surprisal of it overstates i_diff by -log2(Z). The composite must use the
    selectivity repertoire (the Bayes posterior) instead.
    """
    import numpy as np

    from pyphi.measures import distribution as d

    # Unnormalized cause-side likelihoods (Z = 0.8) and their posterior.
    # The unconstrained repertoire is small at state 0, so the specification
    # term there is large and the differentiation term binds -- the ii value
    # differs between the two conventions (the power check below asserts so).
    forward = np.array([0.72, 0.04, 0.024, 0.016])
    posterior = forward / forward.sum()
    unconstrained = np.array([0.05, 0.35, 0.3, 0.3])

    ii = np.asarray(
        d.intrinsic_information(forward, unconstrained, posterior, state=None)
    ).squeeze()
    spec = np.asarray(
        d.generalized_intrinsic_difference(forward, unconstrained, posterior, state=None)
    ).squeeze()
    correct = np.minimum(spec, d.pointwise_intrinsic_differentiation(posterior))
    wrong = np.minimum(spec, d.pointwise_intrinsic_differentiation(forward))
    # The two conventions genuinely disagree on this input (power check).
    assert not np.allclose(correct, wrong)
    assert np.allclose(ii, correct)


def test_composite_keeps_repertoire_rank():
    """The composite's operands align axis by axis at the repertoire's
    canonical rank; a squeezed differentiation operand would broadcast
    against singleton (non-purview) axes and produce wrong values and
    wrong-rank specified states.
    """
    import numpy as np

    from pyphi.measures import distribution as d

    rng_shape = (2, 1, 2)  # purview nodes 0 and 2; node 1 is a singleton axis
    forward = np.full(rng_shape, 0.25)
    unconstrained = np.full(rng_shape, 0.25)
    selectivity = np.full(rng_shape, 0.25)
    out = np.asarray(
        d.intrinsic_information(forward, unconstrained, selectivity, state=None)
    )
    assert out.shape == rng_shape


def test_public_intrinsic_information_purview_rank_and_value():
    """End to end on ``basic_system``: the specified state has purview
    length, and the cause-side value matches the by-hand Eq. 6/11
    computation min(GID, surprisal of the posterior)."""
    import numpy as np

    from pyphi import examples
    from pyphi.conf import presets
    from pyphi.core.repertoire_algebra import forward_repertoire
    from pyphi.core.repertoire_algebra import unconstrained_forward_repertoire
    from pyphi.measures import distribution as d

    with config.override(**presets.iit4_2023):
        system = examples.basic_system()
        mechanism, purview = (0,), (1,)
        measure = resolve_mechanism_measure("INTRINSIC_INFORMATION")
        result = system.intrinsic_information(
            Direction.CAUSE, mechanism, purview, specification_measure=measure
        )
        assert len(result.state) == len(purview)
        # By-hand reference.
        posterior = system.repertoire(Direction.CAUSE, mechanism, purview)
        fwd = forward_repertoire(system, Direction.CAUSE, mechanism, purview, None)
        unc = unconstrained_forward_repertoire(
            system, Direction.CAUSE, mechanism, purview
        )
        spec = np.asarray(
            d.generalized_intrinsic_difference(fwd, unc, posterior)
        ).squeeze()
        diff = d.pointwise_intrinsic_differentiation(
            np.asarray(posterior, dtype=float)
        ).squeeze()
        expected = float(np.minimum(spec, diff).max())
        assert float(result.intrinsic_information) == expected

        # Effect side: the repertoire equals the forward repertoire (Z = 1),
        # so the operand switch is a no-op there (positive control).
        eff_rep = np.asarray(
            system.repertoire(Direction.EFFECT, mechanism, purview)
        ).squeeze()
        eff_fwd = np.asarray(
            forward_repertoire(system, Direction.EFFECT, mechanism, purview, None)
        ).squeeze()
        assert np.allclose(eff_rep, eff_fwd)


def test_composite_specification_measure_distinction_does_not_crash():
    """The config-routed path: computing a distinction with
    ``specification_measure="INTRINSIC_INFORMATION"`` used to raise
    IndexError from the rank-mismatched specified state."""
    from pyphi import examples
    from pyphi.conf import presets

    with config.override(
        **presets.iit4_2023,
        **{"iit.specification_measure": "INTRINSIC_INFORMATION"},
    ):
        system = examples.basic_system()
        distinction = system.distinction((0,))
        repr(distinction)
