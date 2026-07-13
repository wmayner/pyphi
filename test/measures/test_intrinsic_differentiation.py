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
