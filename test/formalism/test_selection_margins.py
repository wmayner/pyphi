"""Tests for selection-margin reporting on the IIT 4.0 SIA."""

import pytest

import pyphi
from pyphi import examples
from pyphi.conf import config
from pyphi.core import repertoire_algebra as ra
from pyphi.direction import Direction
from pyphi.measures.distribution import resolve_mechanism_measure


@pytest.fixture(autouse=True)
def _quiet():
    with pyphi.config.override(progress_bars=False):
        yield


@pytest.fixture(scope="module")
def basic_sia():
    with pyphi.config.override(progress_bars=False):
        return examples.basic_system().sia()


@pytest.fixture(scope="module")
def xor_sia():
    with pyphi.config.override(progress_bars=False):
        return examples.xor_system().sia()


def _per_state_ii(system, direction):
    """Brute force: intrinsic information of every candidate system state."""
    measure = resolve_mechanism_measure(config.formalism.iit.specification_measure)
    alphabet = system.substrate.factored_tpm.alphabet_sizes
    from pyphi.utils import all_states

    sizes = tuple(alphabet[i] for i in system.node_indices)
    return {
        state: float(
            ra.intrinsic_information(
                system,
                direction,
                mechanism=system.node_indices,
                purview=system.node_indices,
                specification_measure=measure,
                states=[state],
            ).intrinsic_information
        )
        for state in all_states(sizes)
    }


@pytest.mark.parametrize("direction", [Direction.CAUSE, Direction.EFFECT])
def test_state_runner_up_matches_brute_force(basic_sia, direction):
    system = examples.basic_system()
    values = _per_state_ii(system, direction)
    ranked = sorted(values.values(), reverse=True)
    spec = basic_sia.system_state[direction]
    assert float(spec.intrinsic_information) == pytest.approx(ranked[0])
    assert float(spec.runner_up_intrinsic_information) == pytest.approx(ranked[1])
    assert float(spec.state_margin) == pytest.approx(ranked[0] - ranked[1])
    assert spec.runner_up_state in values
    assert values[spec.runner_up_state] == pytest.approx(ranked[1])


def test_state_margin_zero_for_exactly_tied_states(xor_sia):
    # xor at (0, 0, 0): the specified cause state ties exactly (2 tied specs)
    spec = xor_sia.system_state.cause
    assert len(spec.ties) > 1
    assert float(spec.state_margin) == pytest.approx(0.0)
    assert float(spec.runner_up_intrinsic_information) == pytest.approx(
        float(spec.intrinsic_information)
    )


def test_tie_members_share_runner_up_fields(xor_sia):
    specs = xor_sia.system_state.cause.ties
    values = {float(s.runner_up_intrinsic_information) for s in specs}
    assert len(values) == 1


def test_state_margin_none_when_no_competitor():
    system = examples.basic_system()
    measure = resolve_mechanism_measure(config.formalism.iit.specification_measure)
    spec = ra.intrinsic_information(
        system,
        Direction.CAUSE,
        mechanism=system.node_indices,
        purview=system.node_indices,
        specification_measure=measure,
        states=[system.proper_state],
    )
    assert spec.runner_up_intrinsic_information is None
    assert spec.runner_up_state is None
    assert spec.state_margin is None
