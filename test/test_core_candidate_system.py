"""Tests for pyphi.core.candidate_system — CandidateSystem value type."""

from __future__ import annotations

import dataclasses

import pytest


@pytest.fixture
def basic_cs():
    from pyphi import examples
    from pyphi.core.candidate_system import CandidateSystem
    from pyphi.core.causal_model import CausalModel

    cm = CausalModel.from_network(examples.basic_network())
    return CandidateSystem(causal_model=cm, state=(1, 0, 0), node_indices=(0, 1, 2))


def test_candidate_system_is_frozen(basic_cs) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        basic_cs.state = (0, 0, 0)


def test_candidate_system_default_cut_is_null(basic_cs) -> None:
    from pyphi.models.cuts import NullCut

    assert isinstance(basic_cs.cut, NullCut)


def test_candidate_system_validates_state_length() -> None:
    from pyphi import examples
    from pyphi.core.candidate_system import CandidateSystem
    from pyphi.core.causal_model import CausalModel

    cm = CausalModel.from_network(examples.basic_network())
    with pytest.raises(ValueError):
        CandidateSystem(causal_model=cm, state=(1, 0), node_indices=(0, 1))


def test_candidate_system_validates_node_states() -> None:
    from pyphi import examples
    from pyphi.core.candidate_system import CandidateSystem
    from pyphi.core.causal_model import CausalModel

    cm = CausalModel.from_network(examples.basic_network())
    with pytest.raises(ValueError):
        CandidateSystem(causal_model=cm, state=(1, 0, 7), node_indices=(0, 1, 2))


def test_candidate_system_equality_includes_cut(basic_cs) -> None:
    from pyphi.core.candidate_system import CandidateSystem
    from pyphi.direction import Direction
    from pyphi.models.cuts import SystemPartition

    other = CandidateSystem(
        causal_model=basic_cs.causal_model,
        state=basic_cs.state,
        node_indices=basic_cs.node_indices,
    )
    assert basic_cs == other
    cut = SystemPartition(Direction.EFFECT, (0,), (1, 2))
    cut_cs = CandidateSystem(
        causal_model=basic_cs.causal_model,
        state=basic_cs.state,
        node_indices=basic_cs.node_indices,
        cut=cut,
    )
    assert basic_cs != cut_cs


def test_candidate_system_is_hashable(basic_cs) -> None:
    from pyphi.core.candidate_system import CandidateSystem

    other = CandidateSystem(
        causal_model=basic_cs.causal_model,
        state=basic_cs.state,
        node_indices=basic_cs.node_indices,
    )
    assert hash(basic_cs) == hash(other)


@pytest.fixture
def cs_and_subsystem():
    """Paired CandidateSystem + legacy Subsystem with identical inputs."""
    from pyphi import Subsystem
    from pyphi import examples
    from pyphi.core.candidate_system import CandidateSystem
    from pyphi.core.causal_model import CausalModel

    network = examples.basic_network()
    state = (1, 0, 0)
    nodes = (0, 1, 2)
    cm = CausalModel.from_network(network)
    cs = CandidateSystem(causal_model=cm, state=state, node_indices=nodes)
    sub = Subsystem(network, state, nodes)
    return cs, sub


@pytest.mark.parametrize(
    "attr",
    [
        "external_indices",
        "node_indices",
        "state",
        "proper_state",
        "size",
        "tpm_size",
        "is_cut",
        "cut_indices",
    ],
)
def test_candidate_system_property_parity(cs_and_subsystem, attr) -> None:
    cs, sub = cs_and_subsystem
    cs_val = getattr(cs, attr)
    sub_val = getattr(sub, attr)
    assert cs_val == sub_val, f"{attr}: {cs_val!r} != {sub_val!r}"


def test_candidate_system_node_labels_parity(cs_and_subsystem) -> None:
    cs, sub = cs_and_subsystem
    assert tuple(cs.node_labels) == tuple(sub.node_labels)


def test_candidate_system_cause_tpm_parity(cs_and_subsystem) -> None:
    import numpy as np

    cs, sub = cs_and_subsystem
    np.testing.assert_array_equal(cs.cause_tpm.to_array(), np.asarray(sub.cause_tpm))


def test_candidate_system_effect_tpm_parity(cs_and_subsystem) -> None:
    import numpy as np

    cs, sub = cs_and_subsystem
    np.testing.assert_array_equal(cs.effect_tpm.to_array(), np.asarray(sub.effect_tpm))


def test_candidate_system_cm_parity(cs_and_subsystem) -> None:
    import numpy as np

    cs, sub = cs_and_subsystem
    np.testing.assert_array_equal(cs.cm, sub.cm)
    np.testing.assert_array_equal(cs.proper_cm, sub.proper_cm)
