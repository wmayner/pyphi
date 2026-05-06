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
