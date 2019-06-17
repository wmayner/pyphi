#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pyphi import (Direction, Network, Subsystem, config, constants,
                   exceptions, macro, validate)


def test_validate_direction():
    validate.direction(Direction.CAUSE)
    validate.direction(Direction.EFFECT)

    with pytest.raises(ValueError):
        validate.direction("dogeeeee")

    validate.direction(Direction.BIDIRECTIONAL, allow_bi=True)
    with pytest.raises(ValueError):
        validate.direction(Direction.BIDIRECTIONAL)


def test_validate_tpm_wrong_shape():
    tpm = np.arange(3**3).reshape(3, 3, 3)
    with pytest.raises(ValueError):
        assert validate.tpm(tpm)


def test_validate_tpm_nonbinary_nodes():
    tpm = np.arange(3*3*2).reshape(3, 3, 2)
    with pytest.raises(ValueError):
        assert validate.tpm(tpm)


def test_validate_tpm_conditional_independence():
    tpm = np.array([
        [1, 0.0, 0.0, 0],
        [0, 0.5, 0.5, 0],
        [0, 0.5, 0.5, 0],
        [0, 0.0, 0.0, 1],
    ])
    with pytest.raises(exceptions.ConditionallyDependentError):
        validate.conditionally_independent(tpm)
    with pytest.raises(exceptions.ConditionallyDependentError):
        validate.tpm(tpm)
    validate.tpm(tpm, check_independence=False)


def test_validate_connectivity_matrix_valid(s):
    assert validate.connectivity_matrix(s.network.cm)


def test_validate_connectivity_matrix_not_square():
    cm = np.random.binomial(1, 0.5, (4, 5))
    with pytest.raises(ValueError):
        assert validate.connectivity_matrix(cm)


def test_validate_connectivity_matrix_not_2D():
    cm = np.arange(8).reshape(2, 2, 2)
    with pytest.raises(ValueError):
        assert validate.connectivity_matrix(cm)


def test_validate_connectivity_matrix_not_binary():
    cm = np.arange(16).reshape(4, 4)
    with pytest.raises(ValueError):
        assert validate.connectivity_matrix(cm)


def test_validate_network_wrong_cm_size(s):
    with pytest.raises(ValueError):
        Network(s.network.tpm, np.ones(16).reshape(4, 4))


def test_validate_is_network(s):
    with pytest.raises(ValueError):
        validate.is_network(s)
    validate.is_network(s.network)


def test_validate_state_no_error_1(s):
    validate.state_reachable(s)


def test_validate_state_error(s):
    with pytest.raises(exceptions.StateUnreachableError):
        state = (0, 1, 0)
        Subsystem(s.network, state, s.node_indices)


def test_validate_state_no_error_2():
    tpm = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ])
    net = Network(tpm)
    # Globally impossible state.
    state = (1, 1, 0, 0)
    # But locally possible for first two nodes.
    subsystem = Subsystem(net, state, (0, 1))
    validate.state_reachable(subsystem)


def test_validate_node_labels():
    validate.node_labels(['A', 'B'], (0, 1))

    with pytest.raises(ValueError):
        validate.node_labels(['A'], (0, 1))
    with pytest.raises(ValueError):
        validate.node_labels(['A', 'B'], (0,))
    with pytest.raises(ValueError):
        validate.node_labels(['A', 'A'], (0, 1))


def test_validate_time_scale():
    with pytest.raises(ValueError):
        validate.time_scale(1.3)
    with pytest.raises(ValueError):
        validate.time_scale(-1)
    with pytest.raises(ValueError):
        validate.time_scale(0)
    validate.time_scale(1)
    validate.time_scale(2)
    # ... etc


def test_validate_coarse_grain():
    # Good:
    cg = macro.CoarseGrain(((2,), (3,)), (((0,), (1,)), (((0,), (1,)))))
    validate.coarse_grain(cg)

    # Mismatched output and state lengths
    cg = macro.CoarseGrain(((2,),), (((0,), (1,)), (((0,), (1,)))))
    with pytest.raises(ValueError):
        validate.coarse_grain(cg)

    # Missing 1-node-on specification in second state grouping
    cg = macro.CoarseGrain(((2,), (3,)), (((0,), (1,)), (((0,), ()))))
    with pytest.raises(ValueError):
        validate.coarse_grain(cg)

    # Two partitions contain same element
    cg = macro.CoarseGrain(((5,), (5,)), (((0,), (1,)), (((0,), (1,)))))
    with pytest.raises(ValueError):
        validate.coarse_grain(cg)


def test_validate_blackbox():
    validate.blackbox(macro.Blackbox(((0, 1),), (1,)))

    # Unsorted output indices
    with pytest.raises(ValueError):
        validate.blackbox(macro.Blackbox(((0, 1),), (1, 0)))

    # Two boxes may not contain the same elements
    with pytest.raises(ValueError):
        validate.blackbox(macro.Blackbox(((0,), (0, 1)), (0, 1)))

    # Every box must have an output
    with pytest.raises(ValueError):
        validate.blackbox(macro.Blackbox(((0,), (1,)), (0,)))


def test_validate_partition():
    # Micro-element appears in two macro-elements
    with pytest.raises(ValueError):
        validate.partition(((0,), (0, 1)))


def test_validate_blackbox_and_coarsegrain():
    blackbox = None
    coarse_grain = macro.CoarseGrain(((0, 1), (2,)), ((0, 1), (2,)))
    validate.blackbox_and_coarse_grain(blackbox, coarse_grain)

    blackbox = macro.Blackbox(((0, 1), (2,)), (0, 2))
    coarse_grain = None
    validate.blackbox_and_coarse_grain(blackbox, coarse_grain)

    blackbox = macro.Blackbox(((0, 1), (2,)), (0, 1, 2))
    coarse_grain = macro.CoarseGrain(((0, 1), (2,)), ((0, 1), (2,)))
    validate.blackbox_and_coarse_grain(blackbox, coarse_grain)

    # Blackboxing with multiple outputs must be coarse-grained
    blackbox = macro.Blackbox(((0, 1), (2,)), (0, 1, 2))
    coarse_grain = None
    with pytest.raises(ValueError):
        validate.blackbox_and_coarse_grain(blackbox, coarse_grain)

    # Coarse-graining does not group multiple outputs of a box into the same
    # macro element
    blackbox = macro.Blackbox(((0, 1), (2,)), (0, 1, 2))
    coarse_grain = macro.CoarseGrain(((0,), (1, 2)), ((0, 1), (2,)))
    with pytest.raises(ValueError):
        validate.blackbox_and_coarse_grain(blackbox, coarse_grain)
