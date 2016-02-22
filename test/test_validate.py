#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pyphi import macro, Network, Subsystem, validate


def test_validate_direction():
    with pytest.raises(ValueError):
        assert validate.direction("dogeeeee")


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
        [1,  0,  0,  0],
        [0, .5, .5,  0],
        [0, .5, .5,  0],
        [0,  0,  0,  1],
    ])
    with pytest.raises(ValueError):
        validate.tpm(tpm)


def test_validate_cm_valid(s):
    assert validate.connectivity_matrix(s.network.connectivity_matrix)


def test_validate_cm_not_square():
    cm = np.random.binomial(1, 0.5, (4, 5))
    with pytest.raises(ValueError):
        assert validate.connectivity_matrix(cm)


def test_validate_cm_not_2D():
    cm = np.arange(8).reshape(2, 2, 2)
    with pytest.raises(ValueError):
        assert validate.connectivity_matrix(cm)


def test_validate_cm_not_binary():
    cm = np.arange(16).reshape(4, 4)
    with pytest.raises(ValueError):
        assert validate.connectivity_matrix(cm)


def test_validate_network_wrong_cm_size(s):
    with pytest.raises(ValueError):
        Network(s.network.tpm, np.ones(16).reshape(4, 4))


def test_validate_state_no_error_1(s):
    validate.state_reachable(s)


def test_validate_state_error(s):
    with pytest.raises(validate.StateUnreachableError):
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
