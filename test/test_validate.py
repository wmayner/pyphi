#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pyphi import Network, Subsystem, validate


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


def test_validate_network_wrong_cm_size(standard):
    with pytest.raises(ValueError):
        Network(standard.tpm, np.ones(16).reshape(4, 4))


def test_validate_state_no_error_1(s, standard):
    validate.state_reachable(s)


def test_validate_state_error(s, standard):
    with pytest.raises(validate.StateUnreachableError):
        state = (0, 1, 0)
        Subsystem(standard, state, range(standard.size))


def test_validate_state_no_error_2(s, standard):
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
