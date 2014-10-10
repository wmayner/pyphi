#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from pyphi.network import Network
from pyphi import validate


def test_validate_nodelist_noniterable():
    with pytest.raises(ValueError):
        validate.nodelist(2, "it's a doge")


def test_validate_nodelist_nonnode():
    with pytest.raises(ValueError):
        validate.nodelist([0, 1, 2], 'invest in dogecoin!')


def test_validate_nodelist_nontuple_sequence(s):
    nodes = validate.nodelist(list(s.nodes), 'such phi')
    assert nodes == s.nodes


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
        Network(standard.tpm, standard.current_state, standard.past_state,
                np.ones(16).reshape(4, 4))


def test_validate_state_wrong_size(standard):
    with pytest.raises(ValueError):
        Network(standard.tpm, (0, 0, 0, 0), standard.past_state)


def test_validate_state_not_reachable_at_all(standard):
    with pytest.raises(validate.StateUnreachableError):
        Network(standard.tpm, (0, 1, 1), standard.past_state)


def test_validate_state_not_reachable_from_given(standard):
    with pytest.raises(validate.StateUnreachableError):
        Network(standard.tpm, (0, 0, 0), (1, 1, 1))
