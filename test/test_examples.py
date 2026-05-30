# -*- coding: utf-8 -*-
import numpy as np
import pytest

from pyphi import examples
from pyphi.actual import Transition
from pyphi.network import Network
from pyphi.subsystem import Subsystem


def _example_items(kind):
    return sorted(examples.EXAMPLES[kind].items())


def test_examples_registry_contains_expected_categories():
    expected = {'network', 'subsystem', 'tpm', 'transition'}
    assert expected.issubset(examples.EXAMPLES.keys())


@pytest.mark.parametrize('name, func', _example_items('network'))
def test_example_networks_construct(name, func):
    network = func()
    assert isinstance(network, Network)


@pytest.mark.parametrize('name, func', _example_items('subsystem'))
def test_example_subsystems_construct(name, func):
    subsystem = func()
    assert isinstance(subsystem, Subsystem)


@pytest.mark.parametrize('name, func', _example_items('tpm'))
def test_example_tpms_construct(name, func):
    tpm = func()
    assert isinstance(tpm, np.ndarray)
    assert tpm.ndim == 2
    assert tpm.shape[0] == tpm.shape[1]


@pytest.mark.parametrize('name, func', _example_items('transition'))
def test_example_transitions_construct(name, func):
    transition = func()
    assert isinstance(transition, Transition)
