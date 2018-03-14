#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_iit_2_0.py

import pytest
import numpy as np

from pyphi import Network
from pyphi.distribution import max_entropy_distribution
from pyphi.subsystem_2_0 import Subsystem_2_0, Partition, generate_partitions


@pytest.fixture
def and_network():
    """A 3-node AND network, from Figure 1.

    Diagram::

                +~~~~~~~+
          +~~~~>|   A   |<~~~~+
          |     | (AND) +~~~+ |
          |     +~~~~~~~+   | |
          |                 | |
          |                 v |
        +~+~~~~~~+      +~~~~~+~+
        |   B    |<~~~~~+   C   |
        | (AND) +~~~~~->| (AND) |
        +~~~~~~~~+      +~~~~~~~+
    """
    tpm = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 1]
    ])
    return Network(tpm, node_labels="ABC")


@pytest.fixture
def and_whole_system(and_network):
    return Subsystem_2_0(and_network, (0, 0, 1), (0, 1, 2))


def test_prior_repertoire(and_whole_system):
    assert np.array_equal(and_whole_system.prior_repertoire(),
                          np.ones([2, 2, 2]) / 8)


def test_posterior_repertoire(and_whole_system):
    """x_0 = (1, 1, 0) is the unique cause of x_1 = (0, 0, 1)"""
    r = np.array([[[0, 0], [0, 0]], [[0, 0], [1, 0]]])
    assert np.array_equal(and_whole_system.posterior_repertoire(), r)


def test_effective_information(and_whole_system):
    assert and_whole_system.effective_information() == 3


def test_effective_information_and_000(and_network):
    """From Figure 2B, AND-network entering state (0, 0, 0)"""
    system = Subsystem_2_0(and_network, (0, 0, 0), (0, 1, 2))
    assert system.effective_information() == 1


@pytest.fixture
def disjoint_couples_network():
    """The network of disjoint COPY gates from Figure 3."""
    tpm = np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 1, 1]])
    return Network(tpm, node_labels="ABCD")


@pytest.fixture
def disjoint_subsystem(disjoint_couples_network):
    """The entire disjoint network in state (0, 1, 1, 0)."""
    return Subsystem_2_0(disjoint_couples_network, (0, 1, 1, 0), (0, 1, 2, 3))


def test_prior_repertoire_disjoint_couples(disjoint_subsystem):
    assert np.array_equal(disjoint_subsystem.prior_repertoire((0, 1)),
                          max_entropy_distribution((0, 1), 4))
    assert np.array_equal(disjoint_subsystem.prior_repertoire((2, 3)),
                          max_entropy_distribution((2, 3), 4))


def test_posterior_repertoire_disjoint_couples(disjoint_subsystem):
    assert np.array_equal(disjoint_subsystem.posterior_repertoire((0, 1)),
                          np.array([[[[0]], [[0]]], [[[1]], [[0]]]]))
    assert np.array_equal(disjoint_subsystem.posterior_repertoire((2, 3)),
                          np.array([[[[0, 1], [0, 0]]]]))


def test_effective_information_disjoint_couples(disjoint_subsystem):
    assert disjoint_subsystem.effective_information() == 4
    assert disjoint_subsystem.effective_information((0, 1)) == 2
    assert disjoint_subsystem.effective_information((2, 3)) == 2


def test_effective_information_partition(disjoint_subsystem):
    mip = Partition((0, 1), (2, 3))
    assert disjoint_subsystem.effective_information_partition(mip) == 0

    not_mip = Partition((0, 2), (1, 3))
    assert disjoint_subsystem.effective_information_partition(not_mip) == 4


def test_partition_indices():
    assert Partition((2, 3), (0, 1)).indices == (0, 1, 2, 3)


def test_effective_information_total_partition(disjoint_subsystem):
    total = Partition((0, 1, 2, 3))
    assert disjoint_subsystem.effective_information_partition(total) == 4


@pytest.mark.parametrize('partition,normalization', [
    (Partition((0, 1), (2, 3)), 2),
    (Partition((0,), (1, 2, 3)), 1),
    (Partition((0,), (1,), (2, 3)), 2),
    (Partition((0, 1), (1, 2), (3, 4, 5)), 4),
    (Partition((0, 1, 2, 3)), 4)])
def test_normalization(partition, normalization):
    assert partition.normalization == normalization


def test_generate_partitions():
    assert set(generate_partitions((0, 1, 2))) == set([
        Partition((0, 1, 2)),
        Partition((0,), (1, 2)),
        Partition((1,), (0, 2)),
        Partition((2,), (0, 1))])
