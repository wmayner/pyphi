#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_iit_2_0.py

import pytest
import numpy as np

from pyphi import Network, jsonify, examples
from pyphi.distribution import max_entropy_distribution
from pyphi.subsystem_2_0 import (
    Subsystem_2_0, Partition, generate_partitions, all_complexes,
    all_subsystems, main_complexes)


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


@pytest.fixture
def and_1_2_system(and_network):
    return Subsystem_2_0(and_network, (0, 0, 1), (1, 2))


def test_prior_repertoire(and_whole_system, and_1_2_system):
    assert np.array_equal(and_whole_system.prior_repertoire(),
                          np.ones([2, 2, 2]) / 8)
    assert np.array_equal(and_1_2_system.prior_repertoire((1, 2)),
                          np.array([[[0.25, 0.25], [0.25, 0.25]]]))


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


def test_mip(disjoint_subsystem):
    assert disjoint_subsystem.find_mip().partition == Partition((0, 1), (2, 3))


def test_phi(disjoint_subsystem):
    assert disjoint_subsystem.phi == 0


def test_complexes(disjoint_couples_network):
    assert set(all_complexes(disjoint_couples_network, (0, 1, 1, 0))) == set([
        Subsystem_2_0(disjoint_couples_network, (0, 1, 1, 0), (0, 1)),
        Subsystem_2_0(disjoint_couples_network, (0, 1, 1, 0), (2, 3))])


@pytest.fixture
def counting_network():
    """Binary counting network from Figure 11."""
    tpm = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 0]])

    return Network(tpm)


@pytest.mark.parametrize('state,phi', [
    ((0, 0, 0, 0), 4),
    ((0, 0, 1, 1), 0.192645)])
def test_binary_counting(state, phi, counting_network):
    subsystem = Subsystem_2_0(counting_network, state, (0, 1, 2, 3))
    assert subsystem.phi == phi


@pytest.fixture
def modular_network():
    """Modular network from Figure 13. Generated with the following code:

    import pyphi
    import graphiit

    # Elements fire if they receive >= 2 spikes
    def THRESHOLD(inputs):
        return sum(inputs) >= 2

    g = graphiit.Graph([
        ("A1", THRESHOLD, "B1", "C1"),
        ("B1", THRESHOLD, "A1", "C1"),
        ("C1", THRESHOLD, "A1", "B1", "C2", "C4"),
        ("A2", THRESHOLD, "B2", "C2"),
        ("B2", THRESHOLD, "A2", "C2"),
        ("C2", THRESHOLD, "A2", "B2", "C1", "C3"),
        ("A3", THRESHOLD, "B3", "C3"),
        ("B3", THRESHOLD, "A3", "C3"),
        ("C3", THRESHOLD, "A3", "B3", "C2", "C4"),
        ("A4", THRESHOLD, "B4", "C4"),
        ("B4", THRESHOLD, "A4", "C4"),
        ("C4", THRESHOLD, "A4", "B4", "C1", "C3")])

    network = g.pyphi_network()

    with open('./test/data/iit_2.0_fig_13.json', 'w') as f:
        pyphi.jsonify.dump(network.to_json(), f, indent=2)
    """
    with open('./test/data/iit_2.0_fig_13.json') as f:
        data = jsonify.load(f)

    return Network(data['tpm'], data['cm'], data['labels'])


@pytest.mark.parametrize('nodes,phi', [
    (range(12), 0.700186),
    ((0, 1, 2), 1.188722),
    ((3, 4, 5), 1.188722),
    ((6, 7, 8), 1.188722),
    ((9, 10, 11), 1.188722)])
def test_modular_network(nodes, phi, modular_network):
    subsystem = Subsystem_2_0(modular_network, [0] * 12, nodes)
    assert subsystem.phi == phi


def test_all_complexes():
    """Complexes of IIT 3.0 Fig 16 example."""
    network = examples.fig16()
    state = (1, 0, 0, 1, 1, 1, 0)
    assert all_complexes(network, state) == [
        Subsystem_2_0(network, state, (0, 1, 2, 3, 4)),
        Subsystem_2_0(network, state, (0, 1, 2, 3)),
        Subsystem_2_0(network, state, (0, 1, 2)),
        Subsystem_2_0(network, state, (5, 6)),
        Subsystem_2_0(network, state, (3, 4))]


def test_main_complexes():
    network = examples.fig16()
    state = (1, 0, 0, 1, 1, 1, 0)
    assert main_complexes(network, state) == [
        Subsystem_2_0(network, state, (0, 1, 2)),
        Subsystem_2_0(network, state, (5, 6)),
        Subsystem_2_0(network, state, (3, 4))]


@pytest.fixture
def figure6_network():
    """Example network from Figure 6."""
    tpm = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1]])

    cm = np.array([
        [0, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]])

    return Network(tpm, cm, node_labels="ABCDEF")


def test_main_complexes_figure6(figure6_network):
    state = (0, 0, 1, 1, 1, 0)
    assert main_complexes(figure6_network, state) == [
        Subsystem_2_0(figure6_network, state, (0, 1, 2)),
        Subsystem_2_0(figure6_network, state, (2, 5)),
        Subsystem_2_0(figure6_network, state, (1, 4)),
        Subsystem_2_0(figure6_network, state, (0, 3))
    ]
