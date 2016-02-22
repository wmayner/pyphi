#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyphi
from pyphi import macro

pyphi.config.CACHE_POTENTIAL_PURVIEWS = False

Nodes = 6

tpm = np.zeros((2 ** Nodes, Nodes))
state = tuple(0 for i in range(Nodes))
network = pyphi.Network(tpm)


# No external, blackbox, or coarse grain
subsystem1 = macro.MacroSubsystem(network, state, network.node_indices)

# External elements
subsystem2 = macro.MacroSubsystem(network, state, (0, 2, 4, 5))

# Hidden elements, no External
hidden_indices = (3, 4)
subsystem3 = macro.MacroSubsystem(network, state, network.node_indices,
                                  hidden_indices=hidden_indices)

# Hidden and External elements
hidden_indices = (1, 3)
subsystem4 = macro.MacroSubsystem(network, state, (0, 1, 3, 5),
                                  hidden_indices=hidden_indices)

# Only macro
output_grouping = ((0, 1, 2), (3, 4, 5))
state_grouping = (((0, 1), (2, 3)), ((0,), (1, 2, 3)))
coarse_grain = macro.CoarseGrain(output_grouping, state_grouping)
subsystem5 = macro.MacroSubsystem(network, state, network.node_indices,
                                  coarse_grain=coarse_grain)
# Macro and External
output_grouping = ((1, 3), (2, 5))
state_grouping = (((0, 1), (2,)), ((0,), (1, 2)))
coarse_grain = macro.CoarseGrain(output_grouping, state_grouping)
subsystem6 = macro.MacroSubsystem(network, state, (1, 2, 3, 5),
                                  coarse_grain=coarse_grain)

# Macro and Hidden
hidden_indices = (3, 4)
output_grouping = ((0, 5), (1, 2))
state_grouping = (((0, 1), (2,)), ((0,), (1, 2)))
coarse_grain = macro.CoarseGrain(output_grouping, state_grouping)
subsystem7 = macro.MacroSubsystem(network, state, network.node_indices,
                                  hidden_indices=hidden_indices,
                                  coarse_grain=coarse_grain)

# External, Macro and Hidden
hidden_indices = (4,)
output_grouping = ((1,), (2,), (3, 5))
state_grouping = (((0,), (1,)), ((0,), (1,)), ((0,), (1, 2)))
coarse_grain = macro.CoarseGrain(output_grouping, state_grouping)
subsystem8 = macro.MacroSubsystem(network, state, (1, 2, 3, 4, 5),
                                  hidden_indices=hidden_indices,
                                  coarse_grain=coarse_grain)


pyphi.config.CACHE_POTENTIAL_PURVIEWS = True


def test_size():
    assert subsystem1.size == 6
    assert subsystem2.size == 4
    assert subsystem3.size == 4
    assert subsystem4.size == 2
    assert subsystem5.size == 2
    assert subsystem6.size == 2
    assert subsystem7.size == 2
    assert subsystem8.size == 3


def test_external_indices():
    assert subsystem1.external_indices == ()
    assert subsystem2.external_indices == (1, 3)
    assert subsystem3.external_indices == ()
    assert subsystem4.external_indices == (2, 4)
    assert subsystem5.external_indices == ()
    assert subsystem6.external_indices == (0, 4)
    assert subsystem7.external_indices == ()
    assert subsystem8.external_indices == (0,)


def test_internal_indices():
    assert subsystem1.internal_indices == (0, 1, 2, 3, 4, 5)
    assert subsystem2.internal_indices == (0, 2, 4, 5)
    assert subsystem3.internal_indices == (0, 1, 2, 3, 4, 5)
    assert subsystem4.internal_indices == (0, 1, 3, 5)
    assert subsystem5.internal_indices == (0, 1, 2, 3, 4, 5)
    assert subsystem6.internal_indices == (1, 2, 3, 5)
    assert subsystem7.internal_indices == (0, 1, 2, 3, 4, 5)
    assert subsystem8.internal_indices == (1, 2, 3, 4, 5)


def test_micro_indices():
    assert subsystem1.micro_indices == (0, 1, 2, 3, 4, 5)
    assert subsystem2.micro_indices == (0, 1, 2, 3)
    assert subsystem3.micro_indices == (0, 1, 2, 3, 4, 5)
    assert subsystem4.micro_indices == (0, 1, 2, 3)
    assert subsystem5.micro_indices == (0, 1, 2, 3, 4, 5)
    assert subsystem6.micro_indices == (0, 1, 2, 3)
    assert subsystem7.micro_indices == (0, 1, 2, 3, 4, 5)
    assert subsystem8.micro_indices == (0, 1, 2, 3, 4)


def test_hidden_indices():
    assert subsystem1.hidden_indices == ()
    assert subsystem2.hidden_indices == ()
    assert subsystem3.hidden_indices == (3, 4)
    assert subsystem4.hidden_indices == (1, 2)
    assert subsystem5.hidden_indices == ()
    assert subsystem6.hidden_indices == ()
    assert subsystem7.hidden_indices == (3, 4)
    assert subsystem8.hidden_indices == (3,)


def test_output_indices():
    assert subsystem1.output_indices == (0, 1, 2, 3, 4, 5)
    assert subsystem2.output_indices == (0, 1, 2, 3)
    assert subsystem3.output_indices == (0, 1, 2, 5)
    assert subsystem4.output_indices == (0, 3)
    assert subsystem5.output_indices == (0, 1, 2, 3, 4, 5)
    assert subsystem6.output_indices == (0, 1, 2, 3)
    assert subsystem7.output_indices == (0, 1, 2, 5)
    assert subsystem8.output_indices == (0, 1, 2, 4)


def test_grouping_indices():
    assert subsystem1.coarse_grain.output_grouping == ()
    assert subsystem2.coarse_grain.output_grouping == ()
    assert subsystem3.coarse_grain.output_grouping == ()
    assert subsystem4.coarse_grain.output_grouping == ()
    assert subsystem5.coarse_grain.output_grouping == ((0, 1, 2), (3, 4, 5))
    assert subsystem6.coarse_grain.output_grouping == ((0, 2), (1, 3))
    assert subsystem7.coarse_grain.output_grouping == ((0, 3), (1, 2))
    assert subsystem8.coarse_grain.output_grouping == ((0,), (1,), (2, 3))


def test_node_indices():
    assert subsystem1.node_indices == (0, 1, 2, 3, 4, 5)
    assert subsystem2.node_indices == (0, 1, 2, 3)
    assert subsystem3.node_indices == (0, 1, 2, 3)
    assert subsystem4.node_indices == (0, 1)
    assert subsystem5.node_indices == (0, 1)
    assert subsystem6.node_indices == (0, 1)
    assert subsystem7.node_indices == (0, 1)
    assert subsystem8.node_indices == (0, 1, 2)


def test_size():
    assert subsystem1.size == 6
    assert subsystem2.size == 4
    assert subsystem3.size == 4
    assert subsystem4.size == 2
    assert subsystem5.size == 2
    assert subsystem6.size == 2
    assert subsystem7.size == 2
    assert subsystem8.size == 3


def test_subsystem_micro():
    assert subsystem1.micro
    assert subsystem2.micro
    assert not subsystem3.micro
    assert not subsystem4.micro
    assert not subsystem5.micro
    assert not subsystem6.micro
    assert not subsystem7.micro
    assert not subsystem8.micro


def test_subsystem_cm():
    assert np.array_equal(subsystem3.connectivity_matrix, np.ones((4, 4)))
