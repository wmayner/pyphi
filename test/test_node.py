#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_node.py

import numpy as np

from pyphi.node import node, expand_node_tpm, generate_nodes
from pyphi.subsystem import Subsystem
from pyphi.tpm import ExplicitTPM


def test_node_init_tpm(s):
    # fmt: off
    answer = [
        np.array([
            [[[1, 0],
              [0, 1]],
             [[0, 1],
              [0, 1]]],
        ]),
        np.array([
            [[[1, 0],
              [0, 1]]],
        ]),
        np.array([
            [[[1, 0]],
              [[0, 1]]],
             [[[0, 1]],
               [[1, 0]]],
        ]),
    ]
    answer = [ExplicitTPM(tpm) for tpm in answer]
    # fmt: on
    for node in s.nodes:
        assert node.pyphi.tpm.array_equal(answer[node.pyphi.index])


def test_node_init_inputs(s):
    answer = [s.node_indices[1:], s.node_indices[2:3], s.node_indices[:2]]
    for node in s.nodes:
        assert set(node.pyphi.inputs) == set(answer[node.pyphi.index])


def test_node_eq(s):
    assert s.nodes[1] == node(s.tpm, s.cm, 1, 0, "B")


def test_node_neq_by_index(s):
    assert s.nodes[0] != node(s.tpm, s.cm, 1, 0, "B")


def test_node_neq_by_state(s):
    other_s = Subsystem(s.network, (1, 1, 1), s.node_indices)
    assert other_s.nodes[1] != node(s.tpm, s.cm, 1, 0, "B")


def test_repr(s):
    print(repr(s.nodes[0]))


def test_str(s):
    print(str(s.nodes[0]))


def test_expand_tpm():
    # fmt: off
    tpm = ExplicitTPM(
        np.array([
            [[0, 1]],
        ])
    )
    answer = ExplicitTPM(
        np.array([
            [[0, 1],
             [0, 1]],
            [[0, 1],
             [0, 1]],
        ])
    )
    # fmt: on
    assert expand_node_tpm(tpm).array_equal(answer)


def test_generate_nodes(s):
    nodes = generate_nodes(s.tpm, s.cm, s.state, s.node_indices, s.node_labels)

    # fmt: off
    node0_tpm = ExplicitTPM(
        np.array([
            [[[1, 0],
              [0, 1]],
             [[0, 1],
              [0, 1]]],
        ])
    )
    # fmt: on
    assert nodes[0].pyphi.tpm.array_equal(node0_tpm)
    assert nodes[0].pyphi.inputs == set([1, 2])
    assert nodes[0].pyphi.outputs == set([2])
    assert nodes[0].pyphi.label == "A"

    # fmt: off
    node1_tpm = ExplicitTPM(
        np.array([
            [[[1, 0],
              [0, 1]]],
        ])
    )
    # fmt: on
    assert nodes[1].pyphi.tpm.array_equal(node1_tpm)
    assert nodes[1].pyphi.inputs == set([2])
    assert nodes[1].pyphi.outputs == set([0, 2])
    assert nodes[1].pyphi.label == "B"

    # fmt: off
    node2_tpm = ExplicitTPM(
        np.array([
            [[[1, 0]],
             [[0, 1]]],
            [[[0, 1]],
             [[1, 0]]]
        ])
    )
    # fmt: on
    assert nodes[2].pyphi.tpm.array_equal(node2_tpm)
    assert nodes[2].pyphi.inputs == set([0, 1])
    assert nodes[2].pyphi.outputs == set([0, 1])
    assert nodes[2].pyphi.label == "C"


def test_generate_nodes_default_labels(s):
    nodes = generate_nodes(s.tpm, s.cm, s.state, s.node_indices)
    assert [n.pyphi.label for n in nodes] == ["n0", "n1", "n2"]
