#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_macro.py

import numpy as np
import pytest

from pyphi import convert, macro
from pyphi.exceptions import ConditionallyDependentError

# flake8: noqa


def test_all_partitions():
    assert list(macro.all_partitions(())) == []
    assert list(macro.all_partitions((0, 1, 2, 3))) == [
        ((0, 1, 2), (3,)),
        ((0, 1, 3), (2,)),
        ((0, 1), (2, 3)),
        ((0, 1), (2,), (3,)),
        ((0, 2, 3), (1,)),
        ((0, 2), (1, 3)),
        ((0, 2), (1,), (3,)),
        ((0, 3), (1, 2)),
        ((0,), (1, 2, 3)),
        ((0,), (1, 2), (3,)),
        ((0, 3), (1,), (2,)),
        ((0,), (1, 3), (2,)),
        ((0,), (1,), (2, 3)),
        ((0, 1, 2, 3),)
    ]


def test_all_groupings():
    assert list(macro.all_groupings(())) == [()]
    partition = ((0, 1), (2, 3))
    assert list(macro.all_groupings(partition)) == [
        (((0, 1), (2,)), ((0, 1), (2,))),
        (((0, 1), (2,)), ((0, 2), (1,))),
        (((0, 1), (2,)), ((0,), (1, 2))),
        (((0, 2), (1,)), ((0, 1), (2,))),
        (((0, 2), (1,)), ((0, 2), (1,))),
        (((0, 2), (1,)), ((0,), (1, 2))),
        (((0,), (1, 2)), ((0, 1), (2,))),
        (((0,), (1, 2)), ((0, 2), (1,))),
        (((0,), (1, 2)), ((0,), (1, 2)))
    ]


def test_all_coarse_grains():
    assert tuple(macro.all_coarse_grains((1,))) == (
        macro.CoarseGrain(partition=((1,),),
                          grouping=(((0,), (1,)),)),)


def test_all_coarse_grains_for_blackbox():
    blackbox = macro.Blackbox(((0, 1),), (0, 1))
    assert list(macro.all_coarse_grains_for_blackbox(blackbox)) == [
        macro.CoarseGrain(((0, 1),), (((0, 1), (2,)),)),
        macro.CoarseGrain(((0, 1),), (((0, 2), (1,)),)),
        macro.CoarseGrain(((0, 1),), (((0,), (1, 2)),)),
    ]


def test_all_blackboxes():
    assert list(macro.all_blackboxes((1, 2, 3))) == [
        macro.Blackbox(((1, 2), (3,)), (1, 3)),
        macro.Blackbox(((1, 2), (3,)), (2, 3)),
        macro.Blackbox(((1, 2), (3,)), (1, 2, 3)),
        macro.Blackbox(((1, 3), (2,)), (1, 2)),
        macro.Blackbox(((1, 3), (2,)), (2, 3)),
        macro.Blackbox(((1, 3), (2,)), (1, 2, 3)),
        macro.Blackbox(((1,), (2, 3)), (1, 2)),
        macro.Blackbox(((1,), (2, 3)), (1, 3)),
        macro.Blackbox(((1,), (2, 3)), (1, 2, 3)),
        macro.Blackbox(((1, 2, 3),), (1,)),
        macro.Blackbox(((1, 2, 3),), (2,)),
        macro.Blackbox(((1, 2, 3),), (3,)),
        macro.Blackbox(((1, 2, 3),), (1, 2)),
        macro.Blackbox(((1, 2, 3),), (1, 3)),
        macro.Blackbox(((1, 2, 3),), (2, 3)),
        macro.Blackbox(((1, 2, 3),), (1, 2, 3)),
    ]


def test_make_mapping():
    partition = ((0, 1), (2, 3))
    grouping = (((0, 1), (2,)), ((0, 1), (2,)))
    coarse_grain = macro.CoarseGrain(partition, grouping)
    mapping = coarse_grain.make_mapping()
    assert np.array_equal(mapping, np.array(
        (0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 2., 2., 2., 3.)))

    partition = ((0, 1), (2,))
    grouping = (((0, 2), (1,)), ((0,), (1,)))
    coarse_grain = macro.CoarseGrain(partition, grouping)
    mapping = coarse_grain.make_mapping()
    assert np.array_equal(mapping, np.array((0., 1., 1., 0., 2., 3., 3., 2.)))

    partition = ((0, 1, 2),)
    grouping = (((0, 3), (1, 2)),)
    coarse_grain = macro.CoarseGrain(partition, grouping)
    mapping = coarse_grain.make_mapping()
    assert np.array_equal(mapping, np.array((0., 1., 1., 1., 1., 1., 1., 0.)))


def test_make_macro_tpm():
    answer_tpm = convert.state_by_state2state_by_node(np.array([
        [0.375, 0.375, 0.125, 0.125],
        [0.375, 0.375, 0.125, 0.125],
        [0.375, 0.375, 0.125, 0.125],
        [0.375, 0.375, 0.125, 0.125],
    ]))
    partition = ((0,), (1, 2))
    grouping = (((0,), (1,)), ((0, 1), (2,)))
    coarse_grain = macro.CoarseGrain(partition, grouping)
    assert np.array_equal(coarse_grain.make_mapping(),
                          [0, 1, 0, 1, 0, 1, 2, 3])

    micro_tpm = np.zeros((8, 3)) + 0.5
    macro_tpm = coarse_grain.macro_tpm(micro_tpm)
    assert np.array_equal(answer_tpm, macro_tpm)

    micro_tpm = np.zeros((8, 8)) + 0.125
    macro_tpm = coarse_grain.macro_tpm(micro_tpm)
    assert np.array_equal(answer_tpm, macro_tpm)


def test_make_macro_tpm_conditional_independence_check():
    micro_tpm = np.array([
        [1, 0.0, 0.0, 0],
        [0, 0.5, 0.5, 0],
        [0, 0.5, 0.5, 0],
        [0, 0.0, 0.0, 1],
    ])
    partition = ((0,), (1,))
    grouping = (((0,), (1,)), ((0,), (1,)))
    coarse_grain = macro.CoarseGrain(partition, grouping)
    with pytest.raises(ConditionallyDependentError):
        macro_tpm = coarse_grain.macro_tpm(micro_tpm,
                                           check_independence=True)


# TODO: make a fixture for this conditionally dependent TPM
def test_macro_tpm_sbs():
    micro_tpm = np.array([
        [1, 0.0, 0.0, 0, 0, 0, 0, 0],
        [0, 0.5, 0.5, 0, 0, 0, 0, 0],
        [0, 0.5, 0.5, 0, 0, 0, 0, 0],
        [0, 0.0, 0.0, 1, 0, 0, 0, 0],
        [1, 0.0, 0.0, 0, 0, 0, 0, 0],
        [0, 0.5, 0.5, 0, 0, 0, 0, 0],
        [0, 0.5, 0.5, 0, 0, 0, 0, 0],
        [0, 0.0, 0.0, 1, 0, 0, 0, 0],
    ])
    answer_tpm = np.array([
        [1,   0,   0,   0  ],
        [0,   1/2, 1/2, 0  ],
        [1/3, 1/3, 1/3, 0  ],
        [0,   1/6, 1/6, 2/3]
    ])
    partition = ((0,), (1, 2))
    grouping = (((0,), (1,)), ((0,), (1, 2,)))
    coarse_grain = macro.CoarseGrain(partition, grouping)
    macro_tpm = coarse_grain.macro_tpm_sbs(micro_tpm)
    assert np.array_equal(answer_tpm, macro_tpm)


def test_coarse_grain_indices():
    partition = ((1, 2),)  # Node 0 not in system
    grouping = (((0,), (1, 2)),)
    cg = macro.CoarseGrain(partition, grouping)
    assert cg.micro_indices == (1, 2)
    assert cg.macro_indices == (0,)
    assert cg.reindex() == macro.CoarseGrain(((0, 1),), grouping)


def test_coarse_grain_state():
    partition = ((0, 1),)
    grouping = (((0,), (1, 2)),)
    cg = macro.CoarseGrain(partition, grouping)
    with pytest.raises(AssertionError):
        assert cg.macro_state((1, 1, 0)) == (1,)

    assert cg.macro_state((0, 0)) == (0,)
    assert cg.macro_state((0, 1)) == (1,)
    assert cg.macro_state((1, 1)) == (1,)

    partition = ((1,), (2,))
    grouping = (((0,), (1,)), ((1,), (0,)))
    cg = macro.CoarseGrain(partition, grouping)
    assert cg.macro_state((0, 1)) == (0, 0)
    assert cg.macro_state((1, 1)) == (1, 0)


def test_coarse_grain_len():
    partition = ((1, 2),)
    grouping = (((0,), (1, 2)),)
    cg = macro.CoarseGrain(partition, grouping)
    assert len(cg) == 1


@pytest.fixture
def bb():
    partition = ((1, 3), (4,))
    output_indices = (3, 4)
    return macro.Blackbox(partition, output_indices)


@pytest.fixture
def cg_bb():
    """A blackbox with multiple outputs for a box, which must be coarse-
    grained."""
    partition = ((1, 3), (4,), (5,))
    output_indices = (1, 3, 4, 5)
    return macro.Blackbox(partition, output_indices)


def test_blackbox_indices(bb):
    assert bb.micro_indices == (1, 3, 4)
    assert bb.macro_indices == (0, 1)
    assert bb.reindex() == macro.Blackbox(((0, 1), (2,)), (1, 2))


def test_blackbox_state(bb):
    with pytest.raises(AssertionError):
        bb.macro_state((0, 1, 1, 1))
    assert bb.macro_state((0, 1, 0)) == (1, 0)
    assert bb.macro_state((1, 0, 0)) == (0, 0)


def test_blackbox_same_box(bb):
    # Nodes not in Blackox
    with pytest.raises(AssertionError):
        bb.in_same_box(2, 4)
    with pytest.raises(AssertionError):
        bb.in_same_box(4, 19)

    assert bb.in_same_box(1, 3)
    assert not bb.in_same_box(3, 4)
    assert not bb.in_same_box(4, 3)


def test_blackbox_hidden_from(bb):
    assert bb.hidden_from(1, 4)
    assert not bb.hidden_from(1, 3)
    assert not bb.hidden_from(3, 4)


def test_blackbox_outputs_of(cg_bb):
    assert cg_bb.outputs_of(0) == (1, 3)
    assert cg_bb.outputs_of(1) == (4,)


def test_blackbox_len(bb, cg_bb):
    assert len(bb) == 2
    assert len(cg_bb) == 3


def test_rebuild_system_tpm(s):
    node0_tpm = np.array([
        [0, 1],
        [0, 0],
    ])
    node1_tpm = np.array([
        [0, 1],  # Singleton first dimension
    ])
    node_tpms = [node0_tpm, node1_tpm]

    answer = np.array([
        [[0, 0],
         [1, 1]],
        [[0, 0],
         [0, 1]]
    ])
    assert np.array_equal(macro.rebuild_system_tpm(node_tpms), answer)

    node_tpms = [node.tpm_on for node in s.nodes]
    assert np.array_equal(macro.rebuild_system_tpm(node_tpms), s.tpm)


def test_remove_singleton_dimensions():
    # Don't squeeze out last dimension of single-node tpm
    tpm = np.array([[0], [1]])
    assert macro.tpm_indices(tpm) == (0,)
    assert np.array_equal(macro.remove_singleton_dimensions(tpm), tpm)

    tpm = np.array([
        [[[0.,  0.,  1.]],
         [[1.,  0.,  0.]]]])
    assert macro.tpm_indices(tpm) == (1,)
    assert np.array_equal(macro.remove_singleton_dimensions(tpm), np.array([
        [0], [0]]))

    tpm = np.array([
        [[[0., 0., 0.],
          [1., 1., 0.]]],
        [[[0., 0., 1.],
          [1., 1., 1.]]]])
    assert macro.tpm_indices(tpm) == (0, 2)
    assert np.array_equal(macro.remove_singleton_dimensions(tpm), np.array([
        [[0., 0.],
         [1., 0.]],
        [[0., 1.],
         [1., 1.]]]))


def test_pack_attrs(s):
    attrs = macro.SystemAttrs.pack(s)
    assert np.array_equal(attrs.tpm, s.tpm)
    assert np.array_equal(attrs.cm, s.cm)
    assert attrs.node_indices == s.node_indices
    assert attrs.state == s.state
    assert attrs.nodes == s.nodes


def test_apply_attrs(s):
    attrs = macro.SystemAttrs.pack(s)

    class SomeSystem:
        pass
    target = SomeSystem()

    attrs.apply(target)
    assert np.array_equal(target.tpm, s.tpm)
    assert np.array_equal(target.cm, s.cm)
    assert target.node_indices == s.node_indices
    assert target.state == s.state
    assert target.nodes == s.nodes
