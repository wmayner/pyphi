#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_tpm_obj.py

import numpy as np
import pytest
import pandas as pd
from itertools import product

from pyphi.__tpm import TPM
from pyphi.__tpm import SbN


# TODO for harsher tests, may want to ensure that node labels properly direct you to the correct position
# But only necessary if signficant changes are made to shaping as results in notebook testing 
# indicate it works
def test_init():
    tpm = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Nothing but data given
    test = TPM(tpm)
    assert test.tpm.shape == (2,2,2,2)
    assert test.tpm.dims == ("n0_p", "n1_p", "n0_n", "n1_n")

    # Symmetric, binary
    test = TPM(tpm, p_nodes=["A", "B"])

    assert test.tpm.shape == (2, 2, 2, 2)
    assert test.tpm.dims == ("A_p", "B_p", "n0_n", "n1_n")

    # Symmetric, nonbinary, wrong shape
    with pytest.raises(ValueError):
        TPM(tpm, p_nodes=["A", "B"], p_states=[2, 3])

    # Asymmetric, binary
    tpm = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
    test = TPM(tpm, p_nodes=["A", "B"], n_nodes=["C"])

    assert test.tpm.shape == (2, 2, 2)
    assert test.tpm.dims == ("A_p", "B_p", "C_n")

    # Asymmetric, nonbinary
    tpm = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0]])
    test = TPM(tpm, p_nodes=["A", "B"], n_nodes=["C"], p_states=[2, 2], n_states=[3])

    assert test.tpm.shape == (2, 2, 3)
    assert test.tpm.dims == ("A_p", "B_p", "C_n")

    # Symmetric, nonbinary, right shape
    tpm = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    test = TPM(tpm, p_nodes=["A", "B"], p_states=[2, 3])

    assert test.tpm.shape == (2, 3, 2, 3)
    assert test.tpm.dims == ("A_p", "B_p", "A_n", "B_n")

    # DataFrame
    tpm = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    base = [2, 2]
    nam = ["A", "B"]

    states_node=[list(range(b)) for b in base]

    sates_all_nodes=[list(x[::-1])  for x in  list(product(*states_node[::-1])) ]
    sates_all_nodes=np.transpose(sates_all_nodes).tolist()
    index = pd.MultiIndex.from_arrays(sates_all_nodes, names=nam)
    columns = pd.MultiIndex.from_arrays(sates_all_nodes, names=nam)

    df = pd.DataFrame(tpm,columns=columns, index=index)
    test = TPM(df, ["C", "D"]) #TODO should only need one argument, test ensures second arg doesn't influence naming
    
    assert test.tpm.shape == (2, 2, 2, 2)
    assert test.tpm.dims == ("A_p", "B_p", "A_n", "B_n")

    # SbN Network, check both init paths give same results
    can = np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0]])

    can_sbn = np.array([[0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 0],
                        [0, 0, 0],
                        [1, 0, 1],
                        [1, 1, 1],
                        [1, 0, 0],
                        [1, 1, 0]])

    can_obj = SbN(can, p_nodes=["A", "B", "C"], p_states=[2, 2, 2])

    can_sbn_obj = SbN(can_sbn, p_nodes=["A", "B", "C"], p_states=[2, 2, 2], n_nodes=["A", "B", "C"])
    
    assert np.array_equal(can_obj.tpm.data, can_sbn_obj.tpm.data)
    assert can_obj.p_nodes == can_sbn_obj.p_nodes
    assert can_obj.n_nodes == can_sbn_obj.n_nodes

    # Test only data given
    can_obj = SbN(can)
    can_sbn_obj = SbN(can_sbn)

    assert np.array_equal(can_obj.tpm.data, can_sbn_obj.tpm.data)
    assert can_obj.p_nodes == can_sbn_obj.p_nodes
    assert can_obj.n_nodes == can_sbn_obj.n_nodes

def test_marginalize_out_obj():
    p53 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])

    p53_obj = TPM(p53, p_nodes=["P", "Mc", "Mn"], p_states=[3, 2, 2])

    marginalized = p53_obj.marginalize_out((1, )).data

    solution = np.array([[0, 0, 0, 0, 0, 1],
                         [0, 0, 0.5, 0, 0, 0.5],
                         [0, 0, 0.5, 0, 0, 0.5],
                         [0, 0, 0, 1, 0, 0],
                         [0.5, 0, 0, 0.5, 0, 0],
                         [0.5, 0, 0, 0.5, 0, 0]]).reshape(3, 1, 2, 3, 1, 2, order="F")

    assert np.array_equal(marginalized, solution)

def test_marginalize_out_sbn(s):
    sbn = SbN(s.tpm, p_nodes=["A", "B", "C"], n_nodes=["A", "B", "C"], p_states=[2,2,2], n_states=[2,2,2])

    marginalized_distribution = sbn.marginalize_out((0, )).data

    answer = np.array([
        [[[0.0, 0.0, 0.5],
          [1.0, 1.0, 0.5]],
         [[1.0, 0.0, 0.5],
          [1.0, 1.0, 0.5]]],
    ])
    assert np.array_equal(marginalized_distribution, answer)

    marginalized_distribution = sbn.marginalize_out((0, 1)).data

    answer = np.array([
        [[[0.5, 0.0, 0.5],
          [1.0, 1.0, 0.5]]],
    ])

    assert np.array_equal(marginalized_distribution, answer)

def test_infer_cm_obj():
    # Check infer_cm functions on both TPM (state-by-state)
    # and SbN (state-by-node) objects 
    p53 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])

    test_p53 = TPM(p53, p_nodes=["P", "Mc", "Mn"], n_nodes=["P", "Mc", "Mn"], p_states=[3,2,2], n_states=[3,2,2])

    solution = np.array([[0, 1, 1],
                         [0, 0, 1],
                         [1, 0, 0]])

    assert np.array_equal(test_p53.infer_cm(), solution)

def test_infer_cm_sbn():
    # SbN Binary network: Node A (c)opies C, node B takes A (a)nd C, and C (n)ots B
    can = np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0]])

    can_obj = SbN(can, p_nodes=["A", "B", "C"], n_nodes=["A", "B", "C"],
                        p_states=[2,2,2], n_states=[2,2,2])
    solution = np.array([[0, 1, 0],
                         [0, 0, 1],
                         [1, 1, 0]])

    assert np.array_equal(can_obj.infer_cm(), solution)

def test_condition_obj():
    p53 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])

    test_p53 = TPM(p53, p_nodes=["P", "Mc", "Mn"], n_nodes=["P", "Mc", "Mn"], p_states=[3,2,2], n_states=[3,2,2])

    conditioned = test_p53.condition((1, ), (1, 0, 1))

    solution = np.array([[0, 0, 0, 0, 0, 1],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [1, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0]]).reshape(3, 1, 2, 3, 1, 2, order="F")
    
    # Dimensions should remain the same, but Mc has been conditioned on making its
    # dimension one unit in size
    assert conditioned.shape == (3, 1, 2, 3, 1, 2)

    assert np.array_equal(conditioned.data, solution)

def test_condition_sbn():
    # SbN form testing
    can_sbn = np.array([[0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 0],
                        [0, 0, 0],
                        [1, 0, 1],
                        [1, 1, 1],
                        [1, 0, 0],
                        [1, 1, 0]])
    
    can_obj = SbN(can_sbn, p_nodes=["A", "B", "C"], n_nodes=["A", "B", "C"], 
                   p_states=[2,2,2])
    
    conditioned = can_obj.condition((1,), (1, 0, 1))

    # At present we don't drop unneeded columns, though could if needed the space
    solution = np.array([[0, 0, 1],
                         [0, 0, 1],
                         [1, 0, 1],
                         [1, 1, 1]]).reshape(2, 1, 2, 3, order="F")

    assert conditioned.shape == (2, 1, 2, 3)

    assert np.array_equal(conditioned.data, solution)
    