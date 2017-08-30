#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tpm.py

'''
Functions for manipulating transition probability matrices.
'''

from itertools import chain

import numpy as np

from .constants import OFF, ON
from .utils import all_states


def tpm_indices(tpm):
    '''Indices of nodes in the TPM.'''
    return tuple(np.where(np.array(tpm.shape[:-1]) == 2)[0])


def is_state_by_state(tpm):
    '''Return ``True`` if ``tpm`` is in state-by-state form, otherwise
    ``False``.'''
    return tpm.ndim == 2 and tpm.shape[0] == tpm.shape[1]


def condition_tpm(tpm, fixed_nodes, state):
    '''Return a TPM conditioned on the given fixed node indices, whose states
    are fixed according to the given state-tuple.

    The dimensions of the new TPM that correspond to the fixed nodes are
    collapsed onto their state, making those dimensions singletons suitable for
    broadcasting. The number of dimensions of the conditioned TPM will be the
    same as the unconditioned TPM.
    '''
    conditioning_indices = [[slice(None)]] * len(state)
    for i in fixed_nodes:
        # Preserve singleton dimensions with `np.newaxis`
        conditioning_indices[i] = [state[i], np.newaxis]
    # Flatten the indices.
    conditioning_indices = list(chain.from_iterable(conditioning_indices))
    # Obtain the actual conditioned TPM by indexing with the conditioning
    # indices.
    return tpm[conditioning_indices]


def expand_tpm(tpm):
    '''Broadcast a state-by-node TPM so that singleton dimensions are expanded
    over the full network.'''
    uc = np.ones([2] * (tpm.ndim - 1) + [tpm.shape[-1]])
    return tpm * uc


def marginalize_out(indices, tpm):
    '''Marginalize out a node from a TPM.

    Args:
        indices (list[int]): The indices of nodes to be marginalized out.
        tpm (np.ndarray): The TPM to marginalize the node out of.

    Returns:
        np.ndarray: A TPM with the same number of dimensions, with the nodes
        marginalized out.
    '''
    return tpm.sum(tuple(indices), keepdims=True) / (
        np.array(tpm.shape)[list(indices)].prod())


def infer_edge(tpm, a, b, contexts):
    '''Infer the presence or absence of an edge from node A to node B.

    Let S be the set of all nodes in a network. Let A' = S - {A}.
    We call the state of A' the context C of A.
    There is an edge from A to B if there exists any context C(A) such that
    p(B | C(A), A=0) =/= p(B | C(A), A=1).

    Args:
        tpm (np.ndarray): The TPM in state-by-node, n-dimensional form.
        a (int): The index of the putative source node.
        b (int): The index of the putative sink node.
    Returns:
        bool: True if the edge A->B exists, False otherwise.
    '''

    def a_in_context(context):
        '''Given a context C(A), return the states of the full system with A
        off and on, respectively.'''
        a_off = context[:a] + OFF + context[a:]
        a_on = context[:a] + ON + context[a:]
        return (a_off, a_on)

    def a_affects_b_in_context(context):
        '''Returns True if A has an effect on B, given a context.'''
        a_off, a_on = a_in_context(context)
        return tpm[a_off][b] != tpm[a_on][b]

    return any(a_affects_b_in_context(context) for context in contexts)


def infer_cm(tpm):
    '''Infer the connectivity matrix associated with a state-by-node TPM in
    n-dimensional form.'''
    network_size = tpm.shape[-1]
    all_contexts = tuple(all_states(network_size - 1))
    cm = np.empty((network_size, network_size), dtype=int)
    for a, b in np.ndindex(cm.shape):
        cm[a][b] = infer_edge(tpm, a, b, all_contexts)
    return cm
