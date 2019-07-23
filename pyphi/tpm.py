#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tpm.py

"""
Functions for manipulating transition probability matrices.
"""

from itertools import chain

import numpy as np

from .constants import OFF, ON
from .utils import all_states


def tpm_indices(tpm):
    """Return the indices of nodes in the TPM."""
    return tuple(np.where(np.array(tpm.shape[:-1]) == 2)[0])


def is_deterministic(tpm):
    """Return whether the TPM is deterministic."""
    return np.all(np.logical_or(tpm == 1, tpm == 0))


def is_state_by_state(tpm):
    """Return ``True`` if ``tpm`` is in state-by-state form, otherwise
    ``False``.
    """
    return tpm.ndim == 2 and tpm.shape[0] == tpm.shape[1]


def condition_tpm(tpm, fixed_nodes, state):
    """Return a TPM conditioned on the given fixed node indices, whose states
    are fixed according to the given state-tuple.

    The dimensions of the new TPM that correspond to the fixed nodes are
    collapsed onto their state, making those dimensions singletons suitable for
    broadcasting. The number of dimensions of the conditioned TPM will be the
    same as the unconditioned TPM.
    """
    conditioning_indices = [[slice(None)]] * len(state)
    for i in fixed_nodes:
        # Preserve singleton dimensions with `np.newaxis`
        conditioning_indices[i] = [state[i], np.newaxis]
    # Flatten the indices.
    conditioning_indices = list(chain.from_iterable(conditioning_indices))
    # Obtain the actual conditioned TPM by indexing with the conditioning
    # indices.
    return tpm[tuple(conditioning_indices)]


def expand_tpm(tpm):
    """Broadcast a state-by-node TPM so that singleton dimensions are expanded
    over the full network.
    """
    unconstrained = np.ones([2] * (tpm.ndim - 1) + [tpm.shape[-1]])
    return tpm * unconstrained


def marginalize_out(node_indices, tpm):
    """Marginalize out nodes from a TPM.

    Args:
        node_indices (list[int]): The indices of nodes to be marginalized out.
        tpm (np.ndarray): The TPM to marginalize the node out of.

    Returns:
        np.ndarray: A TPM with the same number of dimensions, with the nodes
        marginalized out.
    """
    return tpm.sum(tuple(node_indices), keepdims=True) / (
        np.array(tpm.shape)[list(node_indices)].prod())


def infer_edge(tpm, a, b, contexts):
    """Infer the presence or absence of an edge from node A to node B.

    Let |S| be the set of all nodes in a network. Let |A' = S - {A}|. We call
    the state of |A'| the context |C| of |A|. There is an edge from |A| to |B|
    if there exists any context |C(A)| such that |Pr(B | C(A), A=0) != Pr(B |
    C(A), A=1)|.

    Args:
        tpm (np.ndarray): The TPM in state-by-node, multidimensional form.
        a (int): The index of the putative source node.
        b (int): The index of the putative sink node.
    Returns:
        bool: ``True`` if the edge |A -> B| exists, ``False`` otherwise.
    """

    def a_in_context(context):
        """Given a context C(A), return the states of the full system with A
        OFF and ON, respectively.
        """
        a_off = context[:a] + OFF + context[a:]
        a_on = context[:a] + ON + context[a:]
        return (a_off, a_on)

    def a_affects_b_in_context(context):
        """Return ``True`` if A has an effect on B, given a context."""
        a_off, a_on = a_in_context(context)
        return tpm[a_off][b] != tpm[a_on][b]

    return any(a_affects_b_in_context(context) for context in contexts)


def infer_cm(tpm):
    """Infer the connectivity matrix associated with a state-by-node TPM in
    multidimensional form.
    """
    network_size = tpm.shape[-1]
    all_contexts = tuple(all_states(network_size - 1))
    cm = np.empty((network_size, network_size), dtype=int)
    for a, b in np.ndindex(cm.shape):
        cm[a][b] = infer_edge(tpm, a, b, all_contexts)
    return cm


def reconstitute_tpm(subsystem):
    """Reconstitute the TPM of a subsystem using the individual node TPMs."""
    # The last axis of the node TPMs correponds to ON or OFF probabilities
    # (used in the conditioning step when calculating the repertoires); we want
    # ON probabilities.
    node_tpms = [node.tpm[..., 1] for node in subsystem.nodes]
    # Remove the singleton dimensions corresponding to external nodes
    node_tpms = [
        tpm.squeeze(axis=subsystem.external_indices)
        for tpm in node_tpms
    ]
    # We add a new singleton axis at the end so that we can use
    # pyphi.tpm.expand_tpm, which expects a state-by-node TPM (where the last
    # axis corresponds to nodes.)
    node_tpms = [np.expand_dims(tpm, -1) for tpm in node_tpms]
    # Now we expand the node TPMs to the full state space, so we can combine
    # them all (this uses the maximum entropy distribution).
    node_tpms = list(map(expand_tpm, node_tpms))
    # We concatenate the node TPMs along a new axis to get a multidimensional
    # state-by-node TPM (where the last axis corresponds to nodes).
    return np.concatenate(node_tpms, axis=-1)
