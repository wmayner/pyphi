#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tpm.py

"""
Functions for manipulating transition probability matrices.
"""

from itertools import chain, product

import functools

import numpy as np

import pandas as pd

from .constants import OFF, ON
from .utils import all_states, all_possible_states_nb
from .labels import NodeLabels

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


def tpm2df(tpm, num_states_per_node, node_labels):
    if num_states_per_node is None:
        return
    all_states = all_possible_states_nb(num_states_per_node)
    index = pd.MultiIndex.from_tuples(all_states, names=node_labels)
    columns = pd.MultiIndex.from_tuples(all_states, names=node_labels).reorder_levels(
        node_labels
    )
    return pd.DataFrame(tpm, index=index, columns=columns)


def condition_tpm_nb(
    tpm, fixed_nodes, state, num_states_per_node=None, node_labels=None, node_indices=None
):
    tpmdf = tpm2df(tpm, num_states_per_node, node_labels)

    # Generate a tuple where an element is the fixed state
    # if it is a fixed node, or a None slice elsewise.
    indexer = tuple(
        state[node]
        if node in fixed_nodes
        else slice(None)
        for node in range(len(state))
    )
    
    # Provided to the IndexSlice, will return all rows that fit
    # with the fixed states, throwing out the rest.
    tpmdf = tpmdf.loc[pd.IndexSlice[indexer]]
    
    free_node_indices = tuple(sorted(set(node_indices) - set(fixed_nodes)))

    # Shrink num_states_per_node to only free nodes
    num_states_per_node = tuple(
        num_states_per_node[node]
        for node in free_node_indices
    )

    free_nodes = [node_labels[node] for node in free_node_indices]

    # Marginalize out fixed nodes by grouping by free nodes, giving a conditioned tpm
    # with size free_node_states by free_node_states
    tpmdf = tpmdf.groupby(free_nodes, axis="columns").sum()

    # Update the subsystem's total states and free nodes
    return tpmdf, tpmdf.values, num_states_per_node, NodeLabels(free_nodes, free_node_indices, True), tuple(range(len(free_nodes)))


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
        np.array(tpm.shape)[list(node_indices)].prod()
    )


def infer_edge_nb(tpm, a, b, contexts):
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
        in each of its possible states, in order as a list.
        """
        a_states = [
            context[:a] + (i, ) + context[a:]
            for i in range(tpm.index.levels[a].size)
        ]
        return a_states

    # TODO not needed as a seperate method?
    def marginalize_b(state):
        """Return the distribution of possible states of b at t+1"""
        return tpm.groupby(tpm.columns.names[b], axis=1).sum().loc[state]

    def a_affects_b_in_context(context):
        """Return ``True`` if A has an effect on B, given a context."""
        a_states = a_in_context(context)
        comparator = marginalize_b(a_states[0]).round(12)
        return any(not comparator.equals(marginalize_b(state).round(12)) for state in a_states[1:])

    return any(a_affects_b_in_context(context) for context in contexts)


def infer_cm_nb(tpm):
    """Infer the connectivity matrix associated with a state-by-node TPM in
    multidimensional form.
    """
    network_size = len(tpm.index[0]) # Number of nodes in the network
    all_contexts = tuple(all_states(network_size - 1))
    cm = np.empty((network_size, network_size), dtype=int)
    for a, b in np.ndindex(cm.shape):
        cm[a][b] = infer_edge_nb(tpm, a, b, all_contexts)
    return cm

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
    node_tpms = [tpm.squeeze(axis=subsystem.external_indices) for tpm in node_tpms]
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


def tensor(a, b):
    return functools.reduce(
        lambda a, b: np.concatenate((a, b), axis=1),
        [
            np.transpose(np.multiply(np.transpose(a), b[:, c]))
            for c in range(b.shape[-1])
        ],
    )


def cut_tpm(subsystem, from_nodes, to_nodes):
    node_tpms = expand_node_tpms(
        subsystem, cut_node_tpms(subsystem, from_nodes, to_nodes)
    )
    tpm = functools.reduce(
        lambda x, y: tensor(x, y), [node_tpm.values for node_tpm in node_tpms]
    )
    # Normalize so all rows sum to 1
    return tpm * (1 / np.sum(tpm, axis=1))


def cut_node_tpms(subsystem, from_nodes, to_nodes):
    # Get the connectivity of the system in dictionary form
    connections = subsystem.connections()
    node_tpms = []

    from_nodes = [subsystem.node_labels[c] for c in from_nodes]
    to_nodes = [subsystem.node_labels[c] for c in to_nodes]

    # Create the TPM for each element in the subsystem
    for element in subsystem.node_indices:
        inputs = connections[subsystem.node_labels[element]]

        remain_connections = (
            list(set(inputs) - set(from_nodes))
            if subsystem.node_labels[element] in to_nodes
            else inputs
        )
        if remain_connections == list(
            subsystem.node_labels
        ):  # if the elment is connected to everyone (including itself), no marg. of any input
            element_node = subsystem.node_labels[element]
            node_tpms += [subsystem.tpmdf.groupby(element_node, axis=1).sum()]
        else:
            if (
                not remain_connections
            ):  # element got disconnected, so it depends on itself
                remain_connections = [subsystem.node_labels[element]]

            factor_set = tuple(
                [
                    list(subsystem.node_labels).index(i)
                    for i in tuple(
                        set(list(subsystem.node_labels)) - set(remain_connections)
                    )
                ]
            )

            factor = sum([subsystem.network.num_states_per_node[i] for i in factor_set])

            element_node = subsystem.node_labels[element]

            node_tpms += [
                (
                    subsystem.tpmdf.groupby(element_node, axis=1)
                    .sum()
                    .groupby(remain_connections)
                    .sum()
                )
                * (1 / factor)
            ]

    return node_tpms


def expand_node_tpms(subsystem, node_tpms):
    sys = list(subsystem.node_labels)
    expanded_node_tpms = list()
    a = subsystem.tpmdf.reset_index()[sys]
    a.columns = sys

    # Expand each element TPM
    for df in node_tpms:
        b = df.reset_index()
        b.columns = list(b.columns)
        inter = list(set(a.columns).intersection(set(b.columns)))
        inter.sort(reverse=True)
        expanded = pd.merge(a, b, on=inter).sort_values(by=sys[::-1])
        expanded = expanded[list(set(expanded.columns) - set(sys))].values
        expanded_node_tpms.append(
            pd.DataFrame(expanded, columns=df.columns, index=subsystem.tpmdf.index)
        )

    return expanded_node_tpms
