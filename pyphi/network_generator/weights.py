# -*- coding: utf-8 -*-
# network_generator/weights.py
"""Generate different weight matrices."""

import functools
from itertools import product

import numpy as np


def normalize_inputs(W):
    return np.nan_to_num(W / W.sum(axis=0, keepdims=True), nan=0.0)


def normalize_outputs(W):
    return np.nan_to_num(W / W.sum(axis=1, keepdims=True), nan=0.0)


def _optionally_normalize_inputs(func):
    @functools.wraps(func)
    def wrapper(*args, normalize_input_weights=False, **kwargs):
        W = func(*args, **kwargs)
        if normalize_input_weights:
            W = normalize_inputs(W)
        return W

    return wrapper


def pareto_distribution(n, alpha=1.0):
    return np.ones(n) / np.arange(1, n + 1) ** alpha


@_optionally_normalize_inputs
def nearest_neighbor(
    size,
    weights,
    k=1,
    periodic=False,
    **kwargs,
):
    if periodic:
        k_limit = (size + 1) / 2
    else:
        k_limit = size
    if k >= k_limit:
        raise ValueError(f"k must be <= {k_limit} (n={size}, periodic={periodic})")

    weights = np.ones(k + 1) * weights
    # Self loops
    W = weights[0] * np.eye(size)
    # Laterals
    for i in range(1, k + 1):
        for j in [-1, 1]:
            W += weights[i] * np.eye(size, k=j * i)
            # Don't double-weight farthest laterals if periodic
            if periodic and i < (size / 2):
                W += weights[i] * np.eye(size, k=j * (size - i))
    return W


@_optionally_normalize_inputs
def pareto(size, alpha=1.0, periodic=False):
    W = np.zeros([size, size])
    p = pareto_distribution(size, alpha=alpha)
    if periodic:
        middle = size // 2
        if size % 2:
            # Odd
            p = np.concatenate([p[: middle + 1], np.flip(p[1 : middle + 1])])
        else:
            # Even
            p = np.concatenate([p[: middle + 1], np.flip(p[1:middle])])
    for k, w in zip(range(size), p):
        W += np.eye(size, k=k) * w
        if k:
            W += np.eye(size, k=-k) * w
    return W


def potentiate_self(W, node_indicator, amount=0.0):
    potentiation = np.eye(len(W)) * node_indicator * amount
    return W + potentiation


def potentiate_laterals(W, node_indicator, amount=0.0):
    W = W.copy()
    for i, j in product(range(W.shape[0]), range(W.shape[1])):
        if node_indicator[i] != node_indicator[j]:
            W[i, j] += amount
    return W


def lateral_indices_from(W):
    n = len(W)
    idx = list(range(n))
    rows = np.repeat(idx, n - 1)
    cols = np.concatenate([idx[:i] + idx[i + 1 :] for i in idx])
    return rows, cols


def get_laterals(W):
    return W * (1 - np.eye(len(W)))


def separate_weights(W, node_indicator):
    """Separate weights between two sets of nodes."""
    hom = np.zeros(W.shape)
    het = np.zeros(W.shape)
    for i, j in product(range(W.shape[0]), range(W.shape[1])):
        if node_indicator[i] == node_indicator[j]:
            hom[i, j] = W[i, j]
        else:
            het[i, j] = W[i, j]
    return hom, het


def on_weights(W, node_indicator):
    weights = np.zeros(W.shape)
    nodes = np.arange(len(node_indicator))[np.array(node_indicator, dtype=bool)]
    ix = np.ix_(nodes, nodes)
    weights[ix] = W[ix]
    return weights


def get_cross_laterals(W, node_indicator):
    return separate_weights(get_laterals(W), node_indicator)


def symmetric_triu(W):
    return np.triu(W) + np.tril(W.T, k=-1)


def compensatory_potentiation(W, node_indicator, self_amount=0.0, lateral_amount=0.0):
    node_indicator = np.array(node_indicator)
    self_potentiation = self_amount * np.eye(len(W)) * node_indicator

    _, het = separate_weights(W, node_indicator)
    het_laterals = get_laterals(het)
    on_laterals = get_laterals(on_weights(W, node_indicator))

    # Adjust ON-laterals proportionally to output
    on_proportion = normalize_outputs(on_laterals)
    # Make lateral potentiation symmetric
    on_proportion = symmetric_triu(on_proportion)
    on_potentiation = lateral_amount * on_proportion

    # Compensate heterogeneous laterals proportionally to output
    het_proportion = normalize_outputs(het_laterals)
    # Make lateral potentiation symmetric
    het_proportion = symmetric_triu(het_proportion)
    het_potentiation = -1 * (self_amount + lateral_amount) * het_proportion

    potentiation = self_potentiation + on_potentiation + het_potentiation

    # Potentiate self loops to offset reduction in input
    compensatory_self_potentiation = (
        -1 * np.eye(len(W)) * potentiation.sum(axis=0, keepdims=True)
    )
    return W + potentiation + compensatory_self_potentiation


def compensatory_pareto(
    size,
    alpha,
    periodic,
    normalize_input_weights,
    w_self_potentiation,
    w_lateral_potentiation,
    layer_state,
    **kwargs,
):
    return compensatory_potentiation(
        pareto(
            size,
            alpha=alpha,
            periodic=periodic,
            normalize=normalize_input_weights,
        ),
        node_indicator=layer_state,
        self_amount=w_self_potentiation,
        lateral_amount=w_lateral_potentiation,
    )


@_optionally_normalize_inputs
def join_weights(weights_1, weights_2, weights_feedforward, weights_feedback):
    N, M = weights_1.shape[0], weights_2.shape[0]
    W = np.zeros([N + M] * 2)
    W[:N, :N] = weights_1
    W[N:, N:] = weights_2
    W[:N, N:] = weights_feedforward
    W[N:, :N] = weights_feedback
    return W


@_optionally_normalize_inputs
def bridge(weights1, weights2, w_forward=1, w_back=0.0):
    N, M = weights1.shape[0], weights2.shape[0]
    feedforward = np.zeros([N, M])
    feedforward[0, 0] = w_forward
    feedback = np.zeros([M, N])
    feedback[0, 0] = w_back
    return join_weights(weights1, weights2, feedforward, feedback)


def copy_loop(size):
    W = np.eye(size, size, k=1)
    W[-1, 0] = 1.0
    return W
