#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# state_space.py

"""
Constants and utility functions for dealing with the state space of a |Network|.
"""

from typing import Iterable, List, Optional, Union, Tuple

from .data_structures import FrozenMap


PARENT_DIMENSION_PREFIX = "input_"
PROBABILITY_DIMENSION_LABEL = "Pr"


def dimension_labels(node_labels: Iterable[str]) -> List[str]:
    """Generate labels for each dimension in the |ImplicitTPM|.

    data_vars (xr.DataArray node names) and dimension names share the
    same dictionary-like namespace in xr.Dataset. Prepend constant
    string to avoid the conflict.

    Arguments:
        node_labels (Iterable[str]): Textual labels for each node in the network.

    Returns:
        List[str]: Textual labels for each dimension in a multidimensional TPM.
    """
    return [PARENT_DIMENSION_PREFIX + label for label in node_labels] + [
        PROBABILITY_DIMENSION_LABEL
    ]


def build_state_space(
        node_labels: Iterable[str],
        nodes_shape: Iterable[int],
        node_states: Optional[Iterable[Iterable[Union[int, str]]]] = None,
        singleton_state_space: Optional[Iterable[Union[int, str]]] = None,
) -> Tuple[FrozenMap[str, Tuple[Union[int, str]]], int]:
    """Format the passed state space labels or construct defaults if none.

    Arguments:
        node_labels (Iterable[str]): Textual labels for each node in the network.
        nodes_shape (Iterable[int]): The first |n| components in the shape of
            a multidimensional |ExplicitTPM|, where |n| is the number of nodes
            in the network.

    Keyword Args:
        node_states (Optional[Iterable[Iterable[Union[int, str]]]]): The
            network's state space labels as provided by the user.
        singleton_state_space (Optional[Iterable[Union[int, str]]]): The label
            to be used for singleton dimensions. If ``None``, singleton
            dimensions will be discarded.

    Returns:
        Tuple[Tuple[Tuple[Union[int, str]]], int]: State space for the network
            of interest and its hash.
    """
    if node_states is None:
        node_states = [tuple(range(dim)) for dim in nodes_shape]
    else:
        node_states = [tuple(n) for n in node_states]

    # labels-to-states map.
    state_space = zip(dimension_labels(node_labels), node_states)

    # Filter out states of singleton dimensions.
    shape_state_map = zip(nodes_shape, state_space)

    if singleton_state_space is None:
        state_space = dict(
            node_states
            for dim, node_states in shape_state_map
            if dim > 1
        )

    else:
        state_space = dict(
            node_states if dim > 1 else singleton_state_space
            for dim, node_states in shape_state_map
        )

    state_space = FrozenMap(state_space)
    state_space_hash = hash(state_space)
    state_space = FrozenMap({k: list(v) for k,v in state_space.items()})

    return (state_space, state_space_hash)
