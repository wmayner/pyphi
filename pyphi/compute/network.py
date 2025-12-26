# compute/network.py
"""Functions for computing network-level properties."""

import logging

from .. import conf, exceptions, utils, validate
from ..conf import config
from ..models import _null_sia
from ..subsystem import Subsystem
from ..parallel import MapReduce
from .subsystem import sia

# Create a logger for this module.
log = logging.getLogger(__name__)


def reachable_subsystems(network, indices, state, **kwargs):
    """A generator over all subsystems in a valid state."""
    validate.is_network(network)

    # Return subsystems largest to smallest to optimize parallel
    # resource usage.
    for subset in utils.powerset(indices, nonempty=True, reverse=True):
        try:
            yield Subsystem(network, state, subset, **kwargs)
        except exceptions.StateUnreachableError:
            pass


def subsystems(network, state, **kwargs):
    """Return a generator of all **possible** subsystems of a network.

    .. note::
        Does not return subsystems that are in an impossible state (after
        conditioning the subsystem TPM on the state of the other nodes).

    Args:
        network (Network): The |Network| of interest.
        state (tuple[int]): The state of the network (a binary tuple).

    Yields:
        Subsystem: A |Subsystem| for each subset of nodes in the network,
        excluding subsystems that would be in an impossible state.
    """
    return reachable_subsystems(network, network.node_indices, state, **kwargs)


def possible_complexes(network, state, **kwargs):
    """Return a generator of subsystems of a network that could be a complex.

    This is the just powerset of the nodes that have at least one input and
    output (nodes with no inputs or no outputs cannot be part of a main
    complex, because they do not have a causal link with the rest of the
    subsystem in the previous or next timestep, respectively).

    .. note::
        Does not return subsystems that are in an impossible state (after
        conditioning the subsystem TPM on the state of the other nodes).

    Args:
        network (Network): The |Network| of interest.
        state (tuple[int]): The state of the network (a binary tuple).

    Yields:
        Subsystem: The next subsystem that could be a complex.
    """
    return reachable_subsystems(
        network, network.causally_significant_nodes, state, **kwargs
    )


def all_complexes(network, state, parallel_kwargs=None, **kwargs):
    """Return a generator for all complexes of the network.

    .. note::
        Includes reducible, zero-|big_phi| complexes (which are not, strictly
        speaking, complexes at all).

    Args:
        network (Network): The |Network| of interest.
        state (tuple[int]): The state of the network (a binary tuple).

    Yields:
        SystemIrreducibilityAnalysis: A |SIA| for each |Subsystem| of the
        |Network|.
    """
    parallel_kwargs = conf.parallel_kwargs(
        config.PARALLEL_COMPLEX_EVALUATION, **(parallel_kwargs or dict())
    )
    return MapReduce(
        sia,
        possible_complexes(network, state, **kwargs),
        total=2 ** len(network) - 1,
        map_kwargs=dict(progress=False),
        desc="Evaluating complexes",
        **parallel_kwargs,
    ).run()


def complexes(network, state, **kwargs):
    """Return all irreducible complexes of the network.

    Args:
        network (Network): The |Network| of interest.
        state (tuple[int]): The state of the network (a binary tuple).

    Yields:
        SystemIrreducibilityAnalysis: A |SIA| for each |Subsystem| of the
        |Network|, excluding those with |big_phi = 0|.
    """
    return list(filter(None, all_complexes(network, state, **kwargs)))


def major_complex(network, state, **kwargs):
    """Return the major complex of the network.

    Args:
        network (Network): The |Network| of interest.
        state (tuple[int]): The state of the network (a binary tuple).

    Returns:
        SystemIrreducibilityAnalysis: The |SIA| for the |Subsystem| with
        maximal |big_phi|.
    """
    log.info("Calculating major complex...")
    empty_subsystem = Subsystem(network, state, ())
    default = _null_sia(empty_subsystem)
    parallel_kwargs = conf.parallel_kwargs(config.PARALLEL_COMPLEX_EVALUATION, **kwargs)
    result = MapReduce(
        sia,
        possible_complexes(network, state),
        map_kwargs=dict(progress=False),
        reduce_func=max,
        reduce_kwargs=dict(default=default),
        total=2 ** len(network) - 1,
        desc="Evaluating complexes",
        **parallel_kwargs,
    ).run()
    log.info("Finished calculating major complex.")
    return result


def condensed(network, state, **kwargs):
    """Return a list of maximal non-overlapping complexes.

    Args:
        network (Network): The |Network| of interest.
        state (tuple[int]): The state of the network (a binary tuple).

    Returns:
        list[SystemIrreducibilityAnalysis]: A list of |SIA| for non-overlapping
        complexes with maximal |big_phi| values.
    """
    result = []
    covered_nodes = set()

    for c in reversed(sorted(complexes(network, state, **kwargs))):
        if not any(n in covered_nodes for n in c.subsystem.node_indices):
            result.append(c)
            covered_nodes = covered_nodes | set(c.subsystem.node_indices)

    return result
