# compute/network.py
"""Functions for computing network-level properties."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator
from typing import TYPE_CHECKING
from typing import Any

from pyphi import conf
from pyphi import exceptions
from pyphi import utils
from pyphi import validate
from pyphi.conf import config
from pyphi.core import CandidateSystem as Subsystem
from pyphi.models import SystemIrreducibilityAnalysis
from pyphi.models import _null_sia
from pyphi.parallel import MapReduce
from pyphi.types import State

from .subsystem import sia

if TYPE_CHECKING:
    from pyphi.network import Network

# Create a logger for this module.
log = logging.getLogger(__name__)


def reachable_subsystems(
    network: Network,
    indices: tuple[int, ...],
    state: State,
    **kwargs: Any,
) -> Generator[Subsystem]:
    """A generator over all subsystems in a valid state."""
    validate.is_network(network)

    # Return subsystems largest to smallest to optimize parallel
    # resource usage.
    for subset in utils.powerset(indices, nonempty=True, reverse=True):
        with contextlib.suppress(exceptions.StateUnreachableError):
            yield Subsystem.from_network(network, state, subset, **kwargs)


def subsystems(network: Network, state: State, **kwargs: Any) -> Generator[Subsystem]:
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


def possible_complexes(
    network: Network, state: State, **kwargs: Any
) -> Generator[Subsystem]:
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


def all_complexes(
    network: Network,
    state: State,
    parallel_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[SystemIrreducibilityAnalysis]:
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
    pkwargs = conf.parallel_kwargs(
        config.infrastructure.parallel_complex_evaluation, **(parallel_kwargs or {})
    )
    result = MapReduce(
        sia,
        possible_complexes(network, state, **kwargs),
        total=2 ** len(network) - 1,
        map_kwargs={"progress": False},
        desc="Evaluating complexes",
        **pkwargs,  # type: ignore[arg-type]  # parallel_kwargs contains MapReduce params
    ).run()
    assert result is not None
    return result


def complexes(
    network: Network, state: State, **kwargs: Any
) -> list[SystemIrreducibilityAnalysis]:
    """Return all irreducible complexes of the network.

    Args:
        network (Network): The |Network| of interest.
        state (tuple[int]): The state of the network (a binary tuple).

    Yields:
        SystemIrreducibilityAnalysis: A |SIA| for each |Subsystem| of the
        |Network|, excluding those with |big_phi = 0|.
    """
    return list(filter(None, all_complexes(network, state, **kwargs)))


def major_complex(
    network: Network, state: State, **kwargs: Any
) -> SystemIrreducibilityAnalysis:
    """Return the major complex of the network.

    Args:
        network (Network): The |Network| of interest.
        state (tuple[int]): The state of the network (a binary tuple).

    Returns:
        SystemIrreducibilityAnalysis: The |SIA| for the |Subsystem| with
        maximal |big_phi|.
    """
    log.info("Calculating major complex...")
    empty_subsystem = Subsystem.from_network(network, state, ())
    default = _null_sia(empty_subsystem)
    pkwargs = conf.parallel_kwargs(
        config.infrastructure.parallel_complex_evaluation, **kwargs
    )
    result = MapReduce(
        sia,
        possible_complexes(network, state),
        map_kwargs={"progress": False},
        reduce_func=max,
        reduce_kwargs={"default": default},
        total=2 ** len(network) - 1,
        desc="Evaluating complexes",
        **pkwargs,  # type: ignore[arg-type]  # parallel_kwargs contains MapReduce params
    ).run()
    log.info("Finished calculating major complex.")
    assert result is not None
    return result


def condensed(
    network: Network, state: State, **kwargs: Any
) -> list[SystemIrreducibilityAnalysis]:
    """Return a list of maximal non-overlapping complexes.

    Args:
        network (Network): The |Network| of interest.
        state (tuple[int]): The state of the network (a binary tuple).

    Returns:
        list[SystemIrreducibilityAnalysis]: A list of |SIA| for non-overlapping
        complexes with maximal |big_phi| values.
    """
    result: list[SystemIrreducibilityAnalysis] = []
    covered_nodes: set[int] = set()

    for c in sorted(complexes(network, state, **kwargs), reverse=True):
        if c.subsystem is not None and not any(
            n in covered_nodes for n in c.subsystem.node_indices
        ):
            result.append(c)
            covered_nodes = covered_nodes | set(c.subsystem.node_indices)

    return result
