#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# iit_2_0.py

"""Implementation of IIT 2.0

All page numbers and figures reference the IIT 2.0 paper:

`Integrated Information in Discrete Dynamical Systems: Motivation and
Theoretical Framework <https://doi.org/10.1371/journal.pcbi.1000091>`_
by David Balduzzi and Giulio Tononi.
"""

import collections
import functools
import itertools

import numpy as np

import pyphi
from pyphi import config, exceptions, utils
from pyphi.distribution import entropy, max_entropy_distribution
from pyphi.models import cmp
from pyphi.partition import bipartition


class Subsystem:
    """A subsystem for IIT 2.0 computations.

    This system takes the same arguments as the standard PyPhi subsystem.

    Args:
        network (Network): A PyPhi network.
        state (tuple[int]): The current state of the network.
        node_indices (tuple[int]): The nodes of the network included in
            this subsystem.
    """

    def __init__(self, network, state, node_indices):
        self.network = network
        self.state = tuple(state)
        self.node_indices = tuple(sorted(node_indices))

        # IIT 3.0 subsystem used to compute cause repertoires
        # Note that external elements are *not* frozen - that is handled
        # by the `_external_indices=()` argument.
        self._subsystem_3_0 = pyphi.Subsystem(
            self.network, self.state, self.node_indices, _external_indices=())

        # Memoized MIP
        self._mip = None

    def __len__(self):
        return len(self.node_indices)

    def __repr__(self):
        return "Subsystem2.0(state={}, nodes={})".format(self.state,
                                                         self.node_indices)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return (self.network == other.network and
                self.state == other.state and
                self.node_indices == other.node_indices)

    def __hash__(self):
        return hash((self.network, self.state, self.node_indices))

    def prior_repertoire(self, mechanism=None):
        """The *a priori* repertoire of the system."""
        if mechanism is None:
            mechanism = self.node_indices
        return max_entropy_distribution(mechanism, self.network.size)

    def posterior_repertoire(self, mechanism=None):
        """The *a posteriori* repertoire of the system."""
        if mechanism is None:
            mechanism = self.node_indices
        return self._subsystem_3_0.cause_repertoire(mechanism, mechanism)

    def partitioned_posterior_repertoire(self, partition):
        """The joint posterior repertoire of a partition."""
        return functools.reduce(np.multiply, [
            self.posterior_repertoire(mechanism) for mechanism in partition])

    def effective_information(self, mechanism=None):
        """The effective information of the system."""
        return entropy(self.posterior_repertoire(mechanism),
                       self.prior_repertoire(mechanism))

    def effective_information_partition(self, partition):
        """The effective information across an arbitrary partition.

        The total partition is a special case; otherwise it would produce
        partitioned effective information of 0. See p. 5-6.
        """
        assert partition.indices == self.node_indices

        if partition.is_total:
            return self.effective_information()

        return entropy(self.posterior_repertoire(),
                       self.partitioned_posterior_repertoire(partition))

    def find_mip(self):
        """Compute the minimum information partition of the system.

        This result is cached on the system for reuse.
        """
        if self._mip is None:
            self._mip = min(
                Mip(self.effective_information_partition(partition),
                        partition, self)
                for partition in generate_partitions(self.node_indices))

        return self._mip

    @property
    def phi(self):
        """The integrated information of the system."""
        return self.find_mip().ei


@functools.total_ordering
class Mip:
    """An IIT 2.0 minimum information partition."""

    def __init__(self, ei, partition, subsystem):
        self.ei = round(ei, config.PRECISION)
        self.partition = partition
        self.subsystem = subsystem

    @property
    def ei_normalized(self):
        """The normalized effective information of this MIP."""
        return self.ei / self.partition.normalization

    @property
    def phi(self):
        """An alias for effective information."""
        return self.ei

    @cmp.sametype
    def __eq__(self, other):
        return (self.ei == other.ei and self.partition == other.partition)

    @cmp.sametype
    def __lt__(self, other):
        """If more than one partition has the same minimum normalized value,
        select the partition that generates the lowest _un-normalized_
        quantity of effective information.
        """
        return (self.ei_normalized, self.ei) < (other.ei_normalized, other.ei)

    def __repr__(self):
        return "Mip2.0(ei={}, {})".format(self.ei, self.partition)


# TODO: implement all partitions
def generate_partitions(node_indices):
    """Currently only returns bipartitions."""
    if node_indices:
        # `bipartition` returns the total partition as a pair,
        # so return it properly here.
        yield Partition(node_indices)

    for partition in bipartition(node_indices)[1:]:
        yield Partition(*partition)


class Partition(collections.abc.Sequence):
    """A IIT 2.0 partition of a system.

    The partition must cover the system.

    Args:
        *parts tuple[int]: The disjoint parts of the partition.
    """

    def __init__(self, *parts):
        self.parts = tuple(sorted(parts))

        if len(set(self.indices)) != len(self.indices):
            raise ValueError('IIT 2.0 {} must be disjoint'.format(self))

    def __len__(self):
        return len(self.parts)

    def __getitem__(self, x):
        return self.parts[x]

    @cmp.sametype
    def __eq__(self, other):
        return self.parts == other.parts

    def __hash__(self):
        return hash(self.parts)

    def __repr__(self):
        return "Partition{}".format(self.parts)

    @property
    def indices(self):
        """The indices of all nodes in the partition."""
        return tuple(sorted(itertools.chain.from_iterable(self.parts)))

    @property
    def normalization(self):
        """Normalization factor for this partition.

        See p. 6-7 for a discussion.
        """
        if self.is_total:
            return len(self.indices)
        return (len(self) - 1) * min(len(p) for p in self)

    @property
    def is_total(self):
        """Is this a total (unitary) partition?"""
        return len(self) == 1


def all_subsystems(network, state):
    """Generator over all possible subsystems of the network.

    Excludes subsystems that would be in an invalid state.
    """
    for subset in utils.powerset(network.node_indices, nonempty=True,
                                 reverse=True):
        try:
            yield Subsystem(network, state, subset)
        except exceptions.StateUnreachableError:
            pass


def all_complexes(network, state):
    """
    S is a complex if phi(s) > 0 and S is not contained in some larger set
    with strictly higher phi.
    """
    # Note: all_subsystems yields systems from largest to smallest
    complexes = []

    for s in all_subsystems(network, state):
        if s.phi > 0:
            for c in complexes:
                if (set(s.node_indices) < set(c.node_indices)
                        and s.phi < c.phi):
                    break  # Not a complex
            else:
                complexes.append(s)

    return complexes


def main_complexes(network, state):
    """
    S is a main complex iff S is a complex and phi(S) is greater than the
    phi of all subsets of S.
    """
    complexes = all_complexes(network, state)
    main_complexes = []

    for s in complexes:
        for r in complexes:
            if (set(r.node_indices) < set(s.node_indices)
                    and r.phi >= s.phi):
                break
        else:
            main_complexes.append(s)

    return main_complexes
