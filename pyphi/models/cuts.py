#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/cuts.py

'''Objects that represent partitions of sets of nodes.'''

from collections import namedtuple
from itertools import chain

import numpy as np

from . import fmt
from .. import config, connectivity, utils


class _CutBase:
    '''Base class for all unidirectional system cuts.

    Concrete cut classes must implement a ``cut_matrix`` method and an
    ``indices`` property. See ``Cut`` for a concrete example.
    '''
    @property
    def indices(self):
        '''Return the indices of this cut.'''
        raise NotImplementedError

    def cut_matrix(self, n):
        '''Return the cut matrix for this cut.

        The cut matrix is a square matrix representing  connections severed
        by the cut: if the connection from node `a` to node `b` is cut,
        `cut_matrix[a, b]` is `1`; otherwise it is `0`.

        Args:
           n (int): The size of the network.
        '''
        raise NotImplementedError

    def apply_cut(self, cm):
        '''Return a modified connectivity matrix with all connections that are
        severed by this cut removed.

        Args:
            cm (np.ndarray): A connectivity matrix.
        '''
        # Invert the cut matrix, creating a matrix of preserved connections
        inverse = np.logical_not(self.cut_matrix(cm.shape[0])).astype(int)
        return cm * inverse

    def cuts_connections(self, a, b):
        '''Check if this cut severs any connections from ``a`` to ``b``.

        Args:
            a (tuple[int]): A set of nodes.
            b (tuple[int]): A set of nodes.
        '''
        n = max(self.indices) + 1
        return self.cut_matrix(n)[np.ix_(a, b)].any()

    def splits_mechanism(self, mechanism):
        '''Check if this cut splits a mechanism.

        Args:
            mechanism (tuple[int]): The mechanism in question.

        Returns:
            bool: ``True`` if `mechanism` has elements on both sides of the
            cut; ``False`` otherwise.
        '''
        return self.cuts_connections(mechanism, mechanism)

    def all_cut_mechanisms(self):
        '''Return all mechanisms with elements on both sides of this cut.

        Returns:
            tuple[tuple[int]]
        '''
        all_mechanisms = utils.powerset(self.indices, nonempty=True)
        return tuple(m for m in all_mechanisms if self.splits_mechanism(m))


class Cut(namedtuple('Cut', ['from_nodes', 'to_nodes']), _CutBase):
    '''Represents a unidirectional cut.

    Attributes:
        from_nodes (tuple[int]): Connections from this group of nodes to those
            in ``to_nodes`` are from_nodes.
        to_nodes (tuple[int]): Connections to this group of nodes from those in
            ``from_nodes`` are from_nodes.
    '''
    # Don't construct an attribute dictionary; see
    # https://docs.python.org/3.3/reference/datamodel.html#notes-on-using-slots
    __slots__ = ()

    @property
    def indices(self):
        '''Returns the indices of this cut.'''
        return tuple(sorted(set(self[0] + self[1])))

    def cut_matrix(self, n):
        '''Compute the cut matrix for this cut.

        The cut matrix is a square matrix which represents connections severed
        by the cut.

        Args:
           n (int): The size of the network.

        Example:
            >>> cut = Cut((1,), (2,))
            >>> cut.cut_matrix(3)
            array([[ 0.,  0.,  0.],
                   [ 0.,  0.,  1.],
                   [ 0.,  0.,  0.]])
        '''
        return connectivity.relevant_connections(n, self[0], self[1])

    def __repr__(self):
        return fmt.make_repr(self, ['from_nodes', 'to_nodes'])

    def __str__(self):
        return fmt.fmt_cut(self)

    def to_json(self):
        '''Return a JSON-serializable representation.'''
        return {'from_nodes': self.from_nodes, 'to_nodes': self.to_nodes}


class KCut(_CutBase):
    '''A cut that severs all connections between parts of a K-partition.

    Note: since the ``KCut`` does not have a direction associated with it,
    connectivity is always considered to be from the purview of partition to
    the mechanism of the partition.

    TODO: add a ``direction`` to the cut?
    '''
    def __init__(self, partition):
        assert partition.mechanism == partition.purview
        self.partition = partition

    @property
    def indices(self):
        return self.partition.mechanism

    def cut_matrix(self, n):
        '''The matrix of connections that are severed by this cut.'''
        cm = np.zeros((n, n))

        for part in self.partition:
            # All indices external to this part
            external = tuple(set(self.indices) - set(part.mechanism))
            cm[np.ix_(part.purview, external)] = 1

        return cm

    def __repr__(self):
        return fmt.make_repr(self, ['partition'])

    # TODO: improve
    def __str__(self):
        return "KCut\n{}".format(self.partition)


actual_cut_attributes = ['cause_part1', 'cause_part2', 'effect_part1',
                         'effect_part2']


# TODO: this is a special case of KCut - refactor to reflect that?
class ActualCut(namedtuple('ActualCut', actual_cut_attributes), _CutBase):
    '''Represents an cut for a |Context|.

    This is a bipartition of the cause and effect elements.

    Attributes:
        cause_part1 (tuple[int]): Connections from this group to those in
            ``effect_part2`` are cut.
        cause_part2 (tuple[int]): Connections from this group to those in
            ``effect_part1`` are cut.
        effect_part1 (tuple[int]): Connections to this group from
            ``cause_part2`` are cut.
        effect_part2 (tuple[int]): Connections to this group from
            ``cause_part1`` are cut.
    '''
    __slots__ = ()

    @property
    def indices(self):
        '''tuple[int]: The indices in this cut.'''
        return tuple(sorted(set(chain.from_iterable(self))))

    def cut_matrix(self, n):
        '''The matrix of connections severed by this cut.'''
        cm = np.zeros((n, n))
        cm[np.ix_(self.cause_part1, self.effect_part2)] = 1
        cm[np.ix_(self.cause_part2, self.effect_part1)] = 1
        return cm

    def __repr__(self):
        return fmt.make_repr(self, actual_cut_attributes)

    def __str__(self):
        return fmt.fmt_actual_cut(self)


class Part(namedtuple('Part', ['mechanism', 'purview'])):
    '''Represents one part of a |Bipartition|.

    Attributes:
        mechanism (tuple[int]): The nodes in the mechanism for this part.
        purview (tuple[int]): The nodes in the mechanism for this part.

    Example:
        When calculating |small_phi| of a 3-node subsystem, we partition the
        system in the following way::

            mechanism:  A,C    B
                        ─── ✕ ───
              purview:   B    A,C

        This class represents one term in the above product.
    '''

    __slots__ = ()

    def to_json(self):
        '''Return a JSON-serializable representation.'''
        return {'mechanism': self.mechanism, 'purview': self.purview}


class KPartition(tuple):
    '''A partition with an arbitrary number of parts.'''
    __slots__ = ()

    def __new__(cls, *args):
        '''Construct the base tuple with multiple |Part| arguments.'''
        return super().__new__(cls, args)

    def __getnewargs__(self):
        '''And support unpickling with this ``__new__`` signature.'''
        return tuple(self)

    @property
    def mechanism(self):
        '''tuple[int]: The nodes of the mechanism in the partition.'''
        return tuple(sorted(
            chain.from_iterable(part.mechanism for part in self)))

    @property
    def purview(self):
        '''tuple[int]: The nodes of the purview in the partition.'''
        return tuple(sorted(
            chain.from_iterable(part.purview for part in self)))

    def __str__(self):
        return fmt.fmt_bipartition(self)

    def __repr__(self):
        if config.REPR_VERBOSITY > 0:
            return str(self)

        return '{}{}'.format(self.__class__.__name__, super().__repr__())

    def to_json(self):
        raise NotImplementedError


class Bipartition(KPartition):
    '''A bipartition of a mechanism and purview.

    Attributes:
        part0 (Part): The first part of the partition.
        part1 (Part): The second part of the partition.
    '''
    __slots__ = ()

    def to_json(self):
        '''Return a JSON-serializable representation.'''
        return {'part0': self[0], 'part1': self[1]}

    @classmethod
    def from_json(cls, json):
        return cls(json['part0'], json['part1'])


class Tripartition(KPartition):

    __slots__ = ()
