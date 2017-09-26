#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/cuts.py

'''Objects that represent partitions of sets of nodes.'''

from collections import namedtuple
from itertools import chain

import numpy as np

from . import cmp, fmt
from ..constants import Direction
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
    '''A cut that severs all connections between parts of a K-partition.'''

    def __init__(self, direction, partition):
        self.direction = direction
        self.partition = partition

    @property
    def indices(self):
        assert self.partition.mechanism == self.partition.purview
        return self.partition.mechanism

    def cut_matrix(self, n):
        '''The matrix of connections that are severed by this cut.'''
        cm = np.zeros((n, n))

        for part in self.partition:
            if self.direction is Direction.PAST:
                from_ = part.purview
                to = part.mechanism

            elif self.direction is Direction.FUTURE:
                from_ = part.mechanism
                to = part.purview

            # All indices external to this part
            external = tuple(set(self.indices) - set(to))
            cm[np.ix_(from_, external)] = 1

        return cm

    @cmp.sametype
    def __eq__(self, other):
        return (self.partition == other.partition and
                self.direction == other.direction)

    def __hash__(self):
        return hash((self.direction, self.partition))

    def __repr__(self):
        return fmt.make_repr(self, ['partition'])

    # TODO: improve
    def __str__(self):
        return fmt.fmt_kcut(self)


actual_cut_attributes = ['cause_part1', 'cause_part2', 'effect_part1',
                         'effect_part2']


class ActualCut(KCut):
    '''Represents an cut for a |Transition|.'''

    @property
    def indices(self):
        return tuple(sorted(set(self.partition.mechanism +
                                self.partition.purview)))

    def to_json(self):
        return {'partition': self.partition}

    def invert(self):
        '''Return a cut representing the cut in the opposite direction.

        This is a hacky way to deal with the directionality of actual cuts.
        '''
        return ActualCut(type(self.partition)(
            *(Part(part.purview, part.mechanism) for part in self.partition)))

    def normalize(self):
        '''Normalize this cut.'''
        return ActualCut(self.partition.normalize())

    def __repr__(self):
        return fmt.make_repr(self, actual_cut_attributes)

    def __str__(self):
        return str(self.partition)


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

    def normalize(self):
        '''Normalize the order of parts in the partition.'''
        return type(self)(*sorted(self))

    def __str__(self):
        return fmt.fmt_bipartition(self)

    def __repr__(self):
        if config.REPR_VERBOSITY > 0:
            return str(self)

        return '{}{}'.format(self.__class__.__name__, super().__repr__())

    def to_json(self):
        return {'parts': list(self)}

    @classmethod
    def from_json(cls, dct):
        return cls(*dct['parts'])


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
