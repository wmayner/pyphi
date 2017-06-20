#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/cuts.py

from collections import namedtuple
from itertools import chain

import numpy as np

from . import fmt
from .. import config, connectivity, utils


# TODO: Rename `severed` to `from` and `intact` to `to`
class Cut(namedtuple('Cut', ['severed', 'intact'])):
    """Represents a unidirectional cut.

    Attributes:
        severed (tuple[int]):
            Connections from this group of nodes to those in ``intact`` are
            severed.
        intact (tuple[int]):
            Connections to this group of nodes from those in ``severed`` are
            severed.
    """

    # This allows accessing the namedtuple's ``__dict__``; see
    # https://docs.python.org/3.3/reference/datamodel.html#notes-on-using-slots
    __slots__ = ()

    @property
    def indices(self):
        """Returns the indices of this cut."""
        return tuple(sorted(set(self[0] + self[1])))

    # TODO: cast to bool
    def splits_mechanism(self, mechanism):
        """Check if this cut splits a mechanism.

        Args:
            mechanism (tuple[int]): The mechanism in question

        Returns:
            bool: ``True`` if `mechanism` has elements on both sides of the
            cut, otherwise ``False``.
        """
        # TODO: use cuts_connections
        return ((set(mechanism) & set(self[0])) and
                (set(mechanism) & set(self[1])))

    def cuts_connections(self, a, b):
        """Check if this cut severs any connections from nodes `a` to `b`."""
        return (set(a) & set(self[0])) and (set(b) & set(self[1]))

    def all_cut_mechanisms(self):
        """Return all mechanisms with elements on both sides of this cut.

        Returns:
            tuple[tuple[int]]
        """
        all_mechanisms = utils.powerset(self.indices, nonempty=True)
        return tuple(m for m in all_mechanisms if self.splits_mechanism(m))

    def apply_cut(self, cm):
        """Return a modified connectivity matrix where the connections from one
        set of nodes to the other are destroyed.
        """
        cm = cm.copy()

        for i in self[0]:
            for j in self[1]:
                cm[i][j] = 0

        return cm

    # TODO: pass in `size` arg and keep expanded to full network??
    # TODO: memoize?
    def cut_matrix(self):
        """Compute the cut matrix for this cut.

        The cut matrix is a square matrix which represents connections
        severed by the cut. The matrix is shrunk to the size of the cut
        subsystem--not necessarily the size of the entire network.

        Example:
            >>> cut = Cut((1,), (2,))
            >>> cut.cut_matrix()
            array([[ 0.,  1.],
                   [ 0.,  0.]])
        """
        cut_indices = self.indices

        # Don't pass an empty tuple to `max`
        if not cut_indices:
            return np.array([])

        # Construct a cut matrix large enough for all indices
        # in the cut, then extract the relevant submatrix
        n = max(cut_indices) + 1
        matrix = connectivity.relevant_connections(n, self[0], self[1])
        return matrix[np.ix_(cut_indices, cut_indices)]

    def __repr__(self):
        return fmt.make_repr(self, ['severed', 'intact'])

    def __str__(self):
        return fmt.fmt_cut(self)

    def to_json(self):
        return {'severed': self.severed, 'intact': self.intact}


class ActualCut(namedtuple('ActualCut', ['cause_part1', 'cause_part2',
                                         'effect_part1', 'effect_part2'])):

    """Represents an actual cut for a context.

    Attributes:
        cause_part1 (tuple(int)):
            Connections from this group to those in ``effect_part2`` are cut
        cause_part2 (tuple(int)):
            Connections from this group to those in ``effect_part1`` are cut
        effect_part1 (tuple(int)):
            Connections to this group from ``cause_part2`` are cut
        effect_part2 (tuple(int)):
             Connections to this group from ``cause_part1`` are cut
    """

    __slots__ = ()

    @property
    def indices(self):
        """tuple[int]: The indices in this cut."""
        return tuple(sorted(set(chain.from_iterable(self))))

    # TODO test
    def apply_cut(self, cm):
        """Cut a connectivity matrix.

        Args:
            cm (np.ndarray): A connectivity matrix

        Returns:
            np.ndarray: A copy of the connectivity matrix with connections cut
                across the cause and effect indices.
        """
        cm = cm.copy()

        for i in self.cause_part1:
            for j in self.effect_part2:
                cm[i][j] = 0

        for i in self.cause_part2:
            for j in self.effect_part1:
                cm[i][j] = 0

        return cm

    # TODO implement
    def cut_matrix(self):
        return "DUMMY MATRIX"

    def __repr__(self):
        return fmt.make_repr(self, ['cause_part1', 'cause_part2',
                                    'effect_part1', 'effect_part2'])

    def __str__(self):
        return fmt.fmt_actual_cut(self)


class Part(namedtuple('Part', ['mechanism', 'purview'])):
    """Represents one part of a bipartition.

    Attributes:
        mechanism (tuple[int]):
            The nodes in the mechanism for this part.
        purview (tuple[int]):
            The nodes in the mechanism for this part.

    Example:
        When calculating |small_phi| of a 3-node subsystem, we partition the
        system in the following way::

            mechanism:   A C        B
                        -----  X  -----
              purview:    B        A C

        This class represents one term in the above product.
    """

    __slots__ = ()

    def to_json(self):
        return {'mechanism': self.mechanism, 'purview': self.purview}


class KPartition(tuple):
    """A partition with an arbitrary number of parts."""
    __slots__ = ()

    def __new__(cls, *args):
        """Construct the base tuple with multiple ``Part`` arguments."""
        return super().__new__(cls, args)

    def __getnewargs__(self):
        """And support unpickling with this ``__new__`` signature."""
        return tuple(self)

    @property
    def mechanism(self):
        """tuple[int]: The nodes of the mechanism in the partition."""
        return tuple(sorted(
            chain.from_iterable(part.mechanism for part in self)))

    @property
    def purview(self):
        """tuple[int]: The nodes of the purview in the partition."""
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
    """A bipartition of a mechanism and purview.

    Attributes:
        part0 (Part): The first part of the partition.
        part1 (Part): The second part of the partition.
    """
    __slots__ = ()

    def to_json(self):
        return {'part0': self[0], 'part1': self[1]}

    @classmethod
    def from_json(cls, json):
        return cls(json['part0'], json['part1'])


class Tripartition(KPartition):

    __slots__ = ()
