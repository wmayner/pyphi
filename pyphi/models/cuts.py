#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/cuts.py

from collections import namedtuple

import numpy as np

from .. import utils
from . import fmt


class Cut(namedtuple('Cut', ['severed', 'intact'])):
    """Represents a unidirectional cut.

    Attributes:
        severed (tuple(int)):
            Connections from this group of nodes to those in ``intact`` are
            severed.
        intact (tuple(int)):
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
            mechanism (tuple(int)): The mechanism in question

        Returns:
            (bool): True if `mechanism` has elements on both sides
                of the cut, otherwise False.
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
            tuple(tuple(int)):
        """
        all_mechanisms = utils.powerset(self.indices)
        return tuple(m for m in all_mechanisms if self.splits_mechanism(m))

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
        matrix = utils.relevant_connections(n, self[0], self[1])
        return utils.submatrix(matrix, cut_indices, cut_indices)

    def __repr__(self):
        return fmt.make_repr(self, ['severed', 'intact'])

    def __str__(self):
        return "Cut {self.severed} --//--> {self.intact}".format(self=self)

    def to_json(self):
        return [self.severed, self.intact]


class Part(namedtuple('Part', ['mechanism', 'purview'])):
    """Represents one part of a bipartition.

    Attributes:
        mechanism (tuple(int)):
            The nodes in the mechanism for this part.
        purview (tuple(int)):
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
    pass
