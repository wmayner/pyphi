#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Models
~~~~~~

Lightweight containers for MICE, MIP, cut, partition, and concept data.
"""

from collections import namedtuple, Iterable
from .utils import phi_eq as _phi_eq

# TODO use properties to avoid data duplication
# TODO add proper docstrings with __doc__


class Cut(namedtuple('Cut', ['severed', 'intact'])):
    """Represents a unidirectional cut.

    Attributes:
        severed (tuple(Node)):
            Connections from this group of nodes to those in ``intact`` are
            severed.
        intact (tuple(Node)):
            Connections to this group of nodes from those in ``severed`` are
            severed.
    """
    pass


class Part(namedtuple('Part', ['mechanism', 'purview'])):
    """Represents one part of a bipartition.

    When calculating |phi| of a subsystem, we take partition the system in the
    following way::

        mechanism:   A C        B
                    ~~~~~  X  ~~~~~
            purview:    B        A C

    This class represents one term in the above product.

    Attributes:
        mechanism (tuple(Node): The nodes in the mechanism for this part.
        purview (tuple(Node): The nodes in the mechanism for this part.
    """
    pass


def _numpy_aware_eq(a, b):
    """Return whether two objects are equal via recursion, using all(x == y)
    for comparing numpy arays."""
    if (not (isinstance(a, str) or isinstance(b, str)) and
            isinstance(a, Iterable) and isinstance(b, Iterable)):
        return all(_numpy_aware_eq(x, y) for x, y in zip(a, b))
    return a == b


# TODO cross reference PRECISION in docs
def _general_eq(a, b, attributes):
    """Return whether two objects are equal up to the given attributes.

    If an attribute is called ``'phi'``, it is compared up to
    ``constants.PRECISION``. All other given attributes are compared with
    :func:`_numpy_aware_eq`"""
    if 'phi' in attributes:
        if not _phi_eq(a.phi, b.phi):
            return False
    return all(_numpy_aware_eq(getattr(a, attr), getattr(b, attr)) if attr !=
               'phi' else True for attr in attributes)

# Phi-ordering methods
_phi_lt = lambda self, other: (self.phi < other.phi) if other else False
_phi_gt = lambda self, other: (self.phi > other.phi) if other else True
_phi_le = lambda self, other: ((_phi_lt(self, other) or _phi_eq(self.phi,
                                                                other.phi)) if
                               other else False)
_phi_ge = lambda self, other: ((_phi_gt(self, other) or _phi_eq(self.phi,
                                                                other.phi)) if
                               other else False)


_mip_attributes = ['phi', 'direction', 'mechanism', 'purview', 'partition',
                   'unpartitioned_repertoire', 'partitioned_repertoire']


class Mip(namedtuple('Mip', _mip_attributes)):
    """A minimum information partition for |phi| calculation.

    Attributes:
        phi (float):
        direction (str):
        mechanism (list(Node)):
        purview (list(Node)):
        partition (tuple(Part, Part)):
        unpartitioned_repertoire (np.ndarray):
        partitioned_repertoire (np.ndarray):
    """
    def __eq__(self, other):
        return _general_eq(self, other, _mip_attributes)

    # Order by phi value

    def __lt__(self, other):
        return _phi_lt(self, other)

    def __gt__(self, other):
        return _phi_gt(self, other)

    def __le__(self, other):
        return _phi_le(self, other)

    def __ge__(self, other):
        return _phi_ge(self, other)


_mice_attributes = ['phi', 'direction', 'mechanism', 'purview', 'repertoire',
                    'mip']


class Mice(namedtuple('Mice', _mice_attributes)):
    """A maximally irreducible cause or effect (i.e., "core cause" or "core
    effect").

    Attributes:
        phi (float):
            The difference in information between the partitioned and
            unpartitioned system.
        direction (str):
            Either 'past' or 'future'. If 'past' ('future'), this
            represents a maximally irreducible cause (effect).
        mechanism (list(Node)):
            The mechanism for which the MICE is evaluated.
        purview (list(Node)):
            The purview over which this mechanism's |phi| is
            maximal.
        repertoire (np.ndarray):
            The unpartitioned repertoire of the mechanism over
            the purview.
        mip (Mip):
            The minimum information partition for this mechanism.
    """
    def __eq__(self, other):
        return _general_eq(self, other, _mice_attributes)

    # Order by phi value

    def __lt__(self, other):
        return _phi_lt(self, other)

    def __gt__(self, other):
        return _phi_gt(self, other)

    def __le__(self, other):
        return _phi_le(self, other)

    def __ge__(self, other):
        return _phi_ge(self, other)


_concept_attributes = ['phi', 'mechanism', 'location', 'cause', 'effect']


class Concept(namedtuple('Concept', _concept_attributes)):
    """A star in concept-space.

    `phi` is the small-phi_max value. `cause` and `effect` are the MICE objects
    for the past and future, respectively.

    Attributes:
        phi (float):
        mechanism (list(Node)):
        location (np.ndarray):
            This concept's location in concept space. The first dimension
            corresponds to cause and effect, and the remaining dimensions
            contain the cause and effect repertoire.
        cause (Mice):
            The Mice representing the core cause of this concept.
        effect (Mice):
            The Mice representing the core effect of this concept.

    Examples:
        The location of a concept in concept-space is given by the
        probabilities of each state in its cause and effect repertoires, i.e.
        ``concept.location = array[direction][n_0][n_1]...[n_k]``, where
        `direction` is either `PAST` or `FUTURE` and the rest of the dimensions
        correspond to a node in the network.
    """
    def __eq__(self, other):
        return _general_eq(self, other, _concept_attributes)

    # Order by phi value

    def __lt__(self, other):
        return _phi_lt(self, other)

    def __gt__(self, other):
        return _phi_gt(self, other)

    def __le__(self, other):
        return _phi_le(self, other)

    def __ge__(self, other):
        return _phi_ge(self, other)


_bigmip_attributes = ['phi', 'partition', 'unpartitioned_constellation',
                      'partitioned_constellation', 'subsystem']
# TODO! document comparison methods
# TODO! implement exclusion principle in comparison methods


class BigMip(namedtuple('BigMip', _bigmip_attributes)):
    """A minimum information partition for |big_phi| calculation.

    """
    def __eq__(self, other):
        return _general_eq(self, other, _bigmip_attributes)

    # Order by phi value, then by subsystem size

    def __lt__(self, other):
        if other:
            return (self.subsystem < other.subsystem if _phi_eq(self.phi,
                                                                other.phi)
                    else _phi_le(self, other))
        else:
            return False

    def __gt__(self, other):
        if other:
            return (self.subsystem > other.subsystem if _phi_eq(self.phi,
                                                                other.phi)
                    else _phi_gt(self, other))
        else:
            return True

    def __le__(self, other):
        return self < other or _phi_eq(self.phi, other.phi) if other else False

    def __ge__(self, other):
        return self > other or _phi_eq(self.phi, other.phi) if other else True
