#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Models
~~~~~~

Containers for MICE, MIP, cut, partition, and concept data.
"""

from collections import namedtuple, Iterable
from marbl import MarblSet
import numpy as np

from . import utils

# TODO use properties to avoid data duplication


# TODO make ref in docs
class Mechanism(tuple):

    """Represents an unordered subset of nodes for |phi| evaluation.

    Allows for hashing via the normal form of the nodes' Marbls. See the
    :ref:`Marbl-documentation`.
    """

    def __new__(cls, *iterable):
        self = super(Mechanism, cls).__new__(cls, *iterable)
        # Make the normal form of the Mechanism
        self.marblset = MarblSet(n.marbl for n in self)
        # Compute the canonical hash (once)
        self._hash = hash(MarblSet(n.marbl for n in self))
        return self

    def __hash__(self):
        return self._hash


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

    Attributes:
        mechanism (tuple(Node)):
            The nodes in the mechanism for this part.
        purview (tuple(Node)):
            The nodes in the mechanism for this part.

    Example:
        When calculating |phi| of a 3-node subsystem, we partition the
        system in the following way::

            mechanism:   A C        B
                        -----  X  -----
              purview:    B        A C

        This class represents one term in the above product.
    """

    pass


# Phi-ordering methods
# =============================================================================

# Compare phi
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _phi_eq(self, other):
    try:
        return utils.phi_eq(self.phi, other.phi)
    except AttributeError:
        return False


def _phi_lt(self, other):
    try:
        if not utils.phi_eq(self.phi, other.phi):
            return self.phi < other.phi
        return False
    except AttributeError:
        return False


def _phi_gt(self, other):
    try:
        if not utils.phi_eq(self.phi, other.phi):
            return self.phi > other.phi
        return False
    except AttributeError:
        return False


def _phi_le(self, other):
    return _phi_lt(self, other) or _phi_eq(self, other)


def _phi_ge(self, other):
    return _phi_gt(self, other) or _phi_eq(self, other)


# First compare phi, then mechanism size
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _phi_then_mechanism_size_lt(self, other):
    if _phi_eq(self, other):
        return (len(self.mechanism) < len(other.mechanism)
                if hasattr(other, 'mechanism') else False)
    else:
        return _phi_lt(self, other)


def _phi_then_mechanism_size_gt(self, other):
    return (not _phi_then_mechanism_size_lt(self, other) and
            not self == other)


def _phi_then_mechanism_size_le(self, other):
    return (_phi_then_mechanism_size_lt(self, other) or
            _phi_eq(self, other))


def _phi_then_mechanism_size_ge(self, other):
    return (_phi_then_mechanism_size_gt(self, other) or
            _phi_eq(self, other))


# Equality helpers
# =============================================================================

# TODO use builtin numpy methods here
def _numpy_aware_eq(a, b):
    """Return whether two objects are equal via recursion, using
    :func:`numpy.array_equal` for comparing numpy arays."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if ((isinstance(a, Iterable) and isinstance(b, Iterable))
            and not isinstance(a, str) and not isinstance(b, str)):
        return all(_numpy_aware_eq(x, y) for x, y in zip(a, b))
    return a == b


def _general_eq(a, b, attributes):
    """Return whether two objects are equal up to the given attributes.

    If an attribute is called ``'phi'``, it is compared up to |PRECISION|. All
    other attributes are compared with :func:`_numpy_aware_eq`.

    If an attribute is called ``'mechanism'`` or ``'purview'``, it is compared
    using set equality."""
    try:
        for attr in attributes:
            _a, _b = getattr(a, attr), getattr(b, attr)
            if attr == 'phi':
                if not utils.phi_eq(_a, _b):
                    return False
            elif (attr == 'mechanism' or attr == 'purview'):
                if _a is None or _b is None and not _a == _b:
                    return False
                elif not set(_a) == set(_b):
                    return False
            else:
                if not _numpy_aware_eq(_a, _b):
                    return False
        return True
    except AttributeError:
        return False

# =============================================================================

_mip_attributes = ['phi', 'direction', 'mechanism', 'purview', 'partition',
                   'unpartitioned_repertoire', 'partitioned_repertoire']


class Mip(namedtuple('Mip', _mip_attributes)):

    """A minimum information partition for |phi| calculation.

    MIPs may be compared with the built-in Python comparison operators
    (``<``, ``>``, etc.). First, ``phi`` values are compared. Then, if these
    are equal up to |PRECISION|, the size of the mechanism is compared
    (exclusion principle).

    Attributes:
        phi (float):
            This is the difference between the mechanism's unpartitioned and
            partitioned repertoires.
        direction (str): The temporal direction specifiying whether this MIP
            should be calculated with cause or effect repertoires.
        mechanism (list(Node)): The mechanism over which to evaluate the MIP.
        purview (list(Node)): The purview over which the unpartitioned
            repertoire differs the least from the partitioned repertoire.
        partition (tuple(Part, Part)): The partition that makes the least
            difference to the mechanism's repertoire.
        unpartitioned_repertoire (np.ndarray): The unpartitioned repertoire of
            the mecanism.
        partitioned_repertoire (np.ndarray): The partitioned repertoire of the
            mechanism. This is the product of the repertoires of each part of
            the partition.
    """

    def __eq__(self, other):
        return _general_eq(self, other, _mip_attributes)

    def __hash__(self):
        return hash((self.phi, self.direction, self.mechanism, self.purview,
                     self.partition,
                     utils.np_hash(self.unpartitioned_repertoire),
                     utils.np_hash(self.partitioned_repertoire)))

    # Order by phi value, then by mechanism size
    __lt__ = _phi_then_mechanism_size_lt
    __gt__ = _phi_then_mechanism_size_gt
    __le__ = _phi_then_mechanism_size_le
    __ge__ = _phi_then_mechanism_size_ge


# =============================================================================

_mice_attributes = ['phi', 'direction', 'mechanism', 'purview', 'repertoire',
                    'mip']


class Mice(namedtuple('Mice', _mice_attributes)):

    """A maximally irreducible cause or effect (i.e., "core cause" or "core
    effect").

    MICEs may be compared with the built-in Python comparison operators
    (``<``, ``>``, etc.). First, ``phi`` values are compared. Then, if these
    are equal up to |PRECISION|, the size of the mechanism is compared
    (exclusion principle).

    Attributes:
        phi (float):
            The difference between the mechanism's unpartitioned and
            partitioned repertoires.
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

    def __hash__(self):
        return hash((self.phi, self.direction, self.mechanism, self.purview,
                     utils.np_hash(self.repertoire), self.mip))

    # Order by phi value, then by mechanism size
    __lt__ = _phi_then_mechanism_size_lt
    __gt__ = _phi_then_mechanism_size_gt
    __le__ = _phi_then_mechanism_size_le
    __ge__ = _phi_then_mechanism_size_ge


# =============================================================================

_concept_attributes = ['phi', 'mechanism', 'location', 'cause', 'effect']


class Concept(namedtuple('Concept', _concept_attributes)):

    """A star in concept-space.

    `phi` is the small-phi_max value. `cause` and `effect` are the MICE objects
    for the past and future, respectively.

    Concepts may be compared with the built-in Python comparison operators
    (``<``, ``>``, etc.). First, ``phi`` values are compared. Then, if these
    are equal up to |PRECISION|, the size of the mechanism is compared.

    Attributes:
        phi (float):
            The size of the concept. This is the minimum of the |phi| values of
            the concept's core cause and core effect.
        mechanism (tuple(Node)):
            The mechanism that the concept consists of.
        location (np.ndarray):
            The concept's location in concept space. The first dimension
            corresponds to cause and effect, and the remaining dimensions
            contain the cause and effect repertoire; i.e., ``concept.location =
            array[direction][n_0][n_1]...[n_k]``, where `direction` is either
            `PAST` or `FUTURE` and the rest of the dimensions correspond to a
            node in the network.
        cause (Mice):
            The :class:`Mice` representing the core cause of this concept.
        effect (Mice):
            The :class:`Mice` representing the core effect of this concept.
    """

    def __eq__(self, other):
        return _general_eq(self, other, _concept_attributes)

    def __hash__(self):
        return hash((self.phi, self.mechanism, utils.np_hash(self.location),
                     self.cause, self.effect))

    # Order by phi value, then by mechanism size
    __lt__ = _phi_then_mechanism_size_lt
    __gt__ = _phi_then_mechanism_size_gt
    __le__ = _phi_then_mechanism_size_le
    __ge__ = _phi_then_mechanism_size_ge


# =============================================================================

_bigmip_attributes = ['phi', 'cut', 'unpartitioned_constellation',
                      'partitioned_constellation', 'subsystem']


class BigMip(namedtuple('BigMip', _bigmip_attributes)):

    """A minimum information partition for |big_phi| calculation.

    BigMIPs may be compared with the built-in Python comparison operators
    (``<``, ``>``, etc.). First, ``phi`` values are compared. Then, if these
    are equal up to |PRECISION|, the size of the mechanism is compared
    (exclusion principle).

    Attributes:
        phi (float): The |big_phi| value for the subsystem when taken against
            this MIP, *i.e.* the difference between the unpartitioned
            constellation and this MIP's partitioned constellation.
        cut (Cut): The unidirectional cut that makes the least difference to
            the subsystem.
        unpartitioned_constellation (tuple(Concept)): The constellation of the
            whole subsystem.
        partitioned_constellation (tuple(Concept)): The constellation when the
            subsystem is cut.
        subsystem (Subsystem): The subsystem this MIP was calculated for.
    """

    def __eq__(self, other):
        return _general_eq(self, other, _bigmip_attributes)

    def __hash__(self):
        return hash((self.phi, self.cut, self.unpartitioned_constellation,
                     self.partitioned_constellation, self.subsystem))

    # First compare phi, then subsystem size
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __lt__(self, other):
        if _phi_eq(self, other):
            try:
                return self.subsystem < other.subsystem
            except AttributeError:
                return False
        else:
            return _phi_lt(self, other)

    def __gt__(self, other):
        return not self.__lt__(other) and not self == other

    def __le__(self, other):
        return (self.__lt__(other) or
                _phi_eq(self, other))

    def __ge__(self, other):
        return (self.__gt__(other) or
                _phi_eq(self, other))
