#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# models.py
"""
Containers for MICE, MIP, cut, partition, and concept data.
"""

from collections import namedtuple, Iterable
import numpy as np

from . import utils, constants, convert, json

# TODO use properties to avoid data duplication


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
    def json_dict(self):
        return {
            'severed': json.make_encodable(self.severed),
            'intact': json.make_encodable(self.intact)
        }


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
    def json_dict(self):
        return {
            'mechanism': json.make_encodable(self.mechanism),
            'purview': json.make_encodable(self.purview)
        }


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
                # Don't use `set` because hashes may be different (contexts are
                # included in node hashes); we want to use Node.__eq__.
                elif not (all(n in _b for n in _a) and len(_a) == len(_b)):
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
_mip_attributes_for_eq = ['phi', 'direction', 'mechanism',
                          'unpartitioned_repertoire']


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
        direction (str):
            The temporal direction specifiying whether this MIP should be
            calculated with cause or effect repertoires.
        mechanism (list(Node)):
            The mechanism over which to evaluate the MIP.
        purview (list(Node)):
            The purview over which the unpartitioned repertoire differs the
            least from the partitioned repertoire.
        partition (tuple(Part, Part)):
            The partition that makes the least difference to the mechanism's
            repertoire.
        unpartitioned_repertoire (np.ndarray):
            The unpartitioned repertoire of the mecanism.
        partitioned_repertoire (np.ndarray):
            The partitioned repertoire of the mechanism. This is the product of
            the repertoires of each part of the partition.
    """

    def __eq__(self, other):
        # We don't count the partition and partitioned repertoire in checking
        # for MIP equality, since these are lost during normalization.
        # We also don't count the mechanism and purview, since these may be
        # different depending on the order in which purviews were evaluated.
        # TODO!!! clarify the reason for that
        # We do however check whether the size of the mechanism or purview is
        # the same, since that matters (for the exclusion principle).
        return (_general_eq(self, other, _mip_attributes_for_eq) and
                len(self.mechanism) == len(other.mechanism) and
                len(self.purview) == len(other.purview))

    def __bool__(self):
        """A Mip is truthy if it is not reducible; i.e. if it has a significant
        amount of |small_phi|."""
        return self.phi > constants.EPSILON

    def __hash__(self):
        return hash((self.phi, self.direction, self.mechanism, self.purview,
                     utils.np_hash(self.unpartitioned_repertoire)))

    def json_dict(self):
        return {
            attr: json.make_encodable(getattr(self, attr))
            for attr in _mip_attributes
        }

    # Order by phi value, then by mechanism size
    __lt__ = _phi_then_mechanism_size_lt
    __gt__ = _phi_then_mechanism_size_gt
    __le__ = _phi_then_mechanism_size_le
    __ge__ = _phi_then_mechanism_size_ge


# =============================================================================


class Mice:

    """A maximally irreducible cause or effect (i.e., "core cause" or "core
    effect").

    MICEs may be compared with the built-in Python comparison operators
    (``<``, ``>``, etc.). First, ``phi`` values are compared. Then, if these
    are equal up to |PRECISION|, the size of the mechanism is compared
    (exclusion principle).
    """

    def __init__(self, mip):
        self._mip = mip
        # TODO remove?
        if (self.repertoire is not None and
            any(self.repertoire.shape[i] != 2 for i in
                convert.nodes2indices(self.purview))):
            raise Exception("Attempted to create MICE with mismatched purview "
                            "and repertoire.")

    @property
    def phi(self):
        """
        ``float`` -- The difference between the mechanism's unpartitioned and
        partitioned repertoires.
        """
        return self._mip.phi

    @property
    def direction(self):
        """
        ``str`` -- Either 'past' or 'future'. If 'past' ('future'), this
        represents a maximally irreducible cause (effect).
        """
        return self._mip.direction

    @property
    def mechanism(self):
        """
        ``list(Node)`` -- The mechanism for which the MICE is evaluated.
        """
        return self._mip.mechanism

    @property
    def purview(self):
        """
        ``list(Node)`` -- The purview over which this mechanism's |phi| is
        maximal.
        """
        return self._mip.purview

    @property
    def repertoire(self):
        """
        ``np.ndarray`` -- The unpartitioned repertoire of the mechanism over
        the purview.
        """
        return self._mip.unpartitioned_repertoire

    @property
    def mip(self):
        """
        ``Mip`` -- The minimum information partition for this mechanism.
        """
        return self._mip

    def __str__(self):
        return "Mice(" + str(self._mip) + ")"

    def __repr__(self):
        return "Mice(" + repr(self._mip) + ")"

    def __eq__(self, other):
        return self.mip == other.mip

    def __hash__(self):
        return hash(('Mice', self._mip))

    def json_dict(self):
        return {
            "mip": json.make_encodable(self._mip)
        }

    # Order by phi value, then by mechanism size
    __lt__ = _phi_then_mechanism_size_lt
    __gt__ = _phi_then_mechanism_size_gt
    __le__ = _phi_then_mechanism_size_le
    __ge__ = _phi_then_mechanism_size_ge


# =============================================================================

_concept_attributes = ['phi', 'mechanism', 'cause', 'effect', 'subsystem',
                       'normalized']


# TODO: make mechanism a property
# TODO: make phi a property
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
        cause (Mice):
            The :class:`Mice` representing the core cause of this concept.
        effect (Mice):
            The :class:`Mice` representing the core effect of this concept.
        subsystem (Subsystem):
            This Concept's parent subsystem.
    """

    def __new__(cls, phi=None, mechanism=None, cause=None, effect=None,
                subsystem=None, normalized=None):
        return super(Concept, cls).__new__(
            cls, phi, mechanism, cause, effect, subsystem, normalized)

    @property
    def location(self):
        """
        ``tuple(np.ndarray)`` -- The concept's location in concept space. The
        two elements of the tuple are the cause and effect repertoires.
        """
        return (self.cause.repertoire, self.effect.repertoire)

    def __eq__(self, other):
        return _general_eq(self, other, _concept_attributes)

    def __hash__(self):
        return hash((self.phi, self.mechanism, self.cause, self.effect,
                     self.subsystem))

    def __str__(self):
        return ('Concept(' +
                ', '.join([str(self.mechanism), str(self.phi),
                           str(self.location)])
                + ')')

    def __bool__(self):
        """A Concept is truthy if it is not reducible; i.e. if it has a
        significant amount of |big_phi|."""
        return self.phi > constants.EPSILON

    def eq_repertoires(self, other):
        """Return whether this concept has the same cause and effect
        repertoires as another."""
        if self.subsystem.network != other.subsystem.network:
            raise Exception("Can't compare repertoires of concepts from "
                            "different networks.")
        return (
            np.array_equal(self.cause.repertoire, other.cause.repertoire) and
            np.array_equal(self.effect.repertoire, other.effect.repertoire))

    def emd_eq(self, other):
        """Return whether this concept is equal to another in the context of an
        EMD calculation."""
        return self.mechanism == other.mechanism and self.eq_repertoires(other)

    # TODO Rename to expanded_cause_repertoire, etc
    def expand_cause_repertoire(self, new_purview=None):
        """Expands a cause repertoire to be a distribution over an entire
        network."""
        return self.subsystem.expand_cause_repertoire(self.cause.purview,
                                                      self.cause.repertoire,
                                                      new_purview)

    def expand_effect_repertoire(self, new_purview=None):
        """Expands an effect repertoire to be a distribution over an entire
        network."""
        return self.subsystem.expand_effect_repertoire(self.effect.purview,
                                                       self.effect.repertoire,
                                                       new_purview)

    def expand_partitioned_cause_repertoire(self):
        """Expands a partitioned cause repertoire to be a distribution over an
        entire network."""
        return self.subsystem.expand_cause_repertoire(
            self.cause.purview,
            self.cause.mip.partitioned_repertoire)

    def expand_partitioned_effect_repertoire(self):
        """Expands a partitioned effect repertoire to be a distribution over an
        entire network."""
        return self.subsystem.expand_effect_repertoire(
            self.effect.purview,
            self.effect.mip.partitioned_repertoire)

    def json_dict(self):
        d = {
            attr: json.make_encodable(getattr(self, attr))
            for attr in ['phi', 'mechanism', 'cause', 'effect']
        }
        # Expand the repertoires.
        d['cause']['repertoire'] = json.make_encodable(
            self.expand_cause_repertoire().flatten())
        d['effect']['repertoire'] = json.make_encodable(
            self.expand_effect_repertoire().flatten())
        d['cause']['partitioned_repertoire'] = json.make_encodable(
            self.expand_partitioned_cause_repertoire().flatten())
        d['effect']['partitioned_repertoire'] = json.make_encodable(
            self.expand_partitioned_effect_repertoire().flatten())
        return d

    # Order by phi value, then by mechanism size
    __lt__ = _phi_then_mechanism_size_lt
    __gt__ = _phi_then_mechanism_size_gt
    __le__ = _phi_then_mechanism_size_le
    __ge__ = _phi_then_mechanism_size_ge


# =============================================================================

_bigmip_attributes = ['phi', 'unpartitioned_constellation',
                      'partitioned_constellation', 'subsystem',
                      'cut_subsystem']


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
        cut_subsystem (Subsystem): The subsystem with the minimal cut applied.
        cut (Cut): The minimal cut.
    """

    @property
    def cut(self):
        return self.cut_subsystem.cut

    def __eq__(self, other):
        return _general_eq(self, other, _bigmip_attributes)

    def __bool__(self):
        """A BigMip is truthy if it is not reducible; i.e. if it has a
        significant amount of |big_phi|."""
        return self.phi > constants.EPSILON

    def __hash__(self):
        return hash((self.phi, self.unpartitioned_constellation,
                     self.partitioned_constellation, self.subsystem,
                     self.cut_subsystem))

    # First compare phi, then subsystem size
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __lt__(self, other):
        if _phi_eq(self, other):
            if len(self.subsystem) == len(other.subsystem):
                # Compare actual Phi values up to maximum precision, for
                # more determinism in things like max and min
                return self.phi < other.phi
            else:
                return len(self.subsystem) < len(other.subsystem)
        else:
            return _phi_lt(self, other)

    def __gt__(self, other):
        if _phi_eq(self, other):
            if len(self.subsystem) == len(other.subsystem):
                # Compare actual Phi values up to maximum precision, for
                # more determinism in things like max and min
                return self.phi > other.phi
            else:
                return len(self.subsystem) > len(other.subsystem)
        else:
            return _phi_gt(self, other)

    def __le__(self, other):
        return (self.__lt__(other) or
                _phi_eq(self, other))

    def __ge__(self, other):
        return (self.__gt__(other) or
                _phi_eq(self, other))

    def json_dict(self):
        return {
            attr: json.make_encodable(getattr(self, attr))
            for attr in _bigmip_attributes
        }
