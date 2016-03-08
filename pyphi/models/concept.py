#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/concept.py

from collections import namedtuple

import numpy as np

from . import cmp, fmt
from .. import config, jsonify, utils
from ..constants import DIRECTIONS, PAST, FUTURE

_mip_attributes = ['phi', 'direction', 'mechanism', 'purview', 'partition',
                   'unpartitioned_repertoire', 'partitioned_repertoire']


class Mip(cmp._Orderable, namedtuple('Mip', _mip_attributes)):
    """A minimum information partition for |small_phi| calculation.

    MIPs may be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, ``phi`` values are compared. Then, if these are equal
    up to |PRECISION|, the size of the mechanism is compared (exclusion
    principle).

    Attributes:
        phi (float):
            This is the difference between the mechanism's unpartitioned and
            partitioned repertoires.
        direction (str):
            Either |past| or |future|. The temporal direction specifiying
            whether this MIP should be calculated with cause or effect
            repertoires.
        mechanism (tuple(int)):
            The mechanism over which to evaluate the MIP.
        purview (tuple(int)):
            The purview over which the unpartitioned repertoire differs the
            least from the partitioned repertoire.
        partition (tuple(Part, Part)):
            The partition that makes the least difference to the mechanism's
            repertoire.
        unpartitioned_repertoire (np.ndarray):
            The unpartitioned repertoire of the mechanism.
        partitioned_repertoire (np.ndarray):
            The partitioned repertoire of the mechanism. This is the product of
            the repertoires of each part of the partition.
    """

    __slots__ = ()

    _unorderable_unless_eq = ['direction']

    def _order_by(self):
        return [self.phi, len(self.mechanism), len(self.purview)]

    def __eq__(self, other):
        # We don't count the partition and partitioned repertoire in checking
        # for MIP equality, since these are lost during normalization.
        return cmp._general_eq(self, other, ['phi', 'direction', 'mechanism',
                                         'purview', 'unpartitioned_repertoire'])

    def __bool__(self):
        """A Mip is truthy if it is not reducible.

        (That is, if it has a significant amount of |small_phi|.)
        """
        return not utils.phi_eq(self.phi, 0)

    def __hash__(self):
        return hash((self.phi,
                     self.direction,
                     self.mechanism,
                     self.purview,
                     utils.np_hash(self.unpartitioned_repertoire)))

    def to_json(self):
        d = self.__dict__
        # Flatten the repertoires.
        d['partitioned_repertoire'] = self.partitioned_repertoire.flatten()
        d['unpartitioned_repertoire'] = self.unpartitioned_repertoire.flatten()
        return d

    def __repr__(self):
        return fmt.make_repr(self, _mip_attributes)

    def __str__(self):
        return "Mip\n" + fmt.indent(fmt.fmt_mip(self))


def _null_mip(direction, mechanism, purview):
    """The null mip (of a reducible mechanism)."""
    # TODO Use properties here to infer mechanism and purview from
    # partition yet access them with .mechanism and .partition
    return Mip(direction=direction,
               mechanism=mechanism,
               purview=purview,
               partition=None,
               unpartitioned_repertoire=None,
               partitioned_repertoire=None,
               phi=0.0)


# =============================================================================

class Mice(cmp._Orderable):
    """A maximally irreducible cause or effect (i.e., “core cause” or “core
    effect”).

    MICEs may be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, ``phi`` values are compared. Then, if these are equal
    up to |PRECISION|, the size of the mechanism is compared (exclusion
    principle).
    """

    def __init__(self, mip):
        self._mip = mip

    @property
    def phi(self):
        """``float`` -- The difference between the mechanism's unpartitioned
        and partitioned repertoires.
        """
        return self._mip.phi

    @property
    def direction(self):
        """``str`` -- Either |past| or |future|. If |past| (|future|), this
        represents a maximally irreducible cause (effect).
        """
        return self._mip.direction

    @property
    def mechanism(self):
        """``list(int)`` -- The mechanism for which the MICE is evaluated."""
        return self._mip.mechanism

    @property
    def purview(self):
        """``list(int)`` -- The purview over which this mechanism's |small_phi|
        is maximal.
        """
        return self._mip.purview

    @property
    def repertoire(self):
        """``np.ndarray`` -- The unpartitioned repertoire of the mechanism over
        the purview.
        """
        return self._mip.unpartitioned_repertoire

    @property
    def mip(self):
        """``Mip`` -- The minimum information partition for this mechanism."""
        return self._mip

    def __repr__(self):
        return fmt.make_repr(self, ['mip'])

    def __str__(self):
        return "Mice\n" + fmt.indent(fmt.fmt_mip(self.mip))

    _unorderable_unless_eq = Mip._unorderable_unless_eq

    def _order_by(self):
        return self.mip._order_by()

    def __eq__(self, other):
        return self.mip == other.mip

    def __hash__(self):
        return hash(('Mice', self._mip))

    def to_json(self):
        return {'mip': self._mip}

    # TODO: benchmark and memoize?
    # TODO: pass in subsystem indices only?
    def _relevant_connections(self, subsystem):
        """Identify connections that “matter” to this concept.

        For a core cause, the important connections are those which connect the
        purview to the mechanism; for a core effect they are the connections
        from the mechanism to the purview.

        Returns an |n x n| matrix, where `n` is the number of nodes in this
        corresponding subsystem, that identifies connections that “matter” to
        this MICE:

        ``direction == 'past'``:
            ``relevant_connections[i,j]`` is ``1`` if node ``i`` is in the
            cause purview and node ``j`` is in the mechanism (and ``0``
            otherwise).

        ``direction == 'future'``:
            ``relevant_connections[i,j]`` is ``1`` if node ``i`` is in the
            mechanism and node ``j`` is in the effect purview (and ``0``
            otherwise).

        Args:
            subsystem (Subsystem): The subsystem of this mice

        Returns:
            cm (np.ndarray): A |n x n| matrix of connections, where `n` is the
                size of the subsystem.
        """
        if self.direction == DIRECTIONS[PAST]:
            _from, to = self.purview, self.mechanism
        elif self.direction == DIRECTIONS[FUTURE]:
            _from, to = self.mechanism, self.purview

        cm = utils.relevant_connections(subsystem.network.size, _from, to)
        # Submatrix for this subsystem's nodes
        idxs = subsystem.node_indices
        return utils.submatrix(cm, idxs, idxs)

    # TODO: pass in `cut` instead? We can infer
    # subsystem indices from the cut itself, validate, and check.
    def damaged_by_cut(self, subsystem):
        """Return True if this |Mice| is affected by the subsystem's cut.

        The cut affects the |Mice| if it either splits the |Mice|'s
        mechanism or splits the connections between the purview and
        mechanism.
        """
        return (subsystem.cut.splits_mechanism(self.mechanism) or
                np.any(self._relevant_connections(subsystem) *
                       subsystem.cut_matrix == 1))


# =============================================================================

_concept_attributes = ['phi', 'mechanism', 'cause', 'effect', 'subsystem',
                       'normalized']


# TODO: make mechanism a property
# TODO: make phi a property
class Concept(cmp._Orderable):
    """A star in concept-space.

    The ``phi`` attribute is the |small_phi_max| value. ``cause`` and
    ``effect`` are the MICE objects for the past and future, respectively.

    Concepts may be compared with the built-in Python comparison operators
    (``<``, ``>``, etc.). First, ``phi`` values are compared. Then, if these
    are equal up to |PRECISION|, the size of the mechanism is compared.

    Attributes:
        phi (float):
            The size of the concept. This is the minimum of the |small_phi|
            values of the concept's core cause and core effect.
        mechanism (tuple(int)):
            The mechanism that the concept consists of.
        cause (|Mice|):
            The |Mice| representing the core cause of this concept.
        effect (|Mice|):
            The |Mice| representing the core effect of this concept.
        subsystem (Subsystem):
            This concept's parent subsystem.
        time (float):
            The number of seconds it took to calculate.
    """

    def __init__(self, phi=None, mechanism=None, cause=None, effect=None,
                 subsystem=None, normalized=False):
        self.phi = phi
        self.mechanism = mechanism
        self.cause = cause
        self.effect = effect
        self.subsystem = subsystem
        self.normalized = normalized
        self.time = None

    def __repr__(self):
        return fmt.make_repr(self, _concept_attributes)

    def __str__(self):
        return "Concept\n""-------\n" + fmt.fmt_concept(self)

    @property
    def location(self):
        """
        ``tuple(np.ndarray)`` -- The concept's location in concept space. The
        two elements of the tuple are the cause and effect repertoires.
        """
        if self.cause and self.effect:
            return (self.cause.repertoire, self.effect.repertoire)
        else:
            return (self.cause, self.effect)

    _unorderable_unless_eq = ['subsystem']

    def _order_by(self):
        return [self.phi, len(self.mechanism)]

    def __eq__(self, other):
        self_cause_purview = getattr(self.cause, 'purview', None)
        other_cause_purview = getattr(other.cause, 'purview', None)
        self_effect_purview = getattr(self.effect, 'purview', None)
        other_effect_purview = getattr(other.effect, 'purview', None)
        return (self.phi == other.phi
                and self.mechanism == other.mechanism
                and (utils.state_of(self.mechanism, self.subsystem.state) ==
                     utils.state_of(self.mechanism, other.subsystem.state))
                and self_cause_purview == other_cause_purview
                and self_effect_purview == other_effect_purview
                and self.eq_repertoires(other)
                and self.subsystem.network == other.subsystem.network)

    def __hash__(self):
        return hash((self.phi,
                     self.mechanism,
                     utils.state_of(self.mechanism, self.subsystem.state),
                     self.cause.purview,
                     self.effect.purview,
                     utils.np_hash(self.cause.repertoire),
                     utils.np_hash(self.effect.repertoire),
                     self.subsystem.network))

    def __bool__(self):
        """A concept is truthy if it is not reducible.

        (That is, if it has a significant amount of |big_phi|.)
        """
        return not utils.phi_eq(self.phi, 0)

    def eq_repertoires(self, other):
        """Return whether this concept has the same cause and effect
        repertoires as another.

        .. warning::
            This only checks if the cause and effect repertoires are equal as
            arrays; mechanisms, purviews, or even the nodes that node indices
            refer to, might be different.
        """
        this_cr = getattr(self.cause, 'repertoire', None)
        this_er = getattr(self.effect, 'repertoire', None)
        other_cr = getattr(other.cause, 'repertoire', None)
        other_er = getattr(other.effect, 'repertoire', None)
        return (np.array_equal(this_cr, other_cr) and
                np.array_equal(this_er, other_er))

    def emd_eq(self, other):
        """Return whether this concept is equal to another in the context of an
        EMD calculation.
        """
        return (self.phi == other.phi
                and self.mechanism == other.mechanism
                and self.eq_repertoires(other))

    # TODO Rename to expanded_cause_repertoire, etc
    def expand_cause_repertoire(self, new_purview=None):
        """Expand a cause repertoire into a distribution over an entire
        network.
        """
        return self.subsystem.expand_cause_repertoire(self.cause.purview,
                                                      self.cause.repertoire,
                                                      new_purview)

    def expand_effect_repertoire(self, new_purview=None):
        """Expand an effect repertoire into a distribution over an entire
        network.
        """
        return self.subsystem.expand_effect_repertoire(self.effect.purview,
                                                       self.effect.repertoire,
                                                       new_purview)

    def expand_partitioned_cause_repertoire(self):
        """Expand a partitioned cause repertoire into a distribution over an
        entire network.
        """
        return self.subsystem.expand_cause_repertoire(
            self.cause.purview,
            self.cause.mip.partitioned_repertoire)

    def expand_partitioned_effect_repertoire(self):
        """Expand a partitioned effect repertoire into a distribution over an
        entire network.
        """
        return self.subsystem.expand_effect_repertoire(
            self.effect.purview,
            self.effect.mip.partitioned_repertoire)

    def to_json(self):
        d = jsonify.jsonify(self.__dict__)
        # Attach the expanded repertoires to the jsonified MICEs.
        d['cause']['repertoire'] = self.expand_cause_repertoire().flatten()
        d['effect']['repertoire'] = self.expand_effect_repertoire().flatten()
        d['cause']['partitioned_repertoire'] = \
            self.expand_partitioned_cause_repertoire().flatten()
        d['effect']['partitioned_repertoire'] = \
            self.expand_partitioned_effect_repertoire().flatten()
        return d


class Constellation(tuple):
    """A constellation of concepts.

    This is a wrapper around a tuple to provide a nice string representation
    and place to put constellation methods. Previously, constellations were
    represented as ``tuple(|Concept|)``; this usage still works in all
    functions.
    """

    def __repr__(self):
        if config.READABLE_REPRS:
            return self.__str__()
        return "Constellation({})".format(
            super(Constellation, self).__repr__())

    def __str__(self):
        return "\nConstellation\n*************" + fmt.fmt_constellation(self)

    def to_json(self):
        return list(self)
