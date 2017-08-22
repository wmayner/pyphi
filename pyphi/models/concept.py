#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/concept.py

'''Objects that represent parts of cause-effect structures.'''

import numpy as np

from . import cmp, fmt
from .. import config, connectivity, distribution, utils, validate
from ..constants import Direction

_mip_attributes = ['phi', 'direction', 'mechanism', 'purview', 'partition',
                   'unpartitioned_repertoire', 'partitioned_repertoire']


class Mip(cmp.Orderable):
    '''A minimum information partition for |small_phi| calculation.

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, |small_phi| values are compared. Then, if these are
    equal up to |PRECISION|, the size of the mechanism is compared (see the
    |PICK_SMALLEST_PURVIEW| option in |config|.)
    '''

    def __init__(self, phi, direction, mechanism, purview, partition,
                 unpartitioned_repertoire, partitioned_repertoire,
                 subsystem=None):
        self._phi = phi
        self._direction = direction
        self._mechanism = mechanism
        self._purview = purview
        self._partition = partition

        def _repertoire(repertoire):
            if repertoire is None:
                return None
            return np.array(repertoire)

        self._unpartitioned_repertoire = _repertoire(unpartitioned_repertoire)
        self._partitioned_repertoire = _repertoire(partitioned_repertoire)

        # Optional subsystem - only used to generate nice labeled reprs
        self._subsystem = subsystem

    @property
    def phi(self):
        '''float: This is the difference between the mechanism's unpartitioned
        and partitioned repertoires.
        '''
        return self._phi

    @property
    def direction(self):
        '''Direction: |PAST| or |FUTURE|.'''
        return self._direction

    @property
    def mechanism(self):
        '''tuple[int]: The mechanism over which to evaluate the MIP.'''
        return self._mechanism

    @property
    def purview(self):
        '''tuple[int]: The purview over which the unpartitioned repertoire
        differs the least from the partitioned repertoire.'''
        return self._purview

    @property
    def partition(self):
        '''KPartition: The partition that makes the least difference to the
        mechanism's repertoire.'''
        return self._partition

    @property
    def unpartitioned_repertoire(self):
        '''np.ndarray: The unpartitioned repertoire of the mechanism.'''
        return self._unpartitioned_repertoire

    @property
    def partitioned_repertoire(self):
        '''np.ndarray: The partitioned repertoire of the mechanism. This is the
        product of the repertoires of each part of the partition.
        '''
        return self._partitioned_repertoire

    @property
    def subsystem(self):
        '''Subsystem: The |Subsystem| this MIP belongs to.'''
        return self._subsystem

    unorderable_unless_eq = ['direction']

    def order_by(self):
        if config.PICK_SMALLEST_PURVIEW:
            return [self.phi, len(self.mechanism), -len(self.purview)]

        return [self.phi, len(self.mechanism), len(self.purview)]

    def __eq__(self, other):
        # We don't consider the partition and partitioned repertoire in
        # checking for MIP equality.
        attrs = ['phi', 'direction', 'mechanism', 'purview',
                 'unpartitioned_repertoire']
        return cmp.general_eq(self, other, attrs)

    def __bool__(self):
        '''A |Mip| is ``True`` if it has |small_phi > 0|.'''
        return not utils.eq(self.phi, 0)

    def __hash__(self):
        return hash((self.phi,
                     self.direction,
                     self.mechanism,
                     self.purview,
                     utils.np_hash(self.unpartitioned_repertoire)))

    def __repr__(self):
        return fmt.make_repr(self, _mip_attributes)

    def __str__(self):
        return "MIP\n" + fmt.indent(fmt.fmt_mip(self))

    def to_json(self):
        return {attr: getattr(self, attr) for attr in _mip_attributes}


def _null_mip(direction, mechanism, purview, unpartitioned_repertoire=None):
    '''The null MIP (of a reducible mechanism).'''
    # TODO Use properties here to infer mechanism and purview from
    # partition yet access them with .mechanism and .partition
    return Mip(direction=direction,
               mechanism=mechanism,
               purview=purview,
               partition=None,
               unpartitioned_repertoire=unpartitioned_repertoire,
               partitioned_repertoire=None,
               phi=0.0)


# =============================================================================

class Mice(cmp.Orderable):
    '''A maximally irreducible cause or effect.

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, |small_phi| values are compared. Then, if these are
    equal up to |PRECISION|, the size of the mechanism is compared (see the
    |PICK_SMALLEST_PURVIEW| option in |config|.)
    '''

    def __init__(self, mip):
        self._mip = mip

    @property
    def phi(self):
        '''float: The difference between the mechanism's unpartitioned and
        partitioned repertoires.
        '''
        return self._mip.phi

    @property
    def direction(self):
        '''Direction: |PAST| or |FUTURE|.'''
        return self._mip.direction

    @property
    def mechanism(self):
        '''list[int]: The mechanism for which the MICE is evaluated.'''
        return self._mip.mechanism

    @property
    def purview(self):
        '''list[int]: The purview over which this mechanism's |small_phi| is
        maximal.
        '''
        return self._mip.purview

    @property
    def partition(self):
        '''KPartition: The partition that makes the least difference to the
        mechanism's repertoire.'''
        return self._mip.partition

    @property
    def repertoire(self):
        '''np.ndarray: The unpartitioned repertoire of the mechanism over the
        purview.
        '''
        return self._mip.unpartitioned_repertoire

    @property
    def partitioned_repertoire(self):
        '''np.ndarray: The partitioned repertoire of the mechanism over the
        purview.
        '''
        return self._mip.partitioned_repertoire

    @property
    def mip(self):
        '''MIP: The minimum information partition for this mechanism.'''
        return self._mip

    def __repr__(self):
        return fmt.make_repr(self, ['mip'])

    def __str__(self):
        return "Mice\n" + fmt.indent(fmt.fmt_mip(self.mip))

    unorderable_unless_eq = Mip.unorderable_unless_eq

    def order_by(self):
        return self.mip.order_by()

    def __eq__(self, other):
        return self.mip == other.mip

    def __hash__(self):
        return hash(('Mice', self._mip))

    def to_json(self):
        return {'mip': self.mip}

    # TODO: benchmark and memoize?
    # TODO: pass in subsystem indices only?
    def _relevant_connections(self, subsystem):
        '''Identify connections that “matter” to this concept.

        For a core cause, the important connections are those which connect the
        purview to the mechanism; for a core effect they are the connections
        from the mechanism to the purview.

        Returns an |N x N| matrix, where `N` is the number of nodes in this
        corresponding subsystem, that identifies connections that “matter” to
        this |Mice|:

        ``direction == Direction.PAST``:
            ``relevant_connections[i,j]`` is ``1`` if node ``i`` is in the
            cause purview and node ``j`` is in the mechanism (and ``0``
            otherwise).

        ``direction == Direction.FUTURE``:
            ``relevant_connections[i,j]`` is ``1`` if node ``i`` is in the
            mechanism and node ``j`` is in the effect purview (and ``0``
            otherwise).

        Args:
            subsystem (Subsystem): The |Subsystem| of this |Mice|.

        Returns:
            np.ndarray: A |N x N| matrix of connections, where |N| is the size
            of the subsystem.

        Raises:
            ValueError: If ``direction`` is invalid.
        '''
        if self.direction == Direction.PAST:
            _from, to = self.purview, self.mechanism
        elif self.direction == Direction.FUTURE:
            _from, to = self.mechanism, self.purview
        else:
            validate.direction(self.direction)

        cm = connectivity.relevant_connections(subsystem.network.size,
                                               _from, to)
        # Submatrix for this subsystem's nodes
        return cm[np.ix_(subsystem.node_indices, subsystem.node_indices)]

    # TODO: pass in `cut` instead? We can infer
    # subsystem indices from the cut itself, validate, and check.
    def damaged_by_cut(self, subsystem):
        '''Return ``True`` if this |Mice| is affected by the subsystem's cut.

        The cut affects the |Mice| if it either splits the |Mice|'s mechanism
        or splits the connections between the purview and mechanism.
        '''
        return (subsystem.cut.splits_mechanism(self.mechanism) or
                np.any(self._relevant_connections(subsystem) *
                       subsystem.cut_matrix == 1))


# =============================================================================

_concept_attributes = ['phi', 'mechanism', 'cause', 'effect', 'subsystem']


# TODO: make mechanism a property
# TODO: make phi a property
class Concept(cmp.Orderable):
    '''A the maximally irreducible cause and effect specified by a mechanism.

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, |small_phi| values are compared. Then, if these are
    equal up to |PRECISION|, the size of the mechanism is compared (see the
    |PICK_SMALLEST_PURVIEW| option in |config|.)

    Attributes:
        mechanism (tuple[int]): The mechanism that the concept consists of.
        cause (Mice): The |Mice| representing the core cause of this concept.
        effect (Mice): The |Mice| representing the core effect of this concept.
        subsystem (Subsystem): This concept's parent subsystem.
        time (float): The number of seconds it took to calculate.
    '''

    def __init__(self, mechanism=None, cause=None, effect=None,
                 subsystem=None, time=None):
        self.mechanism = mechanism
        self.cause = cause
        self.effect = effect
        self.subsystem = subsystem
        self.time = time

    def __repr__(self):
        return fmt.make_repr(self, _concept_attributes)

    def __str__(self):
        return fmt.fmt_concept(self)

    @property
    def phi(self):
        '''float: The size of the concept.

        This is the minimum of the |small_phi| values of the concept's core
        cause and core effect.
        '''
        return min(self.cause.phi, self.effect.phi)

    @property
    def cause_purview(self):
        '''tuple[int]: The cause purview.'''
        return getattr(self.cause, 'purview', None)

    @property
    def effect_purview(self):
        '''tuple[int]: The effect purview.'''
        return getattr(self.effect, 'purview', None)

    @property
    def cause_repertoire(self):
        '''np.ndarray: The cause repertoire.'''
        return getattr(self.cause, 'repertoire', None)

    @property
    def effect_repertoire(self):
        '''np.ndarray: The effect repertoire.'''
        return getattr(self.effect, 'repertoire', None)

    unorderable_unless_eq = ['subsystem']

    def order_by(self):
        return [self.phi, len(self.mechanism)]

    def __eq__(self, other):
        return (self.phi == other.phi and
                self.mechanism == other.mechanism and
                (utils.state_of(self.mechanism, self.subsystem.state) ==
                 utils.state_of(self.mechanism, other.subsystem.state)) and
                self.cause_purview == other.cause_purview and
                self.effect_purview == other.effect_purview and
                self.eq_repertoires(other) and
                self.subsystem.network == other.subsystem.network)

    def __hash__(self):
        return hash((self.phi,
                     self.mechanism,
                     utils.state_of(self.mechanism, self.subsystem.state),
                     self.cause_purview,
                     self.effect_purview,
                     utils.np_hash(self.cause_repertoire),
                     utils.np_hash(self.effect_repertoire),
                     self.subsystem.network))

    def __bool__(self):
        '''A concept is ``True`` if |small_phi > 0|.'''
        return not utils.eq(self.phi, 0)

    def eq_repertoires(self, other):
        '''Return whether this concept has the same repertoires as another.

        .. warning::
            This only checks if the cause and effect repertoires are equal as
            arrays; mechanisms, purviews, or even the nodes that the mechanism
            and purview indices refer to, might be different.
        '''
        return (
            np.array_equal(self.cause_repertoire, other.cause_repertoire) and
            np.array_equal(self.effect_repertoire, other.effect_repertoire))

    def emd_eq(self, other):
        '''Return whether this concept is equal to another in the context of
        an EMD calculation.
        '''
        return (self.phi == other.phi and
                self.mechanism == other.mechanism and
                self.eq_repertoires(other))

    # TODO Rename to expanded_cause_repertoire, etc
    def expand_cause_repertoire(self, new_purview=None):
        '''See :meth:`~pyphi.subsystem.Subsystem.expand_repertoire`.'''
        return self.subsystem.expand_cause_repertoire(
            self.cause.repertoire, new_purview)

    def expand_effect_repertoire(self, new_purview=None):
        '''See :meth:`~pyphi.subsystem.Subsystem.expand_repertoire`.'''
        return self.subsystem.expand_effect_repertoire(
            self.effect.repertoire, new_purview)

    def expand_partitioned_cause_repertoire(self):
        '''See :meth:`~pyphi.subsystem.Subsystem.expand_repertoire`.'''
        return self.subsystem.expand_cause_repertoire(
            self.cause.mip.partitioned_repertoire)

    def expand_partitioned_effect_repertoire(self):
        '''See :meth:`~pyphi.subsystem.Subsystem.expand_repertoire`.'''
        return self.subsystem.expand_effect_repertoire(
            self.effect.mip.partitioned_repertoire)

    def to_json(self):
        '''Return a JSON-serializable representation.'''
        dct = {
            attr: getattr(self, attr)
            for attr in _concept_attributes + ['time']
        }
        # These flattened, LOLI-order repertoires are passed to `vphi` via
        # `phiserver`.
        dct.update({
            'expanded_cause_repertoire': distribution.flatten(
                self.expand_cause_repertoire()),
            'expanded_effect_repertoire': distribution.flatten(
                self.expand_effect_repertoire()),
            'expanded_partitioned_cause_repertoire': distribution.flatten(
                self.expand_partitioned_cause_repertoire()),
            'expanded_partitioned_effect_repertoire': distribution.flatten(
                self.expand_partitioned_effect_repertoire()),
        })
        return dct

    @classmethod
    def from_json(cls, dct):
        # Remove extra attributes
        del dct['phi']
        del dct['expanded_cause_repertoire']
        del dct['expanded_effect_repertoire']
        del dct['expanded_partitioned_cause_repertoire']
        del dct['expanded_partitioned_effect_repertoire']

        return cls(**dct)


class Constellation(tuple):
    '''A constellation of concepts.

    This is a wrapper around a tuple to provide a nice string representation
    and place to put constellation methods. Previously, constellations were
    represented as a ``tuple[concept]``; this usage still works in all
    functions.
    '''
    # TODO: compare constellations using set equality

    def __repr__(self):
        if config.REPR_VERBOSITY > 0:
            return self.__str__()

        return "Constellation{}".format(
            super().__repr__())

    def __str__(self):
        return fmt.fmt_constellation(self)

    def to_json(self):
        return {'concepts': list(self)}

    @property
    def mechanisms(self):
        '''The mechanism of each concept.'''
        return [concept.mechanism for concept in self]

    @property
    def phis(self):
        '''The |small_phi| values of each concept.'''
        return [concept.phi for concept in self]

    @property
    def labeled_mechanisms(self):
        '''The labeled mechanism of each concept.'''
        if not self:
            return []
        label = self[0].subsystem.network.indices2labels
        return [list(label(mechanism)) for mechanism in self.mechanisms]

    @classmethod
    def from_json(cls, json):
        return cls(json['concepts'])


def _concept_sort_key(concept):
    return (len(concept.mechanism), concept.mechanism)


def normalize_constellation(constellation):
    '''Deterministically reorder the concepts in a constellation.

    Args:
        constellation (Constellation): The constellation in question.

    Returns:
        Constellation: The constellation, ordered lexicographically by
        mechanism.
    '''
    return Constellation(sorted(constellation, key=_concept_sort_key))
