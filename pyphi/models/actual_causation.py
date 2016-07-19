#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/actual_causation.py

"""
Containers for AcMip and AcMice and AcBigMip.
"""

from collections import namedtuple

from . import cmp, fmt
from .. import config, utils

# TODO use properties to avoid data duplication

# =============================================================================
# Todo: Why do we even need this?
# Todo: add second state
_acmip_attributes = ['alpha', 'state', 'direction', 'mechanism', 'purview',
                     'partition', 'probability', 'partitioned_probability',
                     'unconstrained_probability']
_acmip_attributes_for_eq = ['alpha', 'state', 'direction', 'mechanism',
                            'purview', 'probability']


class AcMip(cmp._Orderable, namedtuple('AcMip', _acmip_attributes)):

    """A minimum information partition for ac_coef calculation.

    MIPs may be compared with the built-in Python comparison operators
    (``<``, ``>``, etc.). First, ``ap_phi`` values are compared. Then, if these
    are equal up to |PRECISION|, the size of the mechanism is compared
    (exclusion principle).

    Attributes:
        alpha (float):
            This is the difference between the mechanism's unpartitioned and
            partitioned actual probability.
        state (tuple(int)):
            state of system in specified direction (past, future)
        direction (str):
            The temporal direction specifiying whether this AcMIP should be
            calculated with cause or effect repertoires.
        mechanism (tuple(int)):
            The mechanism over which to evaluate the AcMIP.
        purview (tuple(int)):
            The purview over which the unpartitioned actual probability differs
            the least from the actual probability of the partition.
        partition (tuple(Part, Part)):
            The partition that makes the least difference to the mechanism's
            repertoire.
        probability (float):
            The probability of the state in the past/future.
        partitioned_probability (float):
            The probability of the state in the partitioned repertoire.
        unconstrained_probability (float):
            The unconstrained probability of the state, used for normalization.
    """
    __slots__ = ()

    _unorderable_unless_eq = ['direction']

    def _order_by(self):
        return [self.alpha, len(self.mechanism)]

    def __eq__(self, other):
        # TODO: include 2nd state here?
        return cmp._general_eq(self, other, _acmip_attributes_for_eq)

    def __bool__(self):
        """An AcMip is truthy if it is not reducible; i.e. if it has a significant
        amount of |ap_phi|."""
        return not utils.phi_eq(self.alpha, 0)

    @property
    def phi(self):
        self.phi = self.alpha

    # def __hash__(self):
    #     return hash((self.ap_phi, self.actual_state, self.direction,
    #                  self.mechanism, self.purview,
    #                  utils.np_hash(self.unpartitioned_ap)))

    def to_json(self):
        d = self.__dict__
        return d

    def __repr__(self):
        return fmt.make_repr(self, _acmip_attributes)

    def __str__(self):
        return "Mip\n" + fmt.indent(fmt.fmt_ac_mip(self))


def _null_ac_mip(state, direction, mechanism, purview):
    return AcMip(state=state,
                 direction=direction,
                 mechanism=mechanism,
                 purview=purview,
                 partition=None,
                 probability=None,
                 partitioned_probability=None,
                 unconstrained_probability=None,
                 alpha=0.0)

# =============================================================================

class AcMice(cmp._Orderable):

    """A maximally irreducible actual cause or effect (i.e., "actual cause” or
    “actual effect”).

    AcMICEs may be compared with the built-in Python comparison operators
    (``<``, ``>``, etc.). First, ``alpha`` values are compared. Then, if these
    are equal up to |PRECISION|, the size of the mechanism is compared
    (exclusion principle).
    """

    def __init__(self, mip):
        self._mip = mip

    @property
    def alpha(self):
        """
        ``float`` -- The difference between the mechanism's unpartitioned and
        partitioned actual probabilities.
        """
        return self._mip.alpha

    @property
    def phi(self):
        """
        Define property phi == alpha, to make use of existing util functions
        """
        return self.alpha

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
        ``list(int)`` -- The mechanism for which the AcMICE is evaluated.
        """
        return self._mip.mechanism

    @property
    def purview(self):
        """
        ``list(int)`` -- The purview over which this mechanism's |ap_phi|
        is maximal.
        """
        return self._mip.purview

    @property
    def mip(self):
        """
        ``AcMip`` -- The minimum information partition for this mechanism.
        """
        return self._mip

    def __repr__(self):
        return fmt.make_repr(self, ['acmip'])

    def __str__(self):
        return "AcMice\n" + fmt.indent(fmt.fmt_ac_mip(self.mip))

    _unorderable_unless_eq = AcMip._unorderable_unless_eq

    def _order_by(self):
        return self.mip._order_by()

    def __eq__(self, other):
        return self.mip == other.mip

    def __hash__(self):
        return hash(('AcMice', self._mip))

    def __bool__(self):
        """An AcMice is truthy if it is not reducible; i.e. if it has a
        significant amount of |ap_phi|."""
        return not utils.phi_eq(self._mip.alpha, 0)

    def to_json(self):
        return {'acmip': self._mip}


# =============================================================================


class Account(tuple):
    """The set of MICE with non-zero alpha for both |past| and |future|."""

    def __repr__(self):
        if config.READABLE_REPRS:
            return self.__str__()
        return "Constellation({})".format(
            super().__repr__())

    def __str__(self):
        return fmt.fmt_account(self)


class DirectedAccount(Account):
    """The set of MICE with non-zero alpha for one direction of a context."""
    pass


# =============================================================================

_acbigmip_attributes = ['alpha', 'direction', 'unpartitioned_account',
                        'partitioned_account', 'context', 'cut']


class AcBigMip(cmp._Orderable):

    """A minimum information partition for |big_ap_phi| calculation.

    BigMIPs may be compared with the built-in Python comparison operators
    (``<``, ``>``, etc.). First, ``ac_diff`` values are compared. Then, if these
    are equal up to |PRECISION|, the size of the mechanism is compared
    (exclusion principle).
    TODO: Check if we do the same, i.e. take the bigger system, or take the
    smaller?

    Attributes:
        alpha (float): The |big_ap_phi| value for the subsystem when taken
        against this MIP, *i.e.* the difference between the unpartitioned
        constellation and this MIP's partitioned constellation.
        unpartitioned_constellation (tuple(Concept)): The constellation of the
            whole subsystem.
        partitioned_constellation (tuple(Concept)): The constellation when the
            subsystem is cut.
        subsystem (Subsystem): The subsystem this MIP was calculated for.
        cut: The minimal cut.
    """

    def __init__(self, alpha=None, direction=None, unpartitioned_account=None,
                 partitioned_account=None, context=None, cut=None):
        self.alpha = alpha
        self.direction = direction
        self.unpartitioned_account = unpartitioned_account
        self.partitioned_account = partitioned_account
        self.context = context
        self.cut = cut

    def __repr__(self):
        return fmt.make_repr(self, _acbigmip_attributes)

    def __str__(self):
        return "\nAcBigMip\n======\n" + fmt.fmt_ac_big_mip(self)

    @property
    def before_state(self):
        '''Return actual past state of the context '''
        return self.context.before_state

    @property
    def after_state(self):
        '''Return actual current state of the context'''
        return self.context.after_state

    _unorderable_unless_eq = ['direction']

    def _order_by(self):
        return [self.alpha, len(self.context)]

    def __eq__(self, other):
        return cmp._general_eq(self, other, _acbigmip_attributes)

    def __bool__(self):
        """A BigMip is truthy if it is not reducible; i.e. if it has a
        significant amount of |big_ap_phi|."""
        return not utils.phi_eq(self.alpha, 0)

    def __hash__(self):
        return hash((self.alpha, self.unpartitioned_account,
                     self.partitioned_account, self.context,
                     self.cut))


def _null_ac_bigmip(context, direction):
    """Returns an ac |BigMip| with zero |big_ap_phi| and empty constellations."""
    return AcBigMip(context=context,
                    direction=direction,
                    alpha=0.0,
                    unpartitioned_account=(),
                    partitioned_account=())
