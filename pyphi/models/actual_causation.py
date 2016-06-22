#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/actual_causation.py

"""
Containers for AcMip and AcMice and AcBigMip.
"""
from collections import namedtuple
from .. import utils
from ..utils import phi_eq
from . import cmp
from .fmt import fmt_ac_mip, fmt_ac_big_mip, make_repr, indent

# TODO use properties to avoid data duplication

# Ac_diff-ordering methods
# =============================================================================


# Compare ac_diff
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Todo: check how that works out also with abs values
def _ap_phi_eq(self, other):
    try:
        return phi_eq(self.alpha, other.alpha)
    except AttributeError:
        return False


def _ap_phi_lt(self, other):
    try:
        if not phi_eq(self.alpha, other.alpha):
            return self.alpha < other.alpha
        return False
    except AttributeError:
        return False


def _ap_phi_gt(self, other):
    try:
        if not phi_eq(self.alpha, other.alpha):
            return self.alpha > other.alpha
        return False
    except AttributeError:
        return False


def _ap_phi_le(self, other):
    return _ap_phi_lt(self, other) or _ap_phi_eq(self, other)


def _ap_phi_ge(self, other):
    return _ap_phi_gt(self, other) or _ap_phi_eq(self, other)


# First compare ap_phi, then mechanism size
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _ap_phi_then_mechanism_size_lt(self, other):
    if _ap_phi_eq(self, other):
        return (len(self.mechanism) < len(other.mechanism)
                if hasattr(other, 'mechanism') else False)
    else:
        return _ap_phi_lt(self, other)


def _ap_phi_then_mechanism_size_gt(self, other):
    return (not _ap_phi_then_mechanism_size_lt(self, other) and
            not self == other)


def _ap_phi_then_mechanism_size_le(self, other):
    return (_ap_phi_then_mechanism_size_lt(self, other) or
            _ap_phi_eq(self, other))


def _ap_phi_then_mechanism_size_ge(self, other):
    return (_ap_phi_then_mechanism_size_gt(self, other) or
            _ap_phi_eq(self, other))


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
        return not phi_eq(self.alpha, 0)

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
        return make_repr(self, _acmip_attributes)

    def __str__(self):
        return "Mip\n" + indent(fmt_ac_mip(self))


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

class AcMice:

    """A maximally irreducible actual cause or effect (i.e., "actual cause” or
    “actual effect”).

    relevant_connections (np.array):
        An ``N x N`` matrix, where ``N`` is the number of nodes in this
        corresponding subsystem, that identifies connections that “matter” to
        this AcMICE.

        ``direction == 'past'``:
            ``relevant_connections[i,j]`` is ``1`` if node ``i`` is in the
            cause purview and node ``j`` is in the mechanism (and ``0``
            otherwise).

        ``direction == 'future'``:
            ``relevant_connections[i,j]`` is ``1`` if node ``i`` is in the
            mechanism and node ``j`` is in the effect purview (and ``0``
            otherwise).

    AcMICEs may be compared with the built-in Python comparison operators
    (``<``, ``>``, etc.). First, ``phi`` values are compared. Then, if these
    are equal up to |PRECISION|, the size of the mechanism is compared
    (exclusion principle).
    """

    def __init__(self, mip, relevant_connections=None):
        self._mip = mip
        self._relevant_connections = relevant_connections

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
        return make_repr(self, ['acmip'])

    def __str__(self):
        return "AcMice\n" + indent(fmt_ac_mip(self.mip))

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

    # Order by ap_phi value, then by mechanism size
    __lt__ = _ap_phi_then_mechanism_size_lt
    __gt__ = _ap_phi_then_mechanism_size_gt
    __le__ = _ap_phi_then_mechanism_size_le
    __ge__ = _ap_phi_then_mechanism_size_ge


# =============================================================================

_acbigmip_attributes = ['alpha', 'direction', 'unpartitioned_account',
                        'partitioned_account', 'context', 'cut']


class AcBigMip:

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
        return make_repr(self, _acbigmip_attributes)

    def __str__(self):
        return "\nAcBigMip\n======\n" + fmt_ac_big_mip(self)

    @property
    def before_state(self):
        '''Return actual past state of the context '''
        return self.context.before_state

    @property
    def after_state(self):
        '''Return actual current state of the context'''
        return self.context.after_state

    def __eq__(self, other):
        return cmp._general_eq(self, other, _acbigmip_attributes)

    def __bool__(self):
        """A BigMip is truthy if it is not reducible; i.e. if it has a
        significant amount of |big_ap_phi|."""
        return not _ap_phi_eq(self.alpha, 0)

    def __hash__(self):
        return hash((self.alpha, self.unpartitioned_account,
                     self.partitioned_account, self.context,
                     self.cut))

    # First compare alpha then context
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __lt__(self, other):
        if _ap_phi_eq(self, other):
            if len(self.context) == len(other.context):
                return False
            else:
                return len(self.context) < len(other.context)
        else:
            return _ap_phi_lt(self, other)

    def __gt__(self, other):
        if _ap_phi_eq(self, other):
            if len(self.context) == len(other.context):
                return False
            else:
                return len(self.context) > len(other.context)
        else:
            return _ap_phi_gt(self, other)

    def __le__(self, other):
        return (self.__lt__(other) or _ap_phi_eq(self, other))

    def __ge__(self, other):
        return (self.__gt__(other) or _ap_phi_eq(self, other))


def _null_ac_bigmip(context, direction):
    """Returns an ac |BigMip| with zero |big_ap_phi| and empty constellations."""
    return AcBigMip(context=context,
                    direction=direction,
                    alpha=0.0,
                    unpartitioned_account=(),
                    partitioned_account=())
