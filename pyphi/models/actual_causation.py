#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/actual_causation.py

"""
Objects that represent structures used in actual causation.
"""

# pylint: disable=too-many-arguments

from collections import namedtuple

from . import cmp, fmt
from .. import Direction, config, utils

# TODO(slipperyhank): Why do we even need this?
# TODO(slipperyhank): add second state
_acmip_attributes = ['alpha', 'state', 'direction', 'mechanism', 'purview',
                     'partition', 'probability', 'partitioned_probability']
_acmip_attributes_for_eq = ['alpha', 'state', 'direction', 'mechanism',
                            'purview', 'probability']

def greater_than_zero(alpha):
    """Return ``True`` if alpha is greater than zero, accounting for
    numerical errors."""
    return alpha > 0 and not utils.eq(alpha, 0)


class AcMechanismIrreducibilityAnalysis(
        cmp.Orderable, namedtuple('AcMechanismIrreducibilityAnalysis',
                                  _acmip_attributes)):
    """A minimum information partition for ac_coef calculation.


    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, |alpha| values are compared. Then, if these are equal
    up to |PRECISION|, the size of the mechanism is compared.

    Attributes:
        alpha (float):
            This is the difference between the mechanism's unpartitioned and
            partitioned actual probability.
        state (tuple[int]):
            state of system in specified direction (cause, effect)
        direction (str):
            The temporal direction specifiying whether this AcMIP should be
            calculated with cause or effect repertoires.
        mechanism (tuple[int]):
            The mechanism over which to evaluate the AcMIP.
        purview (tuple[int]):
            The purview over which the unpartitioned actual probability differs
            the least from the actual probability of the partition.
        partition (tuple[Part, Part]):
            The partition that makes the least difference to the mechanism's
            repertoire.
        probability (float):
            The probability of the state in the previous/next timestep.
        partitioned_probability (float):
            The probability of the state in the partitioned repertoire.
    """
    __slots__ = ()

    unorderable_unless_eq = ['direction']

    def order_by(self):
        if config.PICK_SMALLEST_PURVIEW:
            return [self.alpha, len(self.mechanism), -len(self.purview)]

        return [self.alpha, len(self.mechanism), len(self.purview)]

    def __eq__(self, other):
        # TODO(slipperyhank): include 2nd state here?
        return cmp.general_eq(self, other, _acmip_attributes_for_eq)

    def __bool__(self):
        """An |AcMechanismIrreducibilityAnalysis| is ``True`` if it has
        |alpha > 0|.
        """
        return greater_than_zero(self.alpha)

    @property
    def phi(self):
        """Alias for |alpha| for PyPhi utility functions."""
        return self.alpha

    def __hash__(self):
        attrs = tuple(getattr(self, attr) for attr in _acmip_attributes_for_eq)
        return hash(attrs)

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {attr: getattr(self, attr) for attr in _acmip_attributes}

    def __repr__(self):
        return fmt.make_repr(self, _acmip_attributes)

    def __str__(self):
        return ("MechanismIrreducibilityAnalysis\n" +
                fmt.indent(fmt.fmt_ac_mip(self)))


def _null_ac_mip(state, direction, mechanism, purview):
    return AcMechanismIrreducibilityAnalysis(state=state,
                 direction=direction,
                 mechanism=mechanism,
                 purview=purview,
                 partition=None,
                 probability=None,
                 partitioned_probability=None,
                 alpha=0.0)


class CausalLink(cmp.Orderable):
    """A maximally irreducible actual cause or effect.

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, |alpha| values are compared. Then, if these are equal
    up to |PRECISION|, the size of the mechanism is compared.
    """

    def __init__(self, mip):
        self._mip = mip

    @property
    def alpha(self):
        """float: The difference between the mechanism's unpartitioned and
        partitioned actual probabilities.
        """
        return self._mip.alpha

    @property
    def phi(self):
        """Alias for |alpha| for PyPhi utility functions."""
        return self.alpha

    @property
    def direction(self):
        """Direction: Either |CAUSE| or |EFFECT|."""
        return self._mip.direction

    @property
    def mechanism(self):
        """list[int]: The mechanism for which the action is evaluated."""
        return self._mip.mechanism

    @property
    def purview(self):
        """list[int]: The purview over which this mechanism's |alpha| is
        maximal.
        """
        return self._mip.purview

    @property
    def mip(self):
        """AcMechanismIrreducibilityAnalysis: The minimum information partition
        for this mechanism.
        """
        return self._mip

    def __repr__(self):
        return fmt.make_repr(self, ['mip'])

    def __str__(self):
        return "CausalLink\n" + fmt.indent(fmt.fmt_ac_mip(self.mip))

    unorderable_unless_eq = \
        AcMechanismIrreducibilityAnalysis.unorderable_unless_eq

    def order_by(self):
        return self.mip.order_by()

    def __eq__(self, other):
        return self.mip == other.mip

    def __hash__(self):
        return hash(('CausalLink', self._mip))

    def __bool__(self):
        """An |CausalLink| is ``True`` if |alpha > 0|."""
        return greater_than_zero(self.alpha)

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {'mip': self.mip}


class Event(namedtuple('Event', ['actual_cause', 'actual_effect'])):
    """A mechanism which has both an actual cause and an actual effect.

    Attributes:
        actual_cause (CausalLink): The actual cause of the mechanism.
        actual_effect (CausalLink): The actual effect of the mechanism.
    """
    @property
    def mechanism(self):
        """The mechanism of the event."""
        assert self.actual_cause.mechanism == self.actual_effect.mechanism
        return self.actual_cause.mechanism


class Account(tuple):
    """The set of |CausalLinks| with |alpha > 0|. This includes both actual
    causes and actual effects."""

    @property
    def irreducible_causes(self):
        """The set of irreducible causes in this |Account|."""
        return tuple(link for link in self
                     if link.direction is Direction.CAUSE)

    @property
    def irreducible_effects(self):
        """The set of irreducible effects in this |Account|."""
        return tuple(link for link in self
                     if link.direction is Direction.EFFECT)

    def __repr__(self):
        if config.REPR_VERBOSITY > 0:
            return self.__str__()
        return "{0}({1})".format(
            self.__class__.__name__, super().__repr__())

    def __str__(self):
        return fmt.fmt_account(self)

    def to_json(self):
        return {'causal_links': tuple(self)}

    @classmethod
    def from_json(cls, dct):
        return cls(dct['causal_links'])


class DirectedAccount(Account):
    """The set of |CausalLinks| with |alpha > 0| for one direction of a
    transition."""
    pass


_ac_sia_attributes = ['alpha', 'direction', 'unpartitioned_account',
                        'partitioned_account', 'transition', 'cut']


# TODO(slipperyhank): Check if we do the same, i.e. take the bigger system, or
# take the smaller?
class AcSystemIrreducibilityAnalysis(cmp.Orderable):
    """An analysis of transition-level irreducibility (|big_alpha|).

    Contains the |big_alpha| value of the |Transition|, the causal account, and
    all the intermediate results obtained in the course of computing them.

    Attributes:
        alpha (float): The |big_alpha| value for the transition when taken
            against this MIP, *i.e.* the difference between the unpartitioned
            account and this MIP's partitioned account.
        unpartitioned_account (Account): The account of the whole transition.
        partitioned_account (Account): The account of the partitioned
            transition.
        transition (Transition): The transition this MIP was calculated for.
        cut (ActualCut): The minimal partition.
    """

    def __init__(self, alpha=None, direction=None, unpartitioned_account=None,
                 partitioned_account=None, transition=None, cut=None):
        self.alpha = alpha
        self.direction = direction
        self.unpartitioned_account = unpartitioned_account
        self.partitioned_account = partitioned_account
        self.transition = transition
        self.cut = cut

    def __repr__(self):
        return fmt.make_repr(self, _ac_sia_attributes)

    def __str__(self):
        return fmt.fmt_ac_sia(self)

    @property
    def before_state(self):
        """Return the actual previous state of the |Transition|."""
        return self.transition.before_state

    @property
    def after_state(self):
        """Return the actual current state of the |Transition|."""
        return self.transition.after_state

    unorderable_unless_eq = ['direction']

    # TODO: shouldn't the minimal irreducible account be chosen?
    def order_by(self):
        return [self.alpha, len(self.transition)]

    def __eq__(self, other):
        return cmp.general_eq(self, other, _ac_sia_attributes)

    def __bool__(self):
        """An |AcSystemIrreducibilityAnalysis| is ``True`` if it has
        |big_alpha > 0|.
        """
        return greater_than_zero(self.alpha)

    def __hash__(self):
        return hash((self.alpha, self.unpartitioned_account,
                     self.partitioned_account, self.transition,
                     self.cut))

    def to_json(self):
        return {attr: getattr(self, attr) for attr in _ac_sia_attributes}


def _null_ac_sia(transition, direction, alpha=0.0):
    """Returns an |AcSystemIrreducibilityAnalysis| with zero |big_alpha| and
    empty accounts.
    """
    return AcSystemIrreducibilityAnalysis(transition=transition,
                    direction=direction,
                    alpha=alpha,
                    unpartitioned_account=(),
                    partitioned_account=())
