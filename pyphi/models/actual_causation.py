#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/actual_causation.py

'''
Objects that represent structures used in actual causation.
'''

from collections import namedtuple

from . import cmp, fmt
from .. import config, utils

# TODO(slipperyhank): Why do we even need this?
# TODO(slipperyhank): add second state
_acmip_attributes = ['alpha', 'state', 'direction', 'mechanism', 'purview',
                     'partition', 'probability', 'partitioned_probability']
_acmip_attributes_for_eq = ['alpha', 'state', 'direction', 'mechanism',
                            'purview', 'probability']


class AcMip(cmp.Orderable, namedtuple('AcMip', _acmip_attributes)):

    '''A minimum information partition for ac_coef calculation.

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, |alpha| values are compared. Then, if these are equal
    up to |PRECISION|, the size of the mechanism is compared.

    Attributes:
        alpha (float):
            This is the difference between the mechanism's unpartitioned and
            partitioned actual probability.
        state (tuple[int]):
            state of system in specified direction (past, future)
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
            The probability of the state in the past/future.
        partitioned_probability (float):
            The probability of the state in the partitioned repertoire.
    '''
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
        '''An |AcMip| is ``True`` if it has |alpha > 0|.'''
        return not utils.eq(self.alpha, 0)

    @property
    def phi(self):
        return self.alpha

    def __hash__(self):
        attrs = tuple(getattr(self, attr) for attr in _acmip_attributes_for_eq)
        return hash(attrs)

    def to_json(self):
        '''Return a JSON-serializable representation.'''
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
                 alpha=0.0)


class Occurence(cmp.Orderable):
    '''A maximally irreducible actual cause or effect.

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, |alpha| values are compared. Then, if these are equal
    up to |PRECISION|, the size of the mechanism is compared.
    '''

    def __init__(self, mip):
        self._mip = mip

    @property
    def alpha(self):
        '''float: The difference between the mechanism's unpartitioned and
        partitioned actual probabilities.
        '''
        return self._mip.alpha

    # TODO(slipperyhank): Define property phi == alpha, to make use of existing
    # util functions
    @property
    def phi(self):
        return self.alpha

    @property
    def direction(self):
        '''Direction: Either |PAST| or |FUTURE|.'''
        return self._mip.direction

    @property
    def mechanism(self):
        '''list[int]: The mechanism for which the action is evaluated.'''
        return self._mip.mechanism

    @property
    def purview(self):
        '''list[int]: The purview over which this mechanism's |alpha| is
        maximal.
        '''
        return self._mip.purview

    @property
    def mip(self):
        '''AcMip: The minimum information partition for this mechanism.'''
        return self._mip

    def __repr__(self):
        return fmt.make_repr(self, ['mip'])

    def __str__(self):
        return "Occurence\n" + fmt.indent(fmt.fmt_ac_mip(self.mip))

    unorderable_unless_eq = AcMip.unorderable_unless_eq

    def order_by(self):
        return self.mip.order_by()

    def __eq__(self, other):
        return self.mip == other.mip

    def __hash__(self):
        return hash(('Occurence', self._mip))

    def __bool__(self):
        '''An |Occurence| is ``True`` if |alpha > 0|.'''
        return not utils.eq(self._mip.alpha, 0)

    def to_json(self):
        '''Return a JSON-serializable representation.'''
        return {'acmip': self._mip}


class Event(namedtuple('Event', ['actual_cause', 'actual_effect'])):
    '''A mechanism which has both an actual cause and an actual effect.

    Attributes:
        actual_cause (Occurence): The actual cause of the mechanism.
        actual_effect (Occurence): The actual effect of the mechanism.
    '''

    @property
    def mechanism(self):
        assert self.actual_cause.mechanism == self.actual_effect.mechanism
        return self.actual_cause.mechanism


class Account(tuple):
    '''The set of occurences with |alpha > 0| for both |PAST| and
    |FUTURE|.'''

    def __repr__(self):
        if config.READABLE_REPRS:
            return self.__str__()
        return "{0}({1})".format(
            self.__class__.__name__, super().__repr__())

    def __str__(self):
        return fmt.fmt_account(self)


class DirectedAccount(Account):
    '''The set of occurences with |alpha > 0| for one direction of a
    context.'''
    pass


_acbigmip_attributes = ['alpha', 'direction', 'unpartitioned_account',
                        'partitioned_account', 'context', 'cut']


# TODO(slipperyhank): Check if we do the same, i.e. take the bigger system, or
# take the smaller?
class AcBigMip(cmp.Orderable):

    '''A minimum information partition for |big_alpha| calculation.

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, |alpha| values are compared. Then, if these are equal
    up to |PRECISION|, the size of the mechanism is compared.

    Attributes:
        alpha (float): The |big_alpha| value for the subsystem when taken
            against this MIP, *i.e.* the difference between the unpartitioned
            constellation and this MIP's partitioned constellation.
        unpartitioned_constellation (tuple[Concept]): The constellation of the
            whole subsystem.
        partitioned_constellation (tuple[Concept]): The constellation when the
            subsystem is cut.
        subsystem (Subsystem): The subsystem this MIP was calculated for.
        cut: The minimal cut.
    '''

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
        '''Return the actual past state of the |Context|.'''
        return self.context.before_state

    @property
    def after_state(self):
        '''Return the actual current state of the |Context|.'''
        return self.context.after_state

    unorderable_unless_eq = ['direction']

    def order_by(self):
        return [self.alpha, len(self.context)]

    def __eq__(self, other):
        return cmp.general_eq(self, other, _acbigmip_attributes)

    def __bool__(self):
        '''An |AcBigMip| is ``True`` if it has |big_alpha > 0|.'''
        return not utils.eq(self.alpha, 0)

    def __hash__(self):
        return hash((self.alpha, self.unpartitioned_account,
                     self.partitioned_account, self.context,
                     self.cut))


def _null_ac_bigmip(context, direction, alpha=0.0):
    '''Returns an |AcBigMip| with zero |big_alpha| and empty constellations.'''
    return AcBigMip(context=context,
                    direction=direction,
                    alpha=alpha,
                    unpartitioned_account=(),
                    partitioned_account=())
