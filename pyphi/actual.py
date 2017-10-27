#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# actual.py

'''
Methods for computing actual causation of subsystems and mechanisms.
'''

# pylint: disable=too-many-instance-attributes, too-many-arguments
# pylint: disable=too-many-public-methods

import logging
from itertools import chain
from math import log2 as _log2

import numpy as np

import pyphi

from . import (Direction, compute, config, connectivity, constants, exceptions,
               utils, validate)
from .models import (AcBigMip, Account, AcMip, ActualCut, CausalLink,
                     DirectedAccount, Event, NullCut,
                     _null_ac_bigmip, _null_ac_mip, fmt)
from .subsystem import Subsystem, mip_partitions

log = logging.getLogger(__name__)


def log2(x):
    '''Rounded version of ``log2``.'''
    return round(_log2(x), config.PRECISION)


class Transition:
    '''A state transition between two sets of nodes in a network.

    A |Transition| is implemented with two |Subsystem| objects - one
    representing the system at time |t-1| used to compute effect coefficients,
    and another representing the system at time |t| which is used to compute
    cause coefficients. These subsystems are accessed with the
    ``effect_system`` and ``cause_system`` attributes, and are mapped to the
    causal directions via the ``system`` attribute.

    Args:
        network (Network): The network the subsystem belongs to.
        before_state (tuple[int]): The state of the network at
            time |t-1|.
        after_state (tuple[int]): The state of the network at
            time |t|.
        cause_indices (tuple[int] or tuple[str]): Indices of nodes in the cause
            system. (TODO: clarify)
        effect_indices (tuple[int] or tuple[str]): Indices of nodes in the
            effect system. (TODO: clarify)

    Keyword Args:
        noise_background (bool): If ``True``, background conditions are
            noised instead of frozen.

    Attributes:
        node_indices (tuple[int]): The indices of the nodes in the system.
        network (Network): The network the system belongs to.
        before_state (tuple[int]): The state of the network at time |t-1|.
        after_state (tuple[int]): The state of the network at time |t|.
        effect_system (Subsystem): The system in ``before_state`` used to
            compute effect repertoires and coefficients.
        cause_system (Subsystem): The system in ``after_state`` used to compute
            cause repertoires and coefficients.
        cause_system (Subsystem):
        system (dict): A dictionary mapping causal directions to the system
            used to compute repertoires in that direction.
        cut (ActualCut): The cut that has been applied to this transition.

    .. note::
        During initialization, both the cause and effect systems are
        conditioned on ``before_state`` as the background state. After
        conditioning the ``effect_system`` is then properly reset to
        ``after_state``.
    '''

    def __init__(self, network, before_state, after_state, cause_indices,
                 effect_indices, cut=None, noise_background=False):

        self.network = network
        self.before_state = before_state
        self.after_state = after_state

        parse_nodes = network.parse_node_indices
        self.cause_indices = parse_nodes(cause_indices)
        self.effect_indices = parse_nodes(effect_indices)
        self.node_indices = parse_nodes(cause_indices + effect_indices)

        self.cut = cut if cut is not None else NullCut(self.node_indices)

        # Indices external to the cause system.
        # The TPMs of both systems are conditioned on these background
        # conditions.

        if noise_background:
            # Freeze nothing. Background conditions are noised during
            # repertoire computation
            external_indices = ()
        else:
            # Otherwise, freeze the background conditions.
            external_indices = tuple(sorted(
                set(network.node_indices) - set(cause_indices)))

        # Both are conditioned on the `before_state`, but we then change the
        # state of the cause context to `after_state` to reflect the fact that
        # that we are computing cause repertoires of mechanisms in that state.
        with config.override(VALIDATE_SUBSYSTEM_STATES=False):
            self.effect_system = Subsystem(network, before_state,
                                           self.node_indices, self.cut,
                                           _external_indices=external_indices)

            self.cause_system = Subsystem(network, before_state,
                                          self.node_indices, self.cut,
                                          _external_indices=external_indices)

        self.cause_system.state = after_state
        for node in self.cause_system.nodes:
            node.state = after_state[node.index]

        # Validate the cause system
        # The state of the effect system does not need to be reachable
        # because cause repertoires are never computed for that system.
        validate.state_reachable(self.cause_system)

        # Dictionary mapping causal directions to the system which is used to
        # compute repertoires in that direction
        self.system = {
            Direction.PAST: self.cause_system,
            Direction.FUTURE: self.effect_system
        }

    def __repr__(self):
        return fmt.fmt_transition(self)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return (self.cause_indices == other.cause_indices
                and self.effect_indices == other.effect_indices
                and self.before_state == other.before_state
                and self.after_state == other.after_state
                and self.network == other.network
                and self.cut == other.cut)

    def __hash__(self):
        return hash((self.cause_indices, self.effect_indices, self.before_state,
                     self.after_state, self.network, self.cut))

    def __len__(self):
        return len(self.node_indices)

    def __bool__(self):
        return len(self) > 0

    def to_json(self):
        '''Return a JSON-serializable representation.'''
        return {
            'network': self.network,
            'before_state': self.before_state,
            'after_state': self.after_state,
            'cause_indices': self.cause_indices,
            'effect_indices': self.effect_indices,
            'cut': self.cut
        }

    def apply_cut(self, cut):
        '''Return a cut version of this transition.'''
        return Transition(self.network, self.before_state, self.after_state,
                          self.cause_indices, self.effect_indices, cut)

    def cause_repertoire(self, mechanism, purview):
        '''Return the cause repertoire.'''
        return self.repertoire(Direction.PAST, mechanism, purview)

    def effect_repertoire(self, mechanism, purview):
        '''Return the effect repertoire.'''
        return self.repertoire(Direction.FUTURE, mechanism, purview)

    def unconstrained_cause_repertoire(self, purview):
        '''Return the unconstrained cause repertoire of the occurence.'''
        return self.cause_repertoire((), purview)

    def unconstrained_effect_repertoire(self, purview):
        '''Return the unconstrained effect repertoire of the occurence.'''
        return self.effect_repertoire((), purview)

    def repertoire(self, direction, mechanism, purview):
        '''Returns the cause or effect repertoire function based on a
        direction.

        Args:
            direction (str): The temporal direction, specifiying the cause or
                effect repertoire.
        '''
        system = self.system[direction]

        if not set(purview).issubset(self.purview_indices(direction)):
            raise ValueError('{} is not a {} purview in {}'.format(
                fmt.fmt_mechanism(purview, system), direction, self))

        if not set(mechanism).issubset(self.mechanism_indices(direction)):
            raise ValueError('{} is no a {} mechanism in {}'.format(
                fmt.fmt_mechanism(mechanism, system), direction, self))

        return system.repertoire(direction, mechanism, purview)

    def state_probability(self, direction, repertoire, purview,):
        '''Compute the probability of the purview in its current state given
        the repertoire.

        Collapses the dimensions of the repertoire that correspond to the
        purview nodes onto their state. All other dimension are already
        singular and thus receive 0 as the conditioning index.

        Returns a single probabilty.
        '''
        purview_state = self.purview_state(direction)

        index = tuple(node_state if node in purview else 0
                      for node, node_state in enumerate(purview_state))
        return repertoire[index]

    def probability(self, direction, mechanism, purview):
        '''Probability that the purview is in it's current state given the
        state of the mechanism.'''
        repertoire = self.repertoire(direction, mechanism, purview)

        return self.state_probability(direction, repertoire, purview)

    def unconstrained_probability(self, direction, purview):
        '''Unconstrained probability of the purview.'''
        return self.probability(direction, (), purview)

    def purview_state(self, direction):
        '''The state of the purview when we are computing coefficients in
        ``direction``.

        For example, if we are computing the cause coefficient of a mechanism
        in ``after_state``, the direction is``PAST`` and the ``purview_state``
        is ``before_state``.
        '''
        return {
            Direction.PAST: self.before_state,
            Direction.FUTURE: self.after_state
        }[direction]

    def mechanism_state(self, direction):
        '''The state of the mechanism when we are computing coefficients in
        ``direction``.'''
        return self.system[direction].state

    def mechanism_indices(self, direction):
        '''The indices of nodes in the mechanism system.'''
        return {
            Direction.PAST: self.effect_indices,
            Direction.FUTURE: self.cause_indices
        }[direction]

    def purview_indices(self, direction):
        '''The indices of nodes in the purview system.'''
        return {
            Direction.PAST: self.cause_indices,
            Direction.FUTURE: self.effect_indices
        }[direction]

    def _ratio(self, direction, mechanism, purview):
        return log2(self.probability(direction, mechanism, purview) /
                    self.unconstrained_probability(direction, purview))

    def cause_ratio(self, mechanism, purview):
        '''The cause ratio of the ``purview`` given ``mechanism``.'''
        return self._ratio(Direction.PAST, mechanism, purview)

    def effect_ratio(self, mechanism, purview):
        '''The effect ratio of the ``purview`` given ``mechanism``.'''
        return self._ratio(Direction.FUTURE, mechanism, purview)

    def partitioned_repertoire(self, direction, partition):
        '''Compute the repertoire over the partition in the given direction.'''
        system = self.system[direction]
        return system.partitioned_repertoire(direction, partition)

    def partitioned_probability(self, direction, partition):
        '''Compute the probability of the mechanism over the purview in
        the partition.'''
        repertoire = self.partitioned_repertoire(direction, partition)
        return self.state_probability(direction, repertoire, partition.purview)

    # MIP methods
    # =========================================================================

    # TODO: alias to `irreducible_cause/effect ratio?
    def find_mip(self, direction, mechanism, purview, allow_neg=False):
        '''Find the ratio minimum information partition for a mechanism
        over a purview.

        Args:
            direction (str): |PAST| or |FUTURE|
            mechanism (tuple[int]): A mechanism.
            purview (tuple[int]): A purview.

        Keyword Args:
            allow_neg (boolean): If true, ``alpha`` is allowed to be negative.
                Otherwise, negative values of ``alpha`` will be treated as if
                they were 0.

        Returns:
            AcMip: The found MIP.
        '''
        alpha_min = float('inf')
        probability = self.probability(direction, mechanism, purview)

        for partition in mip_partitions(mechanism, purview):
            partitioned_probability = self.partitioned_probability(
                direction, partition)

            alpha = log2(probability / partitioned_probability)

            # First check for 0
            # Default: don't count contrary causes and effects
            if utils.eq(alpha, 0) or (alpha < 0 and not allow_neg):
                return AcMip(state=self.mechanism_state(direction),
                             direction=direction,
                             mechanism=mechanism,
                             purview=purview,
                             partition=partition,
                             probability=probability,
                             partitioned_probability=partitioned_probability,
                             alpha=0.0)
            # Then take closest to 0
            if (abs(alpha_min) - abs(alpha)) > constants.EPSILON:
                alpha_min = alpha
                acmip = AcMip(state=self.mechanism_state(direction),
                              direction=direction,
                              mechanism=mechanism,
                              purview=purview,
                              partition=partition,
                              probability=probability,
                              partitioned_probability=partitioned_probability,
                              alpha=alpha_min)
        return acmip

    # Phi_max methods
    # =========================================================================

    def potential_purviews(self, direction, mechanism, purviews=False):
        '''Return all purviews that could belong to the core cause/effect.

        Filters out trivially-reducible purviews.

        Args:
            direction (str): Either |PAST| or |FUTURE|.
            mechanism (tuple[int]): The mechanism of interest.

        Keyword Args:
            purviews (tuple[int]): Optional subset of purviews of interest.
        '''
        system = self.system[direction]
        return [purview for purview in system.potential_purviews(
                    direction, mechanism, purviews)
                if set(purview).issubset(self.purview_indices(direction))]

    # TODO: Implement mice cache
    # @cache.method('_mice_cache')
    def find_causal_link(self, direction, mechanism, purviews=False,
                         allow_neg=False):
        '''Return the maximally irreducible cause or effect ratio for a mechanism.

        Args:
            direction (str): The temporal direction, specifying cause or
                effect.
            mechanism (tuple[int]): The mechanism to be tested for
                irreducibility.

        Keyword Args:
            purviews (tuple[int]): Optionally restrict the possible purviews
                to a subset of the subsystem. This may be useful for _e.g._
                finding only concepts that are "about" a certain subset of
                nodes.

        Returns:
            CausalLink: The maximally-irreducible actual cause or effect.
        '''
        purviews = self.potential_purviews(direction, mechanism, purviews)

        # Find the maximal MIP over the remaining purviews.
        if not purviews:
            max_mip = _null_ac_mip(self.mechanism_state(direction),
                                   direction, mechanism, None)
        else:
            # This max should be most positive
            max_mip = max(self.find_mip(direction, mechanism, purview,
                                        allow_neg)
                          for purview in purviews)

        # Construct the corresponding CausalLink
        return CausalLink(max_mip)

    def find_actual_cause(self, mechanism, purviews=False):
        '''Return the actual cause of a mechanism.'''
        return self.find_causal_link(Direction.PAST, mechanism, purviews)

    def find_actual_effect(self, mechanism, purviews=False):
        '''Return the actual effect of a mechanism.'''
        return self.find_causal_link(Direction.FUTURE, mechanism, purviews)

    def find_mice(self, *args, **kwargs):
        '''Backwards-compatible alias for :func:`find_causal_link`.'''
        return self.find_causal_link(*args, **kwargs)


# ============================================================================
# Accounts
# ============================================================================


def directed_account(transition, direction, mechanisms=False, purviews=False,
                     allow_neg=False):
    '''Return the set of all |CausalLinks| of the specified direction.'''
    if mechanisms is False:
        mechanisms = utils.powerset(transition.mechanism_indices(direction),
                                    nonempty=True)
    links = [
        transition.find_causal_link(direction, mechanism, purviews=purviews,
                                    allow_neg=allow_neg)
        for mechanism in mechanisms]

    # Filter out causal links with zero alpha
    return DirectedAccount(filter(None, links))


def account(transition, direction=Direction.BIDIRECTIONAL):
    '''Return the set of all causal links for a |Transition|.

    Args:
        transition (Transition): The transition of interest.

    Keyword Args:
        direction (Direction): By default the account contains actual causes
            and actual effects.
    '''
    if direction != Direction.BIDIRECTIONAL:
        return directed_account(transition, direction)

    return Account(directed_account(transition, Direction.PAST) +
                   directed_account(transition, Direction.FUTURE))


# ============================================================================
# AcBigMips and System cuts
# ============================================================================


def account_distance(A1, A2):
    '''Return the distance between two accounts. Here that is just the
    difference in sum(alpha)

    Args:
        A1 (Account): The first account.
        A2 (Account): The second account

    Returns:
        float: The distance between the two accounts.
    '''
    return (sum([action.alpha for action in A1])
            - sum([action.alpha for action in A2]))


def _evaluate_cut(transition, cut, unpartitioned_account,
                  direction=Direction.BIDIRECTIONAL):
    '''Find the |AcBigMip| for a given cut.'''
    cut_transition = transition.apply_cut(cut)
    partitioned_account = account(cut_transition, direction)

    log.debug("Finished evaluating %s.", cut)
    alpha = account_distance(unpartitioned_account, partitioned_account)

    return AcBigMip(
        alpha=round(alpha, config.PRECISION),
        direction=direction,
        unpartitioned_account=unpartitioned_account,
        partitioned_account=partitioned_account,
        transition=transition,
        cut=cut)


# TODO: implement CUT_ONE approximation?
def _get_cuts(transition, direction):
    '''A list of possible cuts to a transition.'''
    n = transition.network.size

    if direction is Direction.BIDIRECTIONAL:
        yielded = set()
        for cut in chain(_get_cuts(transition, Direction.PAST),
                         _get_cuts(transition, Direction.FUTURE)):
            cm = utils.np_hashable(cut.cut_matrix(n))
            if cm not in yielded:
                yielded.add(cm)
                yield cut

    else:
        mechanism = transition.mechanism_indices(direction)
        purview = transition.purview_indices(direction)
        for partition in mip_partitions(mechanism, purview):
            yield ActualCut(direction, partition)


def big_acmip(transition, direction=Direction.BIDIRECTIONAL):
    '''Return the minimal information partition of a transition in a specific
    direction.

    Args:
        transition (Transition): The candidate system.

    Returns:
        AcBigMip: A nested structure containing all the data from the
        intermediate calculations. The top level contains the basic MIP
        information for the given subsystem.
    '''
    validate.direction(direction, allow_bi=True)
    log.info("Calculating big-alpha for %s...", transition)

    if not transition:
        log.info('Transition %s is empty; returning null MIP '
                 'immediately.', transition)
        return _null_ac_bigmip(transition, direction)

    if not connectivity.is_weak(transition.network.cm, transition.node_indices):
        log.info('%s is not strongly/weakly connected; returning null MIP '
                 'immediately.', transition)
        return _null_ac_bigmip(transition, direction)

    log.debug("Finding unpartitioned account...")
    unpartitioned_account = account(transition, direction)
    log.debug("Found unpartitioned account.")

    if not unpartitioned_account:
        log.info('Empty account; returning null AC MIP immediately.')
        return _null_ac_bigmip(transition, direction)

    cuts = _get_cuts(transition, direction)
    finder = FindBigAcMip(cuts, transition, direction, unpartitioned_account)
    result = finder.run_sequential()
    log.info("Finished calculating big-ac-phi data for %s.", transition)
    log.debug("RESULT: \n%s", result)
    return result


class FindBigAcMip(compute.parallel.MapReduce):
    """Computation engine for AC BigMips."""
    # pylint: disable=unused-argument,arguments-differ

    description = 'Evaluating AC cuts'

    def empty_result(self, transition, direction, unpartitioned_account):
        return _null_ac_bigmip(transition, direction, alpha=float('inf'))

    @staticmethod
    def compute(cut, transition, direction, unpartitioned_account):
        return _evaluate_cut(transition, cut, unpartitioned_account, direction)

    def process_result(self, new_mip, min_mip):
        # Check a new result against the running minimum
        if not new_mip:  # alpha == 0
            self.done = True
            return new_mip

        elif new_mip < min_mip:
            return new_mip

        return min_mip


# ============================================================================
# Complexes
# ============================================================================


# TODO: Fix this to test whether the transition is possible
def transitions(network, before_state, after_state):
    '''Return a generator of all **possible** transitions of a network.
    '''
    # TODO: Does not return subsystems that are in an impossible transitions.

    # Elements without inputs are reducibe effects,
    # elements without outputs are reducible causes.
    possible_causes = np.where(np.sum(network.connectivity_matrix, 1) > 0)[0]
    possible_effects = np.where(np.sum(network.connectivity_matrix, 0) > 0)[0]

    for cause_subset in utils.powerset(possible_causes, nonempty=True):
        for effect_subset in utils.powerset(possible_effects, nonempty=True):
            try:
                yield Transition(network, before_state, after_state,
                                 cause_subset, effect_subset)
            except exceptions.StateUnreachableError:
                pass


def nexus(network, before_state, after_state,
          direction=Direction.BIDIRECTIONAL):
    '''Return a tuple of all irreducible nexus of the network.'''
    validate.is_network(network)

    mips = (big_acmip(transition, direction) for transition in
            transitions(network, before_state, after_state))
    return tuple(sorted(filter(None, mips), reverse=True))


def causal_nexus(network, before_state, after_state,
                 direction=Direction.BIDIRECTIONAL):
    '''Return the causal nexus of the network.'''
    validate.is_network(network)

    log.info("Calculating causal nexus...")
    result = nexus(network, before_state, after_state, direction)
    if result:
        result = max(result)
    else:
        null_transition = Transition(network, before_state, after_state, (), ())
        result = _null_ac_bigmip(null_transition, direction)

    log.info("Finished calculating causal nexus.")
    log.debug("RESULT: \n%s", result)
    return result

# ============================================================================
# True Causes
# ============================================================================


# TODO: move this to __str__
def nice_true_constellation(tc):
    '''Format a true constellation.'''
    past_list = []
    future_list = []
    cause = '<--'
    effect = '-->'
    for event in tc:
        if event.direction == Direction.PAST:
            past_list.append(["{0:.4f}".format(round(event.alpha, 4)),
                              event.mechanism, cause, event.purview])
        elif event.direction == Direction.FUTURE:
            future_list.append(["{0:.4f}".format(round(event.alpha, 4)),
                                event.mechanism, effect, event.purview])
        else:
            validate.direction(event.direction)

    true_list = [(past_list[event], future_list[event])
                 for event in range(len(past_list))]
    return true_list


def _actual_causes(network, past_state, current_state, nodes,
                   mechanisms=False):
    log.info("Calculating true causes ...")
    transition = Transition(network, past_state, current_state, nodes, nodes)

    return directed_account(transition, Direction.PAST, mechanisms=mechanisms)


def _actual_effects(network, current_state, future_state, nodes,
                    mechanisms=False):
    log.info("Calculating true effects ...")
    transition = Transition(network, current_state, future_state, nodes, nodes)

    return directed_account(transition, Direction.FUTURE, mechanisms=mechanisms)


def events(network, past_state, current_state, future_state, nodes,
           mechanisms=False):
    '''Find all events (mechanisms with actual causes and actual effects.'''

    actual_causes = _actual_causes(network, past_state, current_state, nodes,
                                   mechanisms)
    actual_effects = _actual_effects(network, current_state, future_state,
                                     nodes, mechanisms)

    actual_mechanisms = (set(c.mechanism for c in actual_causes) &
                         set(c.mechanism for c in actual_effects))

    if not actual_mechanisms:
        return ()

    def index(actual_causes_or_effects):
        '''Filter out unidirectional occurences and return a
        dictionary keyed by the mechanism of the cause or effect.'''
        return {o.mechanism: o for o in actual_causes_or_effects
                if o.mechanism in actual_mechanisms}

    actual_causes = index(actual_causes)
    actual_effects = index(actual_effects)

    return tuple(Event(actual_causes[m], actual_effects[m])
                 for m in sorted(actual_mechanisms))


# TODO: do we need this? it's just a re-structuring of the `events` results
# TODO: rename to `actual_constellation`?
def true_constellation(subsystem, past_state, future_state):
    '''Set of all sets of elements that have true causes and true effects.

    .. note::
        Since the true constellation is always about the full system,
        the background conditions don't matter and the subsystem should be
        conditioned on the current state.
    '''
    network = subsystem.network
    nodes = subsystem.node_indices
    state = subsystem.state

    _events = events(network, past_state, state, future_state, nodes)

    if not _events:
        log.info("Finished calculating, no echo events.")
        return None

    result = tuple([event.actual_cause for event in _events] +
                   [event.actual_effect for event in _events])
    log.info("Finished calculating true events.")
    log.debug("RESULT: \n%s", result)

    return result


def true_events(network, past_state, current_state, future_state, indices=None,
                main_complex=None):
    '''Return all mechanisms that have true causes and true effects within the
    complex.

    Args:
        network (Network): The network to analyze.
        past_state (tuple[int]): The state of the network at ``t - 1``.
        current_state (tuple[int]): The state of the network at ``t``.
        future_state (tuple[int]): The state of the network at ``t + 1``.

    Keyword Args:
        indices (tuple[int]): The indices of the main complex.
        main_complex (AcBigMip): The main complex. If ``main_complex`` is given
            then ``indices`` is ignored.

    Returns:
        tuple[Event]: List of true events in the main complex.
    '''
    # TODO: validate triplet of states

    if main_complex:
        nodes = main_complex.subsystem.node_indices
    elif indices:
        nodes = indices
    else:
        main_complex = compute.main_complex(network, current_state)
        nodes = main_complex.subsystem.node_indices

    return events(network, past_state, current_state, future_state, nodes)


def extrinsic_events(network, past_state, current_state, future_state,
                     indices=None, main_complex=None):
    '''Set of all mechanisms that are in the main complex but which have true
    causes and effects within the entire network.

    Args:
        network (Network): The network to analyze.
        past_state (tuple[int]): The state of the network at ``t - 1``.
        current_state (tuple[int]): The state of the network at ``t``.
        future_state (tuple[int]): The state of the network at ``t + 1``.

    Keyword Args:
        indices (tuple[int]): The indices of the main complex.
        main_complex (AcBigMip): The main complex. If ``main_complex`` is given
            then ``indices`` is ignored.

    Returns:
        tuple(actions): List of extrinsic events in the main complex.
    '''
    if main_complex:
        mc_nodes = main_complex.subsystem.node_indices
    elif indices:
        mc_nodes = indices
    else:
        main_complex = compute.main_complex(network, current_state)
        mc_nodes = main_complex.subsystem.node_indices

    mechanisms = list(utils.powerset(mc_nodes, nonempty=True))
    all_nodes = network.node_indices

    return events(network, past_state, current_state, future_state, all_nodes,
                  mechanisms=mechanisms)
