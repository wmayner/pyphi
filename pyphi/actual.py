#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# actual.py

"""
Methods for computing actual causation of subsystems and mechanisms.
"""

import itertools
import logging
from math import log2
from pprint import pprint

import numpy as np

from . import compute, config, connectivity, exceptions, utils, validate
from .constants import EPSILON, Direction
from .jsonify import jsonify
from .models import (AcBigMip, Account, AcMip, ActualCut, DirectedAccount,
                     Event, Occurence, _null_ac_bigmip, _null_ac_mip)
from .partition import bipartition, directed_bipartition
from .subsystem import Subsystem, maximal_mip, mip_partitions

log = logging.getLogger(__name__)


class Context:
    """A set of nodes in a network, with state transitions.

    A |Context| contains two |Subsystem| objects - one representing the system
    at time |t-1| used to compute effect coefficients, and another
    representing the system at time |t| which is used to compute cause
    coefficients. These subsystems are accessed with the ``effect_system`` and
    ``cause_system`` attributes, and are mapped to the causal directions via
    the ``system`` attribute.

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

    Attributes:
        node_indices (tuple[int]): The indices of the nodes in the system.
        network (Network): The network the system belongs to.
        before_state (tuple[int]): The state of the network at
            time |t-1|.
        after_state (tuple[int]): The state of the network at
            time |t|.
        effect_system (Subsystem): The system in ``before_state`` used to
            compute effect repertoires and coefficients.
        cause_system (Subsystem): The system in ``after_state`` used to compute
            cause repertoires and coefficients.
        cause_system (Subsystem):
        system (dict): A dictionary mapping causal directions to the system
            used to compute repertoires in that direction.
        cut (ActualCut): The cut that has been applied to this context.

    .. note::
        During initialization, both the cause and effect systems are
        conditioned on the ``before_state`` as the background state. After
        conditioning the ``effect_system`` is then properly reset to
        ``after_state``.
    """

    def __init__(self, network, before_state, after_state, cause_indices,
                 effect_indices, cut=None):

        self.network = network
        self.before_state = before_state
        self.after_state = after_state

        parse_nodes = network.parse_node_indices
        self.cause_indices = parse_nodes(cause_indices)
        self.effect_indices = parse_nodes(effect_indices)
        self.node_indices = parse_nodes(cause_indices + effect_indices)

        # TODO: clarify that `ActualCut` is implemented correctly (esp.
        # cutting connections)
        self.null_cut = ActualCut((), self.cause_indices,
                                  (), self.effect_indices)
        self.cut = cut if cut is not None else self.null_cut

        # Both are conditioned on the `before_state`, but we then change the
        # state of the cause context to `after_state` to reflect the fact that
        # that we are computing cause repertoires of mechanisms in that state.
        # TODO: should we not validate the state of effect system since we
        # never compute cause repertoires for it and therefore do not have to
        # worry about invalid repertoires?
        self.effect_system = Subsystem(network, before_state,
                                       self.node_indices, self.cut)
        self.cause_system = Subsystem(network, before_state,
                                      self.node_indices, self.cut)
        self.cause_system.state = after_state
        for node in self.cause_system.nodes:
            node.state = after_state[node.index]

        # Dictionary mapping causal directions to the system which is used to
        # compute repertoires in that direction
        self.system = {
            Direction.PAST: self.cause_system,
            Direction.FUTURE: self.effect_system
        }

        self._hash = hash((
            self.cause_indices, self.effect_indices, self.before_state,
            self.after_state, self.network, self.cut))

    def __repr__(self):
        return "Context(cause: {}, effect: {})".format(
            self.cause_system.indices2nodes(self.cause_indices),
            self.effect_system.indices2nodes(self.effect_indices))

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return (self.cause_indices == self.effect_indices
                and self.before_state == other.before_state
                and self.after_state == other.after_state
                and self.network == other.network
                and self.cut == other.cut)

    def __hash__(self):
        return self._hash

    def __len__(self):
        return len(self.node_indices)

    def __bool__(self):
        return len(self) > 0

    def to_json(self):
        return {
            'network': jsonify(self.network),
            'before_state': self.before_state,
            'after_state': self.after_state,
            'cause_indices': self.cause_indices,
            'effect_indices': self.effect_indices,
            'cut': jsonify(self.cut),
        }

    def apply_cut(self, cut):
        """Return a cut version of this context."""
        return Context(self.network, self.before_state, self.after_state,
                       self.cause_indices, self.effect_indices, cut)

    # TODO: remove these named repertoire methods and just use `_repertoire`?
    def cause_repertoire(self, mechanism, purview):
        return self.cause_system.cause_repertoire(mechanism, purview)

    def effect_repertoire(self, mechanism, purview):
        return self.effect_system.effect_repertoire(mechanism, purview)

    def unconstrained_cause_repertoire(self, purview):
        return self.cause_repertoire((), purview)

    def unconstrained_effect_repertoire(self, purview):
        return self.effect_repertoire((), purview)

    def _repertoire(self, direction, mechanism, purview):
        """Returns the cause or effect repertoire function based on a
        direction.

        Args:
            direction (str): The temporal direction, specifiying the cause or
                effect repertoire.
        """
        system = self.system[direction]
        return system._repertoire(direction, mechanism, purview)

    def state_probability(self, direction, repertoire, purview,):
        """ The dimensions of the repertoire that correspond to the fixed nodes
        are collapsed onto their state. All other dimension should be singular
        already (repertoire size and fixed_nodes need to match), and thus
        should receive 0 as the conditioning index. A single probability is
        returned.
        """
        purview_state = self.purview_state(direction)

        index = tuple(purview_state[i] if i in purview else 0
                      for i in range(len(purview_state)))
        return repertoire[index]

    def probability(self, direction, mechanism, purview):
        """Probability that the purview is in it's current state given the
        state of the mechanism."""
        repertoire = self._repertoire(direction, mechanism, purview)

        return self.state_probability(direction, repertoire, purview)

    def unconstrained_probability(self, direction, purview):
        """Unconstrained probability of the purview."""
        return self.probability(direction, (), purview)

    def purview_state(self, direction):
        """The state of the purview when we are computing coefficients in
        ``direction``.

        For example, if we are computing the cause coefficient of a mechanism
        in ``after_state``, the direction is``PAST`` and the ``purview_state``
        is ``before_state``.
        """
        if direction == Direction.PAST:
            purview_state = self.before_state
        elif direction == Direction.FUTURE:
            purview_state = self.after_state

        return purview_state

    def mechanism_state(self, direction):
        """The state of the mechanism when we are computing coefficients in
        ``direction``."""
        return self.system[direction].state

    def _normalize(self, probability, direction, purview, norm=True):
        """Normalize the probability of a purview in the given direction."""
        if not norm:
            return probability

        return probability / self.unconstrained_probability(direction, purview)

    def _coefficient(self, direction, mechanism, purview, norm=True):
        """Return the cause or effect coefficient of a mechanism over a
        purview."""
        p = self.probability(direction, mechanism, purview)
        return self._normalize(p, direction, purview, norm)

    def cause_coefficient(self, mechanism, purview, norm=True):
        """ Return the cause coefficient for a mechanism in a state over a
        purview in the actual past state """
        return self._coefficient(Direction.PAST, mechanism, purview, norm)

    def effect_coefficient(self, mechanism, purview, norm=True):
        """ Return the effect coefficient for a mechanism in a state over a
        purview in the actual future state """
        return self._coefficient(Direction.FUTURE, mechanism, purview, norm)

    def partitioned_repertoire(self, direction, partition):
        """Compute the repertoire over the partition in the given direction."""
        system = self.system[direction]
        return system.partitioned_repertoire(direction, partition)

    def partitioned_probability(self, direction, partition):
        """Compute the probability of the mechanism over the purview in
        the partition."""
        repertoire = self.partitioned_repertoire(direction, partition)
        return self.state_probability(direction, repertoire, partition.purview)

    # MIP methods
    # =========================================================================

    def find_mip(self, direction, mechanism, purview,
                 norm=True, allow_neg=False):
        """Find the coefficient minimum information partition for a mechanism
        over a purview.

        Args:
            direction (str): |past| or |future|
            mechanism (tuple[int]): A mechanism.
            purview (tuple[int]): A purview.

        Keyword Args:
            norm (boolean): If true, probabilities will be normalized.
            allow_neg (boolean): If true, ``alpha`` is allowed to be negative.
                Otherwise, negative values of ``alpha`` will be treated as if
                they were 0.

        Returns:
            AcMip: The found MIP.
        """
        alpha_min = float('inf')
        probability = self.probability(direction, mechanism, purview)

        # Loop over possible MIP partitions
        if config.PARTITION_TYPE == 'BI':
            partitions = mip_bipartitions(mechanism, purview)
        elif config.PARTITION_TYPE == 'TRI':
            partitions = wedge_partitions(mechanism, purview)
        elif config.PARTITION_TYPE == 'ALL':
            partitions = all_partitions(mechanism, purview)

        for partition in partitions:
            partitioned_probability = self.partitioned_probability(
                direction, partition)

            if config.PARTITION_TYPE == 'TRI':
                alpha = log2(probability / partitioned_probability)
            else:
                alpha = self._normalize(probability - partitioned_probability,
                                        direction, purview)

            # First check for 0
            # Default: don't count contrary causes and effects
            if utils.phi_eq(alpha, 0) or (alpha < 0 and not allow_neg):
                return AcMip(state=self.mechanism_state(direction),
                             direction=direction,
                             mechanism=mechanism,
                             purview=purview,
                             partition=partition,
                             probability=probability,
                             partitioned_probability=partitioned_probability,
                             alpha=0.0)
            # Then take closest to 0
            if (abs(alpha_min) - abs(alpha)) > EPSILON:
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

    def _potential_purviews(self, direction, mechanism, purviews=False):
        """Return all purviews that could belong to the core cause/effect.

        Filters out trivially-reducible purviews.

        Args:
            direction (str): Either |past| or |future|.
            mechanism (tuple[int]): The mechanism of interest.

        Keyword Argss:
            purviews (tuple[int]): Optional subset of purviews of interest.
        """
        system = self.system[direction]
        return system._potential_purviews(direction, mechanism, purviews)

    # TODO: Implement mice cache
    # @cache.method('_mice_cache')
    def find_occurence(self, direction, mechanism, purviews=False,
                       norm=True, allow_neg=False):
        """Return the maximally irreducible cause or effect coefficient for a mechanism.

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
            |Occurence|: The maximally-irreducible actual cause or effect.

        .. note::
            Strictly speaking, the |Occurence| is a pair of coefficients: the
            actual cause and actual effect of a mechanism. Here, we return only
            information corresponding to one direction, |past| or |future|,
            i.e., we return an actual cause or actual effect coefficient, not
            the pair of them.
        """

        purviews = self._potential_purviews(direction, mechanism, purviews)

        # Find the maximal MIP over the remaining purviews.
        if not purviews:
            max_mip = _null_ac_mip(self.mechanism_state(direction),
                                   direction, mechanism, None)
        else:
            # This max should be most positive
            mips = [self.find_mip(direction, mechanism, purview, norm, allow_neg)
                    for purview in purviews]

            if config.PARTITION_TYPE == 'TRI':
                # In the case of tie, chose the mip with smallest purview.
                # (The default behavior is to chose the larger purview.)
                max_mip = max(mips, key=lambda m: (m.alpha, -len(m.purview)))
            else:
                max_mip = max(mips)

        # Construct the corresponding Occurence
        return Occurence(max_mip)

    def find_mice(self, *args, **kwargs):
        """Backwards-compatible alias for `find_occurence`."""
        return self.find_occurence(*args, **kwargs)


# ===========================================================================
# Printed Results
# ============================================================================


def nice_ac_composition(account):
    if account:
        if account[0].direction == Direction.PAST:
            dir_arrow = '<--'
        elif account[0].direction == Direction.FUTURE:
            dir_arrow = '-->'
        else:
            validate.direction(account.direction)
        actions = [["{0:.4f}".format(round(action.alpha, 4)),
                    action.mechanism, dir_arrow, action.purview]
                   for action in account]
        return actions
    else:
        return None


def multiple_states_nice_ac_composition(network, transitions, cause_indices,
                                        effect_indices, mechanisms=False,
                                        purviews=False, norm=True,
                                        allow_neg=False):
    """Print a nice composition for multiple pairs of states.

    Args:
        transitions (list(2 state tuples)): The first is past the second
            current. For 'past' current belongs to subsystem and past is the
            second state. Vice versa for "future"
    """
    for transition in transitions:
        context = Context(network, transition[0], transition[1], cause_indices,
                          effect_indices)
        cause_account = directed_account(context, Direction.PAST, mechanisms,
                                         purviews, norm, allow_neg)
        effect_account = directed_account(context, Direction.FUTURE,
                                          mechanisms, purviews, norm, allow_neg)
        print('#####################################')
        print(transition)
        print('- cause coefs ----------------------')
        pprint(nice_ac_composition(cause_account))
        print('- effect coefs ----------------------')
        pprint(nice_ac_composition(effect_account))
        print('---------------------------')

# ============================================================================
# Average over mechanisms - constellations
# ============================================================================


def directed_account(context, direction, mechanisms=False, purviews=False,
                     norm=True, allow_neg=False):
    """Set of all Occurence of the specified direction"""
    if mechanisms is False:
        if direction == Direction.PAST:
            mechanisms = utils.powerset(context.effect_indices)
        elif direction == Direction.FUTURE:
            mechanisms = utils.powerset(context.cause_indices)

    actions = [context.find_occurence(direction, mechanism, purviews=purviews,
                                      norm=norm, allow_neg=allow_neg)
               for mechanism in mechanisms]

    # Filter out MICE with zero alpha
    return DirectedAccount(filter(None, actions))


def account(context, direction):
    if direction == Direction.BIDIRECTIONAL:
        return Account(directed_account(context, Direction.PAST) +
                       directed_account(context, Direction.FUTURE))
    return directed_account(context, direction)


# ============================================================================
# AcBigMips and System cuts
# ============================================================================


def account_distance(A1, A2):
    """Return the distance between two accounts. Here that is just the
    difference in sum(alpha)

    Args:
        A1 (Account): The first account.
        A2 (Account): The second account

    Returns:
        float: The distance between the two accounts.
    """
    return (sum([action.alpha for action in A1])
            - sum([action.alpha for action in A2]))


# TODO: single node BigMip
def _evaluate_cut_directed(context, cut, account, direction):
    """ Returns partitioned constellation for one direction past/future of the
    transition. For direction = bidirectional, the uncut subsystem is
    subsystem_past and uncut_subsystem2_or_actual_state is subsystem_future. To
    make cut subsystem: To have the right background, the init state for the
    subsystem should always be the past_state. In past direction after
    subsystem is created the actual state and the system state need to be
    swapped."""

    cut_context = context.apply_cut(cut)

    # TODO: Implement shortcuts to avoid recomputing actions?
    # if config.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS:
    #     mechanisms = set([c.mechanism for c in unpartitioned_constellation])
    # else:
    # mechanisms = set([c.mechanism for c in unpartitioned_constellation] +
    #                      list(cut_mechanism_indices(uncut_subsystem, cut)))
    partitioned_account = directed_account(cut_context, direction)
    return partitioned_account


def _evaluate_cut(context, cut, unpartitioned_account, direction=None):
    """Find the |AcBigMip| for a given cut."""
    cut_context = context.apply_cut(cut)

    if not direction:
        direction = Direction.BIDIRECTIONAL

    partitioned_account = account(cut_context, direction)

    log.debug("Finished evaluating cut {}.".format(cut))
    alpha = account_distance(unpartitioned_account, partitioned_account)

    return AcBigMip(
        alpha=alpha,
        direction=direction,
        unpartitioned_account=unpartitioned_account,
        partitioned_account=partitioned_account,
        context=context,
        cut=cut)


def _get_cuts(context):
    """A list of possible cuts to a context."""

    # TODO: Add one-cut approximation as an option.
    # if config.CUT_ONE_APPROXIMATION:
    #     bipartitions = directed_bipartition_of_one(subsystem.node_indices)
    # else:
    cause_bipartitions = bipartition(context.cause_indices)
    effect_bipartitions = directed_bipartition(context.effect_indices)
    # The first element of the list is the null cut.
    partitions = list(itertools.product(cause_bipartitions,
                                        effect_bipartitions))[1:]
    cuts = [ActualCut(part[0][0], part[0][1], part[1][0], part[1][1])
            for part in partitions]
    return cuts


def big_acmip(context, direction=None):
    """Return the minimal information partition of a context in a specific
    direction.

    Args:
        context (Context): The candidate system.

    Returns:
        AcBigMip: A nested structure containing all the data from the
            intermediate calculations. The top level contains the basic MIP
            information for the given subsystem.
    """
    if not direction:
        direction = Direction.BIDIRECTIONAL
    validate.direction(direction)
    log.info("Calculating big-alpha for {}...".format(context))

    if not context:
        log.info('Context {} is empty; returning null MIP '
                 'immediately.'.format(context))
        return _null_ac_bigmip(context, direction)

    if not connectivity.is_weak(context.network.cm, context.node_indices):
        log.info('{} is not strongly/weakly connected; returning null MIP '
                 'immediately.'.format(context))
        return _null_ac_bigmip(context, direction)
    cuts = _get_cuts(context)

    log.debug("Finding unpartitioned account...")
    unpartitioned_account = account(context, direction)
    log.debug("Found unpartitioned account.")

    if not unpartitioned_account:
        # Short-circuit if there are no actions in the unpartitioned
        # account.
        result = _null_ac_bigmip(context, direction)
    else:
        ac_mip = _null_ac_bigmip(context, direction)
        ac_mip.alpha = float('inf')
        for i, cut in enumerate(cuts):
            new_ac_mip = _evaluate_cut(context, cut, unpartitioned_account,
                                       direction)
            log.debug("Finished {} of {} cuts.".format(
                i + 1, len(cuts)))
            if new_ac_mip < ac_mip:
                ac_mip = new_ac_mip
            # Short-circuit as soon as we find a MIP with effectively 0 phi.
            if not ac_mip:
                break
        result = ac_mip
    log.info("Finished calculating big-ac-phi data for {}.".format(context))
    log.debug("RESULT: \n" + str(result))
    return result

# ============================================================================
# Complexes
# ============================================================================


# TODO: Fix this to test whether the transition is possible
def contexts(network, before_state, after_state):
    """Return a generator of all **possible** contexts of a network.
    """
    # TODO: Does not return subsystems that are in an impossible transitions.

    # Elements without inputs are reducibe effects,
    # elements without outputs are reducible causes.
    possible_causes = np.where(np.sum(network.connectivity_matrix, 1) > 0)[0]
    possible_effects = np.where(np.sum(network.connectivity_matrix, 0) > 0)[0]

    for cause_subset in utils.powerset(possible_causes):
        for effect_subset in utils.powerset(possible_effects):

            if cause_subset and effect_subset:
                try:
                    yield Context(network, before_state, after_state,
                                  cause_subset, effect_subset)
                except exceptions.StateUnreachableError:
                    pass


def nexus(network, before_state, after_state, direction=None):
    """Return a generator for all irreducible nexus of the network.
       Direction options are past, future, bidirectional. """
    validate.is_network(network)

    if not direction:
        direction = Direction.BIDIRECTIONAL

    return tuple(filter(None, (big_acmip(context, direction) for context in
                               contexts(network, before_state, after_state))))


def causal_nexus(network, before_state, after_state, direction=None):
    """Return the causal nexus of the network."""
    validate.is_network(network)

    if not direction:
        direction = Direction.BIDIRECTIONAL

    log.info("Calculating causal nexus...")
    result = nexus(network, before_state, after_state, direction)
    if result:
        result = max(result)
    else:
        empty_context = Context(network, before_state, after_state, (), ())
        result = _null_ac_bigmip(empty_context, direction)
    log.info("Finished calculating causal nexus.")
    log.debug("RESULT: \n" + str(result))
    return result

# ============================================================================
# True Causes
# ============================================================================


def nice_true_constellation(true_constellation):
    # TODO: Make sure the past and future purviews are ordered in the same way
    past_list = []
    future_list = []
    cause = '<--'
    effect = '-->'
    for event in true_constellation:
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
    context = Context(network, past_state, current_state, nodes, nodes)

    return directed_account(context, Direction.PAST, mechanisms=mechanisms)


def _actual_effects(network, current_state, future_state, nodes,
                    mechanisms=False):
    log.info("Calculating true effects ...")
    context = Context(network, current_state, future_state, nodes, nodes)

    return directed_account(context, Direction.FUTURE, mechanisms=mechanisms)


def events(network, past_state, current_state, future_state, nodes,
           mechanisms=False):
    """Find all events (mechanisms with actual causes and actual effects."""

    actual_causes = _actual_causes(network, past_state, current_state, nodes,
                                   mechanisms)
    actual_effects = _actual_effects(network, current_state, future_state,
                                     nodes, mechanisms)

    actual_mechanisms = (set(c.mechanism for c in actual_causes) &
                         set(c.mechanism for c in actual_effects))

    if not actual_mechanisms:
        return ()

    def index(actual_causes_or_effects):
        """Filter out unidirectional occurences and return a
        dictionary keyed by the mechanism of the cause or effect."""
        return {o.mechanism: o for o in actual_causes_or_effects
                if o.mechanism in actual_mechanisms}

    actual_causes = index(actual_causes)
    actual_effects = index(actual_effects)

    return tuple(Event(actual_causes[m], actual_effects[m])
                 for m in sorted(actual_mechanisms))


# TODO: do we need this? it's just a re-structuring of the `events` results
# TODO: rename to `actual_constellation`?
def true_constellation(subsystem, past_state, future_state):
    """Set of all sets of elements that have true causes and true effects.

    .. note::
        Since the true constellation is always about the full system,
        the background conditions don't matter and the subsystem should be
        conditioned on the current state.
    """
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
    log.debug("RESULT: \n" + str(result))

    return result


def true_events(network, past_state, current_state, future_state, indices=None,
                main_complex=None):
    """Return all mechanisms that have true causes and true effects within the
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
    """
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
    """Set of all mechanisms that are in the main complex but which have true
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
    """
    if main_complex:
        mc_nodes = main_complex.subsystem.node_indices
    elif indices:
        mc_nodes = indices
    else:
        main_complex = compute.main_complex(network, current_state)
        mc_nodes = main_complex.subsystem.node_indices

    mechanisms = list(utils.powerset(mc_nodes))[1:]
    all_nodes = network.node_indices

    return events(network, past_state, current_state, future_state, all_nodes,
                  mechanisms=mechanisms)
