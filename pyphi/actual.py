#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# actual.py

"""
Methods for computing actual causation of subsystems and mechanisms.
"""

import logging
import numpy as np

from . import validate, utils, compute
from .network import Network
from .utils import powerset, bipartition, directed_bipartition, phi_eq
from .constants import DIRECTIONS, FUTURE, PAST, EPSILON
from .models import AcMip, AcMice, AcBigMip, _null_ac_mip, ActualCut
from .subsystem import mip_bipartitions, Subsystem

import itertools
from pprint import pprint
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Create a logger for this module.
log = logging.getLogger(__name__)


class Context:
    """A set of nodes in a network, with state transitions.

    Args:
        network (Network): The network the subsystem belongs to.
        before_state (tuple[int]): The state of the network at
            time ``t-1``.
        after_state (tuple[int]): The state of the network at
            time ``t``.
        cause_indices (tuple[int] or tuple[str]): Indices of nodes in the cause
            system. (TODO: clarify)
        effect_indices (tuple[int] or tuple[str]): Indices of nodes in the
            effect system. (TODO: clarify)

    Attributes:
        node_indices (tuple(int)): The indices of the nodes in the subsystem.
        network (Network): The network the subsystem belongs to.
        before_state (tuple[int]): The state of the network at
            time ``t-1``.
        after_state (tuple[int]): The state of the network at
            time ``t``.
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

        self.cause_indices = network.parse_node_indices(cause_indices)
        self.effect_indices = network.parse_node_indices(effect_indices)
        self.node_indices = network.parse_node_indices(cause_indices +
                                                       effect_indices)

        # TODO: pass in cut ????
        # Both are conditioned on the `before_state`, but we then change the
        # state of the cause context to `after_state` to reflect the fact that
        # that we are computing cause repertoires of mechanisms in that state.
        self.effect_system = Subsystem(network, before_state, self.node_indices)
        self.cause_system = Subsystem(network, before_state, self.node_indices)
        self.cause_system.state = after_state
        for node in self.cause_system.nodes:
            node.state = after_state[node.index]

        # Dictionary mapping causal directions to the system which is used to
        # compute repertoires in that direction
        self.system = {
            DIRECTIONS[PAST]: self.cause_system,
            DIRECTIONS[FUTURE]: self.effect_system
        }

        self.null_cut = ActualCut((), self.cause_indices,
                                  (), self.effect_indices)

        # The unidirectional cut applied for phi evaluation within the
        self.cut = cut if cut is not None else self.null_cut

        # Get the subsystem's connectivity matrix. This is the network's
        # connectivity matrix, but with the cut applied, and with all
        # connections to/from external nodes severed.
        # TODO: validate that this is an ActualCut
        self.connectivity_matrix = self.cut.apply_cut(network.cm)
        # Get the perturbation probabilities for each node in the network
        self.perturb_vector = network.perturb_vector

        # TODO: Reimplement the matrix of connections which are severed due to
        # the cut
        # self.cut_matrix = self.cut.cut_matrix()

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

    # TODO jsonify
    # def to_json(self):
    #    return {
    #         'node_indices': jsonify(self.node_indices),
    #         'cut': jsonify(self.cut),
    #    }

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
        is ``before_state.
        """
        if direction == DIRECTIONS[PAST]:
            purview_state = self.before_state
        elif direction == DIRECTIONS[FUTURE]:
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
        return self._coefficient(DIRECTIONS[PAST], mechanism, purview, norm)

    def effect_coefficient(self, mechanism, purview, norm=True):
        """ Return the effect coefficient for a mechanism in a state over a
        purview in the actual future state """
        return self._coefficient(DIRECTIONS[FUTURE], mechanism, purview, norm)

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
        """ Return the cause coef mip minimum information partition for a mechanism
            over a cause purview.
            Returns:
                ap_phi_min: The min. difference of the actual probabilities of
                            the unpartitioned cause and its MIP
            Todo: also return cut etc. ?
        """
        alpha_min = float('inf')
        probability = self.probability(direction, mechanism, purview)
        unconstrained_probability = self.unconstrained_probability(
            direction, purview)

        for partition in mip_bipartitions(mechanism, purview):
            partitioned_probability = self.partitioned_probability(
                direction, partition)
            alpha = self._normalize(probability - partitioned_probability,
                                    direction, purview)

            # First check for 0
            # Default: don't count contrary causes and effects
            if phi_eq(alpha, 0) or (alpha < 0 and not allow_neg):
                return AcMip(state=self.mechanism_state(direction),
                             direction=direction,
                             mechanism=mechanism,
                             purview=purview,
                             partition=partition,
                             probability=probability,
                             partitioned_probability=partitioned_probability,
                             unconstrained_probability=unconstrained_probability,
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
                              unconstrained_probability=unconstrained_probability,
                              alpha=alpha_min)
        return acmip

    # Phi_max methods
    # =========================================================================

    def _potential_purviews(self, direction, mechanism, purviews=False):
        """Return all purviews that could belong to the core cause/effect.

        Filters out trivially-reducible purviews.

        Args:
            direction ('str'): Either |past| or |future|.
            mechanism (tuple(int)): The mechanism of interest.

        Kwargs:
            purviews (tuple(int)): Optional subset of purviews of interest.
        """
        system = self.system[direction]
        return system._potential_purviews(direction, mechanism, purviews)

    # TODO: Implement mice cache
    # @cache.method('_mice_cache')
    def find_mice(self, direction, mechanism, purviews=False,
                  norm=True, allow_neg=False):
        """Return the maximally irreducible cause or effect coefficient for a mechanism.

        Args:
            direction (str): The temporal direction, specifying cause or
                effect.
            mechanism (tuple(int)): The mechanism to be tested for
                irreducibility.

        Keyword Args:
            purviews (tuple(int)): Optionally restrict the possible purviews
                to a subset of the subsystem. This may be useful for _e.g._
                finding only concepts that are "about" a certain subset of
                nodes.
        Returns:
            ac_mice: The maximally-irreducible actual cause or effect.

        .. note::
            Strictly speaking, the AC_MICE is a pair of coefficients: the
            actual cause and actual effect of a mechanism. Here, we return only
            information corresponding to one direction, |past| or |future|,
            i.e., we return an actual cause or actual effect coefficient, not
            the pair of them.
        """

        purviews = self._potential_purviews(direction, mechanism, purviews)

        # Find the maximal MIP over the remaining purviews.
        if not purviews:
            maximal_mip = _null_ac_mip(self.mechanism_state(direction),
                                       direction, mechanism, None)
        else:
            # This max should be most positive
            maximal_mip = max(self.find_mip(direction, mechanism,
                                            purview, norm, allow_neg)
                              for purview in purviews)

        # Construct the corresponding AcMICE.
        return AcMice(maximal_mip)


# ===========================================================================
# Printed Results
# ============================================================================


def nice_ac_composition(account):
    if account:
        if account[0].direction == DIRECTIONS[PAST]:
            dir_arrow = '<--'
        elif account[0].direction == DIRECTIONS[FUTURE]:
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
    """nice composition for multiple pairs of states
    Args: As above
        transitions (list(2 state tuples)):
            The first is past the second current.
            For 'past' current belongs to subsystem and past is the second
            state. Vice versa for "future"
    """
    for transition in transitions:
        context = Context(network, transition[0], transition[1], cause_indices,
                          effect_indices)
        cause_account = directed_account(context, 'past', mechanisms, purviews,
                                         norm, allow_neg)
        effect_account = directed_account(context, 'future', mechanisms,
                                          purviews, norm, allow_neg)
        print('#####################################')
        print(transition)
        print('- cause coefs ----------------------')
        pprint(nice_ac_composition(cause_account, 'past'))
        print('- effect coefs ----------------------')
        pprint(nice_ac_composition(effect_account, 'future'))
        print('---------------------------')

# ============================================================================
# Average over mechanisms - constellations
# ============================================================================


def directed_account(context, direction, mechanisms=False, purviews=False,
                     norm=True, allow_neg=False):
    """Set of all AcMice of the specified direction"""
    if mechanisms is False:
        if direction == 'past':
            mechanisms = powerset(context.effect_indices)
        elif direction == 'future':
            mechanisms = powerset(context.cause_indices)
    actions = [context.find_mice(direction, mechanism, purviews=purviews,
                                 norm=norm, allow_neg=allow_neg)
               for mechanism in mechanisms]
    # Filter out falsy acmices, i.e. those with effectively zero ac_diff.
    return tuple(filter(None, actions))

# ============================================================================
# AcBigMips and System cuts
# ============================================================================


def account_distance(A1, A2):
    """Return the distance between two accounts. Here that is just the
    difference in sum(alpha)

    Args:
        A1 (tuple(Concept)): The first constellation.
        A2 (tuple(Concept)): The second constellation.

    Returns:
        distance (``float``): The distance between the two constellations in
            concept-space.
    """
    return (sum([action.alpha for action in A1])
            - sum([action.alpha for action in A2]))


def _null_ac_bigmip(context, direction):
    """Returns an ac |BigMip| with zero |big_ap_phi| and empty constellations.
    For direction = bidirectional, the subsystem is subsystem_past and
    subsystem2_or_actual_state is subsystem_future. """
    if direction == DIRECTIONS[FUTURE]:
        return AcBigMip(context=context, direction=direction, alpha=0.0,
                        unpartitioned_account=(), partitioned_account=())
    else:
        return AcBigMip(context=context, direction=direction, alpha=0.0,
                        unpartitioned_account=(), partitioned_account=())


# TODO: single node BigMip
def _evaluate_cut_directed(context, cut, account, direction):
    """ Returns partitioned constellation for one direction past/future of the
    transition. For direction = bidirectional, the uncut subsystem is
    subsystem_past and uncut_subsystem2_or_actual_state is subsystem_future. To
    make cut subsystem: To have the right background, the init state for the
    subsystem should always be the past_state. In past direction after
    subsystem is created the actual state and the system state need to be
    swapped."""

    # Important to use make_ac_subsystem because otherwise the past
    # cut_subsystem has the wrong conditioning.
    cut_context = Context(context.network,
                          context.before_state,
                          context.after_state,
                          context.cause_indices,
                          context.effect_indices,
                          cut)
    # TODO: Implement shortcuts to avoid recomputing actions?
    # if config.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS:
    #     mechanisms = set([c.mechanism for c in unpartitioned_constellation])
    # else:
    # mechanisms = set([c.mechanism for c in unpartitioned_constellation] +
    #                      list(cut_mechanism_indices(uncut_subsystem, cut)))
    partitioned_account = directed_account(cut_context, direction)
    return partitioned_account


def _evaluate_cut(context, cut, unpartitioned_account, direction=None):
    """Find the |AcBigMip| for a given cut. For direction = bidirectional, the
    uncut subsystem is subsystem_past and uncut_subsystem2_or_actual_state is
    subsystem_future. """
    cut_context = Context(context.network,
                          context.before_state,
                          context.after_state,
                          context.cause_indices,
                          context.effect_indices,
                          cut)
    if not direction:
        direction = 'bidirectional'
    if direction == 'bidirectional':
        past_partitioned_account = directed_account(cut_context, 'past')
        future_partitioned_account = directed_account(cut_context, 'future')
        partitioned_account = tuple(past_partitioned_account
                                    + future_partitioned_account)
    else:
        partitioned_account = directed_account(cut_context, direction)

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
    """ A list of possible cuts to a context.
    Returns:
        cuts: A list of cuts to evaluate. """

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
        subsystem (Subsystem): The candidate set of nodes.
    Returns:
        big_mip (|BigMip|): A nested structure containing all the data from the
            intermediate calculations. The top level contains the basic MIP
            information for the given subsystem.
    """
    if not direction:
        direction = 'bidirectional'
    validate.direction(direction)
    log.info("Calculating big-alpha for {}...".format(context))

    if not context:
        log.info('Context {} is empty; returning null MIP '
                 'immediately.'.format(context))
        return _null_ac_bigmip(context, direction)

    submatrix_indices = np.ix_(context.node_indices, context.node_indices)
    cm = context.network.connectivity_matrix[submatrix_indices]
    # Get the number of weakly or strongly connected components.
    num_components, _ = connected_components(csr_matrix(cm),
                                             connection='weak')
    if num_components > 1:
        log.info('{} is not strongly/weakly connected; returning null MIP '
                 'immediately.'.format(context))
        return _null_ac_bigmip(context, direction)
    cuts = _get_cuts(context)

    log.debug("Finding unpartitioned account...")
    if direction == 'bidirectional':
        unpartitioned_account = tuple(directed_account(context, 'past')
                                      + directed_account(context, 'future'))
    else:
        unpartitioned_account = directed_account(context, direction)
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
    """Return a generator of all **possible** subsystems of a network.
    """
    # TODO: Does not return subsystems that are in an impossible transitions.
    context_list = []
    # Elements without inputs are reducibe effects,
    # elements without outputs are reducible causes.
    possible_causes = np.where(np.sum(network.connectivity_matrix, 1) > 0)[0]
    possible_effects = np.where(np.sum(network.connectivity_matrix, 0) > 0)[0]
    for cause_subset in powerset(possible_causes):
        for effect_subset in powerset(possible_effects):
            context_list.append(Context(network, before_state, after_state,
                                        cause_subset, effect_subset))

    def not_empty(context):
        """
        Ensures both cause and effect indices are not empty
        """
        return bool(context.cause_indices and context.effect_indices)

    return list(filter(not_empty, context_list))


def nexus(network, before_state, after_state, direction=None):
    """Return a generator for all irreducible nexus of the network.
       Direction options are past, future, bidirectional. """
    if not isinstance(network, Network):
        raise ValueError(
            """Input must be a Network (perhaps you passed a Subsystem
            instead?)""")

    if not direction:
        direction = 'bidirectional'

    return tuple(filter(None, (big_acmip(context, direction) for context in
                               contexts(network, before_state, after_state))))


def causal_nexus(network, before_state, after_state, direction=None):
    """Return the causal nexus of the network."""
    if not isinstance(network, Network):
        raise ValueError(
            """Input must be a Network (perhaps you passed a Subsystem
            instead?)""")
    if not direction:
        direction = 'bidirectional'
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
        if event.direction == DIRECTIONS[PAST]:
            past_list.append(["{0:.4f}".format(round(event.alpha, 4)),
                              event.mechanism, cause, event.purview])
        elif event.direction == DIRECTIONS[FUTURE]:
            future_list.append(["{0:.4f}".format(round(event.alpha, 4)),
                                event.mechanism, effect, event.purview])
        else:
            validate.direction(event.direction)

    true_list = [(past_list[event], future_list[event])
                 for event in range(len(past_list))]
    return true_list


def true_constellation(subsystem, past_state, future_state):
    """Set of all sets of elements that have true causes and true effects.
       Note: Since the true constellation is always about the full system,
       the background conditions don't matter and the subsystem should be
       conditioned on the current state."""
    log.info("Calculating true causes ...")
    network = subsystem.network
    nodes = subsystem.node_indices
    state = subsystem.state
    past_context = Context(network, past_state, state, nodes, nodes)
    true_causes = directed_account(past_context, direction=DIRECTIONS[PAST])
    log.info("Calculating true effects ...")
    future_context = Context(network, state, future_state, nodes, nodes)
    true_effects = directed_account(future_context,
                                    direction=DIRECTIONS[FUTURE])
    true_mechanisms = set([c.mechanism for c in true_causes]).\
        intersection(c.mechanism for c in true_effects)
    if true_mechanisms:
        true_events = true_causes + true_effects
        result = tuple(filter(lambda t: t.mechanism in true_mechanisms,
                              true_events))
        log.info("Finished calculating true events.")
        log.debug("RESULT: \n" + str(result))
        return result
    else:
        log.info("Finished calculating, no true events.")
        return None


def true_events(network, past_state, current_state, future_state, indices=None,
                main_complex=None):
    """Set of all mechanisms that have true causes and true effects within the
    complex.

    Args:
        network (Network):

        past_state (tuple(int)): The state of the network at t-1
        current_state (tuple(int)): The state of the network at t
        future_state (tuple(int)): The state of the network at t+1

    Optional Args:
        indices (tuple(int)): The indices of the main complex
        main_complex (big_mip): The main complex

        Note: If main_complex is given, then indices is ignored.

    Returns:
        events (tuple(actions)): List of true events in the main complex
    """
    # TODO: validate triplet of states
    if not indices and not main_complex:
        main_complex = compute.main_complex(network, current_state)
    elif not main_complex:
        main_complex = compute.big_mip(network, current_state, indices)
    # Calculate true causes
    nodes = main_complex.subsystem.node_indices
    past_context = Context(network, past_state, current_state, nodes, nodes)
    true_causes = directed_account(past_context, direction=DIRECTIONS[PAST])
    # Calculate true_effects
    future_context = Context(network, current_state, future_state, nodes, nodes)
    true_effects = directed_account(future_context,
                                    direction=DIRECTIONS[FUTURE])
    true_mechanisms = set([c.mechanism for c in true_causes]).\
        intersection(c.mechanism for c in true_effects)
    # TODO: Make sort function that sorts events by mechanism so that
    # causes and effects match up.
    if true_mechanisms:
        true_causes = tuple(filter(lambda t: t.mechanism in true_mechanisms,
                                   true_causes))
        true_effects = tuple(filter(lambda t: t.mechanism in true_mechanisms,
                                    true_effects))
        true_events = tuple([true_causes[i], true_effects[i]] for i in
                            range(len(true_mechanisms)))
    else:
        true_events = ()
    return true_events


def extrinsic_events(network, past_state, current_state, future_state,
                     indices=None, main_complex=None):
    """Set of all mechanisms that have true causes and true effects within the
    complex.

    Args:
        network (Network):
        past_state (tuple(int)): The state of the network at t-1
        current_state (tuple(int)): The state of the network at t
        future_state (tuple(int)): The state of the network at t+1

    Optional Args:
        indices (tuple(int)): The indices of the main complex
        main_complex (big_mip): The main complex

        Note: If main_complex is given, then indices is ignored.

    Returns:
        events (tuple(actions)): List of true events in the main complex
    """
    # TODO: validate triplet of states
    if not indices and not main_complex:
        main_complex = compute.main_complex(network, current_state)
    elif not main_complex:
        main_complex = compute.big_mip(network, current_state, indices)
    # Identify the potential mechanisms for extrinsic events within the main
    # complex
    all_nodes = network.node_indices
    mechanisms = list(utils.powerset(main_complex.subsystem.node_indices))[1:]
    # Calculate true causes
    past_context = Context(network, past_state, current_state, all_nodes,
                           all_nodes)
    true_causes = directed_account(past_context, direction=DIRECTIONS[PAST],
                                   mechanisms=mechanisms)
    # Calculate true_effects
    future_context = Context(network, current_state, future_state, all_nodes,
                             all_nodes)
    true_effects = directed_account(future_context,
                                    direction=DIRECTIONS[FUTURE],
                                    mechanisms=mechanisms)
    true_mechanisms = set([c.mechanism for c in true_causes]).\
        intersection(c.mechanism for c in true_effects)
    # TODO: Make sort function that sorts events by mechanism so that
    # causes and effects match up.
    if true_mechanisms:
        true_causes = tuple(filter(lambda t: t.mechanism in true_mechanisms,
                                   true_causes))
        true_effects = tuple(filter(lambda t: t.mechanism in true_mechanisms,
                                    true_effects))
        true_events = tuple([true_causes[i], true_effects[i]] for i in
                            range(len(true_mechanisms)))
    else:
        true_events = ()
    return true_events
