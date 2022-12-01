# -*- coding: utf-8 -*-
# repertoire.py

import functools

import numpy as np

from . import convert, utils, validate
from .data_structures import FrozenMap
from .direction import Direction
from .distribution import max_entropy_distribution, repertoire_shape


# TODO(repertoire) refactor to be more independent of subsystem when TPM
# overhaul is done; e.g. no longer need 'tpm_size' with named dimensions


def forward_repertoire(subsystem, direction, mechanism, purview, **kwargs):
    """Main interface to forward repertoire calculation."""
    if direction == Direction.CAUSE:
        return forward_cause_repertoire(subsystem, mechanism, purview, **kwargs)
    elif direction == Direction.EFFECT:
        return forward_effect_repertoire(subsystem, mechanism, purview, **kwargs)
    return validate.direction(direction)


def forward_effect_repertoire(subsystem, mechanism, purview, **kwargs):
    return subsystem.effect_repertoire(mechanism, purview, **kwargs)


def forward_cause_repertoire(subsystem, mechanism, purview):
    # Preallocate the repertoire with the proper shape, so that
    # probabilities are broadcasted appropriately.
    joint = np.ones(repertoire_shape(purview, subsystem.tpm_size))
    # The effect repertoire is the product of the effect repertoires of the
    # individual nodes.
    joint = joint * functools.reduce(
        np.multiply,
        [
            _single_node_forward_cause_repertoire(subsystem, mechanism_node, purview)
            for mechanism_node in mechanism
        ],
    )
    return joint


def _single_node_forward_cause_repertoire(subsystem, mechanism_node, purview):
    # TODO(repertoire) refactor to take any mechanism state, then maybe default to subsystem state
    # TODO(nonbinary) extend to nonbinary nodes
    repertoire = np.empty([2] * len(purview))
    for purview_state in utils.all_states(len(purview)):
        repertoire[purview_state] = _single_node_forward_cause_probability(
            subsystem, mechanism_node, purview, purview_state
        )
    repertoire = repertoire / 2 ** len(purview)
    return repertoire.reshape(repertoire_shape(purview, subsystem.tpm_size))


def _single_node_forward_cause_probability(
    subsystem, mechanism_node, purview, purview_state
):
    mechanism = (mechanism_node,)
    mechanism_state = (subsystem.state[mechanism_node],)
    return forward_effect_repertoire(
        subsystem=subsystem,
        # Switch mechanism and purview
        mechanism=purview,
        purview=mechanism,
        # TODO(repertoire) refactor
        mechanism_state=purview_state,
        nonvirtualized_units=purview,
    ).squeeze()[mechanism_state]


def unconstrained_forward_repertoire(
    subsystem, direction, mechanism, purview, **kwargs
):
    if direction == Direction.CAUSE:
        return unconstrained_forward_cause_repertoire(subsystem, purview, **kwargs)
    elif direction == Direction.EFFECT:
        return unconstrained_forward_effect_repertoire(
            subsystem, mechanism, purview, **kwargs
        )
    return validate.direction(direction)


def unconstrained_forward_cause_repertoire(subsystem, purview):
    return max_entropy_distribution(purview, subsystem.tpm_size)


def unconstrained_forward_effect_repertoire(subsystem, mechanism, purview, **kwargs):
    if not mechanism:
        return _fully_unconstrained_forward_effect_repertoire(subsystem, purview)
    return _partially_unconstrained_forward_effect_repertoire(
        subsystem, mechanism, purview, **kwargs
    )


def _fully_unconstrained_forward_effect_repertoire(subsystem, purview):
    # Ignore units outside the purview.
    nonpurview_nodes = set(subsystem.node_indices) - set(purview)
    tpm = subsystem.tpm.marginalize_out(nonpurview_nodes).tpm
    tpm = tpm[..., list(purview)]
    # Convert to state-by-state to get explicit joint probabilities.
    joint = convert.sbn2sbs(tpm)
    joint = convert.sbs_to_multidimensional(joint)
    # Marginalize over all states at t to get a marginal repertoire over purview states at t+1.
    repertoire = joint.mean(axis=tuple(purview))
    return repertoire.reshape(repertoire_shape(purview, subsystem.tpm_size))


def _partially_unconstrained_forward_effect_repertoire(
    subsystem, mechanism, purview, **kwargs
):
    # Get the effect repertoire for each mechanism state.
    repertoires = np.stack(
        [
            forward_effect_repertoire(
                subsystem, mechanism, purview, mechanism_state=state, **kwargs
            )
            for state in utils.all_states(len(mechanism))
        ]
    )
    # Marginalize over all mechanism states.
    return repertoires.mean(axis=0)
