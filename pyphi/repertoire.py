# -*- coding: utf-8 -*-
# repertoire.py

import numpy as np
from numpy.typing import ArrayLike
from typing import Callable

from . import utils, validate
from .direction import Direction
from .distribution import repertoire_shape

# TODO(repertoire) refactor to be more independent of subsystem when TPM
# overhaul is done; e.g. no longer need 'tpm_size' with named dimensions


# TODO(4.0) use this pattern with subsystem methods
def _directional_dispatch(cause_func: Callable, effect_func: Callable) -> Callable:
    # Assumes signatures of cause_func and effect_func are compatible
    def wrapper(direction, *args, **kwargs):
        if direction == Direction.CAUSE:
            return cause_func(*args, **kwargs)
        elif direction == Direction.EFFECT:
            return effect_func(*args, **kwargs)
        return validate.direction(direction)

    return wrapper


def forward_effect_repertoire(
    subsystem, mechanism: tuple[int], purview: tuple[int], **kwargs
) -> ArrayLike:
    return subsystem.effect_repertoire(mechanism, purview, **kwargs)


def forward_cause_repertoire(
    subsystem, mechanism: tuple[int], purview: tuple[int]
) -> ArrayLike:
    mechanism_state = utils.state_of(mechanism, subsystem.state)
    if purview:
        repertoire = np.empty([2] * len(purview))
        purview_states = utils.all_states(len(purview))
    else:
        repertoire = np.empty([1])
        purview_states = [()]
    for purview_state in purview_states:
        # We compute forward probabilities, but mechanism and purview roles are
        # switched
        er = subsystem.effect_repertoire(
            mechanism=purview,
            purview=mechanism,
            mechanism_state=purview_state,
        )
        repertoire[purview_state] = er.squeeze()[mechanism_state]
    return repertoire.reshape(repertoire_shape(subsystem.network.node_indices, purview))


forward_repertoire = _directional_dispatch(
    forward_cause_repertoire, forward_effect_repertoire
)


def unconstrained_forward_effect_repertoire(
    subsystem, mechanism: tuple[int], purview: tuple[int]
) -> ArrayLike:
    # Get the effect repertoire for each mechanism state.
    repertoires = np.stack(
        [
            forward_effect_repertoire(
                subsystem, mechanism, purview, mechanism_state=state
            )
            # TODO(nonbinary) extend to nonbinary nodes
            for state in utils.all_states(len(mechanism))
        ]
    )
    # Marginalize over all mechanism states.
    return repertoires.mean(axis=0)


def unconstrained_forward_cause_repertoire(
    subsystem, mechanism: tuple[int], purview: tuple[int]
) -> ArrayLike:
    # See Eq. 32 in 4.0 paper.
    # Here, the roles of `m` and `z` in the equation are switched, so the
    # probability within the average is conditioned on `z`. So here, we are
    # averaging over all states `\Omega_Z`. Since `m` is fixed, this means the
    # probabilities for each `z` do not actually depend on `z`; they are all
    # equal to the average value over all `Z`. So we compute this average value
    # and fill the repertoire with it.
    mean_forward_cause_probability = forward_cause_repertoire(
        subsystem, mechanism, purview
    ).mean()
    repertoire = np.empty(repertoire_shape(subsystem.network.node_indices, purview))
    repertoire.fill(mean_forward_cause_probability)
    return repertoire


unconstrained_forward_repertoire = _directional_dispatch(
    unconstrained_forward_cause_repertoire, unconstrained_forward_effect_repertoire
)


# TODO(4.0) test the following invariants:
# - in a causally perfect system, unconstrained m,z and z,m should be the same (eqs 33, 34)
# - informativeness (ii, not partitioned) of the full system) should be the same
#   between cause and effect
