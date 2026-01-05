# repertoire.py
"""Compute cause-effect repertoires."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from . import utils
from .direction import Direction
from .distribution import repertoire_shape
from .types import Mechanism
from .types import Purview
from .types import Repertoire
from .types import State

if TYPE_CHECKING:
    from .subsystem import Subsystem

# TODO(repertoire) refactor to be more independent of subsystem when TPM
# overhaul is done; e.g. no longer need 'tpm_size' with named dimensions


# TODO(4.0) test the following invariants:
# - in a causally perfect system, unconstrained m,z and z,m should be the same (eqs 33, 34)
# - informativeness (ii, not partitioned) of the full system) should be the same
#   between cause and effect


def forward_effect_probability(
    subsystem: Subsystem,
    mechanism: Mechanism,
    purview: Purview,
    purview_state: State,
    **kwargs: Any,
) -> float:
    return forward_effect_repertoire(subsystem, mechanism, purview, **kwargs).squeeze()[
        purview_state
    ]


def forward_effect_repertoire(
    subsystem: Subsystem, mechanism: Mechanism, purview: Purview, **kwargs: Any
) -> Repertoire:
    return subsystem.effect_repertoire(mechanism, purview, **kwargs)


def forward_cause_probability(
    subsystem: Subsystem,
    mechanism: Mechanism,
    purview: Purview,
    purview_state: State,
    mechanism_state: State | None = None,
) -> float:
    if mechanism_state is None:
        mechanism_state = utils.state_of(mechanism, subsystem.state)
    # We compute forward probabilities, but mechanism and purview roles are
    # switched
    er = subsystem.effect_repertoire(
        mechanism=purview,
        purview=mechanism,
        mechanism_state=purview_state,
        direction=Direction.CAUSE,
    )
    return er.squeeze()[mechanism_state]


def forward_cause_repertoire(
    subsystem: Subsystem,
    mechanism: Mechanism,
    purview: Purview,
    purview_state: State | None = None,
    mechanism_state: State | None = None,
) -> Repertoire:
    if mechanism_state is None:
        mechanism_state = utils.state_of(mechanism, subsystem.state)
    if purview:
        repertoire = np.empty([2] * len(purview))
        if purview_state is None:
            purview_states = utils.all_states(len(purview))
        else:
            purview_states = [purview_state]
    else:
        repertoire = np.array([1])
        purview_states = [()]
    for purview_state in purview_states:
        repertoire[purview_state] = forward_cause_probability(
            subsystem,
            mechanism,
            purview,
            purview_state,
            mechanism_state=mechanism_state,
        )
    return repertoire.reshape(repertoire_shape(subsystem.network.node_indices, purview))


def unconstrained_forward_effect_repertoire(
    subsystem: Subsystem, mechanism: Mechanism, purview: Purview
) -> Repertoire:
    # Get the effect repertoire for each mechanism state.
    repertoires = np.stack(
        [
            subsystem.forward_effect_repertoire(
                mechanism, purview, mechanism_state=state
            )
            # TODO(nonbinary) extend to nonbinary nodes
            for state in utils.all_states(len(mechanism))
        ]
    )
    # Marginalize over all mechanism states.
    return repertoires.mean(axis=0)


def unconstrained_forward_cause_repertoire(
    subsystem: Subsystem, mechanism: Mechanism, purview: Purview
) -> Repertoire:
    # See Eq. 32 in 4.0 paper.
    # Here, the roles of `m` and `z` in the equation are switched, so the
    # probability within the average is conditioned on `z`. So here, we are
    # averaging over all states `\Omega_Z`. Since `m` is fixed, this means the
    # probabilities for each `z` do not actually depend on `z`; they are all
    # equal to the average value over all `Z`. So we compute this average value
    # and fill the repertoire with it.
    mean_forward_cause_probability = subsystem.forward_cause_repertoire(
        mechanism, purview, None
    ).mean()
    repertoire = np.empty(repertoire_shape(subsystem.network.node_indices, purview))
    repertoire.fill(mean_forward_cause_probability)
    return repertoire


def forward_repertoire(
    subsystem,
    direction: Direction,
    mechanism: Tuple[int],
    purview: Tuple[int],
    mechanism_state=None,
) -> ArrayLike:
    """Return the forward repertoire of a mechanism over a purview."""
    if direction is Direction.CAUSE:
        func = forward_cause_repertoire
    elif direction is Direction.EFFECT:
        func = forward_effect_repertoire
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be CAUSE or EFFECT.")
    return func(subsystem, mechanism, purview, mechanism_state=mechanism_state)
