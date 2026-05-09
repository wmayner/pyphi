"""Formalism queries — operations whose math depends on the active formalism.

These are free functions taking a ``System`` as the first argument.
The kernel (``pyphi.core.repertoire_algebra``) holds pure repertoire math;
this module holds the operations whose definition is *formalism-policy*
(IIT 3.0 vs 4.0 vs 4.0-2026 each define MIP, MICE, SIA differently).

The dispatch path is::

    queries.X(cs, ...)
        → FORMALISM_REGISTRY[config.formalism.formalism].evaluate_X(cs, ...)
        → concrete formalism's algorithm

The kernel never imports this module — see
``test_core_layering.test_repertoire_algebra_does_not_import_formalism``.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from pyphi import conf as _conf
from pyphi import resolve_ties
from pyphi import utils as _utils
from pyphi import validate as _validate
from pyphi.conf import config
from pyphi.conf import fallback
from pyphi.core import repertoire_algebra as _ra
from pyphi.direction import Direction
from pyphi.models import Concept
from pyphi.models import Distinctions
from pyphi.models import MaximallyIrreducibleCause
from pyphi.models import MaximallyIrreducibleEffect
from pyphi.models import _null_ria
from pyphi.models.ria import ShortCircuitConditions
from pyphi.parallel import MapReduce
from pyphi.partition import mip_partitions

from .base import FORMALISM_REGISTRY

if TYPE_CHECKING:
    from pyphi.system import System


# ---- mechanism partition evaluation ----


def evaluate_partition(
    cs: System,
    direction: Direction,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    partition: Any,
    repertoire: Any = None,
    partitioned_repertoire: Any = None,
    repertoire_distance: str | None = None,
    partitioned_repertoire_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Evaluate a mechanism partition's |small_phi|.

    Dispatches to the active formalism's ``evaluate_mechanism_partition``.
    """
    formalism = FORMALISM_REGISTRY[config.formalism.formalism]  # pyright: ignore[reportAttributeAccessIssue]
    return formalism.evaluate_mechanism_partition(  # pyright: ignore[reportFunctionMemberAccess]
        cs,
        direction,
        mechanism,
        purview,
        partition,
        repertoire=repertoire,
        partitioned_repertoire=partitioned_repertoire,
        repertoire_distance=repertoire_distance,
        partitioned_repertoire_kwargs=partitioned_repertoire_kwargs,
        **kwargs,
    )


def _find_mip_single_state(
    cs: System,
    specified_state: Any,
    direction: Direction,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    repertoire: Any,
    partitions: Any,
    parallel_kwargs: dict[str, Any],
    **kwargs: Any,
) -> Any:
    """Find the MIP for a single specified-state pin.

    Used by formalism MIP-search routines to evaluate all candidate
    mechanism partitions for a (state, direction, mechanism, purview)
    combination.
    """
    partitions = fallback(partitions, mip_partitions(mechanism, purview, cs.node_labels))

    def _eval(partition: Any) -> Any:
        return evaluate_partition(
            cs,
            direction,
            mechanism,
            purview,
            partition,
            repertoire=repertoire,
            state=specified_state,
            **kwargs,
        )

    candidate_mips = MapReduce(
        _eval,
        partitions,
        shortcircuit_func=_utils.is_falsy,
        desc="Evaluating mechanism partitions",
        **parallel_kwargs,
    ).run()
    assert candidate_mips is not None, "MapReduce.run() should not return None"

    ties = tuple(
        resolve_ties.partitions(
            candidate_mips,  # type: ignore[arg-type]
            default=_null_ria(
                direction,
                mechanism,
                purview,
                phi=0,
                specified_state=specified_state,
            ),
        )
    )
    for tie in ties:
        tie.set_partition_ties(ties)
    return ties[0]


# ---- mechanism MIP search ----


def find_mip(
    cs: System,
    direction: Direction,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    partitions: Any | None = None,
    state: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Return the minimum information partition for a mechanism over a purview."""

    def null_mip(**kw: Any) -> Any:  # noqa: ARG001
        return _null_ria(direction, mechanism, purview, specified_state=state)

    if not purview:
        return null_mip(reasons=(ShortCircuitConditions.EMPTY_PURVIEW,))

    rep = _ra.repertoire(cs, direction, mechanism, purview)

    if direction == Direction.CAUSE and np.all(rep == 0):
        return null_mip(reasons=(ShortCircuitConditions.UNREACHABLE_STATE,))

    if partitions is not None:
        partitions = list(partitions)

    parallel_kwargs = _conf.parallel_kwargs(
        dict(config.infrastructure.parallel_mechanism_partition_evaluation),  # pyright: ignore[reportAttributeAccessIssue]
        **kwargs,
    )
    formalism = FORMALISM_REGISTRY[config.formalism.formalism]  # pyright: ignore[reportAttributeAccessIssue]
    return formalism._find_mechanism_mip(  # pyright: ignore[reportFunctionMemberAccess]
        cs,
        direction,
        mechanism,
        purview,
        repertoire=rep,
        partitions=partitions,
        state=state,
        parallel_kwargs=parallel_kwargs,
        **kwargs,
    )


def cause_mip(
    cs: System,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    **kwargs: Any,
) -> Any:
    return find_mip(cs, Direction.CAUSE, mechanism, purview, **kwargs)


def effect_mip(
    cs: System,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    **kwargs: Any,
) -> Any:
    return find_mip(cs, Direction.EFFECT, mechanism, purview, **kwargs)


def phi_cause_mip(
    cs: System,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    **kwargs: Any,
) -> float:
    mip = cause_mip(cs, mechanism, purview, **kwargs)
    return mip.phi if mip else 0


def phi_effect_mip(
    cs: System,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    **kwargs: Any,
) -> float:
    mip = effect_mip(cs, mechanism, purview, **kwargs)
    return mip.phi if mip else 0


def phi(
    cs: System,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    **kwargs: Any,
) -> float:
    return min(
        phi_cause_mip(cs, mechanism, purview, **kwargs),
        phi_effect_mip(cs, mechanism, purview, **kwargs),
    )


# ---- MICE / MIE search ----


def find_mice(
    cs: System,
    direction: Direction,
    mechanism: tuple[int, ...],
    purviews: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Return the |MIC| or |MIE| for a mechanism."""
    purviews_list = _ra.potential_purviews(cs, direction, mechanism, purviews)

    if direction == Direction.CAUSE:
        mice_class = MaximallyIrreducibleCause
    elif direction == Direction.EFFECT:
        mice_class = MaximallyIrreducibleEffect
    else:
        _validate.direction(direction)
        mice_class = MaximallyIrreducibleCause  # unreachable

    no_purviews = mice_class(
        _null_ria(
            direction,
            mechanism,
            (),
            reasons=(ShortCircuitConditions.NO_PURVIEWS,),
        )
    )

    if not purviews_list:
        return no_purviews

    def _find_mip(purview: tuple[int, ...]) -> Any:
        return find_mip(cs, direction, mechanism, purview)

    parallel_kwargs = _conf.parallel_kwargs(
        dict(config.infrastructure.parallel_purview_evaluation),  # pyright: ignore[reportAttributeAccessIssue]
        **kwargs,
    )
    map_reduce = MapReduce(
        _find_mip,
        purviews_list,
        total=len(purviews_list),
        desc="Evaluating purviews",
        **parallel_kwargs,
    )

    all_mice = map(mice_class, map_reduce.run())  # type: ignore[arg-type]
    ties = tuple(resolve_ties.purviews(all_mice, default=no_purviews))  # type: ignore[arg-type]
    for tie in ties:
        tie.set_purview_ties(ties)
    return ties[0]


def mic(cs: System, mechanism: tuple[int, ...], **kwargs: Any) -> Any:
    return find_mice(cs, Direction.CAUSE, mechanism, **kwargs)


def mie(cs: System, mechanism: tuple[int, ...], **kwargs: Any) -> Any:
    return find_mice(cs, Direction.EFFECT, mechanism, **kwargs)


def phi_max(cs: System, mechanism: tuple[int, ...]) -> float:
    return min(mic(cs, mechanism).phi, mie(cs, mechanism).phi)


# ---- distinctions ----
#
# IIT 4.0 paper terminology: the irreducible mechanism with cause-effect
# power is a *distinction*. The IIT 3.0 *concept*, which has the same
# mathematical role under that formalism, lives in :mod:`pyphi.formalism.iit3`
# along with the rest of IIT 3.0's algorithms.


def distinction(cs: System, mechanism: tuple[int, ...]) -> Any:
    """Return the distinction specified by a mechanism."""
    if not mechanism:
        return _ra.null_concept(cs)
    maximally_irreducible_cause = find_mice(cs, Direction.CAUSE, mechanism)
    maximally_irreducible_effect = find_mice(cs, Direction.EFFECT, mechanism)
    return Concept(
        mechanism=mechanism,
        cause=maximally_irreducible_cause,
        effect=maximally_irreducible_effect,
    )


def all_distinctions(cs: System, **kwargs: Any) -> Any:  # noqa: ARG001
    """Iterate non-empty mechanisms and return the resulting Distinctions."""
    mechanisms: Any = _utils.powerset(cs.node_indices, nonempty=True)
    total = 2 ** len(cs.node_indices) - 1

    if fallback(config.infrastructure.progress_bars):
        with contextlib.suppress(TypeError):
            total = len(mechanisms)
        mechanisms = tqdm(mechanisms, total=total)

    distinctions = filter(None, (distinction(cs, mechanism) for mechanism in mechanisms))
    return Distinctions(distinctions)


# ---- system irreducibility ----


def sia(cs: System, **kwargs: Any) -> Any:
    """Run system irreducibility analysis via the active formalism."""
    formalism = FORMALISM_REGISTRY[config.formalism.formalism]  # pyright: ignore[reportAttributeAccessIssue]
    return formalism.evaluate_system(cs, **kwargs)  # pyright: ignore[reportFunctionMemberAccess]
