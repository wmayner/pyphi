"""Formalism queries — operations whose math depends on the active formalism.

These are free functions taking a ``System`` as the first argument.
The kernel (``pyphi.core.repertoire_algebra``) holds pure repertoire math;
this module holds the operations whose definition is *formalism-policy*
(IIT 3.0 vs 4.0 vs 4.0-2026 each define MIP, MICE, SIA differently).

The dispatch path is::

    queries.X(cs, ...)
        → FORMALISM_REGISTRY[config.formalism.iit.version].evaluate_X(cs, ...)
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
from pyphi import numerics
from pyphi import resolve_ties
from pyphi import utils as _utils
from pyphi import validate as _validate
from pyphi.conf import config
from pyphi.conf import fallback
from pyphi.core import repertoire_algebra as _ra
from pyphi.direction import Direction
from pyphi.models import Concept
from pyphi.models import MaximallyIrreducibleCause
from pyphi.models import MaximallyIrreducibleEffect
from pyphi.models import UnresolvedDistinctions
from pyphi.models import _null_ria
from pyphi.models.explanation import NullResultReason
from pyphi.parallel import map_reduce
from pyphi.partition import mechanism_partitions

from .base import FORMALISM_REGISTRY

if TYPE_CHECKING:
    from pyphi.system import System


def _never_shortcircuit(_result: Any) -> bool:
    """Predicate that never short-circuits a partition sweep.

    Deliberately distinct from :func:`pyphi.parallel.false`: the parallel
    backends collect results in completion order when no short-circuit
    predicate is set, and MIP tie resolution requires the deterministic
    enumeration order that an installed predicate preserves.
    """
    return False


# ---- mechanism partition evaluation ----


def evaluate_partition(
    cs: System,
    direction: Direction,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    partition: Any,
    repertoire: Any = None,
    partitioned_repertoire: Any = None,
    partitioned_repertoire_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Evaluate a mechanism partition's φ.

    Dispatches to the active formalism's ``evaluate_mechanism_partition``.
    The caller is expected to thread ``mechanism_measure`` (and any other
    measure kwargs) through ``**kwargs``; the active formalism's method
    forwards them to its internal helper.
    """
    formalism = FORMALISM_REGISTRY[config.formalism.iit.version]  # pyright: ignore[reportAttributeAccessIssue]
    return formalism.evaluate_mechanism_partition(
        cs,
        direction,
        mechanism,
        purview,
        partition,
        repertoire=repertoire,
        partitioned_repertoire=partitioned_repertoire,
        partitioned_repertoire_kwargs=partitioned_repertoire_kwargs,
        **kwargs,
    )


def _partition_total(
    known: int | None, mechanism: tuple[int, ...], purview: tuple[int, ...]
) -> int:
    """The number of partitions a complete sweep would evaluate.

    ``known`` is the length of a caller-supplied partition list; ``None`` means
    the partitions came from the active scheme, whose count depends only on the
    mechanism and purview sizes and is memoized, so a lazily consumed sweep can
    still tell a complete pass from one the short-circuit truncated.
    """
    if known is not None:
        return known
    from pyphi.cost import partition_sweep_count

    return partition_sweep_count(len(mechanism), len(purview))


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
    # The sweep below stops at the first reducible partition, so the scheme's
    # partitions are consumed lazily and their construction stops with it. A
    # caller-supplied set is a concrete collection and is taken as given.
    if partitions is None:
        partitions = mechanism_partitions(mechanism, purview, cs.node_labels)
        n_partitions = None
    else:
        partitions = list(partitions)
        n_partitions = len(partitions)

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

    candidate_mips = map_reduce(
        _eval,
        partitions,
        shortcircuit_func=(
            _utils.is_falsy
            if config.formalism.iit.shortcircuit_sia
            else _never_shortcircuit
        ),
        desc="Evaluating mechanism partitions",
        **parallel_kwargs,
    )
    assert candidate_mips is not None, "map_reduce() should not return None"
    candidates = list(candidate_mips)

    ties = tuple(
        resolve_ties.partitions(
            candidates,  # type: ignore[arg-type]
            default=_null_ria(
                direction,
                mechanism,
                purview,
                phi=0,
                specified_state=specified_state,
                node_labels=cs.node_labels,
                mechanism_state=_utils.state_of(mechanism, cs.state),
            ),
        )
    )
    for tie in ties:
        tie.set_partition_ties(ties)
    winner = ties[0]
    # The margin is only meaningful when every partition was evaluated: a
    # short-circuited sweep yields a truncated prefix whose gap says nothing
    # about the full partition set.
    others = [c for c in candidates if c is not winner]
    if (
        others
        and winner.normalized_phi is not None
        and all(c.normalized_phi is not None for c in others)
        and len(candidates) == _partition_total(n_partitions, mechanism, purview)
    ):
        # numerics: exact — reported margin, not a selection.
        gap = min(float(c.normalized_phi) for c in others) - float(winner.normalized_phi)
        winner.partition_margin = max(0.0, gap)
    return winner


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
        return _null_ria(
            direction,
            mechanism,
            purview,
            specified_state=state,
            node_labels=cs.node_labels,
            mechanism_state=_utils.state_of(mechanism, cs.state),
        )

    if not purview:
        return null_mip(reasons=(NullResultReason.EMPTY_PURVIEW,))

    rep = _ra.repertoire(cs, direction, mechanism, purview)

    if direction == Direction.CAUSE and np.all(rep == 0):
        return null_mip(reasons=(NullResultReason.UNREACHABLE_STATE,))

    if partitions is not None:
        partitions = list(partitions)

    parallel_kwargs = _conf.parallel_kwargs(
        dict(config.infrastructure.parallel_mechanism_partition_evaluation),  # pyright: ignore[reportAttributeAccessIssue]
        **kwargs,
    )
    formalism = FORMALISM_REGISTRY[config.formalism.iit.version]  # pyright: ignore[reportAttributeAccessIssue]
    return formalism._find_mechanism_mip(  # pyright: ignore[reportAttributeAccessIssue]
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
    """Return the φ of a mechanism over a purview: the minimum of the cause
    and effect MIP φ values. The effect MIP is not evaluated when the cause
    MIP's φ is already 0, since φ values are non-negative."""
    cause_phi = phi_cause_mip(cs, mechanism, purview, **kwargs)
    if not numerics.is_positive(float(cause_phi)):
        return cause_phi
    return min(cause_phi, phi_effect_mip(cs, mechanism, purview, **kwargs))


# ---- MICE / MIE search ----


def find_mice(
    cs: System,
    direction: Direction,
    mechanism: tuple[int, ...],
    purviews: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Return the maximally irreducible cause or effect for a mechanism.

    The result is a :class:`~pyphi.models.MaximallyIrreducibleCause` or
    :class:`~pyphi.models.MaximallyIrreducibleEffect`.

    Over the given ``direction``, searches the candidate purviews for the one
    whose φ (irreducibility) is maximal, resolving ties via
    :mod:`pyphi.resolve_ties`. When ``config.infrastructure.validate_phi_bounds``
    is set and a purview is found, the winning φ is checked against the IIT 4.0
    distinction and partition upper bounds.
    """
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
            reasons=(NullResultReason.NO_PURVIEWS,),
            node_labels=cs.node_labels,
            mechanism_state=_utils.state_of(mechanism, cs.state),
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
    mip_results = map_reduce(
        _find_mip,
        purviews_list,
        total=len(purviews_list),
        desc="Evaluating purviews",
        # Cost ~ repertoire state space over the purview (2**|purview|);
        # heterogeneous across purviews, cheap and exact.
        size_func=lambda purview: 2 ** len(purview),
        **parallel_kwargs,
    )

    # Parallel evaluation returns results in completion / cost-bin order;
    # restore the canonical purview enumeration order so tie resolution
    # selects the same winner as sequential evaluation.
    order = {purview: index for index, purview in enumerate(purviews_list)}
    mip_results = sorted(mip_results, key=lambda ria: order[ria.purview])

    all_mice = [mice_class(result) for result in mip_results]  # type: ignore[arg-type]
    ties = tuple(resolve_ties.purviews(all_mice, default=no_purviews))  # type: ignore[arg-type]
    for tie in ties:
        tie.set_purview_ties(ties)
    winner = ties[0]
    # The purview sweep is always exhaustive, so the margin is exact
    # whenever a competing purview exists.
    other_mice = [m for m in all_mice if m is not winner]
    if other_mice:
        # numerics: exact — reported margin, not a selection.
        best_rival_phi = max(float(m.phi) for m in other_mice)
        # numerics: exact — reported margin, not a selection.
        winner.purview_margin = max(0.0, float(winner.phi) - best_rival_phi)
    if config.infrastructure.validate_phi_bounds and winner.purview:
        from pyphi.formalism.iit4 import bounds

        bounds.check_phi_bound(
            winner.phi,
            lambda: bounds.distinction_phi_upper_bound(mechanism, winner.purview),
            system=cs,
            label=f"MICE phi ({direction.name}, mechanism={mechanism}, "
            f"purview={winner.purview})",
        )
        bounds.check_phi_bound(
            winner.phi,
            lambda: bounds.partition_phi_upper_bound(winner.partition),
            system=cs,
            label=f"MICE MIP phi ({direction.name}, mechanism={mechanism}, "
            f"purview={winner.purview})",
        )
    return winner


def mic(cs: System, mechanism: tuple[int, ...], **kwargs: Any) -> Any:
    return find_mice(cs, Direction.CAUSE, mechanism, **kwargs)


def mie(cs: System, mechanism: tuple[int, ...], **kwargs: Any) -> Any:
    return find_mice(cs, Direction.EFFECT, mechanism, **kwargs)


def phi_max(cs: System, mechanism: tuple[int, ...]) -> float:
    # numerics: exact — φ is the minimum of the MIC and MIE φ.
    return min(mic(cs, mechanism).phi, mie(cs, mechanism).phi)


# ---- distinctions ----
#
# IIT 4.0 paper terminology: the irreducible mechanism with cause-effect
# power is a *distinction*. The IIT 3.0 *concept*, which has the same
# mathematical role under that formalism, lives in :mod:`pyphi.formalism.iit3`
# along with the rest of IIT 3.0's algorithms.


def distinction(
    cs: System,
    mechanism: tuple[int, ...],
    purviews: Any | None = None,
    cause_purviews: Any | None = None,
    effect_purviews: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Return the distinction specified by a mechanism.

    A distinction pairs the mechanism's maximally irreducible cause (its
    MIC) with its maximally irreducible effect (its MIE). The empty
    mechanism specifies the null distinction.

    Parameters
    ----------
    cs : System
        The system the mechanism belongs to.
    mechanism : tuple[int]
        The mechanism for which to determine the distinction.
    purviews : tuple[tuple[int]], optional
        A list of purviews to consider in both directions.
    cause_purviews : tuple[tuple[int]], optional
        A list of cause purviews to consider, overriding ``purviews``.
    effect_purviews : tuple[tuple[int]], optional
        A list of effect purviews to consider, overriding ``purviews``.

    Returns
    -------
    Concept
        The distinction specified by the mechanism.

    Notes
    -----
    When ``config.formalism.iit.shortcircuit_distinctions`` is set,
    evaluation stops as soon as the distinction is known to be reducible:
    if the effect direction has no candidate purviews, or once the cause
    MICE comes out with φ = 0, the remaining search is skipped and the
    unevaluated direction is a null MICE carrying the
    :attr:`~pyphi.models.explanation.NullResultReason.OTHER_DIRECTION_REDUCIBLE`
    reason. The distinction's φ (the minimum across directions) is
    unaffected.
    """
    if not mechanism:
        return _ra.null_concept(cs)
    cause_purviews = cause_purviews if cause_purviews is not None else purviews
    effect_purviews = effect_purviews if effect_purviews is not None else purviews
    shortcircuit = config.formalism.iit.shortcircuit_distinctions  # pyright: ignore[reportAttributeAccessIssue]

    if shortcircuit and not _ra.potential_purviews(
        cs, Direction.EFFECT, mechanism, effect_purviews
    ):
        # The effect side is trivially reducible, so the distinction's φ is 0
        # no matter the cause; the cause search is skipped.
        effect = find_mice(
            cs, Direction.EFFECT, mechanism, purviews=effect_purviews, **kwargs
        )
        cause = MaximallyIrreducibleCause(
            _null_ria(
                Direction.CAUSE,
                mechanism,
                (),
                reasons=(NullResultReason.OTHER_DIRECTION_REDUCIBLE,),
                node_labels=cs.node_labels,
                mechanism_state=_utils.state_of(mechanism, cs.state),
            )
        )
        return Concept(mechanism=mechanism, cause=cause, effect=effect)

    cause = find_mice(cs, Direction.CAUSE, mechanism, purviews=cause_purviews, **kwargs)
    if shortcircuit and not numerics.is_positive(float(cause.phi)):
        effect = MaximallyIrreducibleEffect(
            _null_ria(
                Direction.EFFECT,
                mechanism,
                (),
                reasons=(NullResultReason.OTHER_DIRECTION_REDUCIBLE,),
                node_labels=cs.node_labels,
                mechanism_state=_utils.state_of(mechanism, cs.state),
            )
        )
        return Concept(mechanism=mechanism, cause=cause, effect=effect)

    effect = find_mice(
        cs, Direction.EFFECT, mechanism, purviews=effect_purviews, **kwargs
    )
    return Concept(mechanism=mechanism, cause=cause, effect=effect)


def all_distinctions(cs: System, **kwargs: Any) -> Any:  # noqa: ARG001
    """Iterate non-empty mechanisms and return the resulting Distinctions."""
    mechanisms: Any = _utils.powerset(cs.node_indices, nonempty=True)
    total = 2 ** len(cs.node_indices) - 1

    if fallback(config.infrastructure.progress_bars):
        with contextlib.suppress(TypeError):
            total = len(mechanisms)
        mechanisms = tqdm(mechanisms, total=total)

    distinctions = filter(None, (distinction(cs, mechanism) for mechanism in mechanisms))
    # The active formalism's find_mice may return tied specified states
    # (IIT 4.0). Conservative default: return UnresolvedDistinctions; the
    # caller resolves against a SIA system_state if it needs to flow into
    # relations() or CauseEffectStructure.
    return UnresolvedDistinctions(distinctions)


# ---- system irreducibility ----


def sia(cs: System, **kwargs: Any) -> Any:
    """Run system irreducibility analysis via the active formalism."""
    import time

    from pyphi.provenance import stamp_wall_time

    formalism = FORMALISM_REGISTRY[config.formalism.iit.version]  # pyright: ignore[reportAttributeAccessIssue]
    start = time.perf_counter()
    result = formalism.evaluate_system(cs, **kwargs)
    return stamp_wall_time(result, time.perf_counter() - start)
