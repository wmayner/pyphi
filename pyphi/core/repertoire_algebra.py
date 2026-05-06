"""Stateless repertoire computation over CandidateSystem.

Layer 2 of the kernel. Functions take a CandidateSystem as the first
argument; results are memoized via a per-instance decorator that purges
when the CandidateSystem is garbage-collected.

Numerical bodies are ports of the corresponding Subsystem methods in
pyphi/subsystem.py. Parity tests guard equivalence.
"""

from __future__ import annotations

import functools
import weakref
from collections.abc import Callable
from functools import wraps
from typing import Any
from weakref import WeakValueDictionary

import numpy as np

from pyphi import distribution as _dist
from pyphi import metrics as _metrics
from pyphi import utils as _utils
from pyphi import validate as _validate
from pyphi.conf import config
from pyphi.conf import fallback
from pyphi.data_structures import FrozenMap
from pyphi.data_structures import PyPhiFloat
from pyphi.direction import Direction
from pyphi.distribution import max_entropy_distribution
from pyphi.distribution import repertoire_shape
from pyphi.metrics.distribution import repertoire_distance as _repertoire_distance

# One cache dict per memoized function name.
_caches: dict[str, dict[tuple, Any]] = {}

# Live CandidateSystem references keyed by id, with finalizers that purge
# the corresponding cache entries on GC.
_observers: WeakValueDictionary[int, Any] = WeakValueDictionary()


def _evict(cs_id: int) -> None:
    """Purge cache entries whose first key element is ``cs_id``."""
    for fn_cache in _caches.values():
        for key in [k for k in fn_cache if k and k[0] == cs_id]:
            del fn_cache[key]


def _memoize(fn: Callable) -> Callable:
    """Memoize a function over CandidateSystem instances by ``id()``.

    Uses ``WeakValueDictionary`` + ``weakref.finalize`` so that cache
    entries are purged when the CandidateSystem is collected.
    """
    cache = _caches.setdefault(fn.__name__, {})

    @wraps(fn)
    def wrapper(cs: Any, *args: Any) -> Any:
        cs_id = id(cs)
        key = (cs_id, args)
        if cs_id not in _observers:
            _observers[cs_id] = cs
            weakref.finalize(cs, _evict, cs_id)
        if key in cache:
            return cache[key]
        result = fn(cs, *args)  # raises propagate; key not added on raise
        cache[key] = result
        return result

    return wrapper


def cache_info() -> dict[str, dict[str, int]]:
    """Return per-function cache size."""
    return {name: {"size": len(c)} for name, c in _caches.items()}


def clear_caches(cs: Any | None = None) -> None:
    """Clear cache entries. If ``cs`` given, clear only that instance's entries."""
    if cs is None:
        for c in _caches.values():
            c.clear()
        return
    _evict(id(cs))


# ---- repertoire computation ----


@_memoize
def _single_node_cause_repertoire(
    cs: Any, mechanism_node_index: int, purview_set: frozenset[int]
) -> Any:
    """Single-node cause repertoire — used as a building block for full
    cause repertoires (legacy ``Subsystem._single_node_cause_repertoire``).
    """
    mechanism_node = cs._index2node[mechanism_node_index]
    tpm = mechanism_node.cause_tpm[..., mechanism_node.state]
    return tpm.marginalize_out(mechanism_node.inputs - purview_set).tpm


@_memoize
def _single_node_effect_repertoire(
    cs: Any,
    condition: FrozenMap,
    purview_node_index: int,
    direction: Direction,
) -> Any:
    purview_node = cs._index2node[purview_node_index]
    if direction == Direction.CAUSE:
        tpm = purview_node.cause_tpm.condition_tpm(condition)
    elif direction == Direction.EFFECT:
        tpm = purview_node.effect_tpm.condition_tpm(condition)
    else:
        _validate.direction(direction)
        raise AssertionError("unreachable")
    nonmechanism_inputs = purview_node.inputs - set(condition)
    tpm = tpm.marginalize_out(nonmechanism_inputs)
    return tpm.reshape(
        repertoire_shape(cs.network.node_indices, (purview_node_index,))
    ).tpm


@_memoize
def _cause_repertoire_inner(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...]
) -> Any:
    """Joint cause repertoire for non-empty mechanism and purview.

    The joint distribution is the (normalized) product of the per-node
    cause repertoires. Equivalent to legacy
    ``Subsystem._cause_repertoire``.
    """
    purview_set: frozenset[int] = frozenset(purview)
    joint = np.ones(repertoire_shape(cs.network.node_indices, purview_set))
    joint *= functools.reduce(
        np.multiply,
        [_single_node_cause_repertoire(cs, m, purview_set) for m in mechanism],
    )
    return _dist.normalize(joint)


@_memoize
def _effect_repertoire_inner(
    cs: Any,
    condition: FrozenMap,
    purview: tuple[int, ...],
    direction: Direction,
) -> Any:
    joint = np.ones(repertoire_shape(cs.network.node_indices, purview))
    return joint * functools.reduce(
        np.multiply,
        [_single_node_effect_repertoire(cs, condition, p, direction) for p in purview],
    )


def cause_repertoire(
    cs: Any,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    **kwargs: Any,  # noqa: ARG001
) -> Any:
    """Cause repertoire — IIT 4.0 Eq. 5 / Eq. 7."""
    if not purview:
        return np.array([1.0])
    if not mechanism:
        return max_entropy_distribution(cs.node_indices, purview)
    return _cause_repertoire_inner(cs, mechanism, purview)


def effect_repertoire(
    cs: Any,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    mechanism_state: Any | None = None,
    direction: Direction = Direction.EFFECT,
) -> Any:
    """Effect repertoire — IIT 4.0 Eq. 5 / Eq. 7."""
    if not purview:
        return np.array([1.0])
    if mechanism_state is None:
        mechanism_state = _utils.state_of(mechanism, cs.state)
    condition = FrozenMap(zip(mechanism, mechanism_state, strict=False))
    return _effect_repertoire_inner(cs, condition, purview, direction)


def repertoire(
    cs: Any,
    direction: Direction,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    **kwargs: Any,
) -> Any:
    if direction == Direction.CAUSE:
        return cause_repertoire(cs, mechanism, purview, **kwargs)
    if direction == Direction.EFFECT:
        return effect_repertoire(cs, mechanism, purview, **kwargs)
    _validate.direction(direction)
    raise AssertionError("unreachable")


def unconstrained_repertoire(
    cs: Any, direction: Direction, purview: tuple[int, ...]
) -> Any:
    return repertoire(cs, direction, (), purview)


def unconstrained_cause_repertoire(cs: Any, purview: tuple[int, ...]) -> Any:
    return cause_repertoire(cs, (), purview)


def unconstrained_effect_repertoire(cs: Any, purview: tuple[int, ...]) -> Any:
    return effect_repertoire(cs, (), purview)


def expand_repertoire(
    cs: Any,
    direction: Direction,
    repertoire_array: Any,
    *,
    new_purview: tuple[int, ...] | None = None,
) -> Any:
    if repertoire_array is None:
        return None
    purview = _dist.purview(repertoire_array)
    if purview is None:
        return None
    expanded_purview = cs.node_indices if new_purview is None else new_purview
    if not set(purview).issubset(expanded_purview):
        raise ValueError("Expanded purview must contain original purview.")
    non_purview_indices = tuple(set(expanded_purview) - set(purview))
    uc = unconstrained_repertoire(cs, direction, non_purview_indices)
    expanded = repertoire_array * uc
    return _dist.normalize(expanded)


def expand_cause_repertoire(
    cs: Any, repertoire_array: Any, *, new_purview: tuple[int, ...] | None = None
) -> Any:
    return expand_repertoire(
        cs, Direction.CAUSE, repertoire_array, new_purview=new_purview
    )


def expand_effect_repertoire(
    cs: Any, repertoire_array: Any, *, new_purview: tuple[int, ...] | None = None
) -> Any:
    return expand_repertoire(
        cs, Direction.EFFECT, repertoire_array, new_purview=new_purview
    )


def partitioned_repertoire(
    cs: Any,
    direction: Direction,
    partition: Any,
    repertoire_distance: str | None = None,
    **kwargs: Any,
) -> Any:
    """Compute the repertoire of a partitioned mechanism and purview.

    Routes to the state-aware path (forward probabilities + product of
    scalars) when the configured repertoire distance is GID/II; otherwise
    returns the product of the per-part repertoires.
    """
    repertoire_distance = fallback(repertoire_distance, config.REPERTOIRE_DISTANCE)
    if repertoire_distance in [
        "GENERALIZED_INTRINSIC_DIFFERENCE",
        "INTRINSIC_INFORMATION",
    ]:
        if "state" not in kwargs:
            raise ValueError(
                f"must provide purview state for repertoire distance "
                f"{repertoire_distance}"
            )
        purview_state = kwargs.pop("state")
        prs = [
            forward_probability(
                cs,
                direction,
                part.mechanism,
                part.purview,
                purview_state=_utils.substate(
                    partition.purview, purview_state, part.purview
                ),
                **kwargs,
            )
            for part in partition
        ]
        return float(np.prod(prs))
    repertoires = [
        repertoire(cs, direction, part.mechanism, part.purview, **kwargs)
        for part in partition
    ]
    return functools.reduce(np.multiply, repertoires)


# ---- forward repertoires + probabilities ----


@_memoize
def forward_cause_repertoire(
    cs: Any,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    purview_state: tuple[int, ...] | None = None,
    mechanism_state: tuple[int, ...] | None = None,
) -> Any:
    """Forward cause repertoire — legacy ``_repertoire.forward_cause_repertoire``."""
    if mechanism_state is None:
        mechanism_state = _utils.state_of(mechanism, cs.state)
    if purview:
        result = np.empty([2] * len(purview))
        if purview_state is None:
            purview_states = _utils.all_states(len(purview))
        else:
            purview_states = [purview_state]
    else:
        result = np.array([1])
        purview_states = [()]
    for state in purview_states:
        result[state] = forward_cause_probability(
            cs, mechanism, purview, state, mechanism_state=mechanism_state
        )
    return result.reshape(repertoire_shape(cs.network.node_indices, purview))


def forward_effect_repertoire(
    cs: Any,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    **kwargs: Any,
) -> Any:
    """Forward effect repertoire is identical to the effect repertoire."""
    return effect_repertoire(cs, mechanism, purview, **kwargs)


def forward_repertoire(
    cs: Any,
    direction: Direction,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    purview_state: tuple[int, ...] | None = None,
    **kwargs: Any,
) -> Any:
    if direction == Direction.CAUSE:
        return forward_cause_repertoire(cs, mechanism, purview, purview_state)
    if direction == Direction.EFFECT:
        return forward_effect_repertoire(cs, mechanism, purview, **kwargs)
    _validate.direction(direction)
    raise AssertionError("unreachable")


@_memoize
def unconstrained_forward_effect_repertoire(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...]
) -> Any:
    """Unconstrained forward effect repertoire — average over all mechanism states."""
    repertoires = np.stack(
        [
            forward_effect_repertoire(cs, mechanism, purview, mechanism_state=state)
            for state in _utils.all_states(len(mechanism))
        ]
    )
    return repertoires.mean(axis=0)


@_memoize
def unconstrained_forward_cause_repertoire(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...]
) -> Any:
    """Unconstrained forward cause repertoire — see Eq. 32 of the IIT 4.0 paper.

    Since ``m`` is fixed and we average over ``Z``, the per-state
    probabilities are all equal to the mean — fill with that value.
    """
    mean_forward_cause_probability = forward_cause_repertoire(
        cs, mechanism, purview, None
    ).mean()
    result = np.empty(repertoire_shape(cs.network.node_indices, purview))
    result.fill(mean_forward_cause_probability)
    return result


def unconstrained_forward_repertoire(
    cs: Any,
    direction: Direction,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
) -> Any:
    if direction == Direction.CAUSE:
        return unconstrained_forward_cause_repertoire(cs, mechanism, purview)
    if direction == Direction.EFFECT:
        return unconstrained_forward_effect_repertoire(cs, mechanism, purview)
    _validate.direction(direction)
    raise AssertionError("unreachable")


def forward_effect_probability(
    cs: Any,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    purview_state: Any,
    **kwargs: Any,
) -> float:
    return forward_effect_repertoire(cs, mechanism, purview, **kwargs).squeeze()[
        purview_state
    ]


def forward_cause_probability(
    cs: Any,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    purview_state: Any,
    mechanism_state: Any | None = None,
) -> float:
    if mechanism_state is None:
        mechanism_state = _utils.state_of(mechanism, cs.state)
    er = effect_repertoire(
        cs,
        mechanism=purview,
        purview=mechanism,
        mechanism_state=purview_state,
        direction=Direction.CAUSE,
    )
    return er.squeeze()[mechanism_state]


def forward_probability(
    cs: Any,
    direction: Direction,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    purview_state: Any,
    **kwargs: Any,
) -> float:
    if direction == Direction.CAUSE:
        return forward_cause_probability(cs, mechanism, purview, purview_state, **kwargs)
    if direction == Direction.EFFECT:
        return forward_effect_probability(
            cs, mechanism, purview, purview_state, **kwargs
        )
    _validate.direction(direction)
    raise AssertionError("unreachable")


# ---- info / phi ----


def cause_info(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any
) -> float:
    """Cause information — distance between cause repertoire and unconstrained."""
    return _repertoire_distance(
        cause_repertoire(cs, mechanism, purview),
        unconstrained_cause_repertoire(cs, purview),
        direction=Direction.CAUSE,
        **kwargs,
    )


def effect_info(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any
) -> float:
    """Effect information — distance between effect repertoire and unconstrained."""
    return _repertoire_distance(
        effect_repertoire(cs, mechanism, purview),
        unconstrained_effect_repertoire(cs, purview),
        direction=Direction.EFFECT,
        **kwargs,
    )


def cause_effect_info(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any
) -> float:
    """Cause-effect information — minimum of cause and effect information."""
    return min(
        cause_info(cs, mechanism, purview, **kwargs),
        effect_info(cs, mechanism, purview, **kwargs),
    )


def intrinsic_information(
    cs: Any,
    direction: Direction,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    repertoire_distance: str | None = None,
    states: Any | None = None,
) -> Any:
    """Compute intrinsic information and the maximally specified state."""
    from pyphi.models.mechanism import StateSpecification

    repertoire_distance = fallback(
        repertoire_distance,
        config.REPERTOIRE_DISTANCE_SPECIFICATION,  # pyright: ignore[reportAttributeAccessIssue]
    )
    if states is None:
        states = _utils.all_states(len(purview))

    if repertoire_distance in [
        "GENERALIZED_INTRINSIC_DIFFERENCE",
        "INTRINSIC_INFORMATION",
        "INTRINSIC_SPECIFICATION",
    ]:
        func = _metrics.distribution.measures[repertoire_distance]
        selectivity_repertoire = repertoire(cs, direction, mechanism, purview)
        rep = forward_repertoire(cs, direction, mechanism, purview, None)
        unconstrained_rep = unconstrained_forward_repertoire(
            cs, direction, mechanism, purview
        )
        dist = func(rep, unconstrained_rep, selectivity_repertoire)
        assert not isinstance(dist, (int, float)), (
            "Distance metrics should return array when state is None"
        )
        dist = dist.squeeze()

        def evaluate_state(state: Any) -> float:
            return float(dist[state])
    else:
        rep = repertoire(cs, direction, mechanism, purview)
        unconstrained_rep = unconstrained_repertoire(cs, direction, purview)

        def evaluate_state(state: Any) -> float:
            return _repertoire_distance(rep, unconstrained_rep, state=state)

    state_to_information = {state: evaluate_state(state) for state in states}
    max_information = max(state_to_information.values())
    ties = [
        StateSpecification(
            direction=direction,
            purview=purview,
            state=state,
            intrinsic_information=PyPhiFloat(information),
            repertoire=rep,
            unconstrained_repertoire=unconstrained_rep,
        )
        for state, information in state_to_information.items()
        if information == max_information
    ]
    for tie in ties:
        tie.set_ties(ties)
    return ties[0]


# ---- mechanism / system analysis ----


def evaluate_partition(
    cs: Any,
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

    Thin dispatcher to the active formalism's
    ``evaluate_mechanism_partition`` — same pattern as legacy
    ``Subsystem.evaluate_partition``.
    """
    from pyphi.formalism import FORMALISM_REGISTRY

    formalism = FORMALISM_REGISTRY[config.FORMALISM]  # pyright: ignore[reportAttributeAccessIssue]
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
    cs: Any,
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
    mechanism partitions for a given (state, direction, mechanism, purview)
    combination.
    """
    from pyphi import resolve_ties
    from pyphi.models import _null_ria
    from pyphi.parallel import MapReduce
    from pyphi.partition import mip_partitions

    partitions = fallback(partitions, mip_partitions(mechanism, purview, cs.node_labels))

    def _evaluate_partition(partition: Any) -> Any:
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
        _evaluate_partition,
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


def find_mip(
    cs: Any,
    direction: Direction,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    partitions: Any | None = None,
    state: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Return the minimum information partition for a mechanism over a purview."""
    from pyphi import conf as _conf
    from pyphi.formalism import FORMALISM_REGISTRY
    from pyphi.models import _null_ria
    from pyphi.models.mechanism import ShortCircuitConditions

    def null_mip(**kw: Any) -> Any:  # noqa: ARG001
        return _null_ria(direction, mechanism, purview, specified_state=state)

    if not purview:
        return null_mip(reasons=(ShortCircuitConditions.EMPTY_PURVIEW,))

    rep = repertoire(cs, direction, mechanism, purview)

    if direction == Direction.CAUSE and np.all(rep == 0):
        return null_mip(reasons=(ShortCircuitConditions.UNREACHABLE_STATE,))

    if partitions is not None:
        partitions = list(partitions)

    parallel_kwargs = _conf.parallel_kwargs(
        dict(config.PARALLEL_MECHANISM_PARTITION_EVALUATION),  # pyright: ignore[reportAttributeAccessIssue]
        **kwargs,
    )
    formalism = FORMALISM_REGISTRY[config.FORMALISM]  # pyright: ignore[reportAttributeAccessIssue]
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
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any
) -> Any:
    return find_mip(cs, Direction.CAUSE, mechanism, purview, **kwargs)


def effect_mip(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any
) -> Any:
    return find_mip(cs, Direction.EFFECT, mechanism, purview, **kwargs)


def phi_cause_mip(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any
) -> float:
    mip = cause_mip(cs, mechanism, purview, **kwargs)
    return mip.phi if mip else 0


def phi_effect_mip(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any
) -> float:
    mip = effect_mip(cs, mechanism, purview, **kwargs)
    return mip.phi if mip else 0


def phi(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any
) -> float:
    return min(
        phi_cause_mip(cs, mechanism, purview, **kwargs),
        phi_effect_mip(cs, mechanism, purview, **kwargs),
    )


def potential_purviews(
    cs: Any,
    direction: Direction,
    mechanism: tuple[int, ...],
    purviews: Any | None = None,
) -> list[tuple[int, ...]]:
    """Return all purviews that could belong to the |MIC|/|MIE|.

    Filters out trivially-reducible purviews against the (possibly cut)
    connectivity matrix of this candidate system.
    """
    from pyphi.network import irreducible_purviews

    _potential_purviews = set(cs.network.potential_purviews(direction, mechanism))
    if purviews is None:
        purviews_set = _potential_purviews
    else:
        purviews_set = _potential_purviews & set(purviews)
    purviews_list = [
        purview for purview in purviews_set if set(purview).issubset(cs.node_indices)
    ]
    return irreducible_purviews(cs.cm, direction, mechanism, purviews_list)


def find_mice(
    cs: Any,
    direction: Direction,
    mechanism: tuple[int, ...],
    purviews: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Return the |MIC| or |MIE| for a mechanism."""
    from pyphi import conf as _conf
    from pyphi import resolve_ties
    from pyphi.models import MaximallyIrreducibleCause
    from pyphi.models import MaximallyIrreducibleEffect
    from pyphi.models import _null_ria
    from pyphi.models.mechanism import ShortCircuitConditions
    from pyphi.parallel import MapReduce

    purviews_list = potential_purviews(cs, direction, mechanism, purviews)

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
        dict(config.PARALLEL_PURVIEW_EVALUATION),  # pyright: ignore[reportAttributeAccessIssue]
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


def mic(cs: Any, mechanism: tuple[int, ...], **kwargs: Any) -> Any:
    """Maximally-irreducible cause — alias for find_mice with CAUSE direction."""
    return find_mice(cs, Direction.CAUSE, mechanism, **kwargs)


def mie(cs: Any, mechanism: tuple[int, ...], **kwargs: Any) -> Any:
    """Maximally-irreducible effect — alias for find_mice with EFFECT direction."""
    return find_mice(cs, Direction.EFFECT, mechanism, **kwargs)


def phi_max(cs: Any, mechanism: tuple[int, ...]) -> float:
    """Return |small_phi_max| — minimum of the MIC and MIE phi values."""
    return min(mic(cs, mechanism).phi, mie(cs, mechanism).phi)


def null_concept(cs: Any) -> Any:
    """Return the null concept — point identified with the unconstrained
    cause and effect repertoires of the candidate system.
    """
    from pyphi.models import Concept
    from pyphi.models import MaximallyIrreducibleCause
    from pyphi.models import MaximallyIrreducibleEffect
    from pyphi.models import _null_ria

    cause_rep = cause_repertoire(cs, (), ())
    effect_rep = effect_repertoire(cs, (), ())
    cause = MaximallyIrreducibleCause(_null_ria(Direction.CAUSE, (), (), cause_rep))
    effect = MaximallyIrreducibleEffect(_null_ria(Direction.EFFECT, (), (), effect_rep))
    return Concept(mechanism=(), cause=cause, effect=effect)


def concept(
    cs: Any,
    mechanism: tuple[int, ...],
    purviews: Any | None = None,
    cause_purviews: Any | None = None,
    effect_purviews: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Return the concept specified by a mechanism within the candidate system."""
    from pyphi.models import Concept

    if not mechanism:
        return null_concept(cs)

    cause_purviews = cause_purviews if cause_purviews is not None else purviews
    cause = mic(cs, mechanism, purviews=cause_purviews, **kwargs)

    effect_purviews = effect_purviews if effect_purviews is not None else purviews
    effect = mie(cs, mechanism, purviews=effect_purviews, **kwargs)

    return Concept(mechanism=mechanism, cause=cause, effect=effect)


def distinction(cs: Any, mechanism: tuple[int, ...]) -> Any:
    """Return the distinction (Concept) specified by a mechanism."""
    from pyphi.models import Concept

    maximally_irreducible_cause = find_mice(cs, Direction.CAUSE, mechanism)
    maximally_irreducible_effect = find_mice(cs, Direction.EFFECT, mechanism)
    return Concept(
        mechanism=mechanism,
        cause=maximally_irreducible_cause,
        effect=maximally_irreducible_effect,
    )


def all_distinctions(cs: Any, **kwargs: Any) -> Any:  # noqa: ARG001
    """Iterate non-empty mechanisms and return the resulting CauseEffectStructure."""
    import contextlib

    from tqdm.auto import tqdm

    from pyphi.models import CauseEffectStructure

    mechanisms: Any = _utils.powerset(cs.node_indices, nonempty=True)
    total = 2 ** len(cs.node_indices) - 1

    if fallback(config.PROGRESS_BARS):
        with contextlib.suppress(TypeError):
            total = len(mechanisms)
        mechanisms = tqdm(mechanisms, total=total)

    distinctions = filter(None, (distinction(cs, mechanism) for mechanism in mechanisms))
    return CauseEffectStructure(distinctions)


def sia(cs: Any, **kwargs: Any) -> Any:
    """Run system irreducibility analysis via the active formalism."""
    from pyphi.formalism import FORMALISM_REGISTRY

    formalism = FORMALISM_REGISTRY[config.FORMALISM]  # pyright: ignore[reportAttributeAccessIssue]
    return formalism.evaluate_system(cs, **kwargs)  # pyright: ignore[reportFunctionMemberAccess]


def indices2nodes(cs: Any, indices: tuple[int, ...]) -> Any:
    """Return |Nodes| for these indices."""
    if set(indices) - set(cs.node_indices):
        raise ValueError("`indices` must be a subset of the Subsystem's indices.")
    return tuple(cs._index2node[n] for n in indices)
