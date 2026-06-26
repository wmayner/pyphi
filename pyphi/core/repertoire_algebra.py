"""Stateless repertoire computation over System.

Layer 2 of the kernel. Functions take a System as the first
argument; results are memoized via a per-instance decorator that purges
when the System is garbage-collected.

Threading
---------
The kernel cache is NOT thread-safe — see :mod:`pyphi.cache` for the
process-isolated parallelism assumption.
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
from pyphi import utils as _utils
from pyphi import validate as _validate
from pyphi.data_structures import FrozenMap
from pyphi.data_structures import PyPhiFloat
from pyphi.direction import Direction
from pyphi.distribution import max_entropy_distribution
from pyphi.distribution import repertoire_shape
from pyphi.measures.distribution import repertoire_distance as _repertoire_distance
from pyphi.measures.protocols import CompositeMeasure
from pyphi.measures.protocols import DistributionMeasure
from pyphi.measures.protocols import StateAwareMeasure
from pyphi.measures.protocols import StatefulDistributionMeasure
from pyphi.measures.protocols import satisfies_composite_measure

# One cache dict per memoized function name.
_caches: dict[str, dict[tuple, Any]] = {}

# Per-function ``[hits, misses]`` counters, exposed through the cache
# registry adapters set up in ``_memoize``.
_kernel_stats: dict[str, list[int]] = {}

# Live System references keyed by id, with finalizers that purge
# the corresponding cache entries on GC.
_observers: WeakValueDictionary[int, Any] = WeakValueDictionary()


def _evict(cs_id: int) -> None:
    """Purge cache entries whose first key element is ``cs_id``."""
    for fn_cache in _caches.values():
        for key in [k for k in fn_cache if k and k[0] == cs_id]:
            del fn_cache[key]


def _memoize(fn: Callable) -> Callable:
    """Memoize a function over System instances by ``id()``.

    Uses ``WeakValueDictionary`` + ``weakref.finalize`` so that cache
    entries are purged when the System is collected. Stops
    inserting new entries when ``cache_utils.memory_full()`` reports
    process memory above ``MAXIMUM_CACHE_MEMORY_PERCENTAGE`` — already
    computed values are still returned, just not cached.
    """
    from pyphi.cache.policy import _DictCacheAdapter
    from pyphi.cache.registry import register as _register_policy

    cache = _caches.setdefault(fn.__name__, {})
    stats = _kernel_stats.setdefault(fn.__name__, [0, 0])

    _register_policy(
        _DictCacheAdapter(
            name=f"kernel.{fn.__name__}",
            backing=cache,
            stats=lambda s=stats: (s[0], s[1]),
        )
    )

    @wraps(fn)
    def wrapper(cs: Any, *args: Any) -> Any:
        from pyphi.cache.cache_utils import memory_full

        cs_id = id(cs)
        key = (cs_id, args)
        if cs_id not in _observers:
            _observers[cs_id] = cs
            weakref.finalize(cs, _evict, cs_id)
        if key in cache:
            stats[0] += 1
            return cache[key]
        stats[1] += 1
        result = fn(cs, *args)  # raises propagate; key not added on raise
        if not memory_full():
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
    """Single-node cause repertoire — building block for full cause
    repertoires.

    Indexes the mechanism node's per-unit cause factor at
    ``mechanism_node.cause_marginal[..., mechanism_node.state]`` to extract
    the slice corresponding to the node's observed state, then
    marginalizes out the node's inputs that are not in the purview.
    The returned array is the unnormalized per-node contribution;
    normalization is applied in :func:`_cause_repertoire_inner` after
    the per-node contributions are multiplied.

    Alphabet-generic: ``mechanism_node.state`` is an integer index into
    the trailing per-node alphabet axis, so the indexing works
    uniformly for any per-node alphabet size.
    """
    mechanism_node = cs._index2node[mechanism_node_index]
    tpm = mechanism_node.cause_marginal[..., mechanism_node.state]
    # The result is size 1 on every purview node that is not an input to this
    # mechanism node. It is NOT self-contained canonical: it relies on the
    # ``joint = np.ones(repertoire_shape(...))`` allocation in
    # ``_cause_repertoire_inner`` to broadcast those size-1 axes up to the full
    # purview alphabet. Keeping them size 1 (rather than broadcasting here) is
    # deliberate — the product over mechanism nodes stays cheap.
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
        tpm = purview_node.cause_marginal.condition_tpm(condition)
    elif direction == Direction.EFFECT:
        tpm = purview_node.effect_marginal.condition_tpm(condition)
    else:
        _validate.direction(direction)
        raise AssertionError("unreachable")
    nonmechanism_inputs = purview_node.inputs - set(condition)
    tpm = tpm.marginalize_out(nonmechanism_inputs)
    alphabet_sizes = cs.substrate.factored_tpm.alphabet_sizes
    # Unlike the cause builder, the effect builder reshapes to canonical here,
    # so its output is self-describingly canonical (this purview node at full
    # alphabet, every other axis size 1) regardless of the caller's allocation.
    return tpm.reshape(
        repertoire_shape(
            cs.substrate.node_indices,
            (purview_node_index,),
            alphabet_sizes=alphabet_sizes,
        )
    ).tpm


@_memoize
def _cause_repertoire_inner(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...]
) -> Any:
    """Joint cause repertoire for non-empty mechanism and purview.

    The joint distribution is the (normalized) product of the per-node
    cause repertoires.
    """
    purview_set: frozenset[int] = frozenset(purview)
    alphabet_sizes = cs.substrate.factored_tpm.alphabet_sizes
    # Load-bearing: this canonical-shaped allocation establishes the full
    # purview shape, so per-mechanism-node contributions (which are size 1 on
    # purview nodes they do not constrain — see _single_node_cause_repertoire)
    # broadcast up correctly. Do not replace with a bare product of the
    # per-node contributions.
    joint = np.ones(
        repertoire_shape(
            cs.substrate.node_indices, purview_set, alphabet_sizes=alphabet_sizes
        )
    )
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
    alphabet_sizes = cs.substrate.factored_tpm.alphabet_sizes
    joint = np.ones(
        repertoire_shape(
            cs.substrate.node_indices, purview, alphabet_sizes=alphabet_sizes
        )
    )
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
        alphabet_sizes = cs.substrate.factored_tpm.alphabet_sizes
        return max_entropy_distribution(
            cs.substrate.node_indices, purview, alphabet_sizes
        )
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
    *,
    mechanism_measure: (
        DistributionMeasure
        | StateAwareMeasure
        | StatefulDistributionMeasure
        | CompositeMeasure
    ),
    **kwargs: Any,
) -> Any:
    """Compute the repertoire of a partitioned mechanism and purview.

    For composite measures (multi-argument signature consuming forward,
    partitioned, and selectivity repertoires at a specific state), the
    result is a scalar product of forward probabilities evaluated at the
    purview state. For other measure shapes, the result is the product
    of per-part repertoires as a distribution.
    """
    if satisfies_composite_measure(mechanism_measure):
        if "state" not in kwargs:
            raise ValueError(
                f"must provide purview state for repertoire distance "
                f"{mechanism_measure.name}"
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
    """Forward cause repertoire."""
    import itertools

    if mechanism_state is None:
        mechanism_state = _utils.state_of(mechanism, cs.state)
    alphabet_sizes = cs.substrate.factored_tpm.alphabet_sizes
    if purview:
        # Per-purview-node alphabet sizes determine the result shape.
        purview_k = [alphabet_sizes[i] for i in purview]
        result = np.empty(purview_k)
        if purview_state is None:
            purview_states = itertools.product(*[range(k) for k in purview_k])
        else:
            purview_states = iter([purview_state])
    else:
        result = np.array([1])
        purview_states = iter([()])
    for state in purview_states:
        result[state] = forward_cause_probability(
            cs, mechanism, purview, state, mechanism_state=mechanism_state
        )
    return result.reshape(
        repertoire_shape(
            cs.substrate.node_indices, purview, alphabet_sizes=alphabet_sizes
        )
    )


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
    alphabet_sizes = cs.substrate.factored_tpm.alphabet_sizes
    mech_k = tuple(alphabet_sizes[i] for i in mechanism)
    all_mech_states: list[tuple[int, ...]] = list(_utils.all_states(mech_k))
    repertoires = np.stack(
        [
            forward_effect_repertoire(cs, mechanism, purview, mechanism_state=state)
            for state in all_mech_states
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
    alphabet_sizes = cs.substrate.factored_tpm.alphabet_sizes
    result = np.empty(
        repertoire_shape(
            cs.substrate.node_indices, purview, alphabet_sizes=alphabet_sizes
        )
    )
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
    *,
    specification_measure: (
        DistributionMeasure
        | StateAwareMeasure
        | StatefulDistributionMeasure
        | CompositeMeasure
    ),
    states: Any | None = None,
) -> Any:
    """Compute intrinsic information and the maximally specified state.

    Composite measures take the multi-argument path (forward, partitioned,
    selectivity repertoires evaluated at a state); other measure shapes
    fall through to :func:`repertoire_distance`.
    """
    from pyphi.models.state_specification import StateSpecification

    if states is None:
        alphabet_sizes = cs.substrate.factored_tpm.alphabet_sizes
        purview_k = tuple(alphabet_sizes[i] for i in purview)
        states = list(_utils.all_states(purview_k))

    if satisfies_composite_measure(specification_measure):
        from typing import cast

        composite = cast(CompositeMeasure, specification_measure)
        selectivity_repertoire = repertoire(cs, direction, mechanism, purview)
        rep = forward_repertoire(cs, direction, mechanism, purview, None)
        unconstrained_rep = unconstrained_forward_repertoire(
            cs, direction, mechanism, purview
        )
        dist = composite(rep, unconstrained_rep, selectivity_repertoire)
        assert not isinstance(dist, (int, float)), (
            "Distance measures should return array when state is None"
        )
        dist = dist.squeeze()

        def evaluate_state(state: Any) -> float:
            return float(dist[state])
    else:
        rep = repertoire(cs, direction, mechanism, purview)
        unconstrained_rep = unconstrained_repertoire(cs, direction, purview)

        def evaluate_state(state: Any) -> float:
            return _repertoire_distance(
                rep,
                unconstrained_rep,
                state=state,
                repertoire_distance=specification_measure,
            )

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


# ---- purview enumeration (kernel) ----


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
    from pyphi.substrate import irreducible_purviews

    _potential_purviews = set(cs.substrate.potential_purviews(direction, mechanism))
    if purviews is None:
        purviews_set = _potential_purviews
    else:
        purviews_set = _potential_purviews & set(purviews)
    purviews_list = [
        purview for purview in purviews_set if set(purview).issubset(cs.node_indices)
    ]
    return irreducible_purviews(cs.cm, direction, mechanism, purviews_list)


def null_distinction(cs: Any) -> Any:
    """Return the null distinction — the point identified with the
    unconstrained cause and effect repertoires of the candidate system.
    """
    from pyphi.models import Distinction
    from pyphi.models import MaximallyIrreducibleCause
    from pyphi.models import MaximallyIrreducibleEffect
    from pyphi.models import _null_ria

    cause_rep = cause_repertoire(cs, (), ())
    effect_rep = effect_repertoire(cs, (), ())
    cause = MaximallyIrreducibleCause(_null_ria(Direction.CAUSE, (), (), cause_rep))
    effect = MaximallyIrreducibleEffect(_null_ria(Direction.EFFECT, (), (), effect_rep))
    return Distinction(mechanism=(), cause=cause, effect=effect)


# IIT 3.0 paper terminology calls a distinction a "concept"; the alias
# preserves that vocabulary for IIT 3.0-native callers.
null_concept = null_distinction


def indices2nodes(cs: Any, indices: tuple[int, ...]) -> Any:
    """Return |Nodes| for these indices."""
    if set(indices) - set(cs.node_indices):
        raise ValueError("`indices` must be a subset of the System's indices.")
    return tuple(cs._index2node[n] for n in indices)
