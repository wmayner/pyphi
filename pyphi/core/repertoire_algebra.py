"""Stateless repertoire computation over System.

Layer 2 of the kernel. Functions take a System as the first
argument; results are memoized via a per-instance decorator that purges
when the System is garbage-collected.

Threading
---------
The kernel cache is the thread-safe :class:`pyphi.cache.ContentCache`
(eviction and refcount bookkeeping locked; the hot read path lock-free) —
see :mod:`pyphi.cache`.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np

from pyphi import distribution as _dist
from pyphi import numerics
from pyphi import utils as _utils
from pyphi import validate as _validate
from pyphi.cache.content import ContentCache
from pyphi.conf import config
from pyphi.core.tpm import _node_ops
from pyphi.data_structures import FrozenMap
from pyphi.direction import Direction
from pyphi.distribution import max_entropy_distribution
from pyphi.distribution import repertoire_shape
from pyphi.measures.distribution import repertoire_distance as _repertoire_distance
from pyphi.measures.protocols import CompositeMeasure
from pyphi.measures.protocols import DistributionMeasure
from pyphi.measures.protocols import StateAwareMeasure
from pyphi.measures.protocols import StatefulDistributionMeasure
from pyphi.measures.protocols import satisfies_composite_measure

# One ContentCache per memoized function name.
_kernel_caches: dict[str, ContentCache] = {}


def _freeze(result: Any) -> Any:
    """Make an ndarray result read-only; other values pass through."""
    if isinstance(result, np.ndarray):
        return _utils.np_immutable(result)
    return result


def _memoize(fn: Callable) -> Callable:
    """Memoize a function over System instances by content fingerprint.

    Distinct-but-equivalent Systems (re-constructed, or label-distinct) share
    entries via :class:`~pyphi.cache.content.ContentCache`. A fingerprint's
    entries are evicted when its last live carrier is garbage-collected. Once
    resident memory reaches the configured cache ceiling, occupancy is held
    steady by evicting least recently used entries; setting
    ``cache_repertoires`` to false stops caching entirely, in which case
    computed values are returned but not stored. Returned arrays are
    read-only; callers that need a mutable copy must copy explicitly.

    Cache keys carry the resolved cause-side background convention, so
    cause-side entries never cross conventions (effect-side entries are
    duplicated per convention, which only costs anything when a process
    actually flips the option).
    """
    cache = ContentCache(f"kernel.{fn.__name__}")
    _kernel_caches[fn.__name__] = cache

    @wraps(fn)
    def wrapper(cs: Any, *args: Any) -> Any:
        fp = cs._fingerprint
        cache.observe(cs, fp)
        key_args = (cs._resolved_background_conditioning(), *args)
        return cache.get_or_compute(
            fp,
            key_args,
            lambda: _freeze(fn(cs, *args)),
            store=config.infrastructure.cache_repertoires,
        )

    return wrapper


def cache_info() -> dict[str, dict[str, int]]:
    """Return per-function cache size."""
    return {name: {"size": c.size} for name, c in _kernel_caches.items()}


def clear_caches(cs: Any | None = None) -> None:
    """Clear cache entries. If ``cs`` given, clear only that instance's entries."""
    if cs is None:
        for c in _kernel_caches.values():
            c.clear()
        return
    fp = cs._fingerprint
    for c in _kernel_caches.values():
        c.evict(fp)


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
    return _node_ops.marginalize_out(tpm, mechanism_node.inputs - purview_set)


@_memoize
def _single_node_effect_repertoire(
    cs: Any,
    condition: FrozenMap,
    purview_node_index: int,
    direction: Direction,
) -> Any:
    """Single-node effect repertoire — building block for full effect
    repertoires.

    Conditions the purview node's per-unit marginal on the mechanism state
    (``condition``), marginalizes out the node's inputs that are not part of
    the mechanism, then reshapes to the canonical purview shape. Unlike
    :func:`_single_node_cause_repertoire`, the result is self-describingly
    canonical (this purview node at full alphabet, every other axis size 1),
    so it does not rely on the caller's allocation to broadcast.

    ``direction`` selects whether the effect (``EFFECT``) or cause
    (``CAUSE``) per-unit marginal supplies the factor.
    """
    purview_node = cs._index2node[purview_node_index]
    if direction == Direction.CAUSE:
        tpm = _node_ops.condition(purview_node.cause_marginal, condition)
    elif direction == Direction.EFFECT:
        tpm = _node_ops.condition(purview_node.effect_marginal, condition)
    else:
        _validate.direction(direction)
        raise AssertionError("unreachable")
    nonmechanism_inputs = purview_node.inputs - set(condition)
    tpm = _node_ops.marginalize_out(tpm, nonmechanism_inputs)
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
    )


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
    **kwargs: Any,
) -> Any:
    """Cause repertoire πc(z | m) of a mechanism over a purview.

    The distribution over purview (cause) states specified by the mechanism,
    obtained by Bayesian inversion of the per-node cause factors under a
    uniform prior over cause states (Albantakis et al. 2023, Eq. 33).

    Returns
    -------
    numpy.ndarray
        The normalized cause repertoire. An empty purview yields
        ``array([1.0])``; an empty mechanism yields the maximum-entropy
        (unconstrained) distribution over the purview.
    """
    if kwargs:
        raise TypeError(
            f"cause_repertoire got unexpected keyword arguments "
            f"{sorted(kwargs)}; it computes at the system's current state "
            f"and accepts no state overrides"
        )
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
    """Effect repertoire πe(z | m) of a mechanism over a purview.

    The distribution over purview (effect) states specified by the mechanism,
    formed as the product of the per-node effect factors conditioned on the
    mechanism state (Albantakis et al. 2023, Eq. 29). An empty purview yields
    ``array([1.0])``. When ``mechanism_state`` is ``None`` it is read from
    ``cs.state``; ``direction`` selects which per-node marginal (effect or
    cause) supplies the factors.
    """
    if not purview:
        return np.array([1.0])
    if mechanism_state is None:
        mechanism_state = _utils.state_of(mechanism, cs.state)
    if len(mechanism_state) != len(mechanism):
        raise ValueError(
            f"mechanism_state has {len(mechanism_state)} entries but the "
            f"mechanism has {len(mechanism)} nodes; provide one state entry "
            "per mechanism node."
        )
    condition = FrozenMap(zip(mechanism, mechanism_state, strict=True))
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
        # NaN-fill so that when ``purview_state`` restricts the computation to
        # a single state, the entries that were never computed are loud NaNs
        # rather than uninitialized memory.
        result = np.full(purview_k, np.nan)
        if purview_state is None:
            purview_states = itertools.product(*[range(k) for k in purview_k])
        else:
            purview_states = iter([purview_state])
        for state in purview_states:
            result[state] = forward_cause_probability(
                cs, mechanism, purview, state, mechanism_state=mechanism_state
            )
        # The buffer's axes follow the purview-argument order; restore
        # ascending node-index order to match the repertoire shape.
        result = result.transpose(tuple(np.argsort(purview)))
    else:
        # An empty purview constrains nothing; the repertoire is the
        # multiplicative identity, as for ``cause_repertoire``.
        result = np.array([1.0])
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
        return forward_cause_repertoire(cs, mechanism, purview, purview_state, **kwargs)
    if direction == Direction.EFFECT:
        return forward_effect_repertoire(cs, mechanism, purview, **kwargs)
    _validate.direction(direction)
    raise AssertionError("unreachable")


_MAX_UNCONSTRAINED_FORWARD_STATES = 2**16
"""Largest mechanism-state count the unconstrained forward effect repertoire
will average over. Each state costs one forward effect repertoire, so the time
grows with the state count regardless of memory; above this bound the
computation raises instead of silently grinding for days."""


@_memoize
def unconstrained_forward_effect_repertoire(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...]
) -> Any:
    """Unconstrained forward effect repertoire — average over all mechanism states.

    The average is accumulated one repertoire at a time, so memory stays at a
    single repertoire regardless of the mechanism-state count.

    Notes
    -----
    Sequential accumulation can differ from a stacked ``mean`` in the final
    floating-point bits once numpy's pairwise summation engages (above 128
    states); tolerance-based comparisons downstream absorb this.
    """
    alphabet_sizes = cs.substrate.factored_tpm.alphabet_sizes
    mech_k = tuple(alphabet_sizes[i] for i in mechanism)
    n_states = math.prod(mech_k)
    if n_states > _MAX_UNCONSTRAINED_FORWARD_STATES:
        raise ValueError(
            f"unconstrained forward effect repertoire over mechanism "
            f"{mechanism} is infeasible at this size: it averages over "
            f"{n_states:,} mechanism states, each requiring a full forward "
            f"effect repertoire "
            f"(limit {_MAX_UNCONSTRAINED_FORWARD_STATES:,})"
        )
    total: np.ndarray | None = None
    for state in _utils.all_states(mech_k):
        rep = forward_effect_repertoire(cs, mechanism, purview, mechanism_state=state)
        if total is None:
            total = np.array(rep, dtype=float)
        else:
            total += rep
    assert total is not None
    return total / n_states


def unconstrained_forward_cause_repertoire(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...]
) -> Any:
    """Unconstrained forward cause repertoire — see Eq. 32 of the IIT 4.0 paper.

    Since ``m`` is fixed and we average over ``Z``, the per-state
    probabilities are all equal to the mean — fill with that value.

    Notes
    -----
    Deliberately not memoized, unlike its effect-direction counterpart. The
    work it depends on is :func:`forward_cause_repertoire`, which is memoized;
    what remains is a mean, an allocation, and a fill, worth about 4 µs against
    the roughly 0.7 µs a lookup costs — and that margin does not grow with
    mechanism size, since the cost is set by the purview. Reaching it needs the
    same ``(mechanism, purview)`` evaluated twice, which
    :func:`intrinsic_information` does not do within one analysis.
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
    # Read-only like every other repertoire the kernel returns; the memoizing
    # decorator does this for the functions it wraps.
    return _freeze(result)


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

    States whose intrinsic information ties the maximum within
    ``config.numerics.precision`` form the tie family (available via the
    returned specification's ``ties``); the returned specification is the
    first tied state in enumeration order.
    """
    full_state_space = states is None

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

        if full_state_space:
            # Vectorized over the full state space. The array flattened in
            # Fortran order matches ``all_states`` enumeration order (index 0
            # varies fastest), so the winner and tie family are found without
            # materializing one Python tuple per state — only the (small) tie
            # family is ever materialized.
            flat = np.asarray(dist, dtype=float).ravel(order="F")
            shape = np.asarray(dist).shape

            def flat_state(index: int) -> tuple[int, ...]:
                return tuple(int(c) for c in np.unravel_index(index, shape, order="F"))

            max_information = float(flat.max())
            tied_indices = np.flatnonzero(numerics.eq_mask(flat, max_information))
            tied_states = [(flat_state(int(i)), float(flat[i])) for i in tied_indices]
            winner_index = int(tied_indices[0])
            if flat.size > 1:
                # The highest raw value among non-winner states; ``argmax``
                # takes the first occurrence in enumeration order, matching a
                # stable descending sort.
                others = np.flatnonzero(np.arange(flat.size) != winner_index)
                runner_index = int(others[np.argmax(flat[others])])
                runner_up_state = flat_state(runner_index)
                runner_up_information = float(flat[runner_index])
            else:
                runner_up_state = runner_up_information = None
            return _build_state_specification(
                direction,
                purview,
                tied_states,
                rep,
                unconstrained_rep,
                runner_up_state,
                runner_up_information,
            )

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

    if states is None:
        alphabet_sizes = cs.substrate.factored_tpm.alphabet_sizes
        purview_k = tuple(alphabet_sizes[i] for i in purview)
        states = list(_utils.all_states(purview_k))

    state_to_information = {state: evaluate_state(state) for state in states}
    # The raw maximum anchors the tie cluster; membership is tolerance-based,
    # so states whose values differ from the maximum only by float-path noise
    # still join the family. The winner is the first tied state in enumeration
    # order, which keeps the selection independent of that noise.
    max_information = max(state_to_information.values())
    tied_states = [
        (state, information)
        for state, information in state_to_information.items()
        if numerics.eq(information, max_information)
    ]
    winner_state = tied_states[0][0]
    ranked = sorted(state_to_information.items(), key=lambda kv: kv[1], reverse=True)
    runner_up = next(
        ((state, value) for state, value in ranked if state != winner_state),
        None,
    )
    if runner_up is not None:
        runner_up_state = runner_up[0]
        runner_up_information = float(runner_up[1])
    else:
        runner_up_state = runner_up_information = None
    return _build_state_specification(
        direction,
        purview,
        tied_states,
        rep,
        unconstrained_rep,
        runner_up_state,
        runner_up_information,
    )


def _build_state_specification(
    direction: Direction,
    purview: tuple[int, ...],
    tied_states: list[tuple[tuple[int, ...], float]],
    rep: Any,
    unconstrained_rep: Any,
    runner_up_state: tuple[int, ...] | None,
    runner_up_information: float | None,
) -> Any:
    """Build the tie family of state specifications and return the winner."""
    from pyphi.models.state_specification import StateSpecification

    ties = [
        StateSpecification(
            direction=direction,
            purview=purview,
            state=state,
            intrinsic_information=float(information),
            repertoire=rep,
            unconstrained_repertoire=unconstrained_rep,
            runner_up_state=runner_up_state,
            runner_up_intrinsic_information=runner_up_information,
        )
        for state, information in tied_states
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
    max_order: int | None = None,
) -> list[tuple[int, ...]]:
    """Return all purviews that could belong to the MIC or MIE.

    A purview is a candidate for the maximally irreducible cause (MIC) or
    effect (MIE) only if it is not trivially reducible. Purviews that are
    trivially reducible against the (possibly cut) connectivity matrix of
    this candidate system are filtered out.

    The substrate-level enumeration is bounded by ``max_order`` and, when an
    explicit ``purviews`` list is given, by the largest purview in it — no
    candidate above the bound can survive the intersection, so bounding the
    enumeration is exact and avoids constructing the full powerset.
    """
    from pyphi.substrate import irreducible_purviews

    if purviews is not None:
        given_bound = max((len(p) for p in purviews), default=0)
        max_order = given_bound if max_order is None else min(max_order, given_bound)
    _potential_purviews = set(
        cs.substrate.potential_purviews(direction, mechanism, max_order=max_order)
    )
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
    """Return the ``Node`` objects for these indices.

    Raises
    ------
    ValueError
        If ``indices`` is not a subset of the System's node indices.
    """
    if set(indices) - set(cs.node_indices):
        raise ValueError("`indices` must be a subset of the System's indices.")
    return tuple(cs._index2node[n] for n in indices)
