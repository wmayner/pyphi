# resolve_ties.py
"""Resolve ties between IIT objects."""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from itertools import tee
from typing import Any
from typing import Literal
from typing import Protocol
from typing import TypeVar

from .conf import config
from .conf import fallback
from .registry import Registry
from .utils import NO_DEFAULT
from .utils import iter_with_default

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Cascade primitive
# ---------------------------------------------------------------------------
#
# See ``docs/superpowers/specs/2026-05-13-cascade-execution-model.md`` for
# the design. The cascade walks the IIT postulate hierarchy
# (Existence → Intrinsicality → Information → Integration → Exclusion →
# Composition, plus a pyphi-specific Determinism level) and resolves ties
# at the lowest sufficient postulate. A ``ResolutionContext`` bounds the
# walk's escalation budget and memoizes per-candidate intermediate values.


Postulate = Literal[
    "Existence",
    "Intrinsicality",
    "Information",
    "Integration",
    "Exclusion",
    "Composition",
    "Determinism",
]


# Postulate order. Determinism sits after Composition as a pyphi-specific
# canonicalization fallback for ties that don't violate any postulate
# (e.g., mechanism MIPs tied at unnormalized phi with different
# partitions — extrinsic labeling ties).
_POSTULATE_ORDER: tuple[Postulate, ...] = (
    "Existence",
    "Intrinsicality",
    "Information",
    "Integration",
    "Exclusion",
    "Composition",
    "Determinism",
)


def _postulate_rank(postulate: Postulate) -> int:
    return _POSTULATE_ORDER.index(postulate)


CascadeOp = Literal["argmax", "argmin", "filter"]


@dataclass(frozen=True)
class CascadeLevel:
    """A single named step in a cascade.

    ``postulate`` names the IIT postulate this step instantiates;
    used to consult the ``ResolutionContext``'s escalation budget.
    ``op`` is the reduction (argmax / argmin / filter). ``key`` is a
    callable returning the comparison value for a candidate (or a
    bool for ``filter``).
    """

    postulate: Postulate
    op: CascadeOp
    key: Callable[[Any], Any]


CascadeOutcomeStatus = Literal[
    "RESOLVED",
    "UNRESOLVED_WITHIN_BUDGET",
    "POSTULATE_FAILURE",
]


@dataclass(frozen=True)
class CascadeOutcome[U]:
    """The outcome of running a cascade.

    ``resolved`` is the unique winner when ``outcome == 'RESOLVED'``;
    ``None`` otherwise. ``tied_set`` is the set of candidates that
    survived to the final level the cascade reached. ``cascade_level``
    names that postulate.
    """

    resolved: U | None
    tied_set: tuple[U, ...]
    cascade_level: Postulate
    outcome: CascadeOutcomeStatus
    failure_reason: str | None = None


class NotAComplex(Exception):
    """Raised when a cascade reaches its final level with a tie and
    ``on_unresolved='fail'``. Indicates that the substrate fails the
    postulate at which the tie persists; downstream callers convert
    to a Null* sentinel for the public API.
    """

    def __init__(
        self,
        tied_set: Sequence[Any],
        cascade_level: Postulate,
        failure_reason: str | None = None,
    ) -> None:
        super().__init__(
            f"Cascade exhausted at {cascade_level}: "
            f"{len(tied_set)} tied candidates remain"
        )
        self.tied_set: tuple[Any, ...] = tuple(tied_set)
        self.cascade_level: Postulate = cascade_level
        self.failure_reason: str | None = failure_reason


class ResolutionContext:
    """Per-computation context for cascade tie resolution.

    Carries the entry-point function's escalation budget and a
    memoization cache shared across nested cascade calls.
    """

    def __init__(
        self,
        max_escalation_level: Postulate,
        memo: dict[Any, Any] | None = None,
    ) -> None:
        self.max_escalation_level: Postulate = max_escalation_level
        self._memo: dict[Any, Any] = memo if memo is not None else {}

    def can_escalate_to(self, postulate: Postulate) -> bool:
        """True iff ``postulate`` is at or below ``max_escalation_level``
        in the postulate order."""
        return _postulate_rank(postulate) <= _postulate_rank(self.max_escalation_level)

    def memoize[V](self, key: Any, fn: Callable[[], V]) -> V:
        """Return ``memo[key]``, computing via ``fn()`` if absent."""
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    def child(self) -> "ResolutionContext":
        """Return a child context inheriting parent budget and memo cache."""
        return ResolutionContext(
            max_escalation_level=self.max_escalation_level,
            memo=self._memo,
        )


OnUnresolved = Literal["fail", "defer", "warn"]


def _apply_level[U](
    candidates: Sequence[U],
    level: CascadeLevel,
) -> tuple[U, ...]:
    """Apply ``level``'s op to ``candidates`` and return the winners."""
    if level.op == "filter":
        return tuple(c for c in candidates if level.key(c))
    keys = [level.key(c) for c in candidates]
    if level.op == "argmax":
        extremum = max(keys)
    else:  # argmin
        extremum = min(keys)
    return tuple(c for c, k in zip(candidates, keys, strict=True) if k == extremum)


def cascade[U](
    candidates: Iterable[U],
    levels: Sequence[CascadeLevel],
    *,
    context: ResolutionContext,
    on_unresolved: OnUnresolved = "defer",
) -> CascadeOutcome[U]:
    """Walk a cascade of postulate-level reductions.

    At each level, apply ``level.op`` with ``level.key`` to identify
    surviving candidates. If a single candidate remains, the cascade
    resolves. If multiple remain and the next level is within budget,
    recurse on that level. Otherwise:

    - ``on_unresolved='defer'`` (default): return
      ``UNRESOLVED_WITHIN_BUDGET`` carrying the tied set for downstream
      surfacing.
    - ``on_unresolved='fail'``: raise :class:`NotAComplex`.
    - ``on_unresolved='warn'``: emit a warning and return as 'defer'.
    """
    survivors: tuple[U, ...] = tuple(candidates)
    if not survivors:
        raise ValueError("cascade requires at least one candidate")

    # Track the most recently *processed* level (one whose op was applied).
    # When budget blocks or all levels exhaust, this is the level reported.
    last_processed_level: Postulate | None = None

    for level in levels:
        if not context.can_escalate_to(level.postulate):
            return CascadeOutcome(
                resolved=None,
                tied_set=survivors,
                cascade_level=last_processed_level or level.postulate,
                outcome="UNRESOLVED_WITHIN_BUDGET",
            )
        if len(survivors) == 1:
            return CascadeOutcome(
                resolved=survivors[0],
                tied_set=survivors,
                cascade_level=last_processed_level or level.postulate,
                outcome="RESOLVED",
            )
        pre_apply = survivors
        survivors = _apply_level(pre_apply, level)
        last_processed_level = level.postulate
        if len(survivors) == 1:
            return CascadeOutcome(
                resolved=survivors[0],
                tied_set=pre_apply,
                cascade_level=level.postulate,
                outcome="RESOLVED",
            )

    final_level: Postulate = last_processed_level or "Determinism"
    if len(survivors) == 1:
        return CascadeOutcome(
            resolved=survivors[0],
            tied_set=survivors,
            cascade_level=final_level,
            outcome="RESOLVED",
        )

    # Cascade exhausted all levels with a tie.
    if on_unresolved == "fail":
        raise NotAComplex(survivors, final_level)
    if on_unresolved == "warn":
        import warnings

        warnings.warn(
            f"Cascade exhausted at {final_level} with {len(survivors)} tied candidates",
            stacklevel=2,
        )
    return CascadeOutcome(
        resolved=None,
        tied_set=survivors,
        cascade_level=final_level,
        outcome="UNRESOLVED_WITHIN_BUDGET",
    )


class _StateMIP(Protocol):
    """Structural type for a per-state MIP result consumed by the state-tie cascade.

    Concrete types satisfying this (e.g.,
    :class:`pyphi.formalism.iit4.SystemIrreducibilityAnalysis`) need to
    expose ``phi`` (the Integration-level cascade key). ``big_phi``
    (Composition-level key) is only read when the cascade escalates
    past Integration; callers whose results lack ``big_phi`` may rely
    on the cascade short-circuiting at Integration via
    ``context.max_escalation_level``.
    """

    @property
    def phi(self) -> float: ...


class _ComplexCandidate(Protocol):
    """Structural type for a candidate complex consumed by the
    substrate-exclusion cascade. Requires ``big_phi`` (Composition-level
    integrated information of the cause-effect structure).
    """

    @property
    def big_phi(self) -> float: ...


def resolve_complex_tie[V: _ComplexCandidate](
    candidates: "Iterable[V]",
    *,
    context: ResolutionContext,
    on_unresolved: OnUnresolved = "defer",
) -> CascadeOutcome[V]:
    """Resolve a substrate-exclusion tie via the Composition cascade.

    Per Albantakis et al. 2023 S1 Text: when overlapping substrates tie
    at maximum ``φ_s``, the exclusion postulate is resolved by
    escalating to ``Φ`` (Composition). The substrate with maximum ``Φ``
    qualifies as the complex; overlapping losers are excluded. If
    multiple substrates also tie at ``Φ``, the exclusion postulate
    fails for that group and they do not qualify as complexes — under
    ``on_unresolved='fail'`` this raises :exc:`NotAComplex`; under
    ``'defer'`` the outcome is ``UNRESOLVED_WITHIN_BUDGET`` carrying
    the tied set so the caller can proceed to the next-best by
    ``φ_s``.

    Callers are expected to pre-filter candidates to a single
    ``φ_s``-tied overlap-clique; this function only walks the
    Composition step.
    """
    return cascade(
        candidates,
        levels=[
            CascadeLevel(
                postulate="Composition",
                op="argmax",
                key=lambda c: c.big_phi,
            ),
        ],
        context=context,
        on_unresolved=on_unresolved,
    )


class _CongruentMice(Protocol):
    """Structural type for a MICE consumed by the distinction-state cascade.

    Requires ``is_congruent`` (against a per-direction state spec) and
    ``purview`` (used by the cross-purview heuristic).
    """

    @property
    def purview(self) -> Sequence[int]: ...

    def is_congruent(self, other: Any) -> bool: ...


def resolve_distinction_tie[V: _CongruentMice](
    state_ties: "Sequence[V] | None",
    purview_ties: "Sequence[V] | None",
    system_state_spec: Any,
    *,
    context: ResolutionContext,  # noqa: ARG001
) -> V | None:
    """Resolve a per-direction distinction-state tie per Albantakis et al.
    2023 S1 Text.

    Two cases:

    - **Same-purview state ties** (``state_ties``): MICEs tied at maximum
      ``ii(m, z)`` within a single purview. Returns the MICE whose
      specified state is congruent with ``system_state_spec`` — the
      direction-specific component of the system's specified
      cause-effect state.

    - **Cross-purview ties** (``purview_ties``): MICEs tied at maximum
      ``φ_d(m, Z)`` across different purviews. Returns the largest
      congruent purview (the "typically favors larger purviews"
      heuristic for "supports the most relations with other
      distinctions"). The more expensive joint-relations-count
      computation is captured under ROADMAP P11.95c as opt-in.

    State-tie congruence is preferred over purview-tie heuristic: if any
    state-tie MICE is congruent, it is returned; only when none is
    congruent does the cross-purview branch fire. Returns ``None`` when
    no congruent MICE is found in either branch.
    """
    if state_ties:
        congruent_state = [m for m in state_ties if m.is_congruent(system_state_spec)]
        if congruent_state:
            return congruent_state[0]
    if purview_ties:
        congruent_purview = [
            m for m in purview_ties if m.is_congruent(system_state_spec)
        ]
        if congruent_purview:
            return max(congruent_purview, key=lambda m: len(m.purview))
    return None


def resolve_state_tie[K, V: _StateMIP](
    per_state_mips: "Mapping[K, V]",
    *,
    context: ResolutionContext,
    on_unresolved: OnUnresolved = "defer",
) -> CascadeOutcome[K]:
    """Resolve a state tie via the paper-faithful per-state max-min cascade.

    Per Albantakis et al. 2023 S1 Text + Eq 20 parenthetical: among states
    tied at maximum intrinsic information ``ii``, the canonical winner is
    the state whose per-state ``φ_s`` (the value of integrated information
    at that state's MIP) is greatest. If ``φ_s`` ties too, the cascade
    escalates to per-state ``Φ`` (the cause-effect structure's integrated
    information at Composition). If ``Φ`` ties as well, the substrate
    fails the information postulate — :exc:`NotAComplex` is raised under
    ``on_unresolved='fail'``.

    ``per_state_mips`` maps each candidate state spec to a per-state MIP
    result object (typically a :class:`SystemIrreducibilityAnalysis` or
    similar). The object must expose ``.phi``; ``.big_phi`` is consulted
    only when Composition escalation fires.

    Returns a :class:`CascadeOutcome` whose ``resolved`` field is the
    winning key from ``per_state_mips`` (or ``None`` when budget caps
    escalation short of resolution).
    """
    return cascade(
        list(per_state_mips.keys()),
        levels=[
            CascadeLevel(
                postulate="Integration",
                op="argmax",
                key=lambda spec: per_state_mips[spec].phi,
            ),
            CascadeLevel(
                postulate="Composition",
                op="argmax",
                key=lambda spec: per_state_mips[spec].big_phi,  # pyright: ignore[reportAttributeAccessIssue]
            ),
        ],
        context=context,
        on_unresolved=on_unresolved,
    )


# Suppress "unused" warning for field — used in CascadeOutcome subclasses
# that may extend with diagnostic tables.
_ = field


class PhiObjectTieResolutionRegistry(Registry):
    """Storage for functions for resolving ties among phi-objects."""

    desc = "functions for resolving ties among phi-objects"


phi_object_tie_resolution_strategies = PhiObjectTieResolutionRegistry()


@phi_object_tie_resolution_strategies.register("PURVIEW_SIZE")
def _(m):
    return len(m.purview)


@phi_object_tie_resolution_strategies.register("NEGATIVE_PURVIEW_SIZE")
def _(m):
    return -len(m.purview)


@phi_object_tie_resolution_strategies.register("PHI")
def _(m):
    return m.phi


@phi_object_tie_resolution_strategies.register("NEGATIVE_PHI")
def _(m):
    return -m.phi


@phi_object_tie_resolution_strategies.register("NORMALIZED_PHI")
def _(m):
    return m.normalized_phi


@phi_object_tie_resolution_strategies.register("NEGATIVE_NORMALIZED_PHI")
def _(m):
    return -m.normalized_phi


@phi_object_tie_resolution_strategies.register("NONE")
def _(m):
    raise NotImplementedError(
        'tie resolution strategy "NONE" should never be called; '
        "it must be special-cased in the resolve() function"
    )


@phi_object_tie_resolution_strategies.register("PARTITION_LEX")
def _(m):
    return m.partition.lex_key()


def _strategies_to_key_function(strategies):
    """Convert a tie resolution strategy to a key function."""
    if isinstance(strategies, str):
        # Allow a single strategy to be specified as a bare string
        strategies = [strategies]
    return lambda obj: tuple(
        phi_object_tie_resolution_strategies[s](obj) for s in strategies
    )


# TODO(4.0) docstring
# TODO(4.0) fix this implementation so we only need one pass; currently,
# all_maxima only works if equality semantics are correct for this purpose, and
# RIA equality checks purview equality, so they are not.
# def resolve(objects, strategy, operation=all_maxima, default=NO_DEFAULT):
#     """Filter phi-objects according to a strategy."""
#     if strategy == "NONE":
#         yield from iter_with_default(objects, default=default)
#         return
#     sort_key = _strategies_to_key_function(strategy)
#     key_args, objects = tee(objects)
#     keys = map(sort_key, key_args)
#     if default is not NO_DEFAULT:
#         default = (sort_key(default), default)
#     ties = operation(zip(keys, objects), default=default)
#     for _, obj in ties:
#         yield obj


def resolve[T](
    objects: Iterable[T],
    strategy: str | list[str],
    operation: Callable[..., Any],
    default: Any = NO_DEFAULT,
) -> Iterator[T]:
    """Filter phi-objects according to a strategy."""
    if strategy == "NONE":
        yield from iter_with_default(objects, default=default)
        return
    sort_key = _strategies_to_key_function(strategy)
    objects, to_transform = tee(objects)
    values = list(map(sort_key, to_transform))
    extremum = operation(values, default=default)
    ties = (
        obj for obj, value in zip(objects, values, strict=False) if value == extremum
    )
    yield from iter_with_default(ties, default=default)


def states[T](
    rias: Iterable[T], strategy: str | list[str] | None = None, **kwargs: Any
) -> Iterator[T]:
    """Resolve ties among states (RIAs).

    Controlled by the STATE_TIE_RESOLUTION configuration option.
    """
    strategy = fallback(strategy, config.formalism.iit.state_tie_resolution)
    assert strategy is not None, "STATE_TIE_RESOLUTION config must be set"
    return resolve(rias, strategy, operation=max, **kwargs)


def partitions[T](
    mips: Iterable[T], strategy: str | list[str] | None = None, **kwargs: Any
) -> Iterator[T]:
    """Resolve ties among mechanism partitions (MIPs).

    Controlled by the MIP_TIE_RESOLUTION configuration option.
    """
    strategy = fallback(strategy, config.formalism.iit.mip_tie_resolution)
    assert strategy is not None, "MIP_TIE_RESOLUTION config must be set"
    return resolve(mips, strategy, operation=min, **kwargs)


def purviews[T](
    mice: Iterable[T], strategy: str | list[str] | None = None, **kwargs: Any
) -> Iterator[T]:
    """Resolve ties among purviews (MICEs).

    Controlled by the PURVIEW_TIE_RESOLUTION configuration option.
    """
    strategy = fallback(strategy, config.formalism.iit.purview_tie_resolution)
    assert strategy is not None, "PURVIEW_TIE_RESOLUTION config must be set"
    yield from resolve(mice, strategy, operation=max, **kwargs)


def sias[T](
    analyses: Iterable[T], strategy: str | list[str] | None = None, **kwargs: Any
) -> Iterator[T]:
    """Resolve ties among system-level SIAs.

    Controlled by the ``sia_tie_resolution`` configuration option.
    """
    strategy = fallback(strategy, config.formalism.iit.sia_tie_resolution)
    assert strategy is not None, "sia_tie_resolution config must be set"
    return resolve(analyses, strategy, operation=min, **kwargs)
