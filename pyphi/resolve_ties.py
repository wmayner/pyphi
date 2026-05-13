# resolve_ties.py
"""Resolve ties between IIT objects."""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from itertools import tee
from typing import Any
from typing import Literal
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
