# Cascade execution model — design

**Status:** Implementation spec for Phase C.0.
**Inputs:** [Tie-resolution canonical reading](2026-05-13-tie-resolution-canonical-reading.md).
**Output:** A unified tie-resolution primitive that all per-level
cascades use.

## Summary

Tie resolution in PyPhi is recast as a **cascade** over postulate
levels. Each cascade is a generator that yields a sequence of
`CascadeLevel` results, one per postulate it walks through. A
`ResolutionContext` carried by the entry-point function bounds the
cascade's escalation budget (so a low-level tie doesn't trigger
high-level recomputation when the consumer would compute the same
value anyway) and memoizes per-candidate intermediate values across
cascade steps.

The pattern enforces:
- **Lazy escalation**: a tie at postulate K is resolved by computing
  at postulate K+1, never K+2 unless K+1 also ties.
- **Memoization**: a per-state φ_s table computed during system-MIP
  tie resolution is reused by the system-state cascade and by the
  CES builder, not recomputed.
- **Deferred resolution**: when the consumer's budget can't
  disambiguate a tie, the tied set is surfaced as a first-class
  value (`UnresolvedSIA`, distinction `state_ties`/`purview_ties`)
  for downstream resolution.

## Core types

### `CascadeLevel`

A single named step in a cascade.

```python
@dataclass(frozen=True)
class CascadeLevel:
    postulate: Literal[
        "Existence", "Intrinsicality", "Information",
        "Integration", "Exclusion", "Composition",
        "Determinism",  # pyphi-specific final canonicalization
    ]
    op: Literal["argmax", "argmin", "filter"]
    key: str | Callable  # name of registered strategy or inline fn
    # Optional escape clauses (e.g., intrinsic CES equality check
    # at the Composition level of the system-state cascade).
    escape: Callable[[Sequence[T]], bool] | None = None
```

### `CascadeOutcome`

The result of running a cascade to its budget limit.

```python
@dataclass(frozen=True)
class CascadeOutcome[T]:
    resolved: T | None              # single winner, or None
    tied_set: tuple[T, ...]         # all candidates the cascade saw
    cascade_level: str              # last level reached
    outcome: Literal[
        "RESOLVED",                 # single winner
        "UNRESOLVED_WITHIN_BUDGET", # cascade hit budget limit
        "POSTULATE_FAILURE",        # cascade reached final postulate, still tied
    ]
    failure_reason: str | None = None
    diagnostic_tables: dict[str, dict] = field(default_factory=dict)
```

### `ResolutionContext`

The per-computation context carrying entry-point budget and
memoization.

```python
class ResolutionContext:
    max_escalation_level: PostulateName
    formalism: Formalism
    memo: dict[str, Any]  # cached intermediate values
    unresolved_ledger: list[UnresolvedTie]  # diagnostic surface

    def memoize(self, key: str, fn: Callable[[], V]) -> V:
        """Return memo[key], computing via fn() if absent."""

    def can_escalate_to(self, postulate: PostulateName) -> bool:
        """True iff postulate <= max_escalation_level in the
        postulate order."""

    def child(self) -> ResolutionContext:
        """Return a child context inheriting parent's budget+memo
        for nested computations."""
```

Entry points construct the context once:

```python
def ces(system, ...):
    ctx = ResolutionContext(
        max_escalation_level="Composition",
        formalism=resolve_formalism(...),
    )
    sia = _sia(system, context=ctx)
    ...

def sia(system, ...):
    ctx = ResolutionContext(
        max_escalation_level="Integration",
        formalism=resolve_formalism(...),
    )
    ...
```

When `ces()` calls `sia()` internally, `sia` accepts the parent
context — the cascade can escalate to Composition because the budget
permits, AND because `ces()` was going to compute the per-state Φ
anyway.

When `sia()` is called standalone, the cascade stops at Integration;
remaining state ties are surfaced on the returned SIA.

## The cascade generator

```python
def cascade(
    candidates: Iterable[T],
    levels: Sequence[CascadeLevel],
    *,
    context: ResolutionContext,
    on_unresolved: Literal["fail", "defer", "warn"] = "defer",
) -> CascadeOutcome[T]:
    """Walk a cascade, lazily escalating.

    The walk:
    1. Compute the key for each candidate at the current level.
    2. Apply the op (argmax/argmin/filter) to identify winners.
    3. If single winner → resolved.
    4. Else if context allows escalation to next level → recurse.
    5. Else → surface unresolved (or fail / warn per on_unresolved).
    """
```

The walk yields per-level intermediate state for diagnostic
purposes; the public interface is `CascadeOutcome`.

### Worked example: system-state cascade

```python
def resolve_system_state_ties(
    tied_states: Sequence[State],
    system: System,
    *,
    context: ResolutionContext,
) -> CascadeOutcome[State]:
    # Memoize per-state φ_s — shared with system-MIP work.
    phi_s_table = context.memoize(
        ("per_state_phi_s", system.identity),
        lambda: {c: per_state_phi_s(c, system, context) for c in tied_states},
    )
    # Memoize per-state Φ — only computed if Integration ties.
    def big_phi_table():
        return {c: per_state_big_phi(c, system, context) for c in winners_at_integration}

    return cascade(
        tied_states,
        levels=[
            CascadeLevel("Integration",  "argmax", lambda c: phi_s_table[c]),
            CascadeLevel("Composition",  "argmax", lambda c: big_phi_table()[c]),
            # Escape clause: intrinsic CES equality
            CascadeLevel("Composition",  "filter", _intrinsic_ces_equality,
                         escape=allow_extrinsic_tie),
        ],
        context=context,
        on_unresolved="fail",  # → NullSIA with INFORMATION_TIE reason
    )
```

If `context.max_escalation_level == "Integration"`, the cascade
returns `CascadeOutcome.outcome = "UNRESOLVED_WITHIN_BUDGET"` after
the first level — caller surfaces the tied set on the SIA.

If `context.max_escalation_level == "Composition"` (called from
`ces()`), the cascade walks all levels.

## Type-level Resolved/Unresolved split

Mirror the existing `UnresolvedDistinctions` / `ResolvedDistinctions`
(P11.9) pattern at the SIA level.

```python
class UnresolvedSIA(SystemIrreducibilityAnalysis):
    """SIA whose specified state may carry unresolved ties.

    Internal type. Functions that consume a SIA's specified state
    canonically (e.g., congruence filtering) take ResolvedSIA, so
    passing UnresolvedSIA is a static type error.
    """
    cascade_outcome: CascadeOutcome  # for diagnostic surfacing


class ResolvedSIA(SystemIrreducibilityAnalysis):
    """SIA whose specified state has been disambiguated.

    Returned by entry points where the cascade budget permits full
    resolution. Required for CauseEffectStructure construction.
    """
```

The split is internal: the public `system.sia()` returns
`ResolvedSIA` (cascade runs at `max_escalation_level="Composition"`
internally if needed). Standalone `sia()` documents that returned
SIAs may have a non-empty `system_state.cause.ties` /
`.effect.ties` set — these are the canonical representative + the
tied alternatives, surfaced for inspection but not for further
computation without explicit re-resolution.

## Strategy registry extension

The existing `phi_object_tie_resolution_strategies` registry stays;
new keys are added per the canonical reading:

| Name | Returns | Used by |
|---|---|---|
| `PER_STATE_PHI_S` | precomputed per-state φ_s value | system-state cascade |
| `PER_STATE_BIG_PHI` | precomputed per-state Φ value | system-state cascade |
| `BIG_PHI` | candidate's CES Φ | substrate-exclusion cascade |
| `MINIMAL_OCCURRENCE` | True iff no other candidate is a strict subset | AC cascade |
| `CONGRUENCE_WITH_SYSTEM_STATE` | bool of `is_congruent(system_state[direction])` | distinction cascade |
| `LARGER_PURVIEW` | `-len(mice.purview)` | distinction cross-purview (Q2 default) |
| `MOST_RELATIONS_JOINT` | joint-relation count from CES enumeration | distinction cross-purview (Q2 opt-in) |
| `INTRINSIC_CES_EQUALITY` | CES canonical fingerprint | system-state cascade Composition tie |

`PARTITION_LEX` and `NEGATIVE_PHI` already exist (P11.95a).

## Failed-cascade semantics

When the cascade reaches its final postulate level with ties still
present:

- **System-state cascade** (Composition tie): the substrate fails
  the information postulate **unless the intrinsic CES equality
  escape applies**. In the failing case, return a `NullSIA` with
  `reasons=[INFORMATION_TIE]`. CES returns empty.
- **Substrate-exclusion cascade** (Composition + overlap): the
  substrate fails the exclusion postulate; `complexes()` skips this
  candidate and continues with the next-best unique candidate.
- **Mechanism MIP / distinction state cascade**: lex-canonical
  fallback (`PARTITION_LEX`, `STATE_LEX`). These ties don't violate
  any postulate per the paper; they are labeling ties.

A new `NotAComplex` exception type carries the failure reason and
the tied set; entry-point functions catch it and convert to the
appropriate Null* sentinel for the public API.

## Integration with existing pyphi pieces

### What stays

- `pyphi/resolve_ties.py` — extended with the cascade primitive;
  existing `states`/`partitions`/`purviews`/`sias` resolvers
  refactor to thin wrappers around `cascade`.
- `pyphi/conf/formalism.py` — `*_tie_resolution` config fields
  stay; defaults shift to paper-faithful (Q6); legacy values are
  still valid and translate to flat-list cascades.
- `set_ties` / `.ties` on RIA, MICE, SIA — unchanged. Cascade fills
  these.

### What changes

- `pyphi/formalism/iit4/__init__.py:420-458` `integration_value` —
  remove cruelest-cut; use cascade.
- `pyphi/substrate.py:513-541` `complexes` — replace greedy
  condensation; use substrate-exclusion cascade.
- `pyphi/models/distinction.py:199-225` `resolve_congruence` —
  replace first-congruent filter; use distinction-state cascade.
- `pyphi/actual.py:1086-1229` — route through AC cascade.

### What's new

- `pyphi/resolve_ties.py`:
  - `cascade()` generator + `CascadeLevel` + `CascadeOutcome`.
  - `ResolutionContext`.
  - New strategies per the table above.
- `pyphi/models/sia.py` and `pyphi/formalism/iit4/__init__.py`:
  - `UnresolvedSIA` / `ResolvedSIA` types.
- `pyphi/intrinsic_equality.py` (new):
  - Substrate automorphism group computation.
  - CES canonical fingerprint.
- `pyphi/exceptions.py`:
  - `NotAComplex` exception.

## Testing strategy

Phase C.0 lands the cascade primitive with **TDD**. Tests are
written first against the design specified here; implementation
follows.

Test file: `test/test_resolve_ties.py` extends with:

1. **Cascade primitive** unit tests on synthetic mocks:
   - Single-level resolution (`argmax` over scalar keys).
   - Multi-level cascade with budget exhaustion at level 1.
   - Memoization: a key function is called once per candidate
     across multiple cascade levels.
   - `on_unresolved="fail"` raises `NotAComplex` with tied set.
   - `on_unresolved="defer"` returns `UNRESOLVED_WITHIN_BUDGET`.
2. **ResolutionContext**:
   - Child context inherits parent memo + budget.
   - `can_escalate_to` respects postulate order.
   - `memoize` caches by key tuple.
3. **Type-level UnresolvedSIA / ResolvedSIA**:
   - Construction.
   - Pyright check that ResolvedSIA flows into CES functions.

## Performance considerations

- **Memoization** is the central efficiency mechanism. Per-state
  φ_s and per-state Φ tables are computed lazily and shared across
  cascade levels and across calls within a single `ResolutionContext`.
- **Lazy escalation**: Composition-level keys (per-state Φ) require
  full CES computation and are expensive. The cascade computes them
  only when Integration-level ties exist AND the budget permits.
- **Substrate-exclusion cascade**: Φ is computed only for
  overlapping φ_s-tied candidates, not for all candidates. The
  lazy-Φ payload is structured to allow early termination if a
  non-overlapping tie is found at lower φ_s.
- **Distinction cross-purview ties** (Q2 default = heuristic): the
  larger-purview heuristic costs O(1) per tied candidate. The
  two-pass opt-in costs O(2^|tied_distinctions|) for the joint
  enumeration; users who enable it accept this.

## File layout

```
pyphi/
├── resolve_ties.py            # extended: cascade + strategies
├── exceptions.py              # new: NotAComplex
├── intrinsic_equality.py      # new: substrate automorphism + CES fingerprint
├── models/
│   └── sia.py                 # extended: UnresolvedSIA / ResolvedSIA
├── formalism/iit4/
│   ├── __init__.py            # integration_value → cascade dispatch
│   └── formalism.py           # unchanged
├── substrate.py               # complexes() → substrate-exclusion cascade
├── models/distinction.py      # resolve_congruence → distinction cascade
└── actual.py                  # find_mip + find_causal_link → AC cascade

test/
├── test_resolve_ties.py       # extended: cascade primitive tests
├── test_tie_resolution/       # new dir for per-level tests (Phase B.2)
│   ├── test_specified_state_per_purview.py
│   ├── test_mechanism_mip.py
│   ├── test_system_state.py
│   ├── test_system_mip.py
│   ├── test_substrate_exclusion.py
│   ├── test_distinction_congruence.py
│   ├── test_clamp_signed_phi.py
│   ├── test_actual_causation.py
│   ├── test_iit3.py
│   ├── test_serialization.py
│   ├── test_cross_formalism.py
│   └── test_intrinsic_ces_equality.py
└── fixtures/
    └── tie_networks.py        # new: tie-provoking substrates
```

## Implementation order (Phase C.0 sub-steps)

1. **C.0.1**: Add `CascadeLevel`, `CascadeOutcome`,
   `ResolutionContext`, `NotAComplex` types in `resolve_ties.py` +
   `exceptions.py`. Tests on synthetic mocks.
2. **C.0.2**: Implement `cascade()` generator. Tests for
   single-level, multi-level, memoization, budget exhaustion,
   failure modes.
3. **C.0.3**: Refactor existing `resolve_ties.states`/`.partitions`/
   `.purviews`/`.sias` to thin wrappers over `cascade`. Existing
   tests continue to pass.
4. **C.0.4**: Add new strategies (`PER_STATE_PHI_S`,
   `PER_STATE_BIG_PHI`, etc.) to the registry. Each strategy
   tested in isolation.
5. **C.0.5**: Add `UnresolvedSIA` / `ResolvedSIA` types. Pyright
   checks the type flow.

C.0.1–C.0.5 land as a single commit (or two: types + cascade +
strategies, then existing-resolver refactor). Subsequent phases
(C.1, C.2, C.3, ...) consume this scaffolding.
