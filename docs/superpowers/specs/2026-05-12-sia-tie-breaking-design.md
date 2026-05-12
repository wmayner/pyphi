# Paper-Faithful SIA Tie-Breaking — Design

**Status:** Draft for review
**Branch:** `2.0` (local-only)
**Tracks:** ROADMAP item 14 — Paper-faithful SIA tie-breaking; gates two
currently-xfailed tests at HEAD `24912ca`.

## Goal

Make `Substrate.sia()` produce a single deterministic
`SystemIrreducibilityAnalysis` for any input, including substrates with
state-ties at max `ii(s)` and partition-ties at the MIP key, and align
state-selection with Albantakis et al. 2023, Eq. 12 + S1 Text (the
canonical state among states tied at max intrinsic information is the
one with maximum unnormalized `φ_s`).

## Why this matters

- `test_sia_big_subsys_all_complete_{sequential,parallel}` in
  `test/test_big_phi.py` are xfailed today because the fully-connected
  5-node substrate produces tied cause/effect states and tied partitions.
  Re-running picks different "canonical" representatives across runs;
  the SIA's `__eq__` (which compares the full structure, not just `phi`)
  fails.
- Non-binary units (next in the schedule after this) multiply the number
  of candidate states quadratically and will exacerbate the tie surface.
  Tightening tie semantics before that lands is required.
- The fixture at `test/data/sia/big_subsys_all_complete.json` was captured
  under whichever ordering happened at fixture-write time. Any consumer
  comparing full SIA structure (golden tests, downstream caches, JSON
  round-trip) is affected.

## Two sources of non-determinism (both must be fixed)

### 1. System-state ties — IIT 4.0 only

`pyphi/core/repertoire_algebra.py:586-602` (`intrinsic_information`):

```python
state_to_information = {state: evaluate_state(state) for state in states}
max_information = max(state_to_information.values())
ties = [StateSpecification(...) for state, info in state_to_information.items()
        if info == max_information]
for tie in ties:
    tie.set_ties(ties)
return ties[0]   # ← arbitrary first entry
```

`pyphi/formalism/iit4/__init__.py:62-105` (`system_intrinsic_information`)
calls this once per direction and packages the returned representatives
into a `SystemStateSpecification`. The `.ties` set is preserved on each
`StateSpecification`, but the representative chosen as the canonical
`system_state[direction]` is whichever the dict-iteration order yielded
first.

This representative flows into SIA equality through
`SystemIrreducibilityAnalysis.system_state`.

### 2. MIP partition ties — IIT 3.0 and IIT 4.0

**IIT 4.0** (`pyphi/formalism/iit4/__init__.py:591-735`):

```python
def sia_minimization_key(sia):
    return (sia.normalized_phi, -sia.phi)

# ...
for candidate_mip_sia in sias:
    candidate_key = sia_minimization_key(candidate_mip_sia)
    if candidate_key < mip_key:
        mip_sia = candidate_mip_sia
        ...
    elif candidate_key == mip_key:
        ties.append(candidate_mip_sia)
```

Tied SIAs are collected, but the returned `mip_sia` is whichever
candidate first achieved the min — iteration-order dependent (especially
under parallel `MapReduce`). The pre-existing comment at line 719
acknowledges this: `# TODO(ties) refactor into resolve_ties module`.

**IIT 3.0** (`pyphi/formalism/iit3/__init__.py:286-300`):

```python
result = MapReduce(
    evaluate_partition, cuts,
    map_kwargs={...},
    reduce_func=min,                     # ← OrderableByPhi.__lt__ → phi only
    reduce_kwargs={"default": _null_sia(system)},
    ...
).run()
```

`OrderableByPhi.order_by` returns `self.phi` only (`pyphi/models/cmp.py:100`).
With ties at `phi`, `min()` returns the first-encountered tied element —
non-deterministic across runs.

## Existing infrastructure to reuse

`pyphi/resolve_ties.py` already exposes a strategy registry and resolver
functions for mechanism-level objects (`states`, `partitions`,
`purviews`). Config knobs are in place: `state_tie_resolution`,
`mip_tie_resolution`, `purview_tie_resolution` (all in
`pyphi/conf/formalism.py:53-57`).

What's missing:

- A system-level analogue: `resolve_ties.sias()`.
- A config knob for system-level SIA tie-breaks: `sia_tie_resolution`.
- A registered strategy for structural lex order on partitions:
  `PARTITION_LEX`.
- An IIT 4.0 hook that canonicalises `system_state` per Eq. 12 + S1 Text
  when `SystemStateSpecification.ties` carries more than one
  representative.

## Final design

### Component 1 — Partition lex key

Add a single method on `_PartitionBase`:

```python
def lex_key(self) -> bytes:
    """Return a canonical sortable byte representation of this partition's
    induced edge cut. Two partitions producing the same edge cut on the
    same node set compare equal under ``lex_key`` and sort identically.
    """
    n = max(self.indices) + 1 if self.indices else 0
    return self.cut_matrix(n).tobytes()
```

`cut_matrix(n)` is already on every partition class. `indices` is
already a unifying contract (`NullCut.indices = ()` returns empty;
`max()` is guarded by the `if`).

This is the lowest-common-denominator structural key: two partitions
that sever the same edges sort the same regardless of class. Class
identity is not part of the key — semantically equivalent partitions
canonicalise to the same `lex_key`.

For `NullCut` (empty edge cut), `lex_key()` returns `b""`, which sorts
before any non-null cut.

### Component 2 — `resolve_ties.sias()` and new strategies

Add to `pyphi/resolve_ties.py`:

```python
@phi_object_tie_resolution_strategies.register("PARTITION_LEX")
def _(m):
    return m.partition.lex_key()

@phi_object_tie_resolution_strategies.register("STATE_LEX")
def _(m):
    cause = m.system_state.cause.state if m.system_state and m.system_state.cause else ()
    effect = m.system_state.effect.state if m.system_state and m.system_state.effect else ()
    return (cause, effect)

@phi_object_tie_resolution_strategies.register("UNNORMALIZED_PHI")
def _(m):
    return m.phi

@phi_object_tie_resolution_strategies.register("NEGATIVE_UNNORMALIZED_PHI")
def _(m):
    return -m.phi


def sias[T](
    sias: Iterable[T], strategy: str | list[str] | None = None, **kwargs: Any
) -> Iterator[T]:
    """Resolve ties among system-level SIAs.

    Controlled by the ``sia_tie_resolution`` configuration option.
    """
    strategy = fallback(strategy, config.formalism.iit.sia_tie_resolution)
    assert strategy is not None, "sia_tie_resolution config must be set"
    return resolve(sias, strategy, operation=min, **kwargs)
```

`UNNORMALIZED_PHI` and `NEGATIVE_UNNORMALIZED_PHI` are aliases of the
existing `PHI` / `NEGATIVE_PHI` strategies; the longer names disambiguate
in the SIA context (where `phi` is already the unnormalized system
`φ_s`, distinct from `normalized_phi`).

### Component 3 — Config key

Add to `IITConfig` in `pyphi/conf/formalism.py`:

```python
sia_tie_resolution: list[str] = field(
    default_factory=lambda: ["NORMALIZED_PHI", "NEGATIVE_PHI", "PARTITION_LEX"]
)
```

Primary: minimise `normalized_phi` (the MIP-defining quantity).
Secondary: maximise `phi` (`-phi` minimised) — prefers the partition
whose unnormalized integration is largest among normalised-phi-tied
candidates.
Tertiary: structural lex order on the induced edge cut (deterministic
final fallback).

### Component 4 — IIT 4.0 integration

Replace the manual MIP loop in
`pyphi/formalism/iit4/__init__.py:718-735` with:

```python
sias_iter = sias if sias is not None else []
default_sia = _null_sia(reasons=[ShortCircuitConditions.NO_VALID_PARTITIONS])
sias_list = list(sias_iter) or [default_sia]
ties = tuple(resolve_ties.sias(sias_list, default=default_sia))
mip_sia = ties[0]
for tied_mip in ties:
    tied_mip.resolve_system_state()
    tied_mip.set_ties(ties)
```

The returned `ties` is already strategy-resolved (deterministic via
`PARTITION_LEX` tertiary). `mip_sia = ties[0]` picks the canonical
representative.

### Component 5 — Paper-faithful state canonicalisation (IIT 4.0)

When `system_state[direction].ties` carries multiple representatives at
MIP-time, the canonical state per Albantakis 2023 Eq. 12 + S1 Text is
the one with maximum unnormalized `φ_s` at the MIP partition.

Extend `resolve_system_state` (`pyphi/formalism/iit4/__init__.py:206`)
to accept `system` and `system_measure` as parameters — the call sites
in `sia()` (line 734) already have both in scope, so passing them in
avoids any dataclass change:

```python
def resolve_system_state(self, system, system_measure) -> None:
    """Canonicalise ``system_state`` after MIP selection.

    Among states tied at max intrinsic information, the canonical
    representative is the one with maximum unnormalized phi for the
    MIP partition (Albantakis et al. 2023, Eq. 12 + S1 Text). Structural
    state-tuple lex order breaks any remaining tie.
    """
    if self.system_state is None:
        return
    new_cause = self._canonicalize_tied_state(
        Direction.CAUSE, system, system_measure
    )
    new_effect = self._canonicalize_tied_state(
        Direction.EFFECT, system, system_measure
    )
    if (new_cause is not self.system_state.cause
            or new_effect is not self.system_state.effect):
        self.system_state = replace(
            self.system_state, cause=new_cause, effect=new_effect
        )

def _canonicalize_tied_state(self, direction, system, system_measure):
    spec = self.system_state[direction]
    if spec is None:
        return None
    tied = spec.ties
    if not tied or len(tied) <= 1:
        # No tie at the ii level — keep the MIP's cruelest spec.
        ria = self.cause if direction == Direction.CAUSE else self.effect
        if ria is not None and ria.specified_state is not None:
            return ria.specified_state
        return spec
    # Evaluate the MIP partition once per tied state to find max phi.
    cut_system = system.apply_cut(self.partition)
    evaluated = [
        (_integration_value_for_state(
            direction, system, cut_system, self.partition,
            tied_spec, system_measure,
        ), tied_spec)
        for tied_spec in tied
    ]
    # argmax on (signed_phi, state_tuple) — max phi, state-tuple lex tiebreak.
    _, _, canonical = max(
        (ria.signed_phi, tied_spec.state, tied_spec)
        for ria, tied_spec in evaluated
    )
    return canonical
```

`_integration_value_for_state` (line 376) has the matching signature
already.

The call site in `sia()` at line 734 becomes:

```python
for tied_mip in ties:
    tied_mip.resolve_system_state(system, system_measure)
    tied_mip.set_ties(ties)
```

Cost: at most O(|ties|) extra `forward_repertoire` evaluations at the
MIP — typical tie sizes are 2-8; negligible against the partition loop
that preceded it.

### Component 6 — IIT 3.0 integration

The IIT 3.0 `SystemIrreducibilityAnalysis` in `pyphi/models/sia.py`
inherits from `cmp.OrderableByPhi`, whose `order_by` returns `self.phi`
only. Override on this subclass:

```python
def order_by(self):
    return (self.phi, self.partition.lex_key())
```

`min(sias)` (used by `_sia_map_reduce` via `reduce_func=min` in
`pyphi/formalism/iit3/__init__.py:293`) then produces a deterministic
MIP across runs without any changes to the IIT 3.0 entry point itself.

No state-canonicalisation work for IIT 3.0 — it operates on CES
distances, not state specifications. Mechanism-level state and purview
ties already route through `resolve_ties.states` and
`resolve_ties.purviews`.

## Testing strategy

### Unit tests — `test/test_resolve_ties.py`

Add:

- `test_sias_partition_lex`: two SIAs with identical `(normalized_phi, phi)`
  but different partitions resolve deterministically via `PARTITION_LEX`.
- `test_lex_key_equivalent_partitions_compare_equal`: a
  `DirectedBipartition` and a `DirectedSetPartition` that sever the same
  edges produce equal `lex_key()`.
- `test_lex_key_distinct_partitions_compare_distinct`: distinct edge cuts
  give distinct `lex_key`.

### Integration tests — `test/test_big_phi.py`

Remove the `_BIG_SUBSYS_ALL_COMPLETE_TIE_XFAIL` marker from both
`test_sia_big_subsys_all_complete_{sequential,parallel}`. The tests must
pass with the existing fixture at
`test/data/sia/big_subsys_all_complete.json`.

If the captured fixture state happens to not match the new canonical
state (likely, since the fixture was captured under arbitrary ordering),
regenerate this single fixture via `test/IIT_4.0_make_jsons.ipynb` (the
existing fixture-write notebook; covers `big_subsys_all_complete` at
its `data/sia/big_subsys_all_complete.json` write-site). Document in
the commit message that this fixture was regenerated to match the
canonical ordering.

### Regression tests — full fast and slow lane

- All 17 goldens must continue to pass. Re-running them validates that
  for substrates without state/partition ties the result is unchanged.
- Slow-lane Hypothesis property tests in `test/test_invariants_hypothesis.py`
  must continue to pass.
- For any golden fixture that was captured under symmetric conditions
  and now diverges, regenerate via the fixture-write entry point. Expect
  zero or one such fixture beyond `big_subsys_all_complete`.

### Determinism property test (new)

Add a single property test in `test/test_invariants.py`:

```python
@pytest.mark.slow
def test_sia_is_deterministic_across_runs(big_subsys_all_complete):
    """Running .sia() twice on the same substrate yields equal SIAs."""
    s1 = big_subsys_all_complete.sia()
    s2 = big_subsys_all_complete.sia()
    assert s1 == s2
```

This is the architectural guarantee the project is intended to deliver.

## Files touched

**New:**

- `pyphi/resolve_ties.py` — add `sias()`, `PARTITION_LEX`,
  `STATE_LEX`, `UNNORMALIZED_PHI`, `NEGATIVE_UNNORMALIZED_PHI`.

**Modified:**

- `pyphi/models/partitions.py` — add `lex_key()` method on
  `_PartitionBase`.
- `pyphi/conf/formalism.py` — add `sia_tie_resolution` field on
  `IITConfig`.
- `pyphi/formalism/iit4/__init__.py`:
  - Replace the manual MIP loop (lines 718-735) with
    `resolve_ties.sias`.
  - Extend `resolve_system_state` to accept `system` and `system_measure`
    parameters and canonicalise per Eq. 12 + S1 Text. No SIA-dataclass
    fields added; the caller in `sia()` has both values in scope.
- `pyphi/models/sia.py` — override `order_by` on the IIT 3.0
  `SystemIrreducibilityAnalysis` to include the partition lex key.
  IIT 3.0's `_sia_map_reduce` (`pyphi/formalism/iit3/__init__.py:293`)
  uses `reduce_func=min`, which routes through this override; no
  changes to the iit3 module itself.

**Tests:**

- `test/test_resolve_ties.py` — extend.
- `test/test_big_phi.py` — drop xfail markers.
- `test/test_invariants.py` — add determinism property test.
- `test/data/sia/big_subsys_all_complete.json` — regenerate if needed.

## Changelog

Single fragment, `changelog.d/sia-tie-breaking.fix.md`:

```
Made `Substrate.sia()` results fully deterministic across runs. When
multiple cause/effect states tie at max intrinsic information, the
canonical representative is the one with maximum unnormalized phi
(Albantakis et al. 2023, Eq. 12 + S1 Text). Among partitions tied at the
MIP key, a structural lex break on the induced edge cut selects the
canonical partition. Configurable via the new `sia_tie_resolution`
option.
```

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Captured golden fixtures don't match new canonical ordering | Medium | Regenerate affected fixtures via existing entry point; document in commit |
| `lex_key()` byte comparison is slow for large `cut_matrix` | Low | Tertiary key — only invoked when primary+secondary tied; n is small (n≤8 typical) |
| `_canonicalize_tied_state` re-evaluates `forward_repertoire` at the MIP unnecessarily | Low | Cost dominated by partition loop that preceded it; ties are rare |
| Changing `system_state` after MIP selection breaks downstream consumers expecting the cruelest spec | Low | `resolve_system_state` already does this rewrite; we're changing which state is written, not whether |
| New strategies aliasing existing names (`UNNORMALIZED_PHI` vs `PHI`) confuse users | Low | Aliases are explicit; `PHI` keeps backward compatibility for mechanism-level configs |

## Effort estimate

~2 days of focused work. Mechanical edits dominate. The novel piece is
`_canonicalize_tied_state`; the partition lex key and `resolve_ties.sias`
are mechanical adaptations of existing patterns.
