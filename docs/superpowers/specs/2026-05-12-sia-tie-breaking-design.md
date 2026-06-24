# Deterministic SIA Selection — Design

**Status:** Draft for review
**Branch:** `2.0` (local-only)
**Tracks:** ROADMAP item 14 (partial); gates two currently-xfailed tests
at HEAD `24912ca`.

## Goal

Make `Substrate.sia()` produce a single deterministic
`SystemIrreducibilityAnalysis` for any input, including substrates where
multiple partitions tie at the minimum `(normalized_phi, -phi)` key.
The existing cruelest-cut convention for tied specified states is
preserved; the only behaviour change is that previously-arbitrary
"first-encountered" choices among tied partitions are now resolved
structurally.

**Out of scope (deferred):** Paper-faithful state-tie resolution per
Albantakis et al. 2023, Eq. 12 + S1 Text (max unnormalized `φ_s` among
states tied at max intrinsic information). That is a substantive
correctness change — it would replace the cruelest-cut convention and
shift which distinctions survive congruence filtering in downstream
CES — and is tracked separately on the ROADMAP.

## Why this matters

- `test_sia_big_subsys_all_complete_{sequential,parallel}` in
  `test/test_big_phi.py` are xfailed today because the fully-connected
  5-node substrate produces tied cause/effect states and tied partitions.
  Re-running picks different "canonical" representatives across runs;
  the SIA's `__eq__` (which compares the full structure, not just `phi`)
  fails.
- Non-binary units (next in the schedule after this) multiply the number
  of candidate states quadratically and will exacerbate the tie surface.
  Tightening determinism before that lands keeps the test surface stable.
- The fixture at `test/data/sia/big_subsys_all_complete.json` was captured
  under whichever ordering happened at fixture-write time. Any consumer
  comparing full SIA structure (golden tests, downstream caches, JSON
  round-trip) is affected.

## Source of non-determinism

### State-ties at `intrinsic_information` are not the load-bearing source

`pyphi/core/repertoire_algebra.py:586-602` returns `ties[0]` — the
arbitrary first-encountered tied `StateSpecification`. This propagates
into `system_intrinsic_information` (`pyphi/formalism/iit4/__init__.py:62-105`)
which packages a `SystemStateSpecification` with that representative per
direction. The full tie set is preserved on `.ties`.

However, this initial choice is *irrelevant to the final SIA*:

1. `integration_value` (iit4 line 419-457) iterates `specified.ties`
   for every partition and picks the cruelest tied spec — independent
   of which spec was named canonical.
2. `evaluate_partition` records the cruelest spec as `MIP.cause.specified_state`
   and `MIP.effect.specified_state`.
3. `resolve_system_state` (iit4 line 206) overwrites
   `system_state.cause/effect` with those MIP RIA `specified_state`s.

Once the MIP is fixed, the post-resolve canonical state is the
cruelest-at-MIP — deterministic by construction. The initial `ties[0]`
is overwritten and never escapes the SIA boundary.

### MIP partition ties — IIT 3.0 and IIT 4.0 — are the actual problem

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

### Why downstream-CES considerations don't apply here

`system_state` flows into `pyphi.models.distinction.Concept.resolve_congruence`
(distinction.py:199-225) which filters each concept's `state_ties +
purview_ties` by congruence against the system state. Two different
tied system states project to different sub-tuples on a concept's
purview, so congruence filtering picks different mice. The CES bag of
distinctions can shift.

This is real — but it is *already* true under the existing
cruelest-cut convention. This spec keeps that convention and only
makes the *MIP* selection deterministic; the post-resolve `system_state`
is still the cruelest-at-MIP. CES composition is unchanged from
current behaviour on substrates where the MIP itself was unique;
on substrates with tied MIPs, the canonical CES is now well-defined
where before it was a coin flip among CES bags that all share the
same `φ_s`.

A behavioural switch — replacing cruelest-cut with paper-faithful
max-`φ_s` state resolution per Albantakis 2023 Eq. 12 + S1 Text —
would shift CES composition. That work is deferred (see the
"Deferred follow-up" section below).

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

### Component 2 — `resolve_ties.sias()` and `PARTITION_LEX` strategy

Add to `pyphi/resolve_ties.py`:

```python
@phi_object_tie_resolution_strategies.register("PARTITION_LEX")
def _(m):
    return m.partition.lex_key()


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

The existing `NORMALIZED_PHI` and `NEGATIVE_PHI` strategies cover the
primary and secondary keys. Only `PARTITION_LEX` is new.

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
representative. `resolve_system_state` keeps its current signature and
behaviour — it writes the MIP's cruelest spec into `system_state`,
which is now deterministic because the MIP itself is.

### Component 5 — IIT 3.0 integration

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

## Deferred follow-up — paper-faithful state-tie resolution

Albantakis et al. 2023 Eq. 12 + S1 Text specifies that among states
tied at max intrinsic information, the canonical state is the one with
maximum unnormalized `φ_s`. PyPhi's current cruelest-cut convention
(`integration_value` at iit4 line 449-456) picks the *minimum*
unnormalized phi among ties — explicitly noted in-comment as
"PyPhi-specific, not paper-mandated".

Replacing cruelest-cut with paper-faithful max-`φ_s` is a substantive
correctness change, not a canonicalisation:

- It shifts which `specified_state` is recorded on the MIP RIA.
- That `specified_state` propagates through `resolve_system_state` into
  `SIA.system_state`.
- `SIA.system_state` is the filter passed to
  `Concept.resolve_congruence`, which selects one mice per direction
  per concept from its state/purview tie sets.
- Different filter states pick different mice → CES bag of distinctions
  can change → relations and CES-distance comparisons that ride on top
  inherit the change → `find_complex` rankings across subsystems can
  shift.

This is not in scope here. It is tracked separately on the ROADMAP and
needs its own brainstorm, including:

- Confirming the paper's algorithm with the paper open (Eq. 12 + S1
  Text), since multiple interpretations exist (max-phi at MIP vs. min-P
  max-c construction).
- Deciding the integration-value semantics: keep cruelest-cut for
  computing `φ_s` and use paper-faithful only for selecting the
  reported canonical state, or replace cruelest-cut throughout.
- Quantifying CES-composition drift on existing goldens.
- Planning golden regeneration with the new convention.

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

- `pyphi/resolve_ties.py` — add `sias()` and the `PARTITION_LEX`
  strategy.

**Modified:**

- `pyphi/models/partitions.py` — add `lex_key()` method on
  `_PartitionBase`.
- `pyphi/conf/formalism.py` — add `sia_tie_resolution` field on
  `IITConfig`.
- `pyphi/formalism/iit4/__init__.py`:
  - Replace the manual MIP loop (lines 718-735) with
    `resolve_ties.sias`. `resolve_system_state` is unchanged.
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

Single fragment, `changelog.d/sia-determinism.fix.md`:

```
Made `Substrate.sia()` results deterministic across runs by adding a
structural tie-break on partitions tied at the MIP minimisation key.
The new `sia_tie_resolution` config option exposes the ordering for
users who want to customise it; the default is
`["NORMALIZED_PHI", "NEGATIVE_PHI", "PARTITION_LEX"]`. No change to
phi values, the cruelest-cut convention, or which states are recorded
as canonical on substrates without tied MIPs.
```

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Captured golden fixtures don't match new canonical MIP ordering | Medium | Regenerate affected fixtures via existing entry point; document in commit |
| `lex_key()` byte comparison is slow for large `cut_matrix` | Low | Tertiary key — only invoked when primary+secondary tied; n is small (n≤8 typical) |
| New `PARTITION_LEX` strategy unintentionally applied to mechanism-level resolvers | Low | Strategy is generic, but only `sia_tie_resolution` config defaults to using it; mechanism-level configs unchanged |
| Future paper-faithful state-tie work conflicts with this commit | Low | Cruelest-cut path untouched; deferred work is orthogonal and rewrites different code (`integration_value` + `resolve_system_state`) |

## Effort estimate

~1 day of focused work. All edits are mechanical adaptations of
existing patterns (`resolve_ties.{states,partitions,purviews}` for the
new `sias`, the existing `mip_tie_resolution` config layout for
`sia_tie_resolution`). Fixture regeneration adds a small notebook-run
step.
