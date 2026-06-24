# Paper-faithful state-tie resolution

## Summary

Replace PyPhi's "cruelest-cut" state-tie convention with the rule prescribed
by Albantakis et al. 2023 ("IIT 4.0"): among cause/effect states tied at
maximal intrinsic information, the system specifies the one that maximizes
the system's integrated cause/effect information φ_c/e. This is a
substantive correctness change — it affects reported φ_s values, recorded
system states, distinction congruence filtering, the relation set, and
cross-subsystem CES distance comparisons (e.g., `find_complex` rankings).
Sibling work [P11.95a, landed in commits 2a163b6f..b30e9a47] addressed
*partition*-tie determinism; this spec covers the orthogonal *state*-tie
rule.

## Background

### Current behavior: cruelest-cut

`pyphi/formalism/iit4/__init__.py:419-458` (`integration_value`) iterates
the tied states carried on a `StateSpecification` from
`intrinsic_information`. For each partition θ it picks the tied state that
*minimizes* signed φ — the state the partition "hurts most". The in-source
comment labels this "PyPhi-specific, not paper-mandated".

The selected RIA's `specified_state` flows through
`SystemIrreducibilityAnalysis.resolve_system_state` into
`SIA.system_state`. Downstream, `Concept.resolve_congruence`
(`pyphi/models/distinction.py:199-225`) filters each concept's
`state_ties + purview_ties` to those congruent with the system state. So
which tied state wins propagates into the distinction bag, then relations,
then cross-subsystem CES comparisons.

### Paper's rule

Two passages set the tie-break rules:

- **State ties (parenthetical at Eqs. 19-20, p. 17):** "By the principle of
  maximal existence, if two or more cause-effect states are tied for
  maximal intrinsic information, the system specifies the one that
  maximizes φ_c/e."
- **Partition ties (commentary on Eq. 23, p. 18):** "If two or more
  partitions θ ∈ Θ(S) minimize Eq 23, we select the partition with the
  largest unnormalized φ_s value at θ'."

The partition-tie rule corresponds to the existing `NEGATIVE_PHI` secondary
key already wired into `sia_tie_resolution` by P11.95a. The state-tie rule
is the gap addressed here.

## Reading the state-tie rule

Three candidate readings produce the same outcome on perfectly symmetric
substrates but diverge in general:

| Reading | Rule | Recorded state |
|---|---|---|
| Cruelest-cut (current) | per partition, min over tied c → MIP minimizes over min-c | min-c at the system MIP |
| Per-partition flip | per partition, max over tied c → MIP minimizes over max-c | max-c at the system MIP |
| **Per-state (paper)** | per tied c, MIP-for-c via min over θ; pick c with max φ_d(c) | the globally maximizing c |

The paper says "the system specifies *the one* that maximizes φ_c/e", and
φ_c/e in Eq. 22 is the system-level integrated cause/effect information —
the value at the MIP, *not* per-partition integration. This pegs the
comparison to **max over tied c of min over θ of integration(θ; c)**: each
tied state has its own MIP-search, and the canonical winner is the state
whose own-MIP value is largest. This is the per-state reading.

The per-partition flip is a tempting one-line change but it isn't what the
paper says: at the MIP it would record whichever c won at *that* partition,
which is not necessarily the globally maximal c.

## Algorithm

Replace the existing single-pass `sia()` (build-SIA-per-partition,
reduce-via-MIP) with a three-phase computation. The integration compute is
unchanged: still O(|partitions| × (|cause-ties| + |effect-ties|)) RIA
evaluations.

### Phase 1: Build integration tables (parallel)

For each partition θ and each direction d ∈ {cause, effect}, evaluate
integration for *all* tied states in `system_state[d].ties`:

```
table[(d, θ, c)] = integration_value_for_state(d, system, θ, c, system_measure)
```

Per-partition parallel task returns a `PartitionResult` carrying the
per-direction tied-state table for that θ, not a single SIA. The map step
preserves the existing partition-parallelism; only the reduce step
changes.

### Phase 2: Per-direction state selection (serial, cheap)

For each direction d:

```
per_state_phi_d[c] = min over θ in partitions of table[(d, θ, c)].signed_phi
c_d* = argmax over c in ties of per_state_phi_d[c]
```

Lex tie-break on `c.state` for residual determinism (matches the spirit of
P11.95a's `PARTITION_LEX` tertiary key on the partition side).

If `ties` has a single element (the no-tie case), this collapses to
identity: c_d* = the sole element.

### Phase 3: System MIP search with fixed canonical states (serial)

With `c_c*`, `c_e*` fixed, assemble a candidate SIA for each θ from the
pre-computed table entries:

```
for θ in partitions:
    candidate = build_sia_from_rias(
        partition=θ,
        cause=table[(CAUSE, θ, c_c*)],
        effect=table[(EFFECT, θ, c_e*)],
        ...
    )
```

Apply the existing `resolve_ties.sias` resolver (chain:
`NORMALIZED_PHI → NEGATIVE_PHI → PARTITION_LEX`) to pick the MIP.

`resolve_system_state` continues to back-propagate the winning RIA's
`specified_state` into `SIA.system_state.cause/.effect` without
modification — because those RIAs now carry c_c* / c_e* (the paper-faithful
canonical states), the downstream `Concept.resolve_congruence` filter
receives the correct system state.

## Code surface

### Modified

- **`pyphi/formalism/iit4/__init__.py`**
  - `integration_value(direction, system, partition, system_state, *, system_measure)`:
    collapse the tied-state iteration; this function (or its rename to
    `integration_value_for_state`) becomes single-state.
  - `evaluate_partition(partition, system, system_state, *, system_measure, directions=None)`:
    refactor to accept a fixed `(c_c, c_e)` state pair instead of carrying
    the full tie-set via `system_state`. The integration step uses the
    fixed states; the cap branch (Eq. 23) still keys off
    `system_measure.name == "INTRINSIC_INFORMATION"` and reads the
    direction's `intrinsic_information` (invariant under tie choice).
  - `sia(system, ..., system_measure, specification_measure, ...)`:
    restructure into the three phases above. Phase 1 dispatches the
    parallel partition sweep with a new per-task return shape. Phase 2 is
    serial. Phase 3 reuses `resolve_ties.sias`.
  - Add `_select_specified_state(direction, ties, integration_table) -> StateSpecification`:
    pure helper for Phase 2; takes the table built in Phase 1, returns
    c_d*.

- **`pyphi/parallel`** (only if Phase 1's parallel signature requires it):
  per-task return type becomes the per-partition tied-state table rather
  than a single SIA. The existing MapReduce harness should accept a list
  of records; the reduce step is replaced by serial post-processing in
  Phase 2/3 (i.e., reduce collects the list, no MIP-min inside the
  reducer).

### Unchanged

- `SystemIrreducibilityAnalysis.resolve_system_state` — still pulls
  `specified_state` from `self.cause` / `self.effect` and writes
  `system_state.cause / .effect`. Under the new algorithm, those
  `specified_state` fields already carry c_c* / c_e*.
- `pyphi/core/repertoire_algebra.py:560-602` `intrinsic_information` —
  still returns `ties[0]` with the full tie-set on `.ties`. Consumer
  semantics change; producer is invariant.
- `pyphi/models/state_specification.py:127-133` `StateSpecification.is_congruent` —
  unchanged.
- `pyphi/models/distinction.py:199-225` `Concept.resolve_congruence` —
  unchanged. Receives a correct `system_state` and filters accordingly.

### Removed

- The cruelest-cut comment block at
  `pyphi/formalism/iit4/__init__.py:449-456`. The new algorithm's
  selection rule is structural (Phase 2's argmax helper); a brief
  citation-only comment may replace it, but no design narrative.
- No config knob. Cruelest-cut is removed entirely. (Saved memory:
  "no back-compat shims for unpushed dev work" — 2.0 branch is unreleased.)

## Tests

### New unit tests

- **Per-state selection determinism.** Construct a small substrate
  where two states tie at max ii but yield different per-state-MIP φ
  values. Verify `c_d*` is the argmax, not the cruelest-cut argmin.
- **Interpretation 2 ≠ Interpretation 1.** Construct (or reuse an
  existing) substrate where the per-partition max-c is not the same as
  the per-state max-min-θ winner; verify the algorithm picks the latter.
- **Lex tie-break.** When two tied states share both ii and per-state φ,
  verify the lex-smaller state tuple wins.

### Existing tests

The determinism property test landed in P11.95a
(`test_sia_is_deterministic_across_runs_{sequential,parallel}`) continues
to apply unchanged. No new property test needed.

### Goldens

Run the full golden suite under the new algorithm and identify drift.
Strong suspects (regenerated by P11.95a for partition-tie determinism;
likely affected again by state-tie correction):

- `test/data/sia/big_subsys_all_complete.json`
- `test/data/phi_structure/grid3.json`
- `test/data/phi_structure/rule154.json`
- `test/data/golden/v1/logistic3_k8_iit4_2023.json`

Each affected fixture must be inspected for legitimacy of the drift
before regeneration: same ii, shifted φ_s, plausible state selection,
unchanged where ties don't exist. Regenerate via the existing notebook
entry point (`test/IIT_4.0_make_jsons.ipynb`) with side-by-side
comparison to the pre-change fixture.

### Pre-existing failures unrelated

The slow-lane IIT 3.0 actual-causation failures
(`test_actual.py::TestActualCausationIIT30::test_causal_nexus`,
`::test_true_events`) predate this work and predate P11.95a; they remain
out of scope.

## Performance

Compute is identical to cruelest-cut: same number of integration
evaluations (|partitions| × (|cause-ties| + |effect-ties|)). The reduce
step shifts from "MIP-min over SIAs" to "collect records, then state
selection + MIP-min on assembled candidates". The post-processing is
O(|partitions| × |ties|) per direction (a few hundred floating-point
comparisons even on the symmetric substrates with high tie counts) —
negligible relative to integration computation.

## Determinism across runs

Same guarantees as P11.95a:

- Phase 1: per-partition tasks are independent; parallel reduction is
  order-independent (each task writes to a distinct table entry).
- Phase 2: deterministic — argmax with explicit lex tie-break.
- Phase 3: `resolve_ties.sias` with `PARTITION_LEX` tertiary key.

The slow-lane `test_sia_is_deterministic_across_runs_{sequential,parallel}`
property tests exercise this end-to-end and continue to gate.

## Forward compatibility: P12 non-binary units

Non-binary units multiply the per-direction tied-state count (state space
scales as `b^n` for `b` bits per unit, `n` units). The algorithm scales
linearly in tie count, so compute remains tractable. P11.95b should land
before P12 — locking semantics first reduces the surface that has to be
re-validated under non-binary.

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Golden drift is too widespread to validate by hand | Medium | Diff inspection: same ii on all candidates, only φ_s + recorded state should shift. Distinction/relation set changes audited per fixture. |
| Parallel signature change destabilizes the MapReduce harness | Medium | Phase 1 implementation incrementally — keep the per-task SIA return briefly, then switch to per-task tables only after the serial post-processing works in unit tests. |
| Per-state MIP search inside Phase 1 is computationally non-trivial under unanticipated tie cardinality | Low | Tie cardinality is bounded by `2^n` (binary) and known empirically (4 fixtures' max is single-digit ties per direction). Non-binary: defer revalidation to P12 spec. |
| Cause/effect coupling: paper's parenthetical is per-direction, but a stricter Cartesian-product reading exists | Low-Medium | Per-direction factorization matches PyPhi's existing per-direction independence in cruelest-cut and the paper's literal text ("φ_c/e", per-direction). If a stricter reading is later required, the Phase 2 helper can grow to enumerate the Cartesian product without changing Phase 1. |
| IIT 3.0 affected | None | IIT 3.0 has no system-level "specified state" — state-tie resolution is not a 3.0 concept. The P11.95a partition-tie work covered the 3.0 surface. |

## Out of scope

- Cartesian-product state-pair tie resolution (independent per-direction
  is consistent with PyPhi's existing factorization and the paper's
  per-direction phrasing).
- IIT 3.0 — no system-level state-tie surface.
- Removal or alteration of `intrinsic_information`'s tie-set production
  in `pyphi/core/repertoire_algebra.py` — the producer is correct; only
  consumer semantics change.

## Acceptance criteria

- All affected goldens regenerated and validated as legitimate drift (same
  ii, shifted φ_s and recorded state).
- New unit tests pass.
- P11.95a determinism property tests pass.
- Fast lane: 0 failures.
- Slow lane: 0 new failures (pre-existing IIT 3.0 actual-causation
  failures permitted).
- Pyright: 0/1 baseline. Ruff: clean.
- Changelog fragment under `changelog.d/`.

## Estimated effort

3-5 days post-spec. The bulk is in Phase 1 parallel-signature refactoring
and per-fixture golden inspection. Algorithm itself is mechanical.
