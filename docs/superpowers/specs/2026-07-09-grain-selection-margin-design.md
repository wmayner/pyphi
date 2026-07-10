# Grain selection margin on `Complex` — design

**Date:** 2026-07-09
**Status:** Approved design, pending implementation plan
**Depends on:** complex unification (landed; spec `2026-07-09-complex-unification-design.md`)
**Relates to:** selection-margin reporting (landed for SIA partition/state selection; spec `2026-07-07-substrate-parameter-landscapes.md` §7)

## Goal

Report how decisively each complex won the exclusion competition: the gap
in φₛ between a complex and the best overlapping rival it beat, plus an
`effectively_tied` flag. This extends the selection-margin family (already
landed for partition and state selection on the IIT 4.0 SIA) to the third
selection in the theory — exclusion among overlapping candidate systems,
which at the macro door is the grain competition.

## Background and constraints

- Both doors already build `pyphi.models.complex.Complex` objects carrying
  `excluded: tuple[ExcludedCandidate, ...]` — every overlapping candidate
  that was not itself accepted, with its φₛ. The margin is therefore a
  **derived property**, computable from data already in hand at both the
  micro door (`pyphi.substrate.complexes`) and the macro door
  (`pyphi.macro.complexes`). No cascade changes, no new plumbing, and
  serialization is free: `excluded` round-trips, and the property
  recomputes.
- Because condensation is recursive, an excluded candidate in a complex's
  records may carry **higher** φₛ than the complex itself (it was carved
  away by a different overlapping complex before this one was accepted —
  pinned as correct behavior in the condensation tests). **Ruling (Will,
  2026-07-09): the margin counts beaten rivals only.** Overlapping excluded
  candidates with higher φₛ ("shadows") stay visible in `.excluded` but do
  not enter the margin.
- **No `PyPhiFloat`.** The wrapper type is being removed in parallel work.
  The new properties use plain `float | None` and `pyphi.utils.eq` for
  precision-aware comparison, matching the idiom already used in
  `models/complex.py`.

## Design

### 1. Fix `exclusion_records` self-identification (bug fix)

`pyphi/condensation.py`, `exclusion_records`: the current code skips any
candidate whose sorted footprint equals an accepted footprint — a proxy for
"skip the accepted candidate itself". At the micro door footprints are
unique per candidate, so this is harmless; at the macro door **multiple
candidates share a footprint** (different grains over the same micro
units), so rival grains with the winner's exact footprint — its closest
rivals — silently vanish from the exclusion records.

Fix: skip only the accepted candidates themselves, by object identity
(`id()`), not by footprint equality. A candidate whose footprint happens to
coincide with an accepted complex's footprint but that was not itself
accepted is a genuinely excluded rival and appears in the records.

This is a user-visible behavior fix (exclusion records gain entries at the
macro door) and gets its own changelog fragment.

### 2. `Complex.exclusion_margin` property

`pyphi/models/complex.py`:

```python
@property
def exclusion_margin(self) -> float | None:
    """The gap in φₛ between this complex and the best overlapping rival
    it beat, or None when it beat none."""
```

Semantics:

- **Rivals** are the members of `self.excluded` whose φₛ is less than or
  precision-equal to the complex's own φₛ:
  `c.phi < phi or utils.eq(c.phi, phi)` with `phi = float(self.phi)`.
- Returns `None` when there are no rivals — the complex was unopposed, or
  every overlapping excluded candidate out-φ's it (the shadow case).
- Otherwise returns `max(0.0, phi - max(rival φₛ))`. The clamp absorbs the
  sub-precision negative gap a precision-equal rival can produce.

### 3. `Complex.effectively_tied` property

```python
@property
def effectively_tied(self) -> bool:
    """Whether the exclusion margin is within ``precision`` of zero."""
```

Returns `self.exclusion_margin is not None and
utils.eq(self.exclusion_margin, 0.0)`. A `True` value means an overlapping
rival's φₛ was within `precision` of the complex's own: either the complex
beat it through escalation within its tie clique (under IIT 4.0, the S1
Composition cascade), or the rival was removed in the same φₛ tier by
overlap with a different accepted complex. Either way the rival is present
in `.excluded` with precision-equal φₛ. Naming matches the SIA's
`effectively_tied`.

### 4. Display and pandas surfaces

- `Complex._describe`: when `exclusion_margin` is not `None`, add rows
  `Row("Selection margin", margin)` and
  `Row("Effectively tied", self.effectively_tied)` after the
  "Excluded candidates" row — wording mirrors the SIA card.
- `Complex._pandas_record`: add `exclusion_margin` (float or None) and
  `effectively_tied` (bool) fields.
- **No schema change**: both values derive from `excluded`, which already
  serializes; the properties recompute after a round-trip.
- **No `ComplexesResult` change**: `.ties` already surfaces exclusion
  failures (Φ-tied cliques), and margins are reachable via
  `result.complexes[i]` / `result.maximal_complex`.

### 5. Shadow demonstration notebook

A short jupytext-paired tutorial `docs/tutorials/recursive-exclusion.md`
(paired `.ipynb`, `md:myst,ipynb`, matching the existing tutorials; added
to the tutorials toctree — `docs/examples/` is legacy and not in the built
tree), demonstrating recursive exclusion and the higher-φₛ-shadow
phenomenon on the decaying-chain substrate (4 units, reciprocal couplings
0.6/0.3/0.15, φₛ landscape {A,B} > {B,C} > {C,D}):

- `substrate.complexes(state)` accepts {A,B} and {C,D};
- {C,D}'s `excluded` records contain {B,C} with **higher** φₛ — {B,C} was
  carved away by {A,B}, so it no longer exists and cannot veto {C,D}
  (recursive exclusion);
- {C,D}'s `exclusion_margin` ignores the shadow per the ruling above
  (reads the beaten-rivals gap, or `None` if {C,D} beat no one).

## Testing

- **Property unit tests** (`test/models/test_complex_model.py`), using
  stub SIAs and hand-built `ExcludedCandidate` records:
  - no excluded candidates → margin `None`, `effectively_tied` `False`;
  - one beaten rival → margin equals the φₛ gap;
  - precision-equal rival → margin `0.0`, `effectively_tied` `True`;
  - only higher-φₛ shadows → margin `None`, `effectively_tied` `False`;
  - mixed shadows and beaten rivals → margin measured against the best
    beaten rival only.
- **Identity-fix regression** (`test/test_condensation.py`): two candidates
  with identical footprints, one accepted — the loser appears in the
  winner's exclusion records.
- **Macro-door integration** (`test/macro/test_macro_search.py`): a grain
  search where a rival grain shares the winner's footprint → the rival
  appears in `winner.excluded` and the winner's `exclusion_margin` reflects
  it.
- **Micro-door integration**: on the decaying chain, {C,D}'s margin
  excludes the higher-φₛ {B,C} shadow.
- Full verification: `uv run pytest` with no path argument.

## Out of scope

- Margins between accepted complexes (they are disjoint and never compete).
- A tie surface at the micro door (`substrate.complexes` drops
  `failed_cliques`; `ComplexesResult.ties` covers the macro door, where the
  grain competition lives).
- Composition-level (Φ) margins for escalated cliques.
- `explain()` integration (the SIA's explain surface; `Complex` has no
  explain framework today).
