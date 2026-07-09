# Complex unification — one exclusion semantics, one winner type

**Project:** unify the micro and macro complex-identification surfaces around a
single condensation core. Today `pyphi.substrate.complexes` (the `Complex`
surface) and `pyphi.macro.complexes` (the grain-search driver) implement
**different exclusion semantics** and return different types. This project
fixes the macro semantics (a correctness fix, not a refactor), shares the one
cascade between both drivers, and makes `Complex` the winner type everywhere.
First step of the grain-discovery exposure work
(`docs/superpowers/specs/2026-07-07-grain-discovery.md` §4 is the performance
side; this is the API/correctness side).

**Sources of truth:** Marshall, Albantakis, Tononi 2023 (*System Integrated
Information*), §2.4 and Appendix C Algorithm A1; Albantakis et al. 2023
(IIT 4.0), Exclusion ("assessed recursively") and the S1 tie-resolution
supplement; Marshall et al. 2024 (*Intrinsic Units*), Eq. 8 discussion and
Eq. 19; the maintainer's ruling in this project (2026-07-09): **the recursive
reading is canonical**.

---

## 1. The defect being fixed

The two shipped drivers disagree about what a complex is.

- **Micro** (`pyphi/substrate.py`, the B16 surface): a descending-φₛ tier
  walk — accept the top candidate, drop candidates overlapping an *accepted*
  complex, continue; ties within a tier escalate up the S1 cascade to
  Composition (big Φ) before failing exclusion. This is Marshall 2023's
  Algorithm A1 plus S1.
- **Macro** (`pyphi/macro/search.py`): the literal Eq. 19 predicate — a member
  of P(u) is a complex iff it strictly beats every overlapping member,
  *whether or not that member is itself excluded*. No recursion; ties are
  reported and mutually excluded with no escalation.

These diverge on chains. Demonstrated on a 4-unit substrate with decaying
reciprocal coupling (0–1 at 0.6, 1–2 at 0.3, 2–3 at 0.15, self 0.05, base
0.05, state all-OFF, `iit4_2023` preset):

| candidate | φₛ |
|---|---|
| {A,B} | 0.3199 |
| {B,C} | 0.1041 |
| {A,B,C} | 0.1000 |
| {A,B,C,D} | 0.0564 |
| {B,C,D} | 0.0546 |
| {C,D} | 0.0371 |

The micro tier walk condenses this to **{A,B} and {C,D}**; the macro driver,
run over the identical candidate space (`SearchBounds(max_depth=0)`, identity
units), returns **{A,B} only** — {C,D} is vetoed by {B,C} and {B,C,D}, every
one of its vetoers itself excluded by {A,B}. Under the literal reading the
C–D region's irreducible cause-effect power belongs to no complex, orphaned
by candidates that do not exist.

**Ruling (maintainer, 2026-07-09): the recursive reading is correct.**
Exclusion is downstream of maximal existence: a candidate excluded by an
accepted complex does not exist and has no standing to exclude anything else.
Textual support: Marshall 2023 §2.4 + Algorithm A1 ("recurse on the
remainder"); IIT 4.0 Exclusion ("assessed recursively") and Conclusion
("condenses into a set of disjoint complexes"); Marshall 2024 §2.2.1 itself
("first-maximal complex, then second-maximal, and so on"). The literal Eq. 19
predicate characterizes the *first* condensation layer — every literal winner
is accepted by the tier walk, and the tier walk is exactly iterated Eq. 19 on
residual candidate pools — so the shipped macro driver is computing layer one
and stopping. Chain topologies are generic (any spatially extended substrate
with decaying coupling), not a corner case. Worth queueing for the 2024
paper's authors alongside the SP1/SP2 findings: Eq. 19's literal form and the
recursive procedure described for Eq. 8 diverge on chains.

## 2. Architecture: one cascade, two doors

A new module **`pyphi/condensation.py`** receives, essentially verbatim, the
private condensation machinery currently in `pyphi/substrate.py`:
`_phi_groups`, `_find_overlap_cliques`, `_accept`,
`_substrate_exclusion_cascade`, `_resolve_clique_by_big_phi`,
`_iit3_exclusion_cascade`, `_resolve_clique_iit3`, and `_exclusion_records`.
`substrate.py` keeps its public functions (`complexes`, `maximal_complex`,
`irreducible_sias`) as thin callers; `macro/search.py` becomes the second
consumer. The name follows the theory's own language ("the universe condenses
into disjoint complexes") and the existing "condensation cascade" wording.

The cascade operates on a small internal candidate record rather than raw
SIAs, because the doors disagree about what a SIA's `node_indices` means
(macro SIAs index macro units of the synthetic substrate):

```python
@dataclass(frozen=True)
class Candidate:
    footprint: frozenset[int]      # micro units — the currency of overlap
    sia: Any                       # micro or macro SIA; φ ordering via order_by
    system: Callable[[], System]   # provider for Composition escalation
```

- `footprint` is what Eq. 19 compares (`U^S ∩ U^S′`) and what
  `validate.non_overlapping` and exclusion records key on; for micro
  candidates it equals the SIA's node indices, for macro candidates it is the
  union of the units' micro constituents.
- `system` replaces the current `_big_phi_of_sia` reconstruction
  (`System.from_substrate(substrate, state, indices)`), which cannot rebuild
  a macro candidate. The micro door passes a `from_substrate` closure; the
  macro door passes the already-memoized `MacroSystem`. Escalation calls
  `candidate.system().ces().big_phi` as today.

Semantics carried over unchanged, now stated in the module docstring:

- Tier walk over φₛ descending, precision-aware tiers (`utils.eq`).
- Within a tier, drop candidates overlapping an accepted complex; group
  survivors into overlap cliques; single-member cliques accept directly.
- Multi-member cliques escalate to Composition (IIT 4.0) via
  `resolve_ties.resolve_complex_tie` with the existing
  `max_escalation_level="Composition"` budget; a clique whose Φ also ties
  fails exclusion — its members are removed but **their units stay
  available** to lower-φ candidates in later tiers (Marshall 2023: "remove
  both, continue recursively").
- IIT 3.0 branch moved verbatim, still reachable only from the micro door.
- The cascade is sequential post-processing of already-evaluated candidates,
  so the parallel ≡ sequential invariant is untouched by construction (and
  asserted by test anyway).

## 3. Tie escalation with the fingerprint dedupe

The macro door adopts the S1 escalation, with content-fingerprint dedupe
built into the clique resolver rather than bolted on:

- Within a tied clique, members are grouped by system content fingerprint
  (`System._fingerprint` / `MacroSystem._fingerprint` — label-free digests of
  exactly the kernel inputs, P9.5). Big Φ is computed **once per distinct
  fingerprint** and shared across the group.
- A clique whose members all share one fingerprint therefore skips the CES
  computations entirely: identical kernel inputs ⇒ identical Φ, certified at
  bit level. This is the common case in grain sweeps, where symmetric
  substrates produce tied grain-variants that are mirror descriptions of the
  same structure.
- Mixed-fingerprint cliques still run genuine Φ comparisons, so coincidental
  φₛ ties resolve exactly as at the micro level.

The dedupe applies to the micro door too (same resolver); it is a pure
optimization there with identical results. Deeper isomorphism skips
(macro-level canonical forms; micro automorphism correspondence between
candidates' unit structures) are **out of scope** — a follow-up optimization
with its own spec, building on `pyphi/automorphism.py`.

## 4. Type unification

**`Complex`** (`pyphi/models/complex.py`):

- New attribute `units: tuple[MacroUnit, ...] | None` — the winner's macro
  unit structure; `None` for micro winners.
- `node_indices` means the **micro footprint** uniformly. For micro winners
  this is unchanged (the SIA's indices); for macro winners the constructor
  receives the footprint explicitly (the SIA's own indices are macro-unit
  positions and must not leak into overlap semantics).
- `substrate` is the substrate condensation was run over — the micro
  substrate in both cases. `phi`, `is_maximal`, ordering, display, pandas all
  inherit unchanged.

**`ExcludedCandidate`**: same `units: tuple[MacroUnit, ...] | None` addition;
`node_indices` is the micro footprint.

**`ComplexesResult`** (`pyphi/macro/search.py`):

- `.complexes` becomes `tuple[Complex, ...]` (each wrapping the macro SIA and
  carrying its `excluded` records), ordered by φₛ descending with
  `is_maximal` on the first.
- `.records` unchanged (every evaluated system with its φₛ — the sweep's
  inspectability is a feature and stays).
- `.ties` narrows to cliques that still tied **after** Composition
  escalation, and its shape changes from pairs to cliques:
  `tuple[tuple[MacroSystem, ...], ...]`, one inner tuple per failed clique
  (members are the candidate systems, not `Complex` objects — they were not
  accepted). Docstring updated accordingly.
- New `maximal_complex` property mirroring the micro door's null-object
  behavior (falsy `Complex` with empty units when nothing is irreducible).

**Serialization:** `Complex` and `ExcludedCandidate` round-trip through
`pyphi.serialize` today; the new `units` field must round-trip as well.
`MacroUnit` is a frozen dataclass of ints/tuples (constituents may nest
`MacroUnit`s), so this is expected to be schema plumbing in
`pyphi/serialize/schema.py` + `convert.py`, but it is a task with its own
test, not an assumption.

## 5. Guards

- **IIT 3.0 + macro is rejected eagerly.** The search drivers —
  `complexes`, `intrinsic_units`, `valid_systems`, `is_intrinsic_unit`,
  `competing_systems` — raise under
  `config.formalism.iit.version == "IIT_3_0"` (matching the B13 eager-reject
  pattern) instead of silently running an uncertified combination.
  `MacroSystem` construction itself is not guarded (it is
  formalism-independent up to `sia()`); the 3.0 cascade branch remains
  micro-only.
- Out of scope for this project: the `grains=` parameter on the substrate
  driver, `analyze()` integration, grain margins, display cards for
  `ComplexesResult`/`UnitVerdict`, isomorphism skip levels 1–2, and the
  grain-construction cache (separate plan, independent of this API work).

## 6. Verification

The macro semantics change is a **fix** and the changelog fragment says so
plainly (the literal Eq. 19 implementation computed only the first
condensation layer; complexes on chain topologies were missing from the
result).

1. **Chain regression (the test that would have caught this).** The decaying
   chain substrate above, pinned through *both* doors:
   `pyphi.substrate.complexes` → {A,B}, {C,D}; `pyphi.macro.complexes` at
   `SearchBounds(max_depth=0)` → the same two complexes as `Complex` objects
   with matching φₛ. Plus the identity assertion that both doors agree on
   this candidate space in general (a small Hypothesis sweep over random
   4-unit substrates comparing the two doors at `max_depth=0`).
2. **Macro goldens re-verified.** SP2/SP3 goldens re-run: Example 1 (winner
   spans the substrate — no remainder to recurse on) and the bu verdict
   ({A}, {B} disjoint) are expected byte-stable. Any golden that moves is
   investigated and understood before regeneration — no deferred
   confirmation.
3. **Tie path.** The `tie_substrate` mirror-twin clique pinned for outcome
   (tie recorded, both excluded, walk continues) and for shadow equality:
   result with the fingerprint dedupe ≡ result with escalation forced on
   every clique member.
4. **Exclusion invariants.** Property tests over sweep results: accepted
   complexes are mutually footprint-disjoint (`validate.non_overlapping`);
   every accepted complex beats or Φ-outranks every overlapping candidate
   that was *not itself excluded earlier*; every `ExcludedCandidate` record
   references an accepted complex with greater-or-equal φₛ.
5. **Parallel ≡ sequential** over the new macro path (N2 extension).
6. **Full-suite gate.** `uv run pytest` with no path argument (doctest sweep
   included) before completion is claimed.

## 7. Files

- `pyphi/condensation.py` — new; cascade machinery moved from
  `pyphi/substrate.py`, `Candidate` record, fingerprint-deduped clique
  resolver.
- `pyphi/substrate.py` — thin callers; ~200 lines removed.
- `pyphi/macro/search.py` — `complexes()` consumes the shared cascade;
  `ComplexesResult` changes; IIT 3.0 guard.
- `pyphi/models/complex.py` — `units` on `Complex`/`ExcludedCandidate`;
  footprint-explicit construction.
- `pyphi/serialize/schema.py`, `pyphi/serialize/convert.py` — round-trip the
  new fields.
- `test/` — chain regression, both-doors equivalence, tie/shadow tests,
  invariant properties; existing macro golden suite re-run.
- `changelog.d/` — one `fix` fragment (macro condensation semantics), one
  `change` fragment (`ComplexesResult.complexes` type; `Complex.units`).
- `ROADMAP.md` — dashboard row for this project; note the upstream Eq. 19
  question in the intrinsic-units project entry.
