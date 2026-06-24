# B16 — First-class `Complex` marker type

- **Date:** 2026-06-15
- **Status:** Draft (awaiting user review)
- **Roadmap item:** B16 (Wave 2, pre-freeze)

## Context

`pyphi.substrate.complexes(substrate, state, …)` returns a `list[<SIA>]` — each
element is either an `IIT3SystemIrreducibilityAnalysis` (`pyphi/models/sia.py`)
or the IIT 4.0 `SystemIrreducibilityAnalysis` (`pyphi/formalism/iit4/__init__.py`).
Both subclass `cmp.OrderableByPhi` and expose `.node_indices` / `.phi`.

The exclusion postulate — among overlapping candidate systems, only the
φ_s-maximal one is a complex — is enforced *imperatively* inside the condensation
cascade (`_substrate_exclusion_cascade` / `_iit3_exclusion_cascade` in
`pyphi/substrate.py`) by a `covered` set and per-tier overlap cliques. The
overlapping candidates that lose are computed and then **discarded**. There is no
runtime object that says "this is a complex," carries which candidates it
excluded, or lets the exclusion postulate be checked as a named invariant rather
than trusted as an emergent loop property.

`maximal_complex()` returns `complexes()[0]`, or — when nothing is irreducible —
a **null SIA** (`node_indices=()`, φ=0). Downstream consumers read `.node_indices`
/ `.phi` directly off the result: `pyphi/actual.py` (the `major_complex` path)
does `substrate.maximal_complex(state).node_indices`; `test/test_complexes.py`
reads `cx[0].node_indices` and `float(cx[0].phi)`.

This is the gap B16 closes, and it advances 2.0 ship-criterion #1 ("every Greek
letter maps to a named runtime type"): the *complex* — the object a cause-effect
structure is properly the structure *of* — currently has no type.

## Goals

1. Introduce a first-class **`Complex`** value type (`pyphi/models/complex.py`)
   wrapping a SIA plus: `is_maximal`, the selecting `Substrate`, and the set of
   overlapping candidates excluded in its favor.
2. `complexes()` returns `tuple[Complex, ...]`; `maximal_complex()` returns a
   `Complex`.
3. Make the exclusion postulate a **named, checked invariant** —
   `validate.non_overlapping(...)` — rather than an emergent property of the
   condensation loop.
4. Preserve all existing math and numeric results exactly (additive wrapper).

## Non-goals

- **No change to the condensation math.** The tier walk, clique resolution,
  Composition escalation, and IIT 3.0 indeterminacy handling are untouched. The
  only new work in the cascade is *recording* what it already computes.
- **Macro is out of scope.** `pyphi/macro/search.py:complexes()` is a *separate*
  function that already returns its own named result type
  (`ComplexesResult` / `MacroSystem`). It is deliberately not touched by B16.
- No new exclusion semantics for IIT 3.0 multi-candidate cliques (still skipped
  as indeterminate); B16 only surfaces the existing outcome.

## Design

### The `Complex` type (`pyphi/models/complex.py`)

A frozen value type subclassing `cmp.OrderableByPhi`, with a **curated public
surface plus a `.sia` escape hatch** (no `__getattr__` fallthrough):

| Member | Meaning |
|---|---|
| `.sia` | the wrapped `SystemIrreducibilityAnalysis` (3.0 or 4.0) — escape hatch for any attribute not on the curated surface |
| `.substrate` | the selecting `Substrate` |
| `.is_maximal` | `bool` — `True` iff this is the global φ_s-max (the head of `complexes()`) |
| `.excluded` | `tuple[ExcludedCandidate, ...]` — overlapping candidates excluded in its favor (see below) |
| `.node_indices` | explicit delegation to `sia.node_indices` (hot accessor) |
| `.phi` | explicit delegation to `sia.phi` (hot accessor) |
| `order_by()` | delegates to `sia.order_by()` so `sorted()` / tie logic work |
| `__bool__` | `utils`-precision-aware `phi > 0` (defined on φ directly, not delegated, so it is robust for the null-object case) |
| `to_json` / `from_json` | serialization, mirroring the other `models/` types |
| `__repr__` / `_repr_html_` | `fmt`-based, consistent with `models/sia.py` |

Anything not in the curated surface is reached through `.sia`. This keeps the
type's surface explicit (the "named type" guarantee) while the two accessors that
existing callers actually use (`.node_indices`, `.phi`) keep working unchanged.

### `ExcludedCandidate` (lightweight exclusion record)

```text
ExcludedCandidate            # tiny frozen record
  .node_indices : tuple[int, ...]   # the excluded candidate's units
  .phi          : float             # its φ_s
```

The record holds plain values only — no back-reference to a `Complex` (that
would re-create the heavy object graph and a reference cycle). Per-complex
attribution is already implicit: a candidate appears in the `.excluded` of each
accepted complex whose units it overlaps, so "which complex excluded it" is the
complex you read `.excluded` from.

`Complex.excluded` holds **lightweight records, not full SIA object graphs.**
Rationale (see Performance analysis): the excluded candidates' full SIAs
(partition + RIAs + CES) are heavy and would be kept alive for as long as the
`complexes()` result is held — a needless lifetime extension for code that sweeps
many states/substrates. The `(node_indices, phi)` record is tens of bytes,
captures *what* was excluded and *why* (the φ ordering that the postulate turns
on), and is exactly what makes the exclusion postulate introspectable. The full
excluded SIA is deterministic and recomputable if ever needed.

`.excluded` records **every** irreducible candidate excluded in favor of this
complex — both the lower-φ_s candidates dropped by the `covered`-set overlap
filter and the same-tier clique tie-losers. Attribution (a candidate may overlap
more than one accepted complex) is by overlap of node indices; a candidate
appears in the `.excluded` of every accepted complex whose units it overlaps.
This adds only O(candidates × accepted) trivial set-intersections to the cascade.

### Where wrapping happens (`pyphi/substrate.py`)

`complexes()` keeps the identical cascade and, at the end, wraps each accepted
SIA into a `Complex`. The internal helpers (`_substrate_exclusion_cascade`,
`_iit3_exclusion_cascade`) gain a small step that records, per accepted complex,
the `ExcludedCandidate` records for the candidates it excluded — values they
already compute (the `covered`-set drops and the clique losers) but currently
throw away. `is_maximal=True` is set on the first element (the φ_s-max), `False`
on the rest. The function returns `tuple[Complex, ...]`.

`maximal_complex()` returns the head of `complexes()` (with `is_maximal=True`),
or — when nothing is irreducible — a **null-object `Complex`** wrapping the
existing null SIA (`node_indices=()`, φ=0, `is_maximal=True`, `excluded=()`).
Because `Complex.__bool__` is `phi > 0`, the null complex is falsy:
`if substrate.maximal_complex(state): …` reads as "is there a complex?" while
`substrate.maximal_complex(state).node_indices` still returns `()` without
raising. This matches the codebase's pervasive null-object pattern
(`NullCauseEffectStructure`, `_null_sia`, `NullCut`) and avoids the
`AttributeError` footgun that returning `None` would introduce in `actual.py`
and user code.

### `validate.non_overlapping(...)`

Add to `pyphi/validate.py`:

```text
non_overlapping(complexes) -> bool
    # returns True if the complexes' node-index sets are pairwise disjoint;
    # raises (consistent with the other validators) otherwise.
```

Called as an always-on postcondition at the end of `complexes()`. It is O(k²)
over a handful of complexes, so it is effectively free, and it converts the
exclusion postulate from a trusted loop property into a named invariant that is
*checked* on every call. The cascade already guarantees disjointness by
construction, so this is a defensive "prove it holds" assertion, not a behavior
change.

### Serialization, ordering, display

- `to_json` serializes `sia`, `is_maximal`, `excluded`, and the substrate
  reference using the existing `jsonify` machinery; `from_json` reconstructs.
  `ExcludedCandidate` gets a trivial `to_json`/`from_json`.
- Ordering (`order_by`) delegates to the wrapped SIA, so a `tuple[Complex, …]`
  sorts identically to the underlying SIAs.
- `__repr__` / `_repr_html_` reuse `fmt` helpers, surfacing `is_maximal`, φ_s,
  node labels, and an exclusion summary.

## Performance analysis (exclusion-set memory)

Measured on the IIT 4.0 default path:

| Net | n | `possible_complexes` | irreducible | per-SIA | all-retained |
|---|---|---|---|---|---|
| basic | 3 | 7 | 3 | ~4.6 KiB | 13.4 KiB |
| propagation_delay | 9 | 511 | — | — | intractable (does not return) |

Two findings drive the `ExcludedCandidate` decision:

1. The irreducible candidates are **already all materialized** in `sorted_sias`
   during condensation, so retaining them would not raise *peak* memory — it
   would only **extend their lifetime** past the function return (today they are
   GC'd immediately).
2. `complexes()` is only computable in a small regime (n=3 → 7 candidates;
   n=9 → 511 candidates already hangs). So the worst *reachable* full-SIA
   retention is single-digit MiB, in a computation that already took minutes.

Peak memory is therefore not the concern; the **lifetime-extension** is (a user
accumulating `complexes()` results across a sweep would keep every irreducible
SIA's full graph alive). Storing lightweight `(node_indices, phi)` records
removes that growth entirely while preserving the useful "all overlapping
excluded" semantics.

## Testing (`test/test_complexes.py`, both formalisms)

- **Regression:** existing `.node_indices` / `.phi` assertions must still pass
  through the wrapper (drop-in compatibility guard).
- `is_maximal`: only the head of `complexes()` is maximal; the rest are not.
- `excluded`: contents verified on the `dual_and_xor` two-tier cascade (the
  single-node candidates excluded by the two 2-node complexes) and on the
  `s`-fixture single-complex case (the two overlapping lower-φ candidates).
- `maximal_complex` null-object case: `bool(...) is False`, `.node_indices == ()`,
  `.phi == 0` when no candidate is irreducible.
- `validate.non_overlapping`: returns `True` on real `complexes()` output; raises
  on a hand-built overlapping pair.
- `to_json` round-trip for `Complex` and `ExcludedCandidate`.

## Risks / mitigations

- **Return-contract change** (`list[SIA]` → `tuple[Complex, …]`): pre-freeze and
  intended. The curated surface keeps `.node_indices` / `.phi` working;
  `actual.py`'s `major_complex` path is exercised by the test suite. Any site
  reading a SIA-only attribute off a complex migrates to `.sia.<attr>` (audited
  during implementation).
- **Two SIA shapes** (3.0 vs 4.0): `Complex` wraps either via the existing
  `_sia_node_indices` accessor pattern; no formalism branching in the type.

## Roadmap update

On landing, flip B16 to ✅ in the ROADMAP.md Status Dashboard and the Wave-2
"Remaining 2.0 Work" entry, and note that it advances ship-criterion #1.
