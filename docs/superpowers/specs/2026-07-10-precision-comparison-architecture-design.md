# Precision-comparison architecture: delete `PyPhiFloat`, consolidate tolerance at decision sites

**Date:** 2026-07-10
**Status:** Approved design (B22 follow-on)
**Branch:** `precision-architecture` (worktree)

## Problem

The B22 audit (five-agent sweep, 2026-07-09) found eight sites where φ/Φ/α
comparisons run on raw floats and can resolve on ~1e-16 summation noise
instead of detecting a tie. Root-cause analysis showed the bugs share one
mechanism: `PyPhiFloat` — a `float` subclass with tolerant `==`/`<`/`hash` —
**silently loses its type through arithmetic** (`-x`, `abs(x)`, `x + y`,
`x / 2` all return plain `float`), and several sites never wrap at all (AC
`alpha`) or explicitly downcast (`float(phi)` before a comparison).

Probing the abstraction itself showed it is unsound independent of those
bugs:

- Tolerant `==` is **non-transitive** (`a == b`, `b == c`, `a != c` for
  values spaced 0.6·tol apart) — approximate equality is not an equivalence
  relation.
- The induced order is **not total**: sorting the same four values in
  different input orders produced eight distinct results.
- Set/dict dedup over the tolerant hash is **insertion-order-dependent**
  (and, verified by grep, unused anywhere in the codebase).

The codebase already contains the sound pattern at ~34 sites: tolerance at
the **decision site** (`utils.eq` / `is_positive`), with object selection
through the `resolve_ties` cascades and structural tie-breaks (`lex_key`).
This design consolidates on that pattern and deletes the unsound type.

## Do we still need tolerance at all?

Re-derived from evidence, because the original justification is dead:

- **Dead:** measure-library noise. The tolerance mechanism dates to the
  pyemd C-library era (~1e-6 noise). B13's confirmation experiment showed
  POT's EMD noise floor is machine epsilon; B9 showed the GID /
  information-density primitives are well-conditioned to ≤ 1 ULP.
- **Alive #1 — cross-structure ties.** IIT produces genuine ties between
  different objects: the 2014 Fig 12 constellation has φ(C) = φ(AB) = 1/4
  from different mechanisms and repertoires. Different algebraic paths to
  the same value bit-differ (`0.5 − 1/3` vs `1/6`: 2.8e-17). No summation
  discipline fixes this — the computations share no intermediates. Exact
  comparison would manufacture a strict ordering inside what the theory
  says is a tie, and ties are first-class in IIT (tie sets; the P11.95e
  audit proved tie-breaks load-bearing for system φ).
- **Alive #2 — reassociation across candidates.** Mirror-isomorphic
  candidates' Φ differed by 5.6e-16 in the S1 condensation bug (live
  reproduction, 2026-07-09); the B5 pre-refactor oracle recorded 3.3e-16
  max reassociation drift.

**Charter:** tolerant comparison exists to detect ties between candidate
objects whose φ values are mathematically equal but computed through
different floating-point paths. It is not for papering over library noise.
Consequence: tolerance lives at decision sites only; values are plain,
exact floats. The default `precision = 13` survives: ~100× headroom above
the ~1e-15 noise floor, far below any real φ gap. Alternatives rejected:
exact rational arithmetic (dies on `log2`; GID values are irrational) and
canonical summand ordering (helps only permutation-symmetric twins, not
cross-structure ties).

## Architecture: two layers

### Layer 1 — `pyphi/numerics.py` (new): scalar predicates

The only tolerant primitives in the library. Reads
`config.numerics.precision` at call time (as `utils.eq` does today).

| Function | Origin | Semantics |
|---|---|---|
| `eq(x, y)` | moved from `utils` | equal within precision |
| `is_zero(x)` | new | `eq(x, 0.0)` |
| `is_positive(x)` | moved from `utils` | `x > 0` and not `is_zero(x)` |
| `is_nonpositive(x)` | moved from `utils` | unchanged |
| `positive_mask(array)` | new | elementwise `is_positive` boolean mask (vectorized) |
| `round_to_precision(x)` | new | names the scattered `round(x, config.numerics.precision)` idiom |

`utils` loses the moved predicates; its ~34 call sites are re-imported
mechanically. No re-export shims (2.0 is a clean break). The module
docstring carries the tolerance charter and the noise-floor derivation of
the default precision. `positive_part` stays in `utils` (a value operation,
not a comparison).

### Layer 2 — `pyphi/resolve_ties.py` (rebuilt core): object selection

The only selection engine. The generic `resolve()` is replaced by a
**tolerant lexicographic cascade** over key tuples:

1. For key component *i* (in order): compute the exact extremum over the
   surviving candidates.
2. Keep every candidate whose component *i* is `numerics.eq`-tied to the
   extremum (float components) or exactly equal (structural components:
   `lex_key` bytes, integers).
3. Recurse on the survivors with component *i + 1*.

Properties: order-independent (the extremum and the eq-cluster are both
permutation-invariant), and correct-by-construction for the `NEGATIVE_PHI`
bug class — strategy functions (`PHI`, `NEGATIVE_PHI`,
`NORMALIZED_PHI`, …) return plain floats, and clustering no longer rides on
the operand's type surviving arithmetic.

**Actual causation gains parity.** AC already has mechanism-level cascades
(`resolve_ac_partition_tie`, `resolve_ac_causal_link_tie`) — broken today
by raw-float clustering (`abs(r.alpha)` key; raw `max`/`==`). Both are
rebuilt on the new engine. What AC lacks entirely and gains:

- `resolve_ac_sia_tie` — a system-level cascade, consumed by
  `formalism/actual_causation/compute.py::_sia()` (the bare
  `reduce_func=min` map-reduce becomes materialize → cascade →
  `set_ties`, the IIT 3.0 `_sia_map_reduce` pattern) and by
  `actual.py::causal_nexus()` (replacing bare `max`/`sorted`).
- Tie bookkeeping (`.ties`) on AC results, which they never had.

### Value types become exact

- **`PyPhiFloat` is deleted** (`pyphi/data_structures/pyphi_float.py`
  removed; `data_structures/__init__.py` export dropped).
- **`DistanceResult` survives, re-based on `float`.** Verified severable:
  its metadata role (method/direction/state provenance, `values_array`,
  repr, serialization round-trip) is documented and user-facing; its
  tolerant-comparison role is the smell. It becomes
  `class DistanceResult(float)` carrying metadata kwargs, inheriting exact
  float comparisons. `min()` over instances still propagates the winner's
  metadata (plain `min` returns the object). The tolerant hash is unused
  anywhere (verified: no sets/dicts keyed on φ values).
- **All `.phi` / `.alpha` properties are plain `float`.** AC's `alpha`
  (never wrapped — one of the audit bugs) is now consistent with IIT's φ
  *by design*, with selection routed through cascades instead of value
  types.
- **`cmp.Orderable` / `order_by` dunders become exact total orders** —
  deterministic and safe for `sorted()` / display iteration. Anything that
  selects a winner routes through a cascade.
- **Result-type `__eq__` keeps its tolerant semantics** via `numerics.eq`
  (behavior unchanged; import updated).

## Audit-site fix map

| Site | Fix |
|---|---|
| `measures/distribution.py:1330` ii-differentiation filter (`> 0` admits 3e-16 surprisal from a certain node; corrupts Eq. 23 capped φ) | `numerics.positive_mask` |
| `resolve_ties.py` `NEGATIVE_PHI` / `NEGATIVE_NORMALIZED_PHI` (negation strips tolerance on the default MIP/SIA tie path) | eliminated by the cascade rebuild (clustering explicit, keys plain floats) |
| `resolve_ties.py:414` `abs(r.alpha)` key; `:474` raw `max`/`==` on alpha | rebuilt on the new engine |
| AC `order_by()` raw alpha → bare `min` MIP selection (`compute.py:638`), bare `max` nexus (`actual.py:906/918`) | `resolve_ac_sia_tie` + materialize-and-cascade; dunders stay exact by design |
| `macro/search.py:858` `phi=float(phi)` downcast before the candidate sort | `condensation.exclusion_cascade` takes over ordering: it clusters candidates into φ-tiers via `numerics.eq` (its existing `_phi_groups`) and is stable within a tier, so dispatch order is preserved among precision-tied candidates; the caller's raw-float pre-sort (and the downcast) are removed |
| `iit4/__init__.py:519` / `distinction.py:155` / `explanation.py:93` binding-direction raw `float() <=` | `numerics.eq` tie detection with an explicit `"tied"` outcome instead of an arbitrary side |
| `models/ces.py:203` `big_phi` (addition strips the type; safe only via a defensive re-wrap in `condensation.py`) | plain float property; consumers use decision-site helpers; the defensive re-wrap is removed |
| `distribution.py:1467/1495` PMI zero-guard; `distribution.py:1079` `approximate_specified_state` (Tier-3 SUSPECTs) | confirmation experiments first (below); fixed only if a witness exists |

## Serialization

`PyPhiFloatSchema` is deleted; `PhiSchema = float | DistanceResultSchema`
(msgspec supports primitive-plus-tagged-struct unions). Committed fixtures
carrying `pyphi_float` tags are enumerated and regenerated. The 2.0 format
is unreleased — clean break, no reader shim.

## Enforcement lint

`test/test_precision_lint.py`: an AST walk over `pyphi/` source flagging

- comparison operators (`<`, `>`, `<=`, `>=`, `==`, `!=`) where an operand
  is an attribute named `phi`, `alpha`, `big_phi`, `normalized_phi`,
  `signed_phi`, `sum_phi`, or `intrinsic_information`, and
- `min` / `max` / `sorted` calls whose arguments or `key` reference those
  attributes,

outside `pyphi/numerics.py` and `pyphi/resolve_ties.py`. Value-definitional
uses (e.g. `phi = min(cause, effect)` in `formalism/queries.py` — a
definition, not a selection between candidates) carry an inline waiver
comment `# numerics: exact — <reason>`; every exception is visible and
greppable. The lint runs in CI as an ordinary test.

## Migration strategy: strangler

1. **Phase 1 — `numerics` lands as a pure move.** New module + mechanical
   re-imports of the ~34 `utils` call sites. Zero behavior change; suite
   green; goldens byte-identical.
2. **Phase 2 — decision-site conversions, one subsystem at a time**
   (resolve_ties core → AC cascades → macro/condensation → binding-direction
   sites → measures fixes), goldens green between steps so any drift bisects
   to one conversion.
3. **Phase 3 — type deletion.** `PyPhiFloat` removed, `DistanceResult`
   re-based, annotations swept, serialization schema changed, fixtures
   regenerated.
4. **Phase 4 — lint on.** Waivers finalized; CI gate active.

Rejected: big-bang (un-reviewable, un-bisectable) and deletion-first
(99 simultaneous type errors, no behavioral signal).

## Testing & verification

**Golden policy.** Goldens byte-identical **except** at four enumerated
drift sites, each individually diffed, explained against the specific bug
it corrects, and deliberately regenerated (the k-ary-cut discipline):

1. AC MIP / causal-nexus selection (previously raw-float-decided),
2. macro tie ordering / `is_maximal` labeling on φ-tied disjoint candidates,
3. tie-set membership (raw comparison silently dropped precision-tied
   members),
4. the ii-differentiation value where a certain-node surprisal artifact was
   selected (a genuine φ-value correction under `INTRINSIC_INFORMATION`).

Any drift outside this set is a stop-the-line bug in the refactor.

**New tests.**

- `numerics` unit tests: predicate boundary semantics; `positive_mask` on
  the audit's `0.9999999999999998` reproduction; `round_to_precision`
  idempotence; doctests.
- **Cascade permutation-invariance property tests** (the invariant every
  audit bug violated): Hypothesis over random key tuples with injected
  sub-tolerance perturbations (~1e-15) — winner set and tie set identical
  under candidate-order permutation; perturbed twins always co-selected.
  Instantiated for each concrete cascade (MIP, SIA, AC partition, AC
  causal-link, AC SIA, complex).
- Noise-tie regression fixtures, one per fixed audit site (the S1
  symmetric-tie fixture generalizes).
- AC tie bookkeeping: `.ties` populated on genuine ties, singleton
  otherwise; `causal_nexus` surfaces tied transitions.
- **Tier-3 confirmation experiments** (don't-defer rule): drive the real
  `actual.py` probability plumbing on deterministic transitions in a seeded
  sweep to test whether a mathematically-zero probability ever reaches the
  PMI guard as a ~1e-16 residue; likewise for `approximate_specified_state`.
  Fix only on a concrete witness; otherwise record SAFE with the experiment
  cited.
- The lint test itself.

**Backstops.** N2 (parallel ≡ sequential) verifies schedule-independence
end-to-end. The perf-counter gate: AC's materialize-and-cascade replaces a
streaming min, so AC call counts shift and are repinned deliberately. B5
cross-formalism invariants stay green untouched.

**Recipe.** `uv run pytest` with no path argument at least once at the end
(doctest sweep; the `eq` doctests move to `numerics`), fast/slow parallel
lanes during iteration. Migration-guide entry (".phi is a plain float;
direct comparisons are exact; use `pyphi.numerics`") and changelog
fragments land with the final phase.

## Documentation updates

- `conf/numerics.py` docstring (references `PyPhiFloat` today).
- CLAUDE.md pitfalls section (`utils.is_zero` is documented but never
  existed; becomes `numerics.is_zero`, real).
- Migration guide addendum; changelog fragments per the `changelog.d/`
  convention.
- All new/edited docstrings follow the enforced NumPy final-state style.

## Out of scope

- Any change to tie-resolution *strategy semantics* (which tie-break wins)
  beyond making clustering precision-correct — the configured cascades keep
  their meaning.
- `automorphism.py`'s fixed `_ROUND = 12` canonicalization keys (a
  deliberate, config-independent technique; noted, not changed).
- Distribution-valued φ, interval arithmetic, or any richer uncertainty
  representation (see `pyphi.estimate` for the sampling approach).
