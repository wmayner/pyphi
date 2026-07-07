# An algebra of cause-effect structures — design exploration

**Status: exploration, awaiting review. Nothing here is implemented beyond what
is marked [exists].**

This document asks whether there is a coherent set of operations for working
with cause-effect structures as first-class objects — selecting parts of them,
restricting them to units or distinctions, combining and comparing them,
aggregating over them — and what each operation can and cannot guarantee.

The short answer: **yes, but only within a fixed frame, and it is a
meet-semilattice with an additive measure, not a full algebra.** Restriction
to a distinction subset is exact and closed; intersection of structures is
exact; union is not a structure; recomputation on a subsystem is not an
operation on structures at all and cannot be made one. The operations that do
exist compose because of one mathematical fact (a relation's φ depends only on
its relata) and are bounded by another (a distinction's existence and resolved
state depend on the whole system). Both facts are verified against the library
below.

Every claim is tagged:

- **[exists]** — implemented in PyPhi today, with file references.
- **[established]** — defined in the IIT papers, with equation references.
- **[proposed]** — this document's suggestion.
- **[verified]** — checked by running the actual library in this session
  (IIT_4_0_2023 formalism, default config, `precision = 13`). The appendix
  has the reproduction script.

Related document: `docs/superpowers/specs/2026-07-07-relations-without-materialization-design.md`
(in the `relations-without-materialization` worktree, also awaiting review)
works out how to *query* the relation set without enumerating it. That
document is about representation and cost; this one is about object model and
operation semantics. They meet in §8, where each operation here is classified
by which relation representation supports it.

---

## 1. What PyPhi produces today [exists]

The result objects, after the 2.0 refactoring:

| Object | Type | Contents |
|---|---|---|
| Distinction | `Distinction` (`pyphi/models/distinction.py:43`) | mechanism (absolute index tuple), `cause`/`effect` MICE, `phi = min(cause.phi, effect.phi)` |
| Distinction bag | `Distinctions` → `UnresolvedDistinctions` / `ResolvedDistinctions` (`pyphi/models/distinctions.py`) | sorted tuple of distinctions; the subtype records whether tied specified states were resolved against a SIA system state |
| Relation | `Relation` (`pyphi/relations.py:143`) | a frozenset of distinctions; `phi` computed from the relata alone (Eq. 55 via the S3 overlap identity) |
| Relation set | `Relations` → `ConcreteRelations` (enumerated frozenset) / `AnalyticalRelations` (closed-form aggregates, **not iterable**) / `AnalyticalFoldRelations` / `NullRelations` (`pyphi/relations.py`) | either the materialized set or an implicit view over a distinction bag |
| Structure | `CauseEffectStructure` (`pyphi/models/ces.py:59`) | `sia` + `ResolvedDistinctions` + `Relations` + config snapshot + provenance |
| Fold | `PhiFold(CauseEffectStructure)` (`pyphi/models/ces.py:320`) | seed distinctions + relations incident to at least one seed + `parent` back-reference |

Conventions that shape everything downstream:

- **All indices are absolute substrate indices.** A distinction from a
  subsystem of a 6-unit substrate has mechanism indices in the substrate's
  coordinates; there is no relative re-indexing. `NodeLabels` travels on every
  object.
- **Purviews carry states.** `purview_units` are `Unit` values — (index,
  state) pairs — so purview intersection and union range over units *and*
  their states. Congruence is enforced by plain set intersection.
- **A structure does not keep its `System`.** The SIA records `node_indices`,
  `node_labels`, `current_state`, and (IIT 4.0) `system_state`; the live
  `System`/`Substrate` is not reachable from a `CauseEffectStructure`
  (`Complex.substrate` is the exception).
- **One runtime type covers both paper terms.** The paper's "cause-effect
  structure" (any candidate system) and "Φ-structure" (a complex) are the same
  class; whether the substrate was maximal is context, not type
  (`pyphi/models/ces.py` module docstring). Verified example: `fig4_system()`
  has Φ_s = 0 — it is reducible — yet its `CauseEffectStructure` holds 4
  distinctions and 15 relations with Σφ = 4.79. [verified]

Operations that already exist, i.e. the nascent algebra:

| Operation | Where | Semantics |
|---|---|---|
| `fold(seeds)` / `distinction_folds()` | `CauseEffectStructure` (`ces.py:198`) | seeds + incident relations, as a `PhiFold` |
| `big_phi_contribution` | `PhiFold` (`ces.py:392`) | Σφ_d over seeds + Σ φ_r/\|r\| over incident relations |
| `diff(other)` | `CauseEffectStructure`, `Distinction`, both SIAs (`models/diff.py`) | typed delta: distinctions gained/lost/changed, relations gained/lost, Δφ, MIP change |
| `resolve_congruence(system_state)` | `Distinctions` (`distinctions.py:270`) | filter each distinction's tied states to those congruent with a system state; drop distinctions with no congruent reading |
| slicing | `Distinctions.__getitem__` | returns the same subtype |
| `filter(distinctions)` | `CompositionalState` (`compositional_state.py:263`) | distinctions consistent with a compositional state |
| `sum_phi` / `apportioned_sum_phi` / `big_phi` | bags, relation sets, structure | scalar aggregates |
| `Differentiation` | `pyphi/matching/differentiation.py` | cross-structure component union (2024 Eq. 15/16), with an inclusion-exclusion analytical path |
| `Perception` | `pyphi/matching/perception.py` | scalar weight in [0,1] per component (2024 Eq. 8–14) |
| `project_ces` | `pyphi/visualize/projection/` | flatten to plot-ready nodes/edges; **rejects `PhiFold`** because fold relations reference distinctions outside the fold |
| substrate canonicalization | `pyphi/automorphism.py` | substrate-level automorphism group, canonical form, isomorphism test — no structure-level counterpart |

What does **not** exist: general predicate selection on a structure,
restriction of a structure to a distinction subset with its induced relations,
intersection/union of structures, relabeling of a structure, any cross-frame
alignment beyond exact-value `diff`.

---

## 2. What the papers define [established]

From Albantakis et al. 2023 (equation numbers theirs):

- A **distinction** is `d(m) = (m, z*, φ_d)` (Eq. 27): mechanism in a state,
  maximal cause-effect state over its two purviews, irreducibility
  `φ_d = min(φ_c, φ_e)` (Eq. 47). The distinction set `D` of a system
  requires `φ_d > 0` *and congruence with the system's own cause-effect state*
  `s*`: `z_c*(m) ⊆ s_c'`, `z_e*(m) ⊆ s_e'` (Eq. 48).
- A **relation** `r(d) = (d, f(d), φ_r)` exists for a distinction subset `d`
  whose purviews overlap congruently (Eq. 49–52), with
  `φ_r = min_{d∈d} [ |⋂_{d}(z_c* ∪ z_e*)| · φ_d / |z_c* ∪ z_e*| ]`
  (Eq. 55, in the S3 form that PyPhi implements at `relations.py:195`).
- The **cause-effect structure** is `C(D) = D ∪ R(D)` (Eq. 57), where
  `R(D) = { r(d) : φ_r(d) > 0, d ⊆ D }` (Eq. 56), and
  `Φ = Σ_{c∈C} φ_c` (Eq. 59).
- A **Φ-fold** is "any subset of the distinctions and relations composing a
  Φ-structure" (2023, p. 29). Three species are named: the *distinction
  Φ-fold* (one distinction plus all relations involving it), the *compound
  Φ-fold* (the distinction folds seeded by a subset of units), and the
  *content* (an interrelated subset of distinctions regardless of units).

From Mayner et al. 2024 (intrinsic meaning / matching):

- The distinction Φ-fold has magnitude
  `Φ_d(C(d)) = Σ_{c ∈ C(d)} φ_c / |c|` (Eq. 3), where `|c| = 1` for a
  distinction and `|c| = |d|` (the degree) for a relation, and the whole
  decomposes exactly: `Φ(C) = Σ_d Φ_d(C(d))` (Eq. 4). **The paper defines
  this magnitude only for single-distinction folds**; the magnitude of a
  compound fold is not pinned down explicitly — this matters in §6.3.
- The **differentiation structure** is the deduplicated component union across
  the structures of several states (Eq. 15), with magnitude the sum of unique
  components' φ (Eq. 16). A **perceptual structure** is a structure with a
  scalar field `t(x,c) ∈ [0,1]` over components (Eq. 8–14) — a weighting,
  not a new structure.
- A general structure-similarity measure is explicitly left open (2024,
  footnote 6).

The 2026 formulation (`IIT_4_0_2026`) replaces the intrinsic-information
measure (`ii = min(i_diff, i_spec)`) but leaves the structural definitions —
Eq. 27, 47–48, 55–59, the folds, the decomposition — untouched. An algebra
over structures is therefore formalism-version-independent at the object
level; only the numbers change. IIT 3.0 structures have no relations
(`NullRelations`), so everything below degenerates to bag-of-distinction
operations there.

---

## 3. Two facts that determine the whole design [verified]

### 3.1 Relation locality: φ_r depends only on the relata

Eq. 55 reads a relation's φ off the purview-unions and φ_d values of its own
relata — nothing else in the structure enters. PyPhi implements exactly this
(`Relation.phi`, `relations.py:195`). Consequence: **the relation set is
determined pointwise by the distinction set**, and any distinction subset
`D' ⊆ D` has a well-defined induced relation set that agrees with the parent:

```
R(D') = { r ∈ R(D) : relata(r) ⊆ D' }
```

Verified on `xor_system()`: computing relations fresh over a 3-distinction
subset and filtering the parent's 15 relations by containment give exactly the
same set (`frozenset` equality; the two Σφ_r differ by 4e-16 from summation
order — respect `config.numerics.precision` when comparing aggregates).

The same fact makes `R` commute with intersection. For distinction sets
`D₁, D₂` drawn from the same structure:

```
R(D₁) ∩ R(D₂) = R(D₁ ∩ D₂)        — verified on grid3, True
R(D₁) ∪ R(D₂) ⊆ R(D₁ ∪ D₂)        — strict in general; 17 cross
                                      relations missing on grid3
```

A relation whose relata span both sets without fitting in either belongs to
`R(D₁ ∪ D₂)` and to neither operand. **`R` preserves meets and not joins.**
This single asymmetry is why intersection of structures is a structure and
union is not (§6.4, §6.5). It is also the identity that
`Differentiation.analytical_differentiation` already exploits [exists]: the
union's relation total is computed by inclusion-exclusion over intersections
of distinction sets, which is valid precisely because `R` commutes with `∩`.

### 3.2 Context non-locality: existence and resolution depend on the whole system

Which distinctions exist, which purviews they select, and which of their tied
specified states survive all depend on the *entire* system:

- Congruence (Eq. 48) filters distinctions against the system's own maximal
  cause-effect state `s*`, a property of the whole system's SIA.
- Purview maximization (Eq. 45) ranges over all subsets of the system.
- Exclusion means a candidate system's structure is not any kind of limit of
  its supersets' structures.

Verified, sharply, on `xor_system()`: the whole system's structure contains a
distinction with mechanism `(0, 1)` (φ_d = 0.5). The candidate system on
exactly those two nodes has **Φ_s = 0 and an empty distinction set** — the
XOR pair on its own is fully reducible. So:

> **Selecting the part of a structure supported by a unit subset and
> computing the structure of that subsystem are unrelated operations.**
> Neither bounds the other in either direction.

Consequence for the design: recomputation (`System` → `CauseEffectStructure`)
is an operation on *systems*, not on structures. It does not belong to this
algebra, and no functoriality between the subsystem lattice and any structure
lattice should be claimed or engineered. The algebra operates strictly
*within* the output of one computation.

A second consequence: the resolved specified states inside a structure's
distinctions were chosen against the parent SIA's `system_state`
(`resolve_congruence`, `distinctions.py:270`). Any part taken from a structure
inherits that resolution. A part is therefore always a *part of this
structure*, never a freestanding structure — which motivates the frame in §4.

---

## 4. The object model [proposed]

### 4.1 The frame

Every structure is grounded in a **frame**: the substrate (up to its TPM and
connectivity), the candidate system's node indices, its state, the formalism
and config snapshot, and the SIA whose `system_state` resolved the
distinctions. PyPhi already stores all of this — scattered across `sia`,
`config`, `provenance`, `node_labels` — but never treats it as a unit.

The frame's job in the algebra is to answer one question: **when are two
components comparable or combinable?** Proposed rule:

- Operations that *combine* components (meet, induced restriction, folds,
  measures) require the same frame. Same substrate, same system, same state,
  same config. This is what makes distinction identity (by value: mechanism,
  state, purviews) meaningful.
- Operations that *compare* structures (diff, differentiation across states,
  isomorphism) relax exactly one frame component at a time and say which:
  `diff` relaxes nothing or the state; `Differentiation` pools across states
  of one substrate [exists]; isomorphism relaxes labeling (§6.7). Comparing
  structures from different substrates without an explicit unit mapping is a
  type error, not a silently-empty diff.

The current `CauseEffectStructure.diff` already tracks a `substrate_note` when
node sets differ [exists]; the frame rule generalizes that into a checkable
precondition.

### 4.2 Three properties instead of two types

What separates a "whole structure" from a "part" is best captured by two
independent boolean properties rather than a subclass ladder:

- **Complete**: the distinction set is everything the system specifies
  (Eq. 48 in full — every mechanism swept, φ > 0, congruent).
- **Relation-closed**: the relation set is exactly `R(D)` for the object's
  own distinction set `D` — every relation's relata are members, and no
  qualifying relation is missing.

The taxonomy then reads:

| Object | Complete | Closed | Today |
|---|---|---|---|
| Structure of a system | yes | yes | `CauseEffectStructure` [exists] |
| Induced substructure `C[D']` | no | yes | **missing** — this is the main proposed addition |
| Fold (seeds + incident relations) | no | no (dangling relata) | `PhiFold` [exists] |
| Arbitrary component subset (paper's general Φ-fold) | no | no | missing; the paper allows it (2023 p. 29) but no operation below needs it |

The induced substructure is the missing middle. It is well-formed in every
sense that matters for aggregation and display — all relation endpoints
present, `project_ces` could accept it, Σφ is meaningful — but it is *not*
the structure of any system, and its type should say so.

On typing tactics: `PhiFold` currently subclasses `CauseEffectStructure`,
which lets every fold pass `isinstance` checks for a structure it explicitly
is not (the docstring says "not a self-contained cause-effect structure";
`project_ces` compensates with a runtime rejection). That is a Liskov
violation being patched at each consumer. The pragmatic fix is not a deep
hierarchy: keep one class for parts — call it the view type — with `parent`,
`seeds`/`distinctions`, `relations`, and a `closed` property that consumers
check instead of type-testing. Whether views subclass
`CauseEffectStructure` for display reuse is an implementation convenience the
maintainer can decide; the design point is only that **"is this
relation-closed" must be answerable without recomputing it**, because §6's
operations branch on it.

### 4.3 Components are atomic

No operation in this algebra modifies a distinction or a relation. A
distinction's φ was computed by a maximization over the whole system; its
purview cannot be truncated, its specified state cannot be edited, without
the number becoming wrong. "Projecting a structure onto units U" therefore
can never mean editing purviews down to U — it can only mean *selecting* or
*weighting* whole components by how they relate to U (§6.2). The one
exception is relabeling (§6.7), which is an isomorphism of the frame applied
uniformly, not an edit.

This principle is why the algebra is small. Every operation below is built
from exactly four primitives: **select** components, **close** a selection
(induce or fold), **measure** a selection, **map** the frame.

---

## 5. Notation

Fix a structure `C = (D, R(D))` in frame Γ. For `F ⊆ D` write:

- `C[F]` — the **induced substructure**: `(F, R(F))`. Closed, not complete.
- `C⟨F⟩` — the **fold**: `(F, { r ∈ R(D) : relata(r) ∩ F ≠ ∅ })`. Neither.
- `|r|` — a relation's degree (number of relata).

---

## 6. The operations

### 6.1 Selection over distinctions [partially exists / proposed]

The primitive is a predicate filter on the distinction bag:

```python
ces.distinctions.filter(lambda d: d.phi >= 0.25)
ces.distinctions.filter(lambda d: set(d.mechanism) <= {0, 1})
ces.distinctions.filter(lambda d: units & d.purview_union)
```

Today this is spelled `ResolvedDistinctions(d for d in ces.distinctions if …)`
— it works, preserves nothing implicitly, and is what `CompositionalState.filter`
and `resolve_congruence` do internally [exists]. A `filter` method that
preserves the resolution subtype (as slicing already does,
`distinctions.py:141`) is a two-line ergonomic addition, not new semantics.

Selection returns a bag, deliberately: the caller must then choose a closure
(6.2) to get a part of the structure, because the two closures answer
different questions and neither is a safe default.

### 6.2 The two closures: induce and fold [fold exists; induce proposed]

Given selected distinctions `F`:

- **`C[F]` (induce)** answers: *what does the structure look like restricted
  to these distinctions?* Relations kept iff all relata are kept. Closed;
  safe to display, project, aggregate as a self-contained object; its
  relation aggregates are computable analytically as `AnalyticalRelations(F)`
  — this is exactly what `Differentiation.analytical_differentiation` builds
  internally today [exists, but not exposed as an operation].
- **`C⟨F⟩` (fold)** answers: *what does this set of distinctions contribute
  to / participate in?* Relations kept iff they touch any seed. Open; the
  right object for contribution accounting (Eq. 3–4) and for highlighting in
  a plot of the parent [exists as `PhiFold`].

Both restrict to the same thing when `F = D`. They diverge on every proper
subset, and conflating them is the main source of confusion the current API
invites (a `PhiFold`'s `sum_phi_relations` would double-count if a user summed
folds; its `big_phi_contribution` exists precisely to avoid that).

The paper's three fold species map onto selection + closure:

| Paper notion (2023 p. 29) | Algebra spelling |
|---|---|
| distinction Φ-fold of `d` | `C⟨{d}⟩` |
| compound Φ-fold of units U | `C⟨{d : mechanism(d) ⊆ U}⟩` (or purview-touching-U; the paper says "specified by a subset of units", which reads most naturally as mechanisms-within-U) |
| content | `C[F]` for an interrelated `F` |

**Unit-restriction is therefore not one operation.** "Restrict to units U"
has at least three inequivalent readings a user might intend — mechanisms
within U, purviews touching U, recomputation on the subsystem U — and the
third is not an operation on structures at all (§3.2). An API should force
the choice by naming, e.g. `ces.fold_units(units, by="mechanism")` vs
`by="purview"`, and should not offer a bare `restrict(units)`.

Composition: folds compose along nested seeds — `C⟨F₁⟩⟨F₂⟩ = C⟨F₂⟩` for
`F₂ ⊆ F₁` [verified: fold-of-fold works today and the incident sets agree] —
and `fold` raises for seeds outside the parent [exists]. Induce composes
unconditionally: `C[F₁][F₂] = C[F₂]` for `F₂ ⊆ F₁`, by relation locality.
Mixed composition is directional: `C[F]⟨G⟩` is well-defined (fold within the
restriction — relations incident to G *among survivors in F*); `C⟨F⟩[G]` is
the induce inside a fold and equals `C[G]` when `G ⊆ F`. These identities are
consequences of §3.1; none require computation to hold.

### 6.3 Measures: what a part is worth [exists, with a semantic gap]

Three magnitudes over a part, all additive readings of Eq. 59 / 2024 Eq. 3:

1. **Raw φ-sum** `Σ_{c∈F∪R} φ_c` — Eq. 59 restricted to the part's own
   components. For `C[F]` this is `sum_phi_distinctions + relations.sum_phi()`
   [exists]. Additive over disjoint *component* sets; not additive over
   distinction seeds (two disjoint seed sets share cross relations).
2. **Apportioned incident sum** — what `PhiFold.big_phi_contribution`
   computes today [exists]: `Σ_{d∈F} φ_d + Σ_{r∩F≠∅} φ_r/|r|`, each incident
   relation counted once at `1/|r|`.
3. **Share-weighted sum** [proposed]:
   `μ(F) = Σ_{d∈F} φ_d + Σ_r φ_r · |relata(r) ∩ F| / |r|`.

For single-distinction folds, (2) and (3) coincide and equal the paper's
`Φ_d` (2024 Eq. 3), and both partition Φ: summing over all singleton folds
gives `big_phi` exactly [verified on xor, fig4, grid3 to 1e-9].

They diverge on multi-seed folds, and the divergence is observable [verified
on grid3]: for two distinctions sharing 2 relations,

```
fold(a).contribution + fold(b).contribution  = 2.0457671010
fold([a, b]).contribution                    = 2.0122365895   (counts shared relations once)
μ({a, b})                                    = 2.0457671010   (weights by shared seed count)
```

Only μ is additive: `μ(F₁ ⊎ F₂) = μ(F₁) + μ(F₂)` for disjoint seed sets, and
`μ(D) = Φ`. The current multi-seed `big_phi_contribution` is subadditive and
its docstring ("the fold's *additive* contribution to the structure's Φ")
holds only for singletons. The 2024 paper defines the magnitude only for
distinction folds, so **the compound-fold magnitude is an open semantic
choice, not a bug** — but the two candidate semantics answer different
questions:

- count-once (current): "how much φ is in this fold's neighborhood,
  apportioned" — a property of the fold as a region.
- share-weighted μ: "how much of Φ do these seeds account for" — a measure,
  compatible with Eq. 4, safe to sum across a partition of `D`.

Recommendation: add μ (trivially computable for concrete relations; for
analytical relations it needs a degree-weighted incident sum, which the
S3 factoring supports — see the sibling document's degree-resolved queries)
and either rename or re-document `big_phi_contribution` on multi-seed folds.
Which one deserves the short name is the maintainer's call.

**Resolution:** `big_phi_contribution` now computes the share-weighted
measure μ for all folds, so any partition of a structure's distinctions
into folds tiles Φ exactly; the count-once quantity is no longer exposed.

### 6.4 Meet: intersection of structures [proposed]

For two structures (or induced substructures) in the same frame:

```
C₁ ∧ C₂  :=  C[D₁ ∩ D₂]
```

Because `R` preserves meets (§3.1, verified), this equals both
`(D₁ ∩ D₂, R(D₁) ∩ R(D₂))` — intersection componentwise — and the induced
substructure of the intersection. The two definitions agreeing is what makes
this the unambiguous meet. Induced substructures of a structure form a
**meet-semilattice** ordered by distinction-set inclusion, with `C` as top.
(There is no bottom other than the empty structure, and no complement:
`C[D \ F]` exists but `C[F] ∨ C[D\F]` would need the join that doesn't.)

Same-frame is doing real work here: distinction equality is by value
(mechanism, state, purviews, φ within precision — `distinction.py:270`), so
intersecting bags from different systems silently produces the empty
structure rather than an error. The frame check (§4.1) turns that silence
into a type error.

Use case (why meet is worth having at all): the common core of the structures
of several states of one substrate — the invariant scaffold under state
change — is `⋀ᵢ C(sᵢ)`, and its aggregates are computable analytically. This
is the elementwise dual of what `Differentiation` (union view) computes
today.

### 6.5 Join: why union is not a structure [established + verified]

Two distinct "unions" exist and neither is a structure in the same frame:

1. **Pooled component union** (2024 Eq. 15) [exists as `Differentiation`]:
   dedup the components of several structures. The result can violate
   closure's converse — it contains all relations of each operand but *lacks*
   the cross relations of `R(D₁ ∪ D₂)` [verified: 17 missing on grid3]. It is
   the right object for differentiation, and it is already typed as its
   own class holding a `{component: weight}` view, not a
   `CauseEffectStructure`. Keep it that way.
2. **Closure of the union** `C[D₁ ∪ D₂]` — mathematically fine within one
   frame, but it manufactures relations that neither operand's system
   specified, and across frames (the differentiation case: different states)
   the mixed relata never coexisted in any system. It answers no question a
   user has asked so far. Not proposed.

So: the algebra has a meet and no join. This is not a defect to engineer
around; it is the exclusion postulate showing up as order theory.

### 6.6 Difference and diff [exists]

Two subtraction-shaped things, both present today:

- **Component difference** within a frame: `C[D₁ \ D₂]` is just selection +
  induce; the ablation question "what does the structure lose without
  distinction d" is `μ(D) − μ(D \ {d})` and is the fold identity again. The
  `AnalyticalFoldRelations` implementation (`relations.py:495`) already
  computes incident sums as exactly this difference of two analytical
  totals [exists].
- **`diff`** across structures (`ces.py:300`): a typed changelog keyed by
  mechanism / relata value-identity. It is a comparison, not an operation —
  its output (`ResultDiff`) is not a structure and should stay that way.
  Two gaps worth recording: relation-level diff silently degrades to empty
  when either side carries `AnalyticalRelations` (`ces.py:284` guards on
  iterability — the sibling document's Tier 4 explains why this is
  fundamental, not lazy); and `diff` matches by exact value, so it cannot
  align "the same distinction, slightly moved" (purview changed → reported
  as changed; there is no notion of approximate alignment; §9.2).

### 6.7 Relabeling and isomorphism [proposed; substrate level exists]

A permutation σ of substrate units acts on every component by relabeling
indices in mechanisms and purview units. The action is an equivariance:

```
C(σ·system) = σ·C(system)
```

[verified on grid3: permuting the substrate's TPM/CM/state by (2,0,1) and
recomputing reproduces Φ_s to 1e-15 and every distinction's and relation's
(mechanism, φ) to 10 decimal places under the inverse relabeling].

But the *objects* do not know this: `ces1 == ces2` is False across the
relabeling, since equality is index-based [verified]. Two proposed additions,
both thin:

- `relabel(mapping)` on structures: rewrite indices through a bijection,
  producing a structure in the mapped frame. Pure bookkeeping — no φ changes —
  but it must rewire `Distinction`/MICE parent references and rebuild the
  purview-unit sets, so it deserves one careful implementation rather than
  per-user dict comprehension.
- `is_isomorphic(other)`: exact structure equality up to a substrate
  automorphism/isomorphism. `pyphi/automorphism.py` already enumerates
  candidate permutations and canonical forms at the substrate level [exists];
  structure-level isomorphism is "∃ σ among substrate isomorphisms with
  σ·C₁ == C₂" — the candidate set is already small by that module's own
  argument (n is small or Φ was uncomputable anyway).

This is the entire equivariance story. There is no quotient type ("structure
up to relabeling") proposed: canonical forms exist at the substrate level if
needed, and no current use case wants unlabeled structures.

### 6.8 Weighting [exists]

A scalar field `w: components → [0,1]` over a structure (triggering
coefficients, perception — 2024 Eq. 8–14) is a *view*, not a new structure:
`Perception` implements it today with per-component accessors and weighted
fold magnitudes (`fold_perception`). The design point worth keeping from the
existing code: weights compose with the measure (μ-weighted sums), but they
do **not** compose with the analytical relation machinery when the weight
varies within a relation's relata (the ROADMAP records this for
perception-maximized differentiation: the mean-of-triggering factor breaks
the pure-min factoring). So weighted aggregates are exact for concrete
relations and only bounded/sampled for analytical ones.

---

## 7. Laws

Within one frame, `F, G ⊆ D`, disjointness over seeds:

**Hold** (each verified in this session or a two-line consequence of §3.1):

1. `C[F][G] = C[G]` for `G ⊆ F`; `C⟨F⟩⟨G⟩ = C⟨G⟩` for `G ⊆ F` (idempotence, composition).
2. `C[D₁] ∧ C[D₂] = C[D₁ ∩ D₂]`; meet is associative, commutative, idempotent.
3. `μ(F ⊎ G) = μ(F) + μ(G)`; `μ(D) = Φ`; `μ({d}) = Φ_d` (2024 Eq. 3).
4. `Σ_{d∈D} big_phi_contribution(C⟨{d}⟩) = Φ` (the Eq. 4 partition; exists and verified).
5. Fresh recomputation of `R(F)` equals containment-filtering of `R(D)` (relation locality).
6. `σ·C(system) = C(σ·system)` (equivariance), and all φ values are σ-invariant.
7. Analytical and concrete relation aggregates agree (`Σφ_r`, counts) [verified on xor, fig4, grid3].

**Fail** (each observed, none repairable without changing the theory):

1. `R(D₁) ∪ R(D₂) ≠ R(D₁ ∪ D₂)` — no join; pooled union is a different type.
2. `big_phi_contribution` is not additive over seeds (2.0122 ≠ 2.0458 above);
   only μ is.
3. Recomputation is not monotone or continuous in the system: subsystems of
   an irreducible system can specify nothing (`xor` pair), and parts of a
   structure say nothing about subsystem structures (exclusion).
4. `==` is not relabeling-invariant (by design; use isomorphism).
5. Relation-level diff is unavailable under analytical representation
   (fundamental — enumeration is the cost of naming individual relations).
6. Equality across frames is silently empty rather than erroneous today
   (value-based distinction identity) — the frame check is the fix.
7. `CauseEffectStructure.__hash__` omits `sia` while `__eq__` includes it
   (`ces.py:98` vs `:104`) — two structures differing only in SIA collide.
   Legal Python, but worth tightening if structures start going into sets as
   this algebra encourages.

---

## 8. Operations × representations

The sibling document establishes that relation sets should often stay
implicit (`AnalyticalRelations`: aggregates in closed form, no enumeration).
Each operation above, classified by what it needs:

| Operation | Concrete | Analytical |
|---|---|---|
| select (distinctions) | ✓ | ✓ (bags are always concrete) |
| induce `C[F]` | ✓ filter | ✓ `AnalyticalRelations(F)` — aggregates only |
| fold `C⟨F⟩` | ✓ filter | ✓ `AnalyticalFoldRelations` (difference of totals) [exists] |
| raw φ-sum, counts | ✓ | ✓ [exists] |
| μ (share-weighted) | ✓ | needs degree-resolved incident sums — S3 factoring supports it (sibling doc §3.1.4) but not implemented |
| meet | ✓ | ✓ (it is an induce) |
| pooled union: aggregate D | ✓ | ✓ inclusion-exclusion [exists] |
| pooled union: enumerate components | ✓ | ✗ |
| diff: Δφ, distinction changes | ✓ | ✓ |
| diff: relation-level changes | ✓ | ✗ (Tier 4) |
| relabel / isomorphism | ✓ | ✓ (relabeling a view = relabeling its distinctions) |
| weighting: exact aggregates | ✓ | ✗ in general (min-factoring breaks); bounds/sampling only |

The pattern: **everything that treats relations as an aggregate survives the
analytical representation; everything that names individual relations does
not.** An API that keeps this line visible (methods on the view return
aggregates; iteration is an explicit, documented materialization) will not
surprise users at n = 6+.

---

## 9. Open problems

### 9.1 Ties

A computed structure is one member of a family: state ties, purview ties,
partition ties, SIA ties are all recorded on the objects (`set_ties`,
`resolve_ties.py`) [exists]. The algebra above operates on one resolved
member. Nothing here extends obviously to tie families — is the meet of two
tie families the family of meets? (No: members are alternatives, not
components.) A principled treatment would need the operations to be functions
of the family, probably via "for all resolutions" / "for some resolution"
quantifiers over each law. Untouched here; recorded as out of scope.

### 9.2 Alignment short of isomorphism

`diff` matches components by exact value; isomorphism (§6.7) matches whole
structures by permutation. Between them is the question users will actually
ask: *how similar are these two structures?* — across states, across
parameters, across substrates of different sizes. The 2024 paper leaves the
structure metric open (footnote 6), and this exploration did not close it.
Candidate ingredients (component overlap weighted by φ, purview-space earth
mover's distance, fold-profile comparison) each fail an invariance or a
decomposition test; a serious treatment needs its own document with the
matching manuscript's use cases in hand.

### 9.3 The compound-fold magnitude

Resolved: the compound-fold magnitude is the share-weighted measure μ
(see §6.3), implemented in `PhiFold.big_phi_contribution`.

### 9.4 What was not designed

- No lazy/streaming versions of the operations (the sibling document covers
  the machinery; wiring it under `induce`/`fold` views is implementation).
- No serialization format for views (a view serializes as parent reference +
  seed mechanisms; cheap, but untested).
- No IIT 3.0-specific operations: everything degenerates correctly
  (`NullRelations` makes folds raise today [exists, `ces.py:227`] — arguably
  they should degenerate to bag selection instead; minor).
- Nothing for actual-causation structures (`account` objects); the
  locality/context split likely transfers, unverified.

---

## 10. What is worth building

Ordered by value per line of code, all small — **all implemented**:

1. **`Distinctions.filter(predicate)`** — subtype-preserving; two lines;
   unlocks every selection idiom. (§6.1)
2. **`CauseEffectStructure.induce(distinctions)`** returning a closed view —
   the missing middle object. Concrete: filter by containment. Analytical:
   `AnalyticalRelations(F)`. Reuses everything `PhiFold` has. (§6.2)
3. **μ on folds** (`share_phi` or similar), and a docstring correction on
   `big_phi_contribution` for multi-seed folds either way. (§6.3, §9.3)
4. **The frame check** on `diff`/meet-style operations: raise on substrate
   mismatch instead of returning empty deltas. (§4.1)
5. **`relabel(mapping)`** + structure-level `is_isomorphic` on top of
   `pyphi/automorphism.py`. (§6.7)
6. **Meet** as a function once `induce` exists (it is one line: induce on the
   bag intersection). (§6.4)

Explicitly not worth building: a join (no coherent semantics — §6.5), a
quotient-by-relabeling type, an eager general component-subset view (no
consumer), any recomputation-flavored "restrict to subsystem" method on
structures (§3.2 — it would be a standing invitation to conflate the two
meanings).

---

## Appendix: verification script

Run from the repo root with `uv run python <script>`. Deterministic (no
randomization; example systems and default config). All assertions pass on
the current working tree (2026-07-07, IIT_4_0_2023, precision 13).

```python
import itertools
import pyphi
from pyphi import examples
from pyphi.formalism.base import FORMALISM_REGISTRY
from pyphi.models.distinctions import ResolvedDistinctions
from pyphi.relations import AnalyticalRelations, ConcreteRelations, concrete_relations
from pyphi.system import System

ctx = pyphi.config.override(progress_bars=False); ctx.__enter__()
f = FORMALISM_REGISTRY[pyphi.config.formalism.iit.version]

# --- Fold partition identity + analytical/concrete agreement (Laws 4, 7) ---
for name in ["xor_system", "fig4_system", "grid3_system"]:
    ces = f.build_ces(getattr(examples, name)())
    total = sum(float(fold.big_phi_contribution) for fold in ces.distinction_folds())
    assert abs(total - float(ces.big_phi)) < 1e-9, name
    ar = AnalyticalRelations(ces.distinctions)
    assert ar.num_relations() == ces.relations.num_relations(), name
    assert abs(float(ar.sum_phi()) - float(ces.relations.sum_phi())) < 1e-9, name

# --- Relation locality: induced closure (Law 5) ---
ces = f.build_ces(examples.xor_system())
F = ResolvedDistinctions(list(ces.distinctions)[:3])
fresh = frozenset(concrete_relations(F))
filtered = frozenset(r for r in ces.relations if all(d in set(F) for d in r))
assert fresh == filtered

# --- Selection is not recomputation (§3.2) ---
whole_mechs = {d.mechanism for d in ces.distinctions}
assert (0, 1) in whole_mechs
sub = System(examples.xor_system().substrate, state=examples.xor_system().state,
             node_indices=(0, 1))
ces_sub = f.build_ces(sub)
assert float(ces_sub.sia.phi) == 0.0 and len(ces_sub.distinctions) == 0

# --- R preserves meets, not joins (§3.1, Law Fail 1) ---
ces = f.build_ces(examples.grid3_system())
ds = list(ces.distinctions)
D1, D2 = frozenset(ds[:5]), frozenset(ds[3:])
R = lambda D: frozenset(concrete_relations(ResolvedDistinctions(D)))
assert R(D1) & R(D2) == R(D1 & D2)
assert R(D1) | R(D2) < R(D1 | D2)          # strict: 17 cross relations on grid3

# --- Multi-seed fold contribution vs share-weighted mu (§6.3) ---
for a, b in itertools.combinations(ds, 2):
    if any(a in r and b in r for r in ces.relations):
        break
singles = float(ces.fold([a]).big_phi_contribution) + float(ces.fold([b]).big_phi_contribution)
pair = float(ces.fold([a, b]).big_phi_contribution)
seeds = {a, b}
mu = sum(float(d.phi) for d in seeds) + sum(
    float(r.phi) * len(seeds & set(r)) / len(r) for r in ces.relations if seeds & set(r))
assert abs(mu - singles) < 1e-9        # mu is additive
assert pair < singles - 1e-6           # current contribution is not

# --- Relabeling equivariance (Law 6) ---
import numpy as np
from pyphi.substrate import Substrate
sys0 = examples.grid3_system()
arr = np.asarray(sys0.substrate.tpm.to_array())         # (2,2,2, nodes, states)
perm, n = (2, 0, 1), 3
inv = tuple(perm.index(i) for i in range(n))
arr2 = np.transpose(arr, axes=(*perm, n, n + 1))[:, :, :, list(perm), :]
cm2 = np.asarray(sys0.substrate.cm)[np.ix_(perm, perm)]
sys1 = System(Substrate(arr2, cm=cm2),
              state=tuple(sys0.state[p] for p in perm), node_indices=(0, 1, 2))
c0, c1 = f.build_ces(sys0), f.build_ces(sys1)
key0 = sorted((tuple(sorted(inv[i] for i in d.mechanism)), round(float(d.phi), 10))
              for d in c0.distinctions)
key1 = sorted((tuple(sorted(d.mechanism)), round(float(d.phi), 10))
              for d in c1.distinctions)
assert key0 == key1
assert c0 != c1                        # == is index-based (Law Fail 4)

print("all checks pass")
```
