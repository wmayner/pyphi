# Φ-folds in core — design

**Project:** P14b sub-project 1 (of 4). Brings `PhiFold` into `pyphi.models`,
closing the deferred P8 item. Reference implementation:
`~/projects/matching/matching/phi_fold.py` (110 lines, pre-2.0 API).

**Paper grounding:**

- IIT 4.0 (Albantakis et al. 2023): the Φ-fold of a distinction — the
  distinction together with all relations it participates in.
- Matching paper (Mayner, Juel & Tononi, in prep;
  `~/projects/matching/manuscript/main.tex`): Eq. `perception_component`
  defines a component's contribution as φ_c / |c| (φ divided by the number
  of distinctions contributing to the component), so that summing
  per-distinction folds tiles the Φ-structure without double-counting
  relations. Eq. `perception_slice` computes perception per subset z as the
  triggering coefficient times the apportioned fold sum — the quantity this
  design makes a first-class property.

## Goals

1. A typed, immutable `PhiFold` consistent with the 2.0 model layer.
2. The apportioned (tiling) quantities as explicitly named properties —
   never as a silent change to what `sum_phi` means (the old code swapped
   `_sum_phi` semantics in a subclass).
3. **Analytical fold sums**: folds over `AnalyticalRelations` parents
   compute their relation sums in closed form, without enumerating
   relations. This removes the old hard error on analytical parents and
   makes downstream perception computable on systems where concrete
   relation enumeration is intractable. Scope note: this covers
   *single-structure* quantities (perception, richness); the
   *cross-structure* projection behind differentiation/matching still
   requires concrete relations (see the P14b follow-on roadmap entry for
   the analytical-projection research direction).
4. `highlight_phi_fold` gains a one-argument form.

Out of scope (later sub-projects): `TriggeredDistinctionFold` and anything
involving triggering coefficients (sub-project 3, `pyphi/matching/`).

## Type design

### `PhiFold` (in `pyphi/models/ces.py`)

```python
@dataclass(frozen=True, eq=False)
class PhiFold(CauseEffectStructure):
    parent: CauseEffectStructure = field(kw_only=True)
```

(`parent` is required but keyword-only, since the inherited `config` field
has a default and Python 3.12 dataclasses forbid a positional required
field after it.)

- Inherits the full CES surface. Field semantics:
  - `distinctions`: the seed subset (a `ResolvedDistinctions` restricted to
    the seeds), in parent order.
  - `relations`: the relations *incident* to the seeds — every relation
    with at least one seed among its relata. Concrete parent → filtered
    `ConcreteRelations`; analytical parent → `AnalyticalFoldRelations`
    (below).
  - `sia`, `config`: carried over from the parent.
  - `parent`: the structure the fold was taken from (provenance; used by
    the one-argument `highlight_phi_fold`).
- `sum_phi_relations` / `sum_phi_distinctions` / `big_phi` keep their
  universal CES meaning (full relation φ).
- New properties:
  - `apportioned_sum_phi_relations` = Σ_r φ_r / |r| over the fold's
    relations (|r| = number of relata; self-relations have |r| = 1).
  - `apportioned_big_phi` = `sum_phi_distinctions +
    apportioned_sum_phi_relations`.
- **Tiling invariant** (tested): summing `apportioned_big_phi` over all
  per-distinction folds of a CES equals the parent's `big_phi`.

A fold is *not* a self-contained CES: its relations may reference
distinctions outside `fold.distinctions`. The class docstring states this;
the visualize layer enforces it (see below).

### Construction

Methods on `CauseEffectStructure`:

```python
def fold(self, distinctions) -> PhiFold: ...
def distinction_folds(self) -> Iterator[PhiFold]: ...
```

- `fold(distinctions)` accepts an iterable of `Distinction` objects or
  mechanism index-tuples (mechanisms are unique keys within a resolved
  CES). Unknown mechanisms raise `ValueError` naming the offender.
- `distinction_folds()` yields the single-distinction folds in distinction
  order.
- No loose module-level constructor; the methods are the API.

The old `PhiFold.from_distinctions` classmethod, `DistinctionFold`, and
`DistinctionFoldRelations` are all subsumed: a single-distinction fold is
just a fold whose seed set has size 1, and the tiling sum is an explicit
property rather than a subclass override.

## Relation sums

### Incidence (concrete parents)

A relation `r` (a frozenset of relata) is incident to seed set F iff
`not F.isdisjoint(r)`. The fold's relations are the incident subset as a
`ConcreteRelations`.

### `Relations.apportioned_sum_phi()`

New method on the `Relations` interface, alongside `sum_phi()`:

- `ConcreteRelations`: Σ_r φ_r / len(r).
- `AnalyticalRelations`: closed form (below).
- `NoRelations`: 0.

Cached like `sum_phi` (`_apportioned_sum_phi_cached`).

### Analytical apportioned sum (new combinatorics helper)

The analytical machinery decomposes relation sums per purview unit u:
each relation S (a distinction set with congruent overlap) contributes
min_{d∈S}(φ_d / |z_d|) once for every unit in its overlap, i.e. for every
u with S ⊆ D_u, where D_u = distinctions whose `purview_union` contains u.
`Relation.phi` = |overlap| × min(φ_d/|z_d|) makes the decomposition exact.

For the apportioned sum the per-relation weight is φ_r / |S|, so the inner
sum becomes Σ_{S ⊆ D_u, |S| ≥ 2} min(S) / |S|. New helper in
`pyphi/combinatorics.py`:

```python
def sum_of_minimum_over_size_among_subsets(values): ...
```

For values sorted ascending v_1 ≤ … ≤ v_n, v_i is the minimum exactly for
subsets S with i ∈ S ⊆ {i, …, n}; with a = n − i remaining elements:

    Σ_{k=2}^{a+1} C(a, k−1) / k
      = (1/(a+1)) Σ_{k=2}^{a+1} C(a+1, k)        [C(a,k−1)/k = C(a+1,k)/(a+1)]
      = (2^(a+1) − 1 − (a+1)) / (a+1)

so the helper is a sorted dot-product with closed-form coefficients,
mirroring `sum_of_minimum_among_subsets`. Self-relations (|r| = 1) are
added separately at full φ, exactly as `AnalyticalRelations._sum_phi`
already does.

### `AnalyticalFoldRelations` (in `pyphi/relations.py`)

Key identity: every analytical quantity here is a sum over relations, and
relations either touch F or they don't. So for any per-relation weight:

    incident_total(D, F) = total(D) − total(D \ F)

`AnalyticalFoldRelations(parent_distinctions, seeds)` therefore implements:

- `_sum_phi()` = parent `sum_phi()` − `AnalyticalRelations(D \ F).sum_phi()`
- `_num_relations()` = same subtraction on `num_relations()`
- `apportioned_sum_phi()` = same subtraction on `apportioned_sum_phi()`

No new per-unit code: the complement term is a plain `AnalyticalRelations`
over the complement distinction set (constructed once, cached). The
self-relation bookkeeping is automatic — self-relations of D \ F appear in
both terms and cancel; self-relations of F survive the difference.

Capability degradation is explicit: iterating relations, `faces_by_degree`,
and anything requiring enumeration raise `NotImplementedError` with a
message pointing to concrete relation computation.

### Float precision

The subtraction introduces cancellation error on the order of float
epsilon times the totals. All equality tests compare with `utils.eq`
(config `precision`), consistent with the rest of the library.

## Error behavior

- Folding an IIT 3.0 (relations-less, `NoRelations`) CES raises
  `ValueError` explaining folds require relations.
- `project_ces` / `plot_ces` raise `TypeError` on `PhiFold` input, pointing
  the user to `highlight_phi_fold` (fold relations may reference
  distinctions outside the fold, so the projection is not well-defined).

## Visualize integration

`highlight_phi_fold` gains a one-argument form:

```python
highlight_phi_fold(fold)                  # background from fold.parent
highlight_phi_fold(ces_, phi_fold)        # existing form, unchanged
```

Implementation: if the second argument is omitted, the first must be a
`PhiFold` and the dimmed background is `fold.parent`. The duck-typed
two-argument form (any object with `.distinctions`) keeps working.

## Testing

Unit (hand-verifiable):

- Incidence filtering on a small concrete CES: fold relations are exactly
  those touching the seeds; relations among non-seeds excluded.
- Apportioned arithmetic verified by hand on a 2–3 distinction example.
- `fold()` input coercion: `Distinction` objects, mechanism tuples, mixed;
  unknown mechanism raises.
- `sum_of_minimum_over_size_among_subsets` against brute-force subset
  enumeration for n ≤ 10, including ties and empty/singleton inputs.

Properties / invariants:

- Tiling: Σ over `distinction_folds()` of `apportioned_big_phi` ==
  parent `big_phi` (concrete and analytical parents).
- Fold of all distinctions reproduces the parent's relation count and
  `sum_phi` (concrete); analytical sums match the parent's totals.
- **Analytical ≡ concrete cross-validation**: on small fixture systems,
  every distinction fold's `sum_phi_relations`, `num_relations`, and
  `apportioned_sum_phi_relations` agree between a concrete-relations
  parent and an analytical-relations parent. This validates the
  subtraction identity, the new helper, and the existing analytical
  formulas against brute force in one test.

Visualize:

- One-argument `highlight_phi_fold(fold)` produces the dimmed+overlay
  figure; `plot_ces(fold)` raises the directing error.

## Files

- `pyphi/models/ces.py` — `PhiFold`, `CauseEffectStructure.fold()`,
  `.distinction_folds()`.
- `pyphi/relations.py` — `Relations.apportioned_sum_phi()` interface +
  concrete/analytical/no-relations implementations,
  `AnalyticalFoldRelations`.
- `pyphi/combinatorics.py` — `sum_of_minimum_over_size_among_subsets`.
- `pyphi/visualize/__init__.py` — one-arg `highlight_phi_fold`,
  fold guard in `plot_ces`.
- `pyphi/visualize/projection/__init__.py` — fold guard in `project_ces`.
- `test/test_phi_fold.py` — new.
- `test/test_combinatorics.py` — extend.
- `test/test_visualize_simplicial_complex.py` (or the visualize test that
  covers highlight) — extend.
- `changelog.d/phi-fold.feature.md` — new.
