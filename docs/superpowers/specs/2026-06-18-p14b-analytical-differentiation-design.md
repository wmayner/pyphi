# P14b — Analytical cross-structure differentiation projection: Design

**Status:** approved
**Date:** 2026-06-18
**Wave:** 2 (pre-freeze, surface-affecting)
**Part of:** the P14b tail. Env-generation landed 2026-06-18; this is the
analytical (φ-maximized) cross-structure projection. The perception-maximized
projection stays open research (see §3).

---

## 1. Motivation

A *differentiation* projects the cause-effect structures triggered by a set of
stimuli into one combined view: the union of their components (distinctions and
relations), deduplicated by identity, each unique component carrying its φ (and,
for the perceptual variant, its maximum perception across structures). The
concrete implementation is `pyphi.matching.Differentiation`
(`pyphi/matching/differentiation.py`), whose `differentiation` property computes
D (Eq 16) by pooling every component of every structure.

That pooling iterates `perception.ces.relations`. Concrete relation enumeration
is the n≥6 bottleneck of IIT 4.0 — and `AnalyticalRelations` (the closed-form
Σφ_r used to sidestep it) is deliberately **not iterable**. So a structure built
with analytical relations cannot be projected by the concrete `differentiation`
at all: it raises when it tries to walk the relations. The analytical
differentiation is therefore not merely a speed-up; it is the only way to compute
D for structures whose relations are represented analytically.

This item adds a closed-form D that never enumerates a concrete relation,
mirroring how `AnalyticalRelations.sum_phi()` already replaces concrete relation
enumeration for a single structure, and cross-validates it against the concrete
`Differentiation.differentiation`.

## 2. Goals

- A closed-form differentiation D (Eq 16) over a tuple of `Perception`s that
  reads only each structure's `distinctions` (never `relations`).
- Exposed as `Differentiation.analytical_differentiation`, a property beside the
  existing `differentiation`, so the concrete and analytical paths to the same
  quantity live on the one object that owns the perceptions.
- Numerically equal to the concrete `differentiation` wherever the concrete path
  is computable (cross-validated on small networks).
- Built entirely on the tested `AnalyticalRelations` / `sum_of_minimum` stack.

## 3. Non-goals

- The **perceptual** differentiation D_p (Eq 19) analytically. Its per-component
  weight is φ_r · mean-of-triggering; the mean factor breaks the pure-min algebra
  that makes the φ sum closed-form. D_p stays concrete and remains the open
  research part of P14b. `analytical_differentiation` returns only D.
- Auto-dispatch: `differentiation` is *not* changed to detect the relations
  representation and silently route to the analytical algorithm. The two paths
  stay explicit (one name per algorithm) so cross-validation is direct and a bug
  in one cannot hide behind the other.
- Any change to the concrete `Differentiation` semantics or to the relations /
  matching machinery.

## 4. Design

### 4.1 The decomposition

The concrete D sums φ over the deduplicated union of components across the unique
triggered structures, which splits into the distinction union and the relation
union:

```
D = Σ_{d ∈ ∪ₖ Dₖ} φ_d  +  Σ_{r ∈ ∪ₖ Rₖ} φ_r
```

where Dₖ / Rₖ are the distinctions / relations of structure k, and the unions
deduplicate by component identity. A component's φ is invariant across the
structures that contain it (φ is a property of the component's identity, not of
the stimulus), so the union sums are well-defined and order-independent — exactly
the property the concrete `Differentiation` docstring already relies on.

**Distinction term** — direct. Deduplicate the distinctions across structures by
identity (the existing `Distinction.__hash__` / `__eq__`: mechanism, mechanism
state, cause/effect purviews) and sum φ_d. Distinctions are few; no enumeration
cost.

**Relation term** — inclusion–exclusion over the structures. A relation lies in
every structure of a subset T iff all its relata distinctions are common to those
structures, so the relations common to T are exactly the relations supported on
the common distinction set D_T = ∩_{k∈T} Dₖ:

```
Σ_{r ∈ ∪ₖ Rₖ} φ_r = Σ_{∅≠T⊆[K]} (−1)^(|T|+1) · Σ_{r ∈ ∩_{k∈T} Rₖ} φ_r
                   = Σ_{∅≠T⊆[K]} (−1)^(|T|+1) · AnalyticalRelations(D_T).sum_phi()
```

where K is the number of **unique** triggered structures and D_T is the
distinctions present (by identity) in every structure in T. Each inner term is
one call to the tested `AnalyticalRelations(...).sum_phi()`; there are `2^K − 1`
of them, none enumerating a concrete relation.

Self-relations (|r| = 1) are handled by the same telescoping: a self-relation on
distinction d is "present in structure k" iff d ∈ Dₖ, and
`Σ_{T: d ∈ D_T} (−1)^(|T|+1) = 1`, so each self-relation is counted exactly once —
consistent with `AnalyticalRelations.sum_phi()` including self-relations and the
concrete CES relations including them.

### 4.2 Algorithm

```
analytical_differentiation(perceptions):
    structures = unique distinction-sets across perceptions     # K of them
    # distinction union term
    d_sum = Σ φ_d over the identity-deduplicated union of all distinctions
    # relation term via inclusion-exclusion over structures
    r_sum = 0
    for each non-empty subset T of the K structures:
        D_T = distinctions common (by identity) to every structure in T
        if D_T non-empty:
            r_sum += (-1)^(|T|+1) * AnalyticalRelations(D_T).sum_phi()
    return d_sum + r_sum
```

`D_T` is assembled by intersecting the structures' distinction sets on identity
and wrapping the result in the distinctions collection `AnalyticalRelations`
expects (the same collection type the concrete CES exposes as `.distinctions`).

### 4.3 API

A `cached_property` on `Differentiation`:

```python
@cached_property
def analytical_differentiation(self) -> float:
    """Differentiation D (Eq 16), computed in closed form without enumerating
    concrete relations.

    Equal to :attr:`differentiation` wherever that is computable, but reads only
    each structure's ``distinctions`` (never ``relations``), so it is the path to
    use when the structures carry ``AnalyticalRelations`` (which are not
    iterable). Cost is ``2**K - 1`` analytical relation-sum calls for ``K``
    unique triggered structures.
    """
```

`differentiation` (concrete) and `perceptual_differentiation` (concrete) are
unchanged. There is no `analytical_perceptual_differentiation` (D_p stays
research, §3).

### 4.4 Cost boundary (documented, not enforced)

The analytical path trades concrete relation enumeration (exponential in the
number of *distinctions* — the n≥6 bottleneck) for `2^K` subset iterations
(exponential in the number of *unique structures* K). It wins when K is small and
distinctions are many — the small-sensory-interface regime this targets, where
many stimuli collapse to the same triggered structure. The docstring states this
so a caller with large K is not surprised.

## 5. Testing

`test/test_differentiation.py` (extend), following the `test_phi_fold.py`
analytical-vs-concrete pattern:

- **Cross-validation (real structures):** on the module fixture (grid3,
  sensory=(0,), system=(1,2), stimuli (0,)/(1,)) and on `basic` / `xor` derived
  perceptions, `analytical_differentiation == differentiation` (the concrete
  oracle) within tolerance.
- **Single structure:** `analytical_differentiation` of one structure equals its
  `ces.big_phi` (the concrete single-structure identity already pinned for
  `differentiation`).
- **Idempotence / duplicates:** adding a duplicate structure does not change
  `analytical_differentiation` (K counts unique structures).
- **Disjoint structures:** structures sharing no distinction identity sum exactly
  (the inclusion–exclusion cross terms vanish), matching the concrete
  `test_disjoint_structures_sum_exactly`.
- **Analytical-relations input:** a `Differentiation` over perceptions whose CESs
  carry `AnalyticalRelations` computes `analytical_differentiation` successfully
  (and equals the concrete value computed from the matching concrete-relations
  CESs) — demonstrating the path that the concrete `differentiation` cannot take.
- **Hand-computed inclusion–exclusion:** a tiny two-structure case with a shared
  distinction and a distinct one, where Σφ_r over the union is computed by hand,
  pins the `(−1)^(|T|+1)` signs and the common-distinction restriction.

Verification runs `uv run pytest` **with no path argument** (public surface;
doctest sweep).

## 6. Risks and mitigations

- **`2^K` blow-up for large K.** Documented cost boundary (§4.4); this is the
  same K-small assumption the roadmap states, and the concrete path is available
  (and cheaper) when relations are concrete and structures many.
- **Identity-based intersection correctness.** D_T uses the same
  `Distinction.__hash__`/`__eq__` the concrete `Differentiation.projection`
  dedups on, so the common-distinction set matches the concrete union semantics.
  Pinned by the cross-validation and hand-computed tests.
- **Self-relation / overlap edge cases.** Covered by the telescoping argument
  (§4.1) and guarded by cross-validation against concrete, which includes
  self-relations.
- **φ-invariance assumption.** If a component's φ differed across structures the
  union sum would be ill-defined — but the concrete `Differentiation` already
  assumes (and documents) invariance, so the analytical path inherits the same
  well-defined contract and is checked against the concrete value.

## 7. Acceptance criteria

- `Differentiation.analytical_differentiation` returns D in closed form, reading
  only `ces.distinctions`, via `2^K − 1` `AnalyticalRelations.sum_phi()` calls.
- Equals the concrete `differentiation` on grid3 / basic / xor within tolerance,
  including the single-structure, duplicate, and disjoint cases.
- Computes successfully when the structures carry `AnalyticalRelations` (where the
  concrete `differentiation` cannot run).
- `differentiation` / `perceptual_differentiation` unchanged; no
  `analytical_perceptual_differentiation` (D_p stays research).
- `uv run pytest` (no path argument) green, including doctests.
- ROADMAP P14b-analytical-projection row updated; changelog fragment present.
