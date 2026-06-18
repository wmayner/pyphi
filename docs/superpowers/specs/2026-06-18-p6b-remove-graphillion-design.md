# P6b — Remove graphillion (pure-Python relations enumeration): Design

**Status:** approved
**Date:** 2026-06-18
**Wave:** 2 (pre-freeze, surface-affecting)
**Unblocks:** P6a (no-GIL CI lane — relations no longer the no-GIL gap), P11 (thread scheduler for relations)

---

## 1. Motivation

The roadmap's P6b item was framed as "reimplement the `setset` family algebra behind a
`ZDDFamily` Protocol with OxiDD as default and graphillion retained as a one-release fallback."
A pre-design spike (`p6b_spike/`) changed the conclusion.

**The graphillion footprint is tiny and single-purpose.** The only use of graphillion/`setset`
anywhere in the library is the *concrete* relations candidate generation:

- `pyphi/relations.py::_combinations_with_nonempty_congruent_overlap` — sets the ZDD universe to
  the distinction index space, groups distinctions by shared purview unit via
  `purview_inclusion(max_order=1)`, and calls `union_powerset_family`.
- `pyphi/combinatorics.py::powerset_family` / `union_powerset_family` — the only `setset` wrappers.
- `test/test_lazy_imports.py` — asserts graphillion is not eagerly imported.

The analytical relations path (`AnalyticalRelations(...).sum_phi()`) never touches graphillion.

**The ZDD is not load-bearing.** `all_relations` builds the family and then *fully enumerates* it
into a `MapReduce` that evaluates each combination into a `Relation` (a φ computation). The ZDD's
only real feature — compact symbolic representation of an astronomically large family — is never
exploited, because every element is materialized and evaluated either way.

**Spike findings (`p6b_spike/spike.py`, seed 20260618).** The family the ZDD path produces is
exactly the set of distinction-combinations that share a common purview unit — which is precisely
`combinations_with_nonempty_intersection` over each distinction's purview-union set. Measured:

- *Equivalence:* byte-identical candidate families on every real network tested (basic, xor,
  residue, grid3) and across a synthetic sweep spanning units 4–8 and families up to 4.6M
  combinations. Zero asymmetric differences in any case.
- *Performance:* the pure-Python path is **faster** up to ~10⁵ combinations (0.4–0.8× the ZDD
  time); they tie around 10⁵–10⁶; the ZDD pulls ~2.7× ahead only at the maximal dense 5-unit case
  (4.6M combinations, 25s vs 68s). That crossover regime is irrelevant: at millions of relations
  the generation time is noise against millions of φ evaluations, and the code uses the analytical
  relations path there instead of concrete enumeration.

**OxiDD is the wrong trade.** OxiDD's Python ZBDD bindings expose the boolean/set algebra
(`union`, `intersection`, `difference`, `complement`, `empty`, `base`, `singleton`) but lack the
three operations this code needs: construction from explicit lists of sets, `set_size(k)`
filtering, and — decisively — enumeration of a family's member sets. Porting would mean
reimplementing those on OxiDD's primitives, i.e. swapping one C-extension dependency for a younger
one *and* writing nontrivial ZDD code, to preserve a structure the spike shows is unnecessary.

Therefore P6b removes graphillion outright and replaces the candidate generation with a lazy
pure-Python enumerator. This still achieves all three of the item's stated goals — eliminate the
bus-factor-1 dependency, close the last no-GIL gap, and drop the macOS libomp source-build — with
no new dependency.

## 2. Goals

- Remove graphillion from the dependency set and from all source/test code.
- Replace concrete relations candidate generation with a lazy, bounded-memory pure-Python
  enumerator producing the identical candidate family.
- Preserve relation results exactly (existing relation-count goldens stay byte-identical).
- Pin the new enumerator against a brute-force oracle with a property test.
- No new third-party dependency. No change to relation semantics, the analytical-relations path,
  or the parallel/MapReduce layer.

## 3. Non-goals

- No `ZDDFamily` Protocol, no OxiDD, no retained ZDD backend of any kind.
- No change to what a relation *is* or how φ_r is computed.
- No change to `AnalyticalRelations`.

## 4. Design

### 4.1 Core algorithm — `pyphi/combinatorics.py`

Add a lazy depth-first generator:

```
combinations_with_nonempty_intersection(sets, min_size=0, max_size=None)
    -> Iterator[frozenset[int]]
```

It yields `frozenset` elements, preserving the existing public return-element type (today the
function returns `chain[frozenset]`), so the existing explicit test and any external caller are
unaffected. It enumerates every combination of indices `i_0 < i_1 < ... < i_k` (into `sets`) whose running
set-intersection `sets[i_0] & ... & sets[i_k]` is non-empty, with `min_size <= k+1 <= max_size`.
The search carries the running intersection and prunes a whole subtree the instant the
intersection becomes empty — exact, because intersection is monotone non-increasing under adding
elements (no superset of an empty-intersection combination can have a non-empty intersection).

Properties:
- **Streaming / bounded memory:** O(recursion depth) state, not O(family size). This restores the
  streaming property graphillion had (the materializing predecessor lacked it).
- **Recursion depth** ≤ `max_size` ≤ `len(sets)`. In any feasible relations computation
  `len(sets)` = number of distinctions ≤ `2^n_units` (≲255 at the n≤8 ceiling Φ reaches), far
  under Python's recursion limit. (No explicit-stack rewrite needed; if a future caller pushes
  past this, that's a separate concern.)
- The lower bound on `min_size` follows the call sites; relations passes `min_degree` (default 2).

`combinations_with_nonempty_intersection_by_order` is **kept** but reimplemented as a thin grouping
over the generator (group yielded combinations by length into a `dict[int, set]`), so there is a
single source of truth for the enumeration. Its existing test and any by-size consumers are
unaffected.

Delete `powerset_family`, `union_powerset_family`, and the `from graphillion import setset`
`TYPE_CHECKING` import.

### 4.2 Relations call site — `pyphi/relations.py`

Rewrite `_combinations_with_nonempty_congruent_overlap(components, min_degree, max_degree)` to:

```
purview_unions = [d.purview_union for d in components]   # positional, aligns with distinctions[i]
return combinatorics.combinations_with_nonempty_intersection(
    purview_unions, min_size=min_degree, max_size=max_degree
)
```

This drops `setset.set_universe(...)`, the `purview_inclusion(max_order=1)` grouping detour, the
`mapping` indirection, and both `from graphillion import setset` imports (module-top noqa import
and the in-function import). `all_relations`'s `worker(combination)` already does
`Relation(distinctions[i] for i in combination)`, and `Distinctions.__getitem__` supports the
positional access, so the yielded index tuples are consumed unchanged. The `MapReduce` consumes
the lazy generator directly.

Equivalence rationale: the old path yields every combination `T` with `T ⊆ {distinctions
containing unit u}` for some `u` — i.e. all distinctions in `T` share a common purview unit. The
new path yields every `T` with `⋂_{i∈T} purview_union(i) ≠ ∅` — the same condition. Congruence
filtering remains downstream in `Relation` construction, identical for both paths.

### 4.3 Dependency removal — `pyproject.toml` + `uv.lock`

- Remove `"Graphillion>=1.5"` from `[project] dependencies`.
- Remove the `[tool.uv.sources]` graphillion git entry and the two-line libgomp comment above it.
- Regenerate `uv.lock`.

### 4.4 Tests

- `test/test_combinatorics.py`: add a Hypothesis property test asserting
  `combinations_with_nonempty_intersection` equals a brute-force oracle (`itertools.combinations`
  over `range(len(sets))` filtered by non-empty intersection and size bounds), over random small
  set-families and random `min_size`/`max_size`. The oracle is ground truth — independent of both
  graphillion and the implementation under test. The existing `_by_order` test stays.
- `test/test_lazy_imports.py`: remove `test_graphillion_not_loaded_at_pyphi_import` and the module
  docstring's P6b/OxiDD references — the eager-import concern is moot once the dependency is gone.
- **Regression safety net:** the relation-count goldens (e.g. IIT 4.0 Fig 2/4 relations; Fig 7's
  13740 / 13111 relations; the paper-reproduction suite) must stay byte-identical. Verification
  runs `uv run pytest` **with no path argument** so the `pyphi/` doctest sweep is included.

### 4.5 Roadmap + changelog

- ROADMAP.md Status Dashboard: P6b ⬜ → ✅, noting OxiDD was rejected (spike showed the ZDD is not
  load-bearing) and graphillion removed outright; update the Wave 2 archive bullet; record that
  this closes the relations no-GIL blocker P6a and P11 were gated on (so P6a's "xfail relations
  until P6b" qualifier is removed).
- `changelog.d/<name>.change.md`: graphillion removed as a dependency; concrete relations
  candidate generation reimplemented in pure Python (no value change).

## 5. Risks and mitigations

- **A relation regression slips through.** Mitigated by the existing relation-count goldens and the
  full-suite (`uv run pytest`, no path arg) verification; the spike already shows byte-identical
  candidate families on real networks.
- **Recursion depth on a pathological caller.** Bounded by `len(sets)` ≤ distinction count in every
  feasible relations computation; documented. Not converting to an explicit stack now (YAGNI).
- **Breaking change for external importers.** `powerset_family` / `union_powerset_family` were
  public in `pyphi.combinatorics`; their removal is a breaking change. 2.0 is already a breaking
  release; called out in the changelog fragment. No deprecation shim (consistent with the project's
  no-back-compat-for-2.0 stance).

## 6. Acceptance criteria

- `grep -rn graphillion pyphi/ test/ pyproject.toml` returns nothing.
- `uv run pytest` (no path argument) is green, including relation goldens and doctests.
- New Hypothesis property test for `combinations_with_nonempty_intersection` passes.
- ROADMAP dashboard shows P6b ✅; changelog fragment present.
