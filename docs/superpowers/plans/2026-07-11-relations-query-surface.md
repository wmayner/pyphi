# Relations Query Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the relations query surface from
`docs/superpowers/specs/2026-07-07-relations-without-materialization-design.md`:
exact closed-form queries (moments, degree spectrum, max φ_r, φ_r histogram,
binding matrix, face counts), a lazy exact-descending-order `strongest()`
stream, seeded coverage-weighted sampling with standard errors, an explicit
`materialize()` escape hatch, and a distinction-importance ranking — with
`ConcreteRelations`/`AnalyticalRelations` parity throughout, and correct
restriction to incident relations on Φ-folds.

**Architecture:** The relation set is a deterministic view of the linear-size
summary `{(u(d), q(d))}` (purview-union atoms and φ densities). Generic
subset-combinatorics primitives go in `pyphi/combinatorics.py`; the query
methods go on the `Relations` class family in `pyphi/relations.py` — iterating
defaults on the base class (correct for `ConcreteRelations` and
`NullRelations`), closed-form overrides on `AnalyticalRelations`, and
difference/filter overrides on `AnalyticalFoldRelations` (which would
otherwise silently inherit parent-set answers). The parity between the
iterating and closed-form backends on small example networks is the test
strategy.

**Tech Stack:** Python 3.13+, numpy, pandas (both already dependencies),
stdlib `heapq`/`math`/`random`/`statistics`/`itertools`. `uv run` for all
commands.

## Global Constraints

- Python 3.13+ only; no backward-compatibility shims.
- All commands via `uv run` (e.g. `uv run pytest`, `uv run python`).
- Docstrings: NumPy style (underlined `Parameters`/`Returns`/`Notes`/`References` sections, never `Args:`), final-state impersonal voice, Unicode symbols (`φ`, `Σφ_r`) written directly. **No doctests in new code** (they run under `--doctest-modules`; avoid slow/fragile examples).
- **No planning-artifact language in source, docstrings, comments, or the changelog fragment**: never "Tier 1/2/3", "N6", "N24", "per the spec/ROADMAP", "§3.1.x". Describe what the code *is*. (ROADMAP.md itself is the one file where N6/N24 labels belong.)
- Cite the literature qualitatively ("the analytical-relations supplement (S3 Appendix) of Albantakis et al. (2023)"); do not cite equation numbers from memory. The Karp–Luby reference in `sample()` is "Karp & Luby (1983)" by name only.
- Raw `<`/`>`/`==` on φ values is forbidden by the repository lint outside the tolerant layer. Thresholds must use tolerant forms (`phi > t or numerics.eq(phi, t)`); exact sorts/heaps get a waiver comment in the established style: `# numerics: exact — <reason>`.
- Counts are Python ints end-to-end (`math.comb`, `2**m`), never numpy floats.
- Randomization: isolated `random.Random(seed)` instance; `seed` is a required keyword-only argument; the seed is stored on the returned sample object.
- Commit messages describe what changed and why; no conversation narrative. Never use `--no-verify`; if a commit doesn't land, read the hook output.
- Do not commit this plan file itself, nor the spec, as part of task commits (they are committed separately only after explicit user approval).
- Execute in a git worktree under `.claude/worktrees/` (create via superpowers:using-git-worktrees). Note: in `.claude/worktrees/`, install into the worktree venv with `env -u VIRTUAL_ENV uv pip install ...` if package installation is ever needed (usually it is not — the worktree shares the source tree layout, `uv run` handles it).
- When making user-facing changes, the changelog fragment lands in `changelog.d/` (Task 12).
- Final verification MUST include a bare `uv run pytest` (no path argument) so the doctest sweep over `pyphi/` runs.

## Reference: existing code facts (verified 2026-07-11)

The implementer should trust these without re-deriving them:

- `pyphi/relations.py` class family:
  - `Relations` (base, `relations.py:335`): caching wrappers `sum_phi()`/`apportioned_sum_phi()`/`num_relations()` delegate to subclass hooks `_sum_phi()`/`_apportioned_sum_phi()`/`_num_relations()`.
  - `ConcreteRelations(frozenset, Relations)` (`relations.py:412`): iterable, holds `Relation` objects.
  - `AnalyticalRelations(Relations)` (`relations.py:446`): NOT iterable; holds `self.distinctions` (a `ResolvedDistinctions`); has `self_relations` cached property (tuple of degree-1 `Relation`s with nonempty cause∩effect overlap).
  - `AnalyticalFoldRelations(AnalyticalRelations)` (`relations.py:506`): `self.distinctions` is the **parent** distinction set; `self._full` and `self._complement` are `AnalyticalRelations` over parent and parent∖seeds; `self._seeds` is a tuple of parent `Distinction` objects. Existing scalars are computed as full − complement.
  - `NullRelations(Relations)`: iterable (empty).
- `Relation` is a `frozenset` of `Distinction`s. `relation.purview` = congruent overlap `O(S)` (a set of atoms); `relation.phi` = `len(purview) * min(d.phi/len(d.purview_union))`; `relation.is_self_relation` ⇔ degree 1; `relation.num_faces` enumerates faces (`3^degree` worst case — fine at test scale). `Relation(iterable_of_distinctions)` constructs lazily.
- Atoms are `UnitState` objects (`pyphi/models/state_specification.py:40`): hashable by `(index, state)`, totally ordered via `__lt__`, display via compact label.
- `distinctions.purview_inclusion(max_order=1)` yields `(frozenset_of_one_atom, set_of_distinctions_containing_it)` pairs — the atom incidence `D_a`. `max_order=None` yields all orders (used by the existing `_num_relations` inclusion–exclusion).
- `d.purview_union` = set of `UnitState` atoms; `d.cause.purview_units` / `d.effect.purview_units` = per-MICE atom sets; `d.mechanism` = index tuple (unique per distinction within a CES).
- `combinatorics.sum_of_minimum_among_subsets(values)` = Σ over subsets of size ≥ 2 of min (`combinatorics.py:106`). `combinatorics.combinations_with_nonempty_intersection(sets, min_size, max_size)` enumerates index-frozensets (`combinatorics.py:41`).
- `numerics.eq(x, y)`, `numerics.round_to_precision(x)` exist (`pyphi/numerics.py`). Waiver-comment style for exact comparisons: see `relations.py:62-65`.
- `CauseEffectStructure.fold(distinctions)` → `PhiFold`; `ces.distinction_folds()` yields single-distinction folds **in distinction order**; `PhiFold.big_phi_contribution` is the additive Φ share and single-distinction contributions tile `big_phi` exactly (tested in `test/models/test_phi_fold.py:76`).
- Test fixture pattern (`test/test_relations.py:103-124`): networks `["grid3", "basic", "xor", "rule110", "fig4"]`, systems via `getattr(examples, f"{name}_system")()`, distinctions via `new_big_phi.ces(system, system_measure=..., specification_measure=...).distinctions`.
- **The `basic` network has ZERO relations** (two distinctions with disjoint purview-unions; its self-relations have φ=0 and are filtered by `Relation.__bool__`). It stays in the fixture list deliberately as the empty-structure case; every test that presumes ≥1 relation must branch on `num_relations() == 0` and assert the empty-case behavior instead (`max_phi() == 0.0`, `phi_mean_std()` raises `ValueError`, empty streams/dicts/frames, sampler `normalization == 0` → zero draws).
- `test/test_combinatorics.py` exists for combinatorics unit tests.

Mathematical identities used (all verified numerically in the spec against
enumeration on xor/grid3/fig4/rule110 and at Fig. 6D scale — 27 distinctions,
1,537,080 relations):

- `φ_r(S) = |O(S)| · min_{d∈S} q(d)` where `O(S) = ⋂ u(d)`, `q(d) = φ_d/|u(d)|`.
- Subsets of a relation are relations with ≥ its φ (antitone): for `S′ ⊂ S`, `O(S′) ⊇ O(S)` and `min_{S′} q ≥ min_S q`. Hence the max φ_r is attained at degree ≤ 2, and best-first search over the pair-seeded extension tree yields exact descending order.
- `Σ_S |O(S)|^k f(min q)` decomposes over ordered k-tuples of atoms (since `|O|^k` counts ordered k-tuples of covering atoms), with each inner term a sum-of-minimum over subsets of `D_{a₁} ∩ … ∩ D_{aₖ}`.
- Unweighted counts need inclusion–exclusion / Möbius inversion over the intersection closure of the purview-unions.

---

### Task 1: Combinatorics primitives

**Files:**
- Modify: `pyphi/combinatorics.py` (add three functions after `sum_of_minimum_over_size_among_subsets`, i.e. after line 140)
- Test: `test/test_combinatorics.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `sum_of_minimum_of_size_among_subsets(values: Sequence[float], size: int) -> float` — Σ over subsets with `|S| == size` of `min(S)`.
  - `intersection_closure(sets: Iterable[frozenset]) -> set[frozenset]` — all nonempty intersections of nonempty subfamilies.
  - `exact_intersection_counts(sets: Sequence[frozenset]) -> dict[frozenset, int]` — closure element `P` → number of index-subfamilies of size ≥ 2 whose intersection is exactly `P`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_combinatorics.py`:

```python
import itertools
import math
import random

import pytest

from pyphi import combinatorics


def _brute_force_min_of_size(values, size):
    return math.fsum(
        min(subset) for subset in itertools.combinations(values, size)
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("size", [1, 2, 3, 5])
def test_sum_of_minimum_of_size_among_subsets(seed, size):
    rng = random.Random(seed)
    values = [rng.uniform(0.0, 2.0) for _ in range(7)]
    assert combinatorics.sum_of_minimum_of_size_among_subsets(
        values, size
    ) == pytest.approx(_brute_force_min_of_size(values, size))


def test_sum_of_minimum_of_size_out_of_range():
    values = [1.0, 2.0]
    assert combinatorics.sum_of_minimum_of_size_among_subsets(values, 0) == 0.0
    assert combinatorics.sum_of_minimum_of_size_among_subsets(values, 3) == 0.0


def _random_set_family(rng, num_sets=5, universe=6):
    return [
        frozenset(
            i for i in range(universe) if rng.random() < 0.5
        )
        for _ in range(num_sets)
    ]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_intersection_closure_matches_brute_force(seed):
    rng = random.Random(seed)
    sets = _random_set_family(rng)
    base = [s for s in sets if s]
    expected = set()
    for r in range(1, len(base) + 1):
        for family in itertools.combinations(base, r):
            intersection = frozenset.intersection(*family)
            if intersection:
                expected.add(intersection)
    assert combinatorics.intersection_closure(sets) == expected


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_exact_intersection_counts_match_brute_force(seed):
    rng = random.Random(seed)
    sets = _random_set_family(rng)
    expected = {}
    for r in range(2, len(sets) + 1):
        for indices in itertools.combinations(range(len(sets)), r):
            intersection = frozenset.intersection(*(sets[i] for i in indices))
            if intersection:
                expected[intersection] = expected.get(intersection, 0) + 1
    assert combinatorics.exact_intersection_counts(sets) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_combinatorics.py -v -k "minimum_of_size or closure or exact_intersection"`
Expected: FAIL with `AttributeError: module 'pyphi.combinatorics' has no attribute ...`

- [ ] **Step 3: Implement the three functions**

In `pyphi/combinatorics.py`, after `sum_of_minimum_over_size_among_subsets`
(ensure `import math` and `from collections.abc import Iterable` are present
in the module's imports; add them if missing):

```python
def sum_of_minimum_of_size_among_subsets(
    values: Sequence[float], size: int
) -> float:
    """Return the sum of ``min(S)`` over all subsets ``S`` with ``|S| == size``.

    For values sorted ascending, the ``i``-th smallest value is the minimum of
    exactly ``C(n − 1 − i, size − 1)`` subsets of size ``size`` (its
    companions must all come from the larger positions), so the result is a
    sorted dot product with binomial coefficients. This is the
    fixed-degree analogue of :func:`sum_of_minimum_among_subsets`.
    """
    if size < 1 or size > len(values):
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    return math.fsum(
        value * math.comb(n - 1 - i, size - 1) for i, value in enumerate(ordered)
    )


def intersection_closure(sets: Iterable[frozenset]) -> set[frozenset]:
    """Return every nonempty intersection of a nonempty subfamily of ``sets``.

    The closure is computed by repeatedly intersecting the frontier with the
    base family until no new element appears. Its size is bounded by
    ``2**|⋃ sets|`` but is typically far smaller for structured families.
    """
    base = [frozenset(s) for s in sets if s]
    closure: set[frozenset] = set()
    frontier = set(base)
    while frontier:
        closure |= frontier
        frontier = {
            intersection
            for p in frontier
            for s in base
            if (intersection := p & s) and intersection not in closure
        }
    return closure


def exact_intersection_counts(sets: Sequence[frozenset]) -> dict[frozenset, int]:
    """Map each intersection-closure element to the number of subfamilies
    whose intersection is exactly that element.

    Subfamilies are index-subsets of ``sets`` of size ≥ 2 (duplicates in
    ``sets`` are distinct members). For a closure element ``P`` with ``m``
    supersets among ``sets``, ``2**m − m − 1`` subfamilies intersect to at
    least ``P``; Möbius inversion down the closure (subtracting the exact
    counts of every strict superset of ``P``) leaves the exact count. All
    counts are Python ints.
    """
    closure = sorted(intersection_closure(sets), key=len, reverse=True)
    exact: dict[frozenset, int] = {}
    for p in closure:
        m = sum(1 for s in sets if p <= s)
        exact[p] = (2**m - m - 1) - sum(
            count for q, count in exact.items() if p < q
        )
    return exact
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_combinatorics.py -v -k "minimum_of_size or closure or exact_intersection"`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add pyphi/combinatorics.py test/test_combinatorics.py
git commit -m "Add fixed-size minimum sums and intersection-closure Möbius counts"
```

---

### Task 2: Base `Relations` iterating defaults + shared test fixtures

**Files:**
- Modify: `pyphi/relations.py` (imports; methods on `Relations` at `relations.py:335`)
- Create: `test/test_relations_queries.py`

**Interfaces:**
- Consumes: `Relation.phi`, `Relation.purview`, `Relation.is_self_relation`, `Relation.num_faces`, `numerics.eq`, `numerics.round_to_precision`.
- Produces (on `Relations`, inherited by `ConcreteRelations`/`NullRelations`; every later task's analytical override must match these semantics exactly):
  - `sum_phi_moment(self, k: int = 2) -> float` — `Σ φ_r^k` over ALL relations including self-relations.
  - `phi_mean_std(self) -> tuple[float, float]` — population mean and std of φ_r; `ValueError` when there are no relations. **Implemented once on the base class in terms of `num_relations()`, `sum_phi()`, `sum_phi_moment(2)`; never overridden** (this makes it automatically correct on folds).
  - `num_relations_of_degree(self, degree: int) -> int`; `sum_phi_of_degree(self, degree: int) -> float` — degree-1 means self-relations.
  - `degree_spectrum(self) -> dict[int, tuple[int, float]]` — degree → `(count, Σφ_r)`, zero-count degrees omitted.
  - `max_phi(self) -> float` — `0.0` when empty.
  - `phi_histogram(self) -> dict[float, int]` — keys are `numerics.round_to_precision(φ_r)`; includes self-relations; counts sum to `num_relations()`.
  - `num_faces(self) -> int` — total face count over all relations.
  - `strongest(self, k: int | None = None, min_phi: float | None = None, max_degree: int | None = None) -> Iterator[Relation]` — relations in descending φ_r order (φ-ties in unspecified but deterministic order); `min_phi` is a tolerant `≥` cutoff; includes self-relations.
  - `materialize(self, max_degree: int | None = None, min_phi: float | None = None) -> ConcreteRelations`.
  - `sample(self, n: int, *, seed: int) -> RelationSample` — base raises `NotImplementedError` (implemented in Task 8 for the analytical backend only).
  - Module helper `_passes(relation, max_degree, min_phi) -> bool` (shared by base/analytical `materialize` and base `strongest`).
- Also produces the shared test fixture `structures` in `test/test_relations_queries.py` (module-scoped, parametrized over the five networks) returning `(name, distinctions, concrete, analytical)`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_relations_queries.py`:

```python
"""Parity and invariant tests for the relations query surface.

The iterating backend (``ConcreteRelations``) and the closed-form backend
(``AnalyticalRelations``) must answer every query identically on systems
small enough to enumerate.
"""

import math

import pytest

from pyphi import config
from pyphi import examples
from pyphi import numerics
from pyphi import relations
from pyphi.formalism import iit4 as new_big_phi
from pyphi.measures.distribution import resolve_mechanism_measure
from pyphi.measures.distribution import resolve_system_measure

NETWORKS = ["xor", "basic", "rule110", "fig4", "grid3"]


@pytest.fixture(scope="module", params=NETWORKS)
def structures(request):
    name = request.param
    with config.override(parallel=False):
        system = getattr(examples, f"{name}_system")()
        distinctions = new_big_phi.ces(
            system,
            system_measure=resolve_system_measure(
                config.formalism.iit.system_phi_measure
            ),
            specification_measure=resolve_mechanism_measure(
                config.formalism.iit.specification_measure
            ),
        ).distinctions
        concrete = relations.ConcreteRelations(relations.all_relations(distinctions))
    analytical = relations.AnalyticalRelations(distinctions)
    return name, distinctions, concrete, analytical


# --- Base (iterating) implementations, exercised via ConcreteRelations ---


def test_concrete_sum_phi_moment_first_moment_is_sum_phi(structures):
    _, _, concrete, _ = structures
    assert concrete.sum_phi_moment(1) == pytest.approx(concrete.sum_phi())


def test_concrete_phi_mean_std_matches_manual(structures):
    _, _, concrete, _ = structures
    phis = [float(r.phi) for r in concrete]
    mean, std = concrete.phi_mean_std()
    assert mean == pytest.approx(sum(phis) / len(phis))
    assert std == pytest.approx(
        math.sqrt(sum(p**2 for p in phis) / len(phis) - (sum(phis) / len(phis)) ** 2)
    )


def test_concrete_degree_spectrum_totals(structures):
    _, _, concrete, _ = structures
    spectrum = concrete.degree_spectrum()
    assert sum(count for count, _ in spectrum.values()) == concrete.num_relations()
    assert math.fsum(s for _, s in spectrum.values()) == pytest.approx(
        concrete.sum_phi()
    )
    assert all(count > 0 for count, _ in spectrum.values())


def test_concrete_degree_queries_match_iteration(structures):
    _, _, concrete, _ = structures
    for degree in range(1, max(len(r) for r in concrete) + 1):
        expected_count = sum(1 for r in concrete if len(r) == degree)
        expected_sum = math.fsum(float(r.phi) for r in concrete if len(r) == degree)
        assert concrete.num_relations_of_degree(degree) == expected_count
        assert concrete.sum_phi_of_degree(degree) == pytest.approx(expected_sum)


def test_concrete_max_phi(structures):
    _, _, concrete, _ = structures
    assert concrete.max_phi() == pytest.approx(max(float(r.phi) for r in concrete))


def test_concrete_phi_histogram_totals(structures):
    _, _, concrete, _ = structures
    hist = concrete.phi_histogram()
    assert sum(hist.values()) == concrete.num_relations()
    assert math.fsum(phi * count for phi, count in hist.items()) == pytest.approx(
        concrete.sum_phi()
    )


def test_concrete_num_faces_matches_iteration(structures):
    _, _, concrete, _ = structures
    assert concrete.num_faces() == sum(r.num_faces for r in concrete)


def test_concrete_strongest_is_descending_and_complete(structures):
    _, _, concrete, _ = structures
    stream = list(concrete.strongest())
    phis = [float(r.phi) for r in stream]
    assert phis == sorted(phis, reverse=True)
    assert set(stream) == set(concrete)


def test_concrete_strongest_options(structures):
    _, _, concrete, _ = structures
    top3 = list(concrete.strongest(k=3))
    assert len(top3) == min(3, concrete.num_relations())
    pairs_only = list(concrete.strongest(max_degree=2))
    assert all(len(r) <= 2 for r in pairs_only)
    threshold = float(top3[-1].phi)
    above = list(concrete.strongest(min_phi=threshold))
    assert all(
        float(r.phi) > threshold or numerics.eq(float(r.phi), threshold)
        for r in above
    )


def test_concrete_materialize_filters(structures):
    _, _, concrete, _ = structures
    assert concrete.materialize() == concrete
    capped = concrete.materialize(max_degree=2)
    assert capped == relations.ConcreteRelations(r for r in concrete if len(r) <= 2)


def test_base_sample_not_implemented(structures):
    _, _, concrete, _ = structures
    with pytest.raises(NotImplementedError):
        concrete.sample(10, seed=0)


def test_null_relations_query_defaults():
    nr = relations.NullRelations()
    assert nr.sum_phi_moment(2) == 0.0
    assert nr.degree_spectrum() == {}
    assert nr.max_phi() == 0.0
    assert nr.phi_histogram() == {}
    assert nr.num_faces() == 0
    assert list(nr.strongest()) == []
    assert nr.materialize() == relations.ConcreteRelations(())
    with pytest.raises(ValueError):
        nr.phi_mean_std()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_relations_queries.py -v -x`
Expected: FAIL with `AttributeError: 'ConcreteRelations' object has no attribute 'sum_phi_moment'`

- [ ] **Step 3: Implement the base-class methods**

In `pyphi/relations.py`:

Add to the module imports (top of file, merged into the existing import
block in alphabetical order) — **only what this task uses**; later tasks add
their own imports as they need them, so each task's commit passes the ruff
unused-import check:

```python
import math
from collections import Counter
from collections.abc import Iterator
```

(`defaultdict` is already imported. Task 3 adds `itertools`; Task 5 adds
`pandas as pd`; Task 7 adds `heapq`; Task 8 adds `random` and `statistics`.)

Add a module-level helper after `_relation_size_func` (around line 250):

```python
def _passes(relation, max_degree, min_phi):
    """Filter predicate shared by ``materialize`` and ``strongest``."""
    if max_degree is not None and len(relation) > max_degree:
        return False
    if min_phi is not None:
        phi = float(relation.phi)
        if not (phi > min_phi or numerics.eq(phi, min_phi)):
            return False
    return True
```

Add the following methods to the `Relations` base class (after
`num_relations()`, i.e. after `relations.py:363`):

```python
    def sum_phi_moment(self, k: int = 2) -> float:
        """Return Σφ_r^k over all relations, including self-relations."""
        return math.fsum(float(relation.phi) ** k for relation in self)  # type: ignore[attr-defined]  # iterable in subclasses

    def phi_mean_std(self) -> tuple[float, float]:
        """Return the population mean and standard deviation of φ_r.

        Derived from the count, Σφ_r, and Σφ_r², so it is exact on any
        backend that answers those queries without enumeration.

        Raises
        ------
        ValueError
            If there are no relations.
        """
        n = self.num_relations()
        if n == 0:
            raise ValueError("no relations to summarize")
        mean = self.sum_phi() / n
        variance = self.sum_phi_moment(2) / n - mean**2
        return mean, math.sqrt(max(variance, 0.0))

    def num_relations_of_degree(self, degree: int) -> int:
        """Return the number of relations with exactly ``degree`` relata.

        Degree 1 counts the self-relations.
        """
        return sum(1 for relation in self if len(relation) == degree)  # type: ignore[attr-defined]  # iterable in subclasses

    def sum_phi_of_degree(self, degree: int) -> float:
        """Return Σφ_r over relations with exactly ``degree`` relata."""
        return math.fsum(
            float(relation.phi)
            for relation in self  # type: ignore[attr-defined]  # iterable in subclasses
            if len(relation) == degree
        )

    def degree_spectrum(self) -> dict[int, tuple[int, float]]:
        """Return ``{degree: (count, Σφ_r)}`` over all relations.

        Degrees with no relations are omitted. The counts sum to
        ``num_relations()`` and the φ sums to ``sum_phi()``.
        """
        counts: Counter[int] = Counter()
        sums: defaultdict[int, list[float]] = defaultdict(list)
        for relation in self:  # type: ignore[attr-defined]  # iterable in subclasses
            counts[len(relation)] += 1
            sums[len(relation)].append(float(relation.phi))
        return {
            degree: (counts[degree], math.fsum(sums[degree]))
            for degree in sorted(counts)
        }

    def max_phi(self) -> float:
        """Return the maximum φ_r over all relations, or ``0.0`` if empty."""
        return max(
            (float(relation.phi) for relation in self),  # type: ignore[attr-defined]  # iterable in subclasses
            default=0.0,
        )

    def phi_histogram(self) -> dict[float, int]:
        """Return ``{φ_r: count}`` over all relations.

        Keys are grouped at the configured precision
        (:func:`pyphi.numerics.round_to_precision`), so mathematically equal
        values that differ by float noise share a bucket. Counts sum to
        ``num_relations()``.
        """
        histogram: Counter[float] = Counter(
            numerics.round_to_precision(float(relation.phi))
            for relation in self  # type: ignore[attr-defined]  # iterable in subclasses
        )
        return dict(histogram)

    def num_faces(self) -> int:
        """Return the total number of faces across all relations."""
        return sum(relation.num_faces for relation in self)  # type: ignore[attr-defined]  # iterable in subclasses

    def strongest(
        self,
        k: int | None = None,
        min_phi: float | None = None,
        max_degree: int | None = None,
    ) -> Iterator[Relation]:
        """Yield relations in descending φ_r order.

        Ties in φ_r yield in an unspecified but deterministic order.

        Parameters
        ----------
        k : int, optional
            Yield at most this many relations. If None, yield all.
        min_phi : float, optional
            Stop once φ_r falls below this threshold (compared tolerantly
            at the configured precision).
        max_degree : int, optional
            Skip relations with more than this many relata.
        """
        # numerics: exact — descending sort for a stream; the min_phi
        # threshold below is tolerant.
        candidates = sorted(self, key=lambda r: float(r.phi), reverse=True)  # type: ignore[attr-defined]  # iterable in subclasses
        yielded = 0
        for relation in candidates:
            if min_phi is not None:
                phi = float(relation.phi)
                if not (phi > min_phi or numerics.eq(phi, min_phi)):
                    return
            if max_degree is not None and len(relation) > max_degree:
                continue
            yield relation
            yielded += 1
            if k is not None and yielded >= k:
                return

    def materialize(
        self, max_degree: int | None = None, min_phi: float | None = None
    ) -> ConcreteRelations:
        """Return the relations as an explicit :class:`ConcreteRelations`.

        The one deliberately loud way to obtain enumerable relation objects
        from a non-enumerating backend. ``max_degree`` and ``min_phi``
        (tolerant ``≥``) bound what is materialized.
        """
        return ConcreteRelations(
            relation
            for relation in self  # type: ignore[attr-defined]  # iterable in subclasses
            if _passes(relation, max_degree, min_phi)
        )

    def sample(self, n: int, *, seed: int):
        """Draw a coverage-weighted sample of relations.

        Implemented on backends that hold the distinction set; see
        :meth:`AnalyticalRelations.sample`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support sampling; "
            "use AnalyticalRelations"
        )
```

Note: the base `sample` deliberately has **no return annotation** —
`RelationSample` does not exist until Task 8, and pyright checks each
commit. Task 8 adds the `-> RelationSample` annotation to this method when
it defines the class. The `Iterator` import is used by `strongest`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_relations_queries.py -v`
Expected: PASS (all tests; five network parametrizations of the module fixture)

- [ ] **Step 5: Run the existing relations tests to check nothing broke**

Run: `uv run pytest test/test_relations.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyphi/relations.py test/test_relations_queries.py
git commit -m "Add iterating query defaults to the Relations base class"
```

---

### Task 3: `AnalyticalRelations` closed forms — atom index, moments, degree queries, max φ_r

**Files:**
- Modify: `pyphi/relations.py` (methods on `AnalyticalRelations` at `relations.py:446`)
- Test: `test/test_relations_queries.py` (append)

**Interfaces:**
- Consumes: Task 1's `combinatorics.sum_of_minimum_of_size_among_subsets`; Task 2's base semantics (must match exactly).
- Adds module import: `import itertools` in `pyphi/relations.py`.
- Produces (overrides on `AnalyticalRelations`):
  - `_atom_index` (cached property) — `dict[UnitState, tuple[Distinction, ...]]`, each group deterministically ordered by mechanism. **Tasks 4–8 build on this.**
  - `_density(distinction) -> float` — `φ_d / |u(d)|`.
  - Closed-form `sum_phi_moment`, `num_relations_of_degree`, `sum_phi_of_degree`, `degree_spectrum`, `max_phi`.

- [ ] **Step 1: Write the failing parity tests**

Append to `test/test_relations_queries.py`:

```python
# --- Analytical closed forms: parity with concrete enumeration ---


@pytest.mark.parametrize("k", [1, 2, 3])
def test_analytical_moments_match_concrete(structures, k):
    _, _, concrete, analytical = structures
    assert analytical.sum_phi_moment(k) == pytest.approx(concrete.sum_phi_moment(k))


def test_analytical_phi_mean_std_matches_concrete(structures):
    _, _, concrete, analytical = structures
    if concrete.num_relations() == 0:
        with pytest.raises(ValueError):
            analytical.phi_mean_std()
        return
    assert analytical.phi_mean_std() == pytest.approx(concrete.phi_mean_std())


def test_analytical_degree_queries_match_concrete(structures):
    _, _, concrete, analytical = structures
    for degree in range(1, max((len(r) for r in concrete), default=0) + 2):
        assert analytical.num_relations_of_degree(
            degree
        ) == concrete.num_relations_of_degree(degree)
        assert analytical.sum_phi_of_degree(degree) == pytest.approx(
            concrete.sum_phi_of_degree(degree)
        )


def test_analytical_degree_spectrum_matches_concrete(structures):
    _, _, concrete, analytical = structures
    analytical_spectrum = analytical.degree_spectrum()
    concrete_spectrum = concrete.degree_spectrum()
    assert analytical_spectrum.keys() == concrete_spectrum.keys()
    for degree in concrete_spectrum:
        assert analytical_spectrum[degree][0] == concrete_spectrum[degree][0]
        assert analytical_spectrum[degree][1] == pytest.approx(
            concrete_spectrum[degree][1]
        )


def test_analytical_max_phi_matches_concrete(structures):
    _, _, concrete, analytical = structures
    assert analytical.max_phi() == pytest.approx(concrete.max_phi())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_relations_queries.py -v -k analytical`
Expected: FAIL — the inherited base defaults iterate `self`, and
`AnalyticalRelations` is not iterable, so each test errors with `TypeError`
(this failure mode is exactly why every base method must be overridden here).

- [ ] **Step 3: Implement the overrides**

Add to `AnalyticalRelations` (after `self_relations`, around `relations.py:453`):

```python
    @cached_property
    def _atom_index(self):
        """Map each atom (a state-tagged unit) to the distinctions whose
        purview-union contains it.

        This incidence, together with each distinction's φ density, generates
        the entire relational structure (Albantakis et al. 2023, S3
        Appendix); every closed-form query below is computed from it. Groups
        are deterministically ordered by mechanism.
        """
        index = {}
        for purview, group in self.distinctions.purview_inclusion(max_order=1):
            (atom,) = purview
            index[atom] = tuple(
                sorted(group, key=lambda d: tuple(d.mechanism))
            )
        return index

    @staticmethod
    def _density(distinction) -> float:
        """The distinction's φ per unique purview unit."""
        return float(distinction.phi) / len(distinction.purview_union)

    def sum_phi_moment(self, k: int = 2) -> float:
        """Return Σφ_r^k over all relations, in closed form.

        Since ``φ_r = |O(S)| · min q`` and ``|O(S)|^k`` counts the ordered
        k-tuples of atoms covering ``S``, the k-th moment decomposes over
        atom k-tuples, each contributing a sum-of-minimum of ``q^k`` over the
        distinctions shared by the tuple. Cost is ``O(|𝒰|^k)`` inner sums for
        ``|𝒰|`` atoms.
        """
        if k < 1:
            raise ValueError(f"moment order must be a positive integer: {k}")
        index = self._atom_index
        atoms = sorted(index)
        total = 0.0
        for combo in itertools.product(atoms, repeat=k):
            group = set(index[combo[0]])
            for atom in combo[1:]:
                group &= set(index[atom])
            if len(group) >= 2:
                total += combinatorics.sum_of_minimum_among_subsets(
                    [self._density(d) ** k for d in group]
                )
        total += math.fsum(
            float(relation.phi) ** k for relation in self.self_relations
        )
        return total

    def num_relations_of_degree(self, degree: int) -> int:
        """Return the number of relations with exactly ``degree`` relata,
        by inclusion–exclusion over shared purview subsets."""
        if degree == 1:
            return len(self.self_relations)
        count = 0
        for purview, group in self.distinctions.purview_inclusion(max_order=None):
            count += (-1) ** (len(purview) - 1) * math.comb(len(group), degree)
        return count

    def sum_phi_of_degree(self, degree: int) -> float:
        """Return Σφ_r over relations with exactly ``degree`` relata, as a
        per-atom sorted dot product with binomial coefficients."""
        if degree == 1:
            return math.fsum(float(r.phi) for r in self.self_relations)
        return math.fsum(
            combinatorics.sum_of_minimum_of_size_among_subsets(
                [self._density(d) for d in group], degree
            )
            for group in self._atom_index.values()
        )

    def degree_spectrum(self) -> dict[int, tuple[int, float]]:
        """Return ``{degree: (count, Σφ_r)}``, in closed form per degree."""
        num_distinctions = sum(1 for _ in self.distinctions)
        spectrum = {}
        for degree in range(1, num_distinctions + 1):
            count = self.num_relations_of_degree(degree)
            if count:
                spectrum[degree] = (count, self.sum_phi_of_degree(degree))
        return spectrum

    def max_phi(self) -> float:
        """Return the maximum φ_r, scanning only pairs and self-relations.

        Notes
        -----
        The maximum over relations of degree ≥ 2 is always attained at
        degree 2: for any relation ``S`` with minimum-density member ``d*``
        and any other member ``d'``, the pair ``{d*, d'}`` has overlap
        ``⊇ O(S)`` and the same minimum density, so its φ_r is at least
        ``φ_r(S)``. The scan is ``O(|D|²)``.
        """
        ds = list(self.distinctions)
        unions = [frozenset(d.purview_union) for d in ds]
        densities = [self._density(d) for d in ds]
        best = max(
            (float(relation.phi) for relation in self.self_relations),
            default=0.0,
        )
        for i, j in itertools.combinations(range(len(ds)), 2):
            overlap = unions[i] & unions[j]
            if overlap:
                # numerics: exact — running max; callers compare tolerantly.
                best = max(best, len(overlap) * min(densities[i], densities[j]))
        return best
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_relations_queries.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/relations.py test/test_relations_queries.py
git commit -m "Add closed-form moment, degree, and max-phi queries to AnalyticalRelations"
```

---

### Task 4: `AnalyticalRelations.phi_histogram` — precision-grouped threshold sweep

**Files:**
- Modify: `pyphi/relations.py` (one method on `AnalyticalRelations`)
- Test: `test/test_relations_queries.py` (append)

**Interfaces:**
- Consumes: Task 1's `combinatorics.exact_intersection_counts`; Task 3's `_density`.
- Produces: closed-form `phi_histogram()` whose buckets match the base implementation's rounded buckets up to one unit in the last rounded place (hence the approx-key comparison in the test).

- [ ] **Step 1: Write the failing test**

Append to `test/test_relations_queries.py`:

```python
def _assert_histograms_match(left, right):
    """Histograms match if their sorted (key, count) sequences align with
    approx-equal keys and equal counts (keys are precision-rounded floats,
    so the two backends may differ by one unit in the last rounded place)."""
    left_items = sorted(left.items())
    right_items = sorted(right.items())
    assert len(left_items) == len(right_items)
    for (left_phi, left_count), (right_phi, right_count) in zip(
        left_items, right_items, strict=True
    ):
        assert left_phi == pytest.approx(right_phi, abs=1e-12)
        assert left_count == right_count


def test_analytical_phi_histogram_matches_concrete(structures):
    _, _, concrete, analytical = structures
    _assert_histograms_match(analytical.phi_histogram(), concrete.phi_histogram())


def test_analytical_phi_histogram_totals(structures):
    _, _, _, analytical = structures
    hist = analytical.phi_histogram()
    assert sum(hist.values()) == analytical.num_relations()
    assert math.fsum(phi * count for phi, count in hist.items()) == pytest.approx(
        analytical.sum_phi()
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_relations_queries.py -v -k histogram`
Expected: `test_analytical_phi_histogram_*` FAIL with `TypeError` (inherited
base method iterates a non-iterable backend).

- [ ] **Step 3: Implement**

Add to `AnalyticalRelations`:

```python
    def phi_histogram(self) -> dict[float, int]:
        """Return ``{φ_r: count}`` over all relations, in closed form.

        φ_r takes at most ``|𝒰| · |D|`` distinct values (overlap size times
        minimum density). The histogram is computed by sweeping density
        thresholds from high to low: at each threshold, relations among the
        distinctions at or above it are counted by exact overlap size via
        Möbius inversion over the intersection closure of their
        purview-unions; differencing consecutive sweeps assigns counts to
        ``overlap × density`` buckets. Densities and bucket keys are grouped
        at the configured precision
        (:func:`pyphi.numerics.round_to_precision`), so mathematically equal
        densities that differ by float noise share a threshold.

        Notes
        -----
        The intersection closure is bounded by ``2**|𝒰|`` but is small for
        structured systems; if it grows pathologically, materialization or
        sampling are the fallbacks.
        """
        histogram: Counter[float] = Counter()
        groups: defaultdict[float, list] = defaultdict(list)
        for distinction in self.distinctions:
            groups[numerics.round_to_precision(self._density(distinction))].append(
                distinction
            )
        cumulative: list = []
        previous: Counter[int] = Counter()
        # numerics: exact — iteration over precision-rounded representatives.
        for threshold in sorted(groups, reverse=True):
            cumulative.extend(groups[threshold])
            counts: Counter[int] = Counter()
            exact = combinatorics.exact_intersection_counts(
                [frozenset(d.purview_union) for d in cumulative]
            )
            for overlap, count in exact.items():
                counts[len(overlap)] += count
            for size in counts.keys() | previous.keys():
                delta = counts[size] - previous[size]
                if delta:
                    histogram[
                        numerics.round_to_precision(size * threshold)
                    ] += delta
            previous = counts
        for relation in self.self_relations:
            histogram[numerics.round_to_precision(float(relation.phi))] += 1
        return dict(histogram)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_relations_queries.py -v -k histogram`
Expected: PASS (including grid3, whose near-equal densities are the
motivating case for precision grouping)

- [ ] **Step 5: Commit**

```bash
git add pyphi/relations.py test/test_relations_queries.py
git commit -m "Add the closed-form phi_r histogram to AnalyticalRelations"
```

---

### Task 5: Binding matrix (base + analytical)

**Files:**
- Modify: `pyphi/relations.py` (add `import pandas as pd` to module imports; one method on `Relations`, one on `AnalyticalRelations`)
- Test: `test/test_relations_queries.py` (append)

**Interfaces:**
- Consumes: `combinatorics.sum_of_minimum_among_subsets`, Task 3's `_atom_index`/`_density`.
- Adds imports: `import pandas as pd` in `pyphi/relations.py`; `import numpy as np` in `test/test_relations_queries.py`.
- Produces: `binding_matrix(self) -> pd.DataFrame` on both backends — symmetric matrix over atoms (`UnitState` index/columns, sorted), entry `(a, b)` = Σ over non-self relations whose overlap contains both atoms of `φ_r/|O|` (the minimum density). Index = atoms incident to at least one non-self relation. Self-relations are excluded (the matrix measures binding *between* distinctions). Task 10's fold override subtracts complement from full on this exact shape.

- [ ] **Step 1: Write the failing tests**

Add `import numpy as np` to the imports of `test/test_relations_queries.py`
(it is first used here), then append:

```python
def test_binding_matrix_parity(structures):
    _, _, concrete, analytical = structures
    concrete_matrix = concrete.binding_matrix()
    analytical_matrix = analytical.binding_matrix()
    assert list(concrete_matrix.index) == list(analytical_matrix.index)
    assert np.allclose(
        concrete_matrix.to_numpy(), analytical_matrix.to_numpy(), atol=1e-10
    )


def test_binding_matrix_is_symmetric_with_positive_diagonal(structures):
    _, _, _, analytical = structures
    matrix = analytical.binding_matrix()
    values = matrix.to_numpy()
    assert np.allclose(values, values.T, atol=1e-12)
    assert (np.diag(values) > 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_relations_queries.py -v -k binding`
Expected: FAIL with `AttributeError: ... no attribute 'binding_matrix'`

- [ ] **Step 3: Implement**

Add `import pandas as pd` to the module imports of `pyphi/relations.py`.

Add to the `Relations` base class:

```python
    def binding_matrix(self) -> pd.DataFrame:
        """Return the atom-pair binding matrix of the relational structure.

        Entry ``(a, b)`` is the total minimum density (``φ_r / |O|``) of the
        non-self relations whose congruent overlap contains both atoms — the
        strength with which the two unit-states are jointly bound by
        relations. The diagonal decomposes the apportioned relation strength
        per atom. Index and columns are the atoms (state-tagged units)
        incident to at least one non-self relation, sorted. Self-relations
        are excluded: the matrix measures binding between distinctions.
        """
        weights: defaultdict[tuple, float] = defaultdict(float)
        atoms = set()
        for relation in self:  # type: ignore[attr-defined]  # iterable in subclasses
            if relation.is_self_relation:
                continue
            purview = sorted(relation.purview)
            atoms.update(purview)
            weight = float(relation.phi) / len(purview)
            for a in purview:
                for b in purview:
                    weights[a, b] += weight
        ordered = sorted(atoms)
        matrix = pd.DataFrame(0.0, index=ordered, columns=ordered)
        for (a, b), weight in weights.items():
            matrix.loc[a, b] = weight
        return matrix
```

Add to `AnalyticalRelations`:

```python
    def binding_matrix(self) -> pd.DataFrame:
        """Return the atom-pair binding matrix, in closed form.

        Each entry is one sum-of-minimum over the distinctions shared by the
        atom pair — ``O(|𝒰|²)`` sorted dot products, never touching a
        relation.
        """
        index = self._atom_index
        atoms = sorted(a for a in index if len(index[a]) >= 2)
        matrix = pd.DataFrame(0.0, index=atoms, columns=atoms)
        for a in atoms:
            members = set(index[a])
            for b in atoms:
                group = [d for d in index[b] if d in members]
                if len(group) >= 2:
                    matrix.loc[a, b] = combinatorics.sum_of_minimum_among_subsets(
                        [self._density(d) for d in group]
                    )
        return matrix
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_relations_queries.py -v -k binding`
Expected: PASS. (If the index-equality assertion fails while values agree,
the two backends disagree about which atoms are incident to a non-self
relation — that is a real bug, not a test artifact: an atom participates in
a non-self relation exactly when at least two distinctions share it.)

- [ ] **Step 5: Commit**

```bash
git add pyphi/relations.py test/test_relations_queries.py
git commit -m "Add the atom-pair binding matrix to the relations query surface"
```

---

### Task 6: `AnalyticalRelations.num_faces` — MICE-granularity Möbius count

**Files:**
- Modify: `pyphi/relations.py` (one method on `AnalyticalRelations`)
- Test: `test/test_relations_queries.py` (append)

**Interfaces:**
- Consumes: Task 1's `combinatorics.exact_intersection_counts`; `d.cause.purview_units` / `d.effect.purview_units`.
- Produces: closed-form `num_faces()` matching `sum(r.num_faces for r in concrete)` exactly (integer).

- [ ] **Step 1: Write the failing test**

Append to `test/test_relations_queries.py`:

```python
def test_analytical_num_faces_matches_concrete(structures):
    _, _, concrete, analytical = structures
    assert analytical.num_faces() == concrete.num_faces()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_relations_queries.py -v -k num_faces`
Expected: `test_analytical_num_faces_matches_concrete` FAILS with `TypeError`
(inherited base iterates).

- [ ] **Step 3: Implement**

Add to `AnalyticalRelations`:

```python
    def num_faces(self) -> int:
        """Return the total number of faces across all relations, in closed
        form.

        A face is a set of two or more causes/effects (one per direction
        choice per relatum) with nonempty state-tagged overlap, so the total
        face count is the same subfamily count that
        :meth:`num_relations` computes over distinctions, run instead over
        the individual causes and effects — Möbius inversion over the
        intersection closure of the per-side purviews. Faces of
        self-relations (a distinction's cause paired with its own effect)
        are included, matching enumeration.
        """
        mice_purviews = [
            frozenset(side.purview_units)
            for distinction in self.distinctions
            for side in (distinction.cause, distinction.effect)
        ]
        return sum(
            combinatorics.exact_intersection_counts(mice_purviews).values()
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_relations_queries.py -v -k num_faces`
Expected: PASS on all five networks (exact integer equality; grid3's answer
is 297 per the design exploration).

- [ ] **Step 5: Commit**

```bash
git add pyphi/relations.py test/test_relations_queries.py
git commit -m "Add the closed-form total face count to AnalyticalRelations"
```

---

### Task 7: `AnalyticalRelations.strongest` — lazy exact-descending stream

**Files:**
- Modify: `pyphi/relations.py` (one method on `AnalyticalRelations`)
- Test: `test/test_relations_queries.py` (append)

**Interfaces:**
- Consumes: Task 3's `_density` logic (inlined for index-based access), `self.self_relations`, `Relation(...)` construction, `numerics.eq`.
- Adds module import: `import heapq` in `pyphi/relations.py`.
- Produces: `strongest(k, min_phi, max_degree)` on the analytical backend — same semantics as the base method (Task 2), but output-sensitive: the first `K` yields cost `O(|D|² + K·|D|)` heap operations, independent of the total relation count. Task 10's fold override filters this stream.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_relations_queries.py`:

```python
def test_analytical_strongest_matches_sorted_concrete(structures):
    _, _, concrete, analytical = structures
    analytical_stream = list(analytical.strongest())
    concrete_sorted = sorted(concrete, key=lambda r: float(r.phi), reverse=True)
    assert [float(r.phi) for r in analytical_stream] == pytest.approx(
        [float(r.phi) for r in concrete_sorted]
    )
    assert set(analytical_stream) == set(concrete)


def test_analytical_strongest_top_k(structures):
    _, _, concrete, analytical = structures
    k = 5
    top = list(analytical.strongest(k=k))
    assert len(top) == min(k, concrete.num_relations())
    concrete_top_phis = sorted(
        (float(r.phi) for r in concrete), reverse=True
    )[: len(top)]
    assert [float(r.phi) for r in top] == pytest.approx(concrete_top_phis)


def test_analytical_strongest_min_phi_and_max_degree(structures):
    _, _, concrete, analytical = structures
    if concrete.num_relations() == 0:
        assert list(analytical.strongest()) == []
        return
    phis = sorted((float(r.phi) for r in concrete), reverse=True)
    threshold = phis[len(phis) // 2]
    above = list(analytical.strongest(min_phi=threshold))
    expected = [
        p for p in phis if p > threshold or numerics.eq(p, threshold)
    ]
    assert [float(r.phi) for r in above] == pytest.approx(expected)
    pairs = list(analytical.strongest(max_degree=2))
    assert all(len(r) <= 2 for r in pairs)
    expected_pairs = sorted(
        (float(r.phi) for r in concrete if len(r) <= 2), reverse=True
    )
    assert [float(r.phi) for r in pairs] == pytest.approx(expected_pairs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_relations_queries.py -v -k strongest`
Expected: the three `analytical_strongest` tests FAIL with `TypeError`
(inherited base sorts `self`, which is not iterable).

- [ ] **Step 3: Implement**

Add to `AnalyticalRelations`:

```python
    def strongest(
        self,
        k: int | None = None,
        min_phi: float | None = None,
        max_degree: int | None = None,
    ) -> Iterator[Relation]:
        """Yield relations in descending φ_r order, lazily.

        Best-first search over the subset lattice: φ_r never increases when
        a relatum is added (the overlap shrinks and the minimum density can
        only fall), so seeding a max-heap with all valid pairs and the
        self-relations, and expanding each popped combination by
        larger-index distinctions only, yields relations in exact descending
        order. The first ``K`` yields cost ``O(|D|²)`` seeding plus
        ``O(K·|D|)`` heap pushes, independent of the total relation count.

        Ties in φ_r yield in an unspecified but deterministic order. The
        heap can grow to ``O(yielded · |D|)`` entries when the stream is
        consumed deeply; full enumeration is better served by
        :meth:`materialize`.

        Parameters
        ----------
        k : int, optional
            Yield at most this many relations. If None, yield all.
        min_phi : float, optional
            Stop once φ_r falls below this threshold (compared tolerantly
            at the configured precision). Sound as an early exit because
            the stream is globally descending.
        max_degree : int, optional
            Do not yield or expand relations with more than this many
            relata.
        """
        ds = list(self.distinctions)
        unions = [frozenset(d.purview_union) for d in ds]
        densities = [self._density(d) for d in ds]

        def phi_of(indices):
            overlap = frozenset.intersection(*(unions[i] for i in indices))
            if not overlap:
                return None
            return len(overlap) * min(densities[i] for i in indices)

        heap: list = []
        counter = itertools.count()

        def push(phi, payload):
            # numerics: exact — heap ordering is a total order over floats;
            # the min_phi threshold at yield time is tolerant.
            heapq.heappush(heap, (-phi, next(counter), payload))

        if max_degree is None or max_degree >= 1:
            for relation in self.self_relations:
                push(float(relation.phi), relation)
        if max_degree is None or max_degree >= 2:
            for i, j in itertools.combinations(range(len(ds)), 2):
                phi = phi_of((i, j))
                if phi is not None:
                    push(phi, (i, j))

        yielded = 0
        while heap:
            negative_phi, _, payload = heapq.heappop(heap)
            phi = -negative_phi
            if min_phi is not None and not (
                phi > min_phi or numerics.eq(phi, min_phi)
            ):
                return
            if isinstance(payload, Relation):
                relation = payload
            else:
                relation = Relation(ds[i] for i in payload)
                if max_degree is None or len(payload) < max_degree:
                    for nxt in range(payload[-1] + 1, len(ds)):
                        extended = (*payload, nxt)
                        extended_phi = phi_of(extended)
                        if extended_phi is not None:
                            push(extended_phi, extended)
            yield relation
            yielded += 1
            if k is not None and yielded >= k:
                return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_relations_queries.py -v -k strongest`
Expected: PASS (all networks; the full-stream test is the strong one — the
lazy stream must reproduce the entire sorted enumeration)

- [ ] **Step 5: Commit**

```bash
git add pyphi/relations.py test/test_relations_queries.py
git commit -m "Add lazy best-first descending relation stream to AnalyticalRelations"
```

---

### Task 8: `RelationSample` + `AnalyticalRelations.sample`

**Files:**
- Modify: `pyphi/relations.py` (new class after `ConcreteRelations`; one method on `AnalyticalRelations`)
- Test: `test/test_relations_queries.py` (append)

**Interfaces:**
- Consumes: Task 3's `_atom_index`/`_density`, `Relation(...)`, `relation.purview`.
- Adds module imports: `import random` and `import statistics` in `pyphi/relations.py`; also add the `-> RelationSample` return annotation to the base `Relations.sample` now that the class exists.
- Produces:
  - `class RelationSample` with attributes `relations: tuple[Relation, ...]` (drawn with replacement), `normalization: int` (Z = Σ_a (2^|D_a|| − |D_a| − 1), the coverage-weighted total over non-self relations), `seed: int`, `num_self_relations: int`, `sum_phi_self_relations: float`; methods `estimate(f) -> tuple[float, float]` (unbiased estimate ± standard error of Σf over **non-self** relations), `num_relations() -> tuple[float, float]` and `sum_phi() -> tuple[float, float]` (estimates over all relations: sampled non-self part plus exact self-relation part), `__len__`, `__iter__`.
  - `AnalyticalRelations.sample(n, *, seed) -> RelationSample`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_relations_queries.py`:

```python
# --- Sampling ---


def test_sample_is_seed_reproducible(structures):
    _, _, _, analytical = structures
    first = analytical.sample(200, seed=42)
    second = analytical.sample(200, seed=42)
    assert first.sum_phi() == second.sum_phi()
    assert first.num_relations() == second.num_relations()
    assert analytical.sample(200, seed=7).relations != first.relations or (
        analytical.num_relations() <= 1
    )


def test_sample_estimates_are_accurate(structures):
    _, _, concrete, analytical = structures
    sample = analytical.sample(2000, seed=42)
    exact_count = concrete.num_relations()
    exact_sum = concrete.sum_phi()
    count_estimate, count_stderr = sample.num_relations()
    sum_estimate, sum_stderr = sample.sum_phi()
    # Deterministic given the seed; generous but meaningful bounds.
    assert abs(count_estimate - exact_count) <= max(5 * count_stderr, 0.05 * exact_count)
    assert abs(sum_estimate - exact_sum) <= max(5 * sum_stderr, 0.05 * exact_sum)


def test_sample_estimate_of_predicate(structures):
    _, _, concrete, analytical = structures
    sample = analytical.sample(2000, seed=42)
    exact = sum(1 for r in concrete if not r.is_self_relation and len(r) == 2)
    estimate, stderr = sample.estimate(lambda r: 1.0 if len(r) == 2 else 0.0)
    assert abs(estimate - exact) <= max(5 * stderr, 0.05 * exact + 1.0)


def test_sample_metadata(structures):
    _, _, _, analytical = structures
    sample = analytical.sample(50, seed=3)
    assert sample.seed == 3
    # A structure with no non-self relations has normalization 0 and draws
    # nothing.
    assert len(sample) == (50 if sample.normalization > 0 else 0)
    assert all(len(r) >= 2 for r in sample)
    assert isinstance(sample.normalization, int)


def test_sample_requires_seed_keyword(structures):
    _, _, _, analytical = structures
    with pytest.raises(TypeError):
        analytical.sample(10, 42)  # seed must be keyword-only
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_relations_queries.py -v -k sample`
Expected: FAIL with `NotImplementedError` (base method) or `AttributeError`.

- [ ] **Step 3: Implement**

Add after `ConcreteRelations` in `pyphi/relations.py`:

```python
class RelationSample:
    """An i.i.d., coverage-weighted sample of non-self relations.

    Relations are drawn with probability proportional to the size of their
    congruent overlap ``|O(S)|`` — the number of atoms covering them — which
    is known exactly per sample, so any sum over non-self relations is
    estimable without bias by Horvitz–Thompson reweighting (the union-of-sets
    sampling scheme of Karp & Luby (1983)). Self-relations are never sampled:
    there are at most ``|D|`` of them, and their exact totals are carried on
    the sample so the convenience estimators cover all relations.

    Attributes
    ----------
    relations : tuple[Relation, ...]
        The sampled relations, drawn with replacement.
    normalization : int
        The exact coverage-weighted total ``Σ_S |O(S)|`` over all non-self
        relations.
    seed : int
        The seed of the isolated random generator that produced the sample.
    num_self_relations : int
        The exact number of self-relations in the structure.
    sum_phi_self_relations : float
        The exact Σφ_r over the self-relations.
    """

    def __init__(
        self,
        relations,
        normalization,
        seed,
        num_self_relations,
        sum_phi_self_relations,
    ):
        self.relations = tuple(relations)
        self.normalization = normalization
        self.seed = seed
        self.num_self_relations = num_self_relations
        self.sum_phi_self_relations = sum_phi_self_relations

    def __len__(self):
        return len(self.relations)

    def __iter__(self):
        return iter(self.relations)

    def __repr__(self):
        return (
            f"{type(self).__name__}(n={len(self.relations)}, "
            f"normalization={self.normalization}, seed={self.seed})"
        )

    def estimate(self, f) -> tuple[float, float]:
        """Return an unbiased estimate and standard error of ``Σ f(S)`` over
        all non-self relations.

        Parameters
        ----------
        f : Callable[[Relation], float]
            The per-relation summand.
        """
        if not self.relations:
            return 0.0, 0.0
        values = [
            self.normalization * float(f(relation)) / len(relation.purview)
            for relation in self.relations
        ]
        mean = math.fsum(values) / len(values)
        stderr = (
            statistics.stdev(values) / math.sqrt(len(values))
            if len(values) > 1
            else float("nan")
        )
        return mean, stderr

    def num_relations(self) -> tuple[float, float]:
        """Return an estimate and standard error of the total relation
        count, including the exact self-relation count."""
        estimate, stderr = self.estimate(lambda relation: 1.0)
        return estimate + self.num_self_relations, stderr

    def sum_phi(self) -> tuple[float, float]:
        """Return an estimate and standard error of Σφ_r over all
        relations, including the exact self-relation total."""
        estimate, stderr = self.estimate(lambda relation: float(relation.phi))
        return estimate + self.sum_phi_self_relations, stderr
```

Add to `AnalyticalRelations`:

```python
    def sample(self, n: int, *, seed: int) -> RelationSample:
        """Draw ``n`` non-self relations, coverage-weighted, i.i.d.

        Sampling walks the atom incidence: an atom is drawn with probability
        proportional to the number of relations inside its distinction
        group (``2**m − m − 1`` for a group of ``m``), then a subset of size
        ≥ 2 of that group is drawn uniformly. The resulting relation is
        drawn with probability proportional to its overlap size ``|O(S)|``,
        which is known per sample, so the returned
        :class:`RelationSample` yields unbiased estimates with standard
        errors for any per-relation sum. No burn-in; exact normalization.

        Parameters
        ----------
        n : int
            The number of draws (with replacement).
        seed : int
            Seed for the isolated random generator. Required.
        """
        rng = random.Random(seed)
        index = self._atom_index
        atoms = sorted(index)
        weights = [2 ** len(index[a]) - len(index[a]) - 1 for a in atoms]
        normalization = sum(weights)
        sampled = []
        if normalization > 0:
            for _ in range(n):
                atom = rng.choices(atoms, weights=weights)[0]
                group = index[atom]
                while True:
                    mask = rng.getrandbits(len(group))
                    if mask.bit_count() >= 2:
                        break
                sampled.append(
                    Relation(
                        distinction
                        for i, distinction in enumerate(group)
                        if mask >> i & 1
                    )
                )
        return RelationSample(
            relations=sampled,
            normalization=normalization,
            seed=seed,
            num_self_relations=len(self.self_relations),
            sum_phi_self_relations=math.fsum(
                float(relation.phi) for relation in self.self_relations
            ),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_relations_queries.py -v -k sample`
Expected: PASS (deterministic given seeds — no flakiness)

- [ ] **Step 5: Commit**

```bash
git add pyphi/relations.py test/test_relations_queries.py
git commit -m "Add seeded coverage-weighted relation sampling with standard errors"
```

---

### Task 9: `AnalyticalRelations.materialize`

**Files:**
- Modify: `pyphi/relations.py` (one method on `AnalyticalRelations`)
- Test: `test/test_relations_queries.py` (append)

**Interfaces:**
- Consumes: `all_relations` (`relations.py:252`), Task 2's `_passes`.
- Produces: `materialize(max_degree, min_phi) -> ConcreteRelations` on the analytical backend (enumerates via `all_relations` rather than iterating `self`). Self-relations are always included regardless of `max_degree` (they have degree 1).

- [ ] **Step 1: Write the failing tests**

Append to `test/test_relations_queries.py`:

```python
def test_analytical_materialize_equals_concrete(structures):
    _, _, concrete, analytical = structures
    assert analytical.materialize() == concrete


def test_analytical_materialize_bounds(structures):
    _, _, concrete, analytical = structures
    capped = analytical.materialize(max_degree=2)
    assert capped == relations.ConcreteRelations(
        r for r in concrete if len(r) <= 2
    )
    threshold = concrete.max_phi()
    top = analytical.materialize(min_phi=threshold)
    assert all(
        float(r.phi) > threshold or numerics.eq(float(r.phi), threshold)
        for r in top
    )
    assert len(top) >= min(1, concrete.num_relations())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_relations_queries.py -v -k materialize`
Expected: the two `analytical_materialize` tests FAIL with `TypeError`
(inherited base iterates `self`).

- [ ] **Step 3: Implement**

Add to `AnalyticalRelations`:

```python
    def materialize(
        self, max_degree: int | None = None, min_phi: float | None = None
    ) -> ConcreteRelations:
        """Enumerate the relations as an explicit
        :class:`ConcreteRelations`.

        The one deliberately loud way to obtain relation objects from this
        backend — the output is exponential in the number of distinctions,
        so ``max_degree`` and ``min_phi`` (tolerant ``≥``) exist to bound
        it. Self-relations are always included (they have degree 1 and
        there are at most ``|D|`` of them).
        """
        return ConcreteRelations(
            relation
            for relation in all_relations(self.distinctions, max_degree=max_degree)
            if _passes(relation, max_degree, min_phi)
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_relations_queries.py -v -k materialize`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/relations.py test/test_relations_queries.py
git commit -m "Add bounded materialization from AnalyticalRelations to ConcreteRelations"
```

---

### Task 10: Fold correctness — `AnalyticalFoldRelations` overrides

**Files:**
- Modify: `pyphi/relations.py` (methods on `AnalyticalFoldRelations` at `relations.py:506`)
- Test: `test/test_relations_queries.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 2–9; `self._full`, `self._complement`, `self._seeds` on `AnalyticalFoldRelations`.
- Produces: every query method on `AnalyticalFoldRelations` answers for the **incident** relation set (relations touching at least one seed), not the parent set:
  - Additive queries by difference (full − complement): `sum_phi_moment`, `num_relations_of_degree`, `sum_phi_of_degree`, `num_faces`, `phi_histogram`, `binding_matrix`. (`phi_mean_std` and `degree_spectrum` are already correct: they are defined in terms of the overridden queries.)
  - `max_phi` by a seed-restricted pair scan.
  - `strongest`/`materialize` by incidence-filtering the full stream/enumeration.
  - `sample` raises `NotImplementedError` (sample the parent and restrict the summand instead).

**Why this task must exist:** `AnalyticalFoldRelations` inherits from
`AnalyticalRelations` with `self.distinctions` = the *parent* distinction
set. Without these overrides, every Task 3–9 method would silently answer
for the parent structure — wrong answers, not errors.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_relations_queries.py`:

```python
# --- Folds: every query restricted to incident relations ---


@pytest.fixture(scope="module")
def fold_structures(structures):
    name, distinctions, concrete, analytical = structures
    seeds = [next(iter(distinctions))]
    seed_set = set(seeds)
    fold = relations.AnalyticalFoldRelations(distinctions, seeds)
    incident_concrete = relations.ConcreteRelations(
        r for r in concrete if not seed_set.isdisjoint(r)
    )
    return name, seeds, fold, incident_concrete


def test_fold_moments_match_incident_concrete(fold_structures):
    _, _, fold, incident = fold_structures
    for k in (1, 2):
        assert fold.sum_phi_moment(k) == pytest.approx(incident.sum_phi_moment(k))


def test_fold_phi_mean_std_matches_incident_concrete(fold_structures):
    _, _, fold, incident = fold_structures
    if incident.num_relations() == 0:
        with pytest.raises(ValueError):
            fold.phi_mean_std()
        return
    assert fold.phi_mean_std() == pytest.approx(incident.phi_mean_std())


def test_fold_degree_spectrum_matches_incident_concrete(fold_structures):
    _, _, fold, incident = fold_structures
    fold_spectrum = fold.degree_spectrum()
    incident_spectrum = incident.degree_spectrum()
    assert fold_spectrum.keys() == incident_spectrum.keys()
    for degree in incident_spectrum:
        assert fold_spectrum[degree][0] == incident_spectrum[degree][0]
        assert fold_spectrum[degree][1] == pytest.approx(
            incident_spectrum[degree][1]
        )


def test_fold_max_phi_matches_incident_concrete(fold_structures):
    _, _, fold, incident = fold_structures
    assert fold.max_phi() == pytest.approx(incident.max_phi())


def test_fold_phi_histogram_matches_incident_concrete(fold_structures):
    _, _, fold, incident = fold_structures
    _assert_histograms_match(fold.phi_histogram(), incident.phi_histogram())


def test_fold_num_faces_matches_incident_concrete(fold_structures):
    _, _, fold, incident = fold_structures
    assert fold.num_faces() == incident.num_faces()


def test_fold_binding_matrix_matches_incident_concrete(fold_structures):
    _, _, fold, incident = fold_structures
    fold_matrix = fold.binding_matrix()
    incident_matrix = incident.binding_matrix()
    aligned = incident_matrix.reindex(
        index=fold_matrix.index, columns=fold_matrix.columns, fill_value=0.0
    )
    assert np.allclose(fold_matrix.to_numpy(), aligned.to_numpy(), atol=1e-10)


def test_fold_strongest_matches_incident_concrete(fold_structures):
    _, _, fold, incident = fold_structures
    stream = list(fold.strongest())
    assert [float(r.phi) for r in stream] == pytest.approx(
        sorted((float(r.phi) for r in incident), reverse=True)
    )
    assert set(stream) == set(incident)


def test_fold_materialize_matches_incident_concrete(fold_structures):
    _, _, fold, incident = fold_structures
    assert fold.materialize() == incident


def test_fold_sample_not_implemented(fold_structures):
    _, _, fold, _ = fold_structures
    with pytest.raises(NotImplementedError):
        fold.sample(10, seed=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_relations_queries.py -v -k fold`
Expected: FAIL — the inherited analytical methods answer for the parent set
(e.g. `test_fold_moments...` reports the parent moment, larger than the
incident one).

- [ ] **Step 3: Implement**

Add to `AnalyticalFoldRelations` (after `_apportioned_sum_phi`, around
`relations.py:539`):

```python
    def _difference(self, query, *args, **kwargs):
        """Evaluate an additive query as full − complement.

        A relation either touches the seed set or it does not, so any
        quantity that is a sum over relations restricts to the incident set
        by differencing the parent total against the seed-free total.
        Self-relations of non-seed distinctions cancel; the seeds' survive.
        """
        return getattr(self._full, query)(*args, **kwargs) - getattr(
            self._complement, query
        )(*args, **kwargs)

    def sum_phi_moment(self, k: int = 2) -> float:
        """Return Σφ_r^k over the incident relations."""
        return self._difference("sum_phi_moment", k)

    def num_relations_of_degree(self, degree: int) -> int:
        """Return the number of incident relations with exactly ``degree``
        relata."""
        return self._difference("num_relations_of_degree", degree)

    def sum_phi_of_degree(self, degree: int) -> float:
        """Return Σφ_r over incident relations with exactly ``degree``
        relata."""
        return self._difference("sum_phi_of_degree", degree)

    def num_faces(self) -> int:
        """Return the total face count over the incident relations."""
        return self._difference("num_faces")

    def phi_histogram(self) -> dict[float, int]:
        """Return ``{φ_r: count}`` over the incident relations.

        Bucket-wise difference of the parent and seed-free histograms;
        bucket keys align because both are grouped at the configured
        precision from the same underlying densities.
        """
        histogram = Counter(self._full.phi_histogram())
        histogram.subtract(self._complement.phi_histogram())
        return {phi: count for phi, count in histogram.items() if count}

    def binding_matrix(self) -> pd.DataFrame:
        """Return the atom-pair binding matrix of the incident relations.

        Entry-wise difference of the parent and seed-free matrices, on the
        parent's atom index (rows for atoms bound only by seed-free
        relations go to zero).
        """
        full = self._full.binding_matrix()
        complement = self._complement.binding_matrix()
        aligned = complement.reindex(
            index=full.index, columns=full.columns, fill_value=0.0
        )
        return full - aligned

    def max_phi(self) -> float:
        """Return the maximum φ_r over the incident relations.

        Notes
        -----
        The incident maximum is attained at an incident pair or a seed's
        self-relation: for any incident relation ``S``, its
        minimum-density member ``d*`` paired with any seed in ``S`` is an
        incident pair with overlap ``⊇ O(S)`` and the same minimum density.
        """
        seed_set = set(self._seeds)
        ds = list(self.distinctions)
        unions = [frozenset(d.purview_union) for d in ds]
        densities = [self._density(d) for d in ds]
        best = max(
            (
                float(relation.phi)
                for relation in self.self_relations
                if not seed_set.isdisjoint(relation)
            ),
            default=0.0,
        )
        for i, j in itertools.combinations(range(len(ds)), 2):
            if ds[i] not in seed_set and ds[j] not in seed_set:
                continue
            overlap = unions[i] & unions[j]
            if overlap:
                # numerics: exact — running max; callers compare tolerantly.
                best = max(best, len(overlap) * min(densities[i], densities[j]))
        return best

    def strongest(
        self,
        k: int | None = None,
        min_phi: float | None = None,
        max_degree: int | None = None,
    ) -> Iterator[Relation]:
        """Yield the incident relations in descending φ_r order.

        Filters the parent's descending stream by seed incidence, so the
        order is exact; non-incident relations are popped and discarded, so
        the cost tracks the parent stream's, not the incident count.
        """
        seed_set = set(self._seeds)
        yielded = 0
        for relation in self._full.strongest(
            k=None, min_phi=min_phi, max_degree=max_degree
        ):
            if seed_set.isdisjoint(relation):
                continue
            yield relation
            yielded += 1
            if k is not None and yielded >= k:
                return

    def materialize(
        self, max_degree: int | None = None, min_phi: float | None = None
    ) -> ConcreteRelations:
        """Return the incident relations as an explicit
        :class:`ConcreteRelations`."""
        seed_set = set(self._seeds)
        return ConcreteRelations(
            relation
            for relation in self._full.materialize(max_degree, min_phi)
            if not seed_set.isdisjoint(relation)
        )

    def sample(self, n: int, *, seed: int) -> RelationSample:
        """Not supported on folds: sample the parent structure and restrict
        the summand to incident relations instead."""
        raise NotImplementedError(
            "sampling a fold is not supported; sample the parent "
            "AnalyticalRelations and restrict the estimated summand to "
            "relations touching the seeds"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_relations_queries.py -v -k fold`
Expected: PASS

- [ ] **Step 5: Run the fold regression tests**

Run: `uv run pytest test/models/test_phi_fold.py -v`
Expected: PASS (existing fold behavior unchanged)

- [ ] **Step 6: Commit**

```bash
git add pyphi/relations.py test/test_relations_queries.py
git commit -m "Restrict every relations query to incident relations on folds"
```

---

### Task 11: `CauseEffectStructure.distinction_importance`

**Files:**
- Modify: `pyphi/models/ces.py` (one method on `CauseEffectStructure`, after `distinction_folds` at `ces.py:269-272`)
- Test: `test/models/test_phi_fold.py` (append)

**Interfaces:**
- Consumes: `self.distinctions`, `self.distinction_folds()` (yields single-distinction folds in distinction order), `PhiFold.big_phi_contribution`.
- Produces: `distinction_importance(self) -> list[tuple[Distinction, float]]` — every distinction paired with its additive Φ contribution, sorted descending by contribution (φ-ties broken by mechanism for determinism). The contributions tile `big_phi` exactly.

- [ ] **Step 1: Write the failing test**

Append to `test/models/test_phi_fold.py` (this file already has `xor_ces`
and `xor_ces_analytical` fixtures — reuse them; check the exact fixture
names at the top of the file and match them):

```python
def test_distinction_importance_ranks_and_tiles(xor_ces):
    ranking = xor_ces.distinction_importance()
    assert len(ranking) == len(xor_ces.distinctions)
    contributions = [contribution for _, contribution in ranking]
    assert contributions == sorted(contributions, reverse=True)
    assert sum(contributions) == pytest.approx(xor_ces.big_phi)
    mechanisms = {tuple(d.mechanism) for d, _ in ranking}
    assert mechanisms == {tuple(d.mechanism) for d in xor_ces.distinctions}


def test_distinction_importance_matches_folds(xor_ces):
    by_mechanism = {
        tuple(d.mechanism): contribution
        for d, contribution in xor_ces.distinction_importance()
    }
    for distinction, fold in zip(
        xor_ces.distinctions, xor_ces.distinction_folds(), strict=True
    ):
        assert by_mechanism[tuple(distinction.mechanism)] == pytest.approx(
            fold.big_phi_contribution
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/models/test_phi_fold.py -v -k importance`
Expected: FAIL with `AttributeError: ... no attribute 'distinction_importance'`

- [ ] **Step 3: Implement**

Add to `CauseEffectStructure` in `pyphi/models/ces.py`, directly after
`distinction_folds`:

```python
    def distinction_importance(self):
        """Rank the distinctions by their additive contribution to Φ.

        Each distinction's importance is its single-distinction Φ-fold
        contribution: its own φ plus its share of each incident relation's
        φ (``φ_r / |r|`` per bound seed). These contributions tile Φ —
        summing over all distinctions recovers ``big_phi`` exactly.

        Returns
        -------
        list[tuple[Distinction, float]]
            ``(distinction, contribution)`` pairs, sorted by descending
            contribution; ties are broken by mechanism for determinism. The
            removal cost of a distinction (everything its relations carry,
            not just its share) is the ``big_phi`` of its fold:
            ``self.fold([distinction]).big_phi``.
        """
        pairs = [
            (distinction, fold.big_phi_contribution)
            for distinction, fold in zip(
                self.distinctions, self.distinction_folds(), strict=True
            )
        ]
        # numerics: exact — deterministic total order for a ranking display;
        # selection among near-ties is the caller's concern.
        return sorted(
            pairs, key=lambda pair: (-pair[1], tuple(pair[0].mechanism))
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/models/test_phi_fold.py -v`
Expected: PASS (new tests and all existing fold tests)

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/ces.py test/models/test_phi_fold.py
git commit -m "Add distinction_importance ranking to CauseEffectStructure"
```

---

### Task 12: Changelog fragment, ROADMAP update, full-suite verification

**Files:**
- Create: `changelog.d/relations-query-surface.feature.md`
- Modify: `ROADMAP.md` (the N6 row at ~line 250 and the N24 row at ~line 406; run `grep -n "N6\|N24" ROADMAP.md` first and update every hit consistently, including any Status Dashboard row that references them)

**Interfaces:**
- Consumes: everything landed in Tasks 1–11.
- Produces: user-facing changelog; ROADMAP rows flipped to landed.

- [ ] **Step 1: Write the changelog fragment**

Create `changelog.d/relations-query-surface.feature.md`:

```markdown
Added a query surface to relation sets, answering structural questions
without enumerating relations. On `AnalyticalRelations` every query is
closed-form over the distinction set (via the analytical-relations
supplement of Albantakis et al. 2023): φ_r moments and mean/std
(`sum_phi_moment`, `phi_mean_std`), per-degree counts and sums
(`num_relations_of_degree`, `sum_phi_of_degree`, `degree_spectrum`), the
maximum φ_r (`max_phi`), the exact φ_r histogram grouped at configured
precision (`phi_histogram`), the atom-pair binding matrix
(`binding_matrix`), and the total face count (`num_faces`). `strongest()`
yields relations lazily in exact descending-φ_r order (top-K or thresholded,
output-sensitively); `sample(n, seed=...)` draws unbiased coverage-weighted
relation samples with standard errors (`RelationSample`); `materialize()`
is the explicit, bounded escape hatch to `ConcreteRelations`.
`ConcreteRelations` answers the same queries by iteration, and Φ-folds
answer them restricted to their incident relations.
`CauseEffectStructure.distinction_importance()` ranks distinctions by their
additive contribution to Φ (the contributions tile `big_phi` exactly).
```

- [ ] **Step 2: Update ROADMAP.md**

Run `grep -n "N6\|N24" ROADMAP.md`. In the N6 row (~line 250), change the
status marker `(N6 — explored, build-ready)` to `(N6 — landed 2026-07-11)`
and change the trailing clause `— and discharges this item (with N24) when
its Tier 1–3 query surface lands` to `— and its query surface is landed on
``pyphi/relations.py`` (closed-form moments/degrees/histogram/binding/faces,
lazy top-K ``strongest()``, seeded sampling, bounded ``materialize()``),
discharging this item together with N24`. In the N24 row (~line 406), change
`*`quick-win`; explored, build-ready.*` to `*`quick-win`; landed
2026-07-11.*` and append a sentence: `Landed as
``CauseEffectStructure.distinction_importance()`` plus the fold-restricted
query surface.` If the Status Dashboard at the top of the file has rows for
these items, flip them to landed with the same date; if it does not, leave
the dashboard untouched.

- [ ] **Step 3: Full-suite verification**

Run (no path argument — this is what runs the doctest sweep over `pyphi/`):

```bash
uv run pytest
```

Expected: PASS across the board. If any pre-existing test interacts with the
new methods (e.g. serialization round-trips or display cards), diagnose
before touching it — the new methods add no state, so failures indicate a
real conflict, not fixture drift.

Then run the linters the pre-commit hook will run:

```bash
uv run ruff check pyphi/relations.py pyphi/combinatorics.py pyphi/models/ces.py
uv run ruff format --check pyphi/relations.py pyphi/combinatorics.py pyphi/models/ces.py
```

Expected: clean (fix any formatting/lint complaints; the repository lint on
raw φ comparisons is satisfied by the tolerant forms and the
`# numerics: exact` waiver comments specified in the tasks above).

- [ ] **Step 4: Commit**

```bash
git add changelog.d/relations-query-surface.feature.md ROADMAP.md
git commit -m "Record the relations query surface in the changelog and roadmap"
```

---

## Deliberately out of scope (do not implement)

- **Visualization rewiring** (`plot_ces` consuming `strongest(k)` for analytical structures) — separate follow-up; the query surface is consumable by it unchanged.
- **`CauseEffectStructure.diff` replacement** (statistic deltas instead of relation-level set diff) — separate follow-up.
- **Degree-stratified sampling** (`max_degree` on `sample()`) — add when a rare-event estimation need appears; the exact per-degree counts to stratify with already land here.
- **`Relations.fold(seeds)`** — redundant with `CauseEffectStructure.fold`.
- **Serialization of `RelationSample`** — it is an analysis artifact, not a result type.
- **Large-`|D_a|` numeric hardening** of `sum_of_minimum_among_subsets` (float64 coefficient overflow past ~1000 members) — pre-existing behavior, unchanged; the closed forms inherit it.
- **Intersection-closure-based speedup of the existing `_num_relations`** — noted in the spec as future work; not needed for correctness.
