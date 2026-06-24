# Φ-folds in Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `PhiFold` into `pyphi.models` — a typed, immutable slice of a cause-effect structure (one distinction plus its incident relations) exposing the paper's Φ_d contribution, with closed-form analytical fold sums — closing the deferred P8 item.

**Architecture:** `PhiFold` subclasses `CauseEffectStructure`, adding a `parent` link and two contribution properties. Relation sums gain an `apportioned_sum_phi()` method (Σ φ_r/|r|) across the concrete/analytical/null relation types; a new `AnalyticalFoldRelations` computes incident sums by the subtraction identity `total(D) − total(D∖F)` over two plain `AnalyticalRelations`. One new combinatorics helper supplies the analytical closed form.

**Tech Stack:** Python 3.12+, numpy, pytest, `uv run` for all commands. Frozen dataclasses, the existing `Relations`/`Distinctions` model layer, and `pyphi.combinatorics`.

**Spec:** `docs/superpowers/specs/2026-06-06-phi-fold-core-design.md`

**Reference math (paper eq. numbers from arXiv 2412.21111v2):** Φ_d = Eq (3); tiling Σ_d Φ_d = Φ = Eq (4).

---

## Background the engineer needs

- **Build a full CES in a test:** `examples.xor_system().ces()` returns a `CauseEffectStructure` (`sia`, `distinctions: ResolvedDistinctions`, `relations: ConcreteRelations`). For xor: 4 distinctions with mechanisms `[(0,1),(0,2),(1,2),(0,1,2)]`, 15 concrete relations, `relations.sum_phi() == 7.0`, `big_phi == 9.5`. This is the standard small fixture; it computes in a few seconds.
- **A `Relation`** (`pyphi/relations.py:126`) is a `frozenset` subclass whose elements are the `Distinction` objects it binds. So `len(relation)` is `|r|` (number of relata; 1 for a self-relation), and `seed in relation` / `seed_set.isdisjoint(relation)` test incidence by distinction identity.
- **`Distinction`** (`pyphi/models/distinction.py:35`) has `.mechanism` (a tuple of unit indices), `.phi`, and `.purview_union` (a `cached_property`, the set of cause∪effect purview units). `Distinction.__eq__`/`__hash__` are by value; distinctions taken from the same CES are the exact objects stored in its relations.
- **`Relations` hierarchy** (`pyphi/relations.py`): base `Relations` (`:270`) caches `sum_phi()`/`num_relations()` by delegating to subclass `_sum_phi()`/`_num_relations()`; `NullRelations` (`:302`, IIT 3.0 — no relations); `ConcreteRelations` (`:330`, a `frozenset` of `Relation`); `AnalyticalRelations` (`:359`, closed-form, holds `.distinctions`, not iterable). `AnalyticalRelations._sum_phi` (`:368`) decomposes per purview unit via `distinctions.purview_inclusion(max_order=1)` and `combinatorics.sum_of_minimum_among_subsets`, then adds `self.self_relations` at full φ.
- **`distinctions.purview_inclusion(max_order=1)`** yields `(frozenset{unit}, {distinctions whose purview_union ⊇ that unit})` pairs — the `D_u` sets.
- **`ResolvedDistinctions([...])`** (`pyphi/models/distinctions.py:254`) constructs a distinctions bag; it sorts internally, so order in equals canonical order out.
- **Run a single test:** `uv run pytest test/test_phi_fold.py::test_name -x -q`. **Commit boundary full check:** `uv run pytest` with no path (doctest-inclusive). Commits: `git -c commit.gpgsign=false commit`; never `--no-verify`; if a commit silently doesn't land, the pre-commit hook reformatted — `git add` the same files again and re-commit.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `pyphi/combinatorics.py` | `sum_of_minimum_over_size_among_subsets` helper | modify |
| `pyphi/relations.py` | `apportioned_sum_phi()` on the `Relations` types; `AnalyticalFoldRelations` | modify |
| `pyphi/models/ces.py` | `PhiFold`; `CauseEffectStructure.fold()` / `.distinction_folds()` | modify |
| `pyphi/visualize/projection/__init__.py` | reject `PhiFold` in `project_ces` | modify |
| `pyphi/visualize/__init__.py` | one-arg `highlight_phi_fold(fold)`; reject `PhiFold` in `plot_ces` | modify |
| `test/test_combinatorics.py` | helper tests | modify |
| `test/test_phi_fold.py` | all PhiFold + apportioned-sum + cross-validation tests | create |
| `test/test_visualize_simplicial_complex.py` | one-arg highlight + plot_ces guard | modify |
| `changelog.d/phi-fold.feature.md` | changelog fragment | create |

---

## Task 1: Combinatorics helper `sum_of_minimum_over_size_among_subsets`

**Files:**
- Modify: `pyphi/combinatorics.py` (add after `sum_of_minimum_among_subsets`, currently ending at line 201)
- Test: `test/test_combinatorics.py`

- [ ] **Step 1: Write the failing tests**

Add to `test/test_combinatorics.py` (top of file already has `from pyphi import combinatorics` — verify; if it imports specific names, add `sum_of_minimum_over_size_among_subsets` there too):

```python
import itertools

import pytest

from pyphi.combinatorics import sum_of_minimum_over_size_among_subsets


def _brute_force_min_over_size(values):
    total = 0.0
    for size in range(2, len(values) + 1):
        for subset in itertools.combinations(values, size):
            total += min(subset) / size
    return total


@pytest.mark.parametrize(
    "values",
    [
        [],
        [3.0],
        [1.0, 2.0],
        [1.0, 2.0, 3.0],
        [3.0, 1.0, 2.0],          # unsorted
        [2.0, 2.0, 2.0],          # ties
        [0.5, 1.5, 0.25, 4.0, 4.0, 0.1],
    ],
)
def test_sum_of_minimum_over_size_matches_brute_force(values):
    assert sum_of_minimum_over_size_among_subsets(values) == pytest.approx(
        _brute_force_min_over_size(values)
    )


def test_sum_of_minimum_over_size_small_inputs_are_zero():
    assert sum_of_minimum_over_size_among_subsets([]) == 0.0
    assert sum_of_minimum_over_size_among_subsets([7.0]) == 0.0


def test_sum_of_minimum_over_size_known_value():
    # subsets of [1,2,3] size>=2: {1,2}=0.5, {1,3}=0.5, {2,3}=1.0, {1,2,3}=1/3
    assert sum_of_minimum_over_size_among_subsets([1.0, 2.0, 3.0]) == pytest.approx(
        0.5 + 0.5 + 1.0 + 1.0 / 3.0
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_combinatorics.py -k over_size -x -q`
Expected: FAIL with `ImportError`/`cannot import name sum_of_minimum_over_size_among_subsets`.

- [ ] **Step 3: Implement the helper**

In `pyphi/combinatorics.py`, immediately after `sum_of_minimum_among_subsets` (after line 201), add:

```python
def sum_of_minimum_over_size_among_subsets(values: Sequence[float]) -> float:
    """Return the sum of ``min(S) / |S|`` over all subsets ``S`` with size > 1.

    For values sorted ascending as ``v_0 <= ... <= v_{n-1}``, ``v_i`` is the
    minimum of exactly those subsets containing ``i`` whose other elements all
    come from the ``a = n - 1 - i`` larger positions. Summing ``1/|S|`` over
    those subsets gives the closed-form coefficient

        Σ_{k=2}^{a+1} C(a, k-1) / k  =  (2^{a+1} - 1 - (a+1)) / (a+1)

    via the hockey-stick identity, so the result is a sorted dot product.
    This is the apportioned (``φ_r / |r|``) analogue of
    :func:`sum_of_minimum_among_subsets`.
    """
    n = len(values)
    if n < 2:
        return 0.0
    sorted_values = np.sort(np.asarray(values, dtype=float))
    coefficients = np.zeros(n)
    for i in range(n):
        a = n - 1 - i
        if a > 0:
            coefficients[i] = (2 ** (a + 1) - 1 - (a + 1)) / (a + 1)
    return float(np.sum(sorted_values * coefficients))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_combinatorics.py -k over_size -x -q`
Expected: PASS (all parametrizations).

- [ ] **Step 5: Commit**

```bash
git add pyphi/combinatorics.py test/test_combinatorics.py
git -c commit.gpgsign=false commit -m "Add sum_of_minimum_over_size_among_subsets combinatorics helper"
```

---

## Task 2: `apportioned_sum_phi()` on the relation types

**Files:**
- Modify: `pyphi/relations.py` (`Relations` `:270`, `NullRelations` `:302`, `ConcreteRelations` `:330`, `AnalyticalRelations` `:359`)
- Test: `test/test_phi_fold.py` (create)

`apportioned_sum_phi()` returns Σ_r φ_r/|r| over the relation set. It is the building block the fold's `sum_phi_relations_contribution` consumes.

- [ ] **Step 1: Write the failing tests**

Create `test/test_phi_fold.py`:

```python
"""Tests for Φ-folds and apportioned relation sums."""

import pytest

from pyphi import examples
from pyphi.relations import (
    AnalyticalRelations,
    ConcreteRelations,
    NullRelations,
)


@pytest.fixture(scope="module")
def xor_ces():
    return examples.xor_system().ces()


def test_concrete_apportioned_sum_phi_matches_manual(xor_ces):
    relations = xor_ces.relations
    expected = sum(r.phi / len(r) for r in relations)
    assert relations.apportioned_sum_phi() == pytest.approx(expected)


def test_concrete_apportioned_sum_phi_at_most_sum_phi(xor_ces):
    relations = xor_ces.relations
    # dividing each term by |r| >= 1 cannot increase the total
    assert relations.apportioned_sum_phi() <= relations.sum_phi() + 1e-12


def test_null_relations_apportioned_sum_phi_is_zero():
    assert NullRelations().apportioned_sum_phi() == 0.0


def test_analytical_apportioned_matches_concrete(xor_ces):
    analytical = AnalyticalRelations(xor_ces.distinctions)
    assert analytical.apportioned_sum_phi() == pytest.approx(
        xor_ces.relations.apportioned_sum_phi()
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_phi_fold.py -k apportioned -x -q`
Expected: FAIL with `AttributeError: 'ConcreteRelations' object has no attribute 'apportioned_sum_phi'`.

- [ ] **Step 3: Implement on the base + each subclass**

In `pyphi/relations.py`, `Relations.__init__` (`:273`) currently sets two cache fields. Add a third:

```python
    def __init__(self, *args, **kwargs):
        self._num_relations_cached = None
        self._sum_phi_cached = None
        self._apportioned_sum_phi_cached = None
```

Add the cached accessor to `Relations`, right after `sum_phi` (after line 280):

```python
    def apportioned_sum_phi(self):
        if self._apportioned_sum_phi_cached is None:
            self._apportioned_sum_phi_cached = self._apportioned_sum_phi()  # type: ignore[attr-defined]  # Defined in subclass
        return self._apportioned_sum_phi_cached
```

In `NullRelations`, after its `_sum_phi` (after line 314):

```python
    def _apportioned_sum_phi(self):
        return 0
```

In `ConcreteRelations`, after its `_sum_phi` (after line 332):

```python
    def _apportioned_sum_phi(self):
        return sum(relation.phi / len(relation) for relation in self)
```

In `AnalyticalRelations`, after its `_sum_phi` (after line 382):

```python
    def _apportioned_sum_phi(self):
        apportioned = 0
        # Apportioned sum excluding self-relations
        for _, overlapping_distinctions in self.distinctions.purview_inclusion(
            max_order=1
        ):
            apportioned += combinatorics.sum_of_minimum_over_size_among_subsets(
                [
                    distinction.phi / len(distinction.purview_union)
                    for distinction in overlapping_distinctions
                ]
            )
        # Self-relations have |r| = 1, so they enter at full phi
        apportioned += sum(relation.phi for relation in self.self_relations)
        return apportioned
```

(`combinatorics` is already imported in `relations.py` — verify with `grep -n "import combinatorics\|from . import combinatorics" pyphi/relations.py`; the existing `_sum_phi` already calls `combinatorics.sum_of_minimum_among_subsets`, so the import is present.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_phi_fold.py -k apportioned -x -q`
Expected: PASS, including `test_analytical_apportioned_matches_concrete` (cross-validates the new helper against brute-force concrete on xor).

- [ ] **Step 5: Commit**

```bash
git add pyphi/relations.py test/test_phi_fold.py
git -c commit.gpgsign=false commit -m "Add apportioned_sum_phi (sum of phi_r/|r|) to relation types"
```

---

## Task 3: `PhiFold` type + `fold()` / `distinction_folds()` (concrete parents)

**Files:**
- Modify: `pyphi/models/ces.py` (`CauseEffectStructure` `:42`; imports at `:34`)
- Test: `test/test_phi_fold.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_phi_fold.py`:

```python
from pyphi.models.ces import CauseEffectStructure, PhiFold


def test_fold_is_phi_fold_with_parent(xor_ces):
    seed = xor_ces.distinctions[0]
    fold = xor_ces.fold([seed])
    assert isinstance(fold, PhiFold)
    assert isinstance(fold, CauseEffectStructure)
    assert fold.parent is xor_ces
    assert [d.mechanism for d in fold.distinctions] == [seed.mechanism]


def test_fold_accepts_mechanism_tuples(xor_ces):
    by_mech = xor_ces.fold([(0, 1)])
    by_obj = xor_ces.fold([d for d in xor_ces.distinctions if d.mechanism == (0, 1)])
    assert [d.mechanism for d in by_mech.distinctions] == [(0, 1)]
    assert by_mech.relations.sum_phi() == pytest.approx(by_obj.relations.sum_phi())


def test_fold_unknown_mechanism_raises(xor_ces):
    with pytest.raises(ValueError, match="not in this cause-effect structure"):
        xor_ces.fold([(9,)])


def test_fold_relations_are_exactly_the_incident_ones(xor_ces):
    seed = xor_ces.distinctions[0]
    fold = xor_ces.fold([seed])
    expected = {r for r in xor_ces.relations if seed in r}
    assert set(fold.relations) == expected
    # every relation in the fold touches the seed
    assert all(seed in r for r in fold.relations)


def test_big_phi_contribution_matches_manual(xor_ces):
    seed = xor_ces.distinctions[0]
    fold = xor_ces.fold([seed])
    expected = seed.phi + sum(r.phi / len(r) for r in xor_ces.relations if seed in r)
    assert fold.big_phi_contribution == pytest.approx(expected)


def test_distinction_folds_tile_big_phi(xor_ces):
    total = sum(fold.big_phi_contribution for fold in xor_ces.distinction_folds())
    assert total == pytest.approx(xor_ces.big_phi)


def test_fold_big_phi_is_universal_not_contribution(xor_ces):
    # big_phi keeps its standalone-CES meaning (full phi_r), distinct from Phi_d
    seed = xor_ces.distinctions[0]
    fold = xor_ces.fold([seed])
    full = seed.phi + sum(r.phi for r in xor_ces.relations if seed in r)
    assert fold.big_phi == pytest.approx(full)
    assert fold.big_phi >= fold.big_phi_contribution


def test_fold_relations_less_ces_raises():
    # A relations-less (e.g. IIT 3.0) structure carries NullRelations.
    from pyphi.models.distinctions import ResolvedDistinctions
    from pyphi.relations import NullRelations

    bare = CauseEffectStructure(
        sia=None, distinctions=ResolvedDistinctions(()), relations=NullRelations()
    )
    with pytest.raises(ValueError, match="requires relations"):
        bare.fold([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_phi_fold.py -k "fold or contribution or tile" -x -q`
Expected: FAIL — `AttributeError: 'CauseEffectStructure' object has no attribute 'fold'` and `cannot import name 'PhiFold'`.

- [ ] **Step 3: Implement `PhiFold` and the constructors**

In `pyphi/models/ces.py`, update the imports near line 27-34. Add `field` and the `Distinction`/`ResolvedDistinctions`/relation types:

```python
from dataclasses import dataclass
from dataclasses import field
```

and (with the existing `from .distinctions import ResolvedDistinctions`) add a TYPE_CHECKING-free local import inside the method to avoid a potential import cycle (see Step note). Add `fold` and `distinction_folds` methods to `CauseEffectStructure` (after `to_json`, end of class at line 117):

```python
    def fold(self, distinctions) -> "PhiFold":
        """Return the Φ-fold seeded by the given distinctions.

        ``distinctions`` is an iterable of :class:`Distinction` objects or
        mechanism index-tuples drawn from this structure. The fold contains
        those distinctions and every relation incident to at least one of
        them.
        """
        from .distinction import Distinction
        from pyphi.relations import (
            AnalyticalRelations,
            ConcreteRelations,
            NullRelations,
        )

        by_mechanism = {tuple(d.mechanism): d for d in self.distinctions}
        seeds = []
        for item in distinctions:
            mechanism = (
                tuple(item.mechanism)
                if isinstance(item, Distinction)
                else tuple(item)
            )
            if mechanism not in by_mechanism:
                raise ValueError(
                    f"mechanism {mechanism} not in this cause-effect structure"
                )
            seeds.append(by_mechanism[mechanism])

        if isinstance(self.relations, NullRelations):
            raise ValueError(
                "folding requires relations; this cause-effect structure has "
                "none (e.g. IIT 3.0)"
            )
        seed_set = set(seeds)
        if isinstance(self.relations, ConcreteRelations):
            incident = ConcreteRelations(
                r for r in self.relations if not seed_set.isdisjoint(r)
            )
        elif isinstance(self.relations, AnalyticalRelations):
            from pyphi.relations import AnalyticalFoldRelations

            incident = AnalyticalFoldRelations(
                self.distinctions, ResolvedDistinctions(seeds)
            )
        else:
            raise TypeError(
                f"cannot fold a structure with {type(self.relations).__name__} "
                "relations"
            )
        return PhiFold(
            sia=self.sia,
            distinctions=ResolvedDistinctions(seeds),
            relations=incident,
            config=self.config,
            parent=self,
        )

    def distinction_folds(self):
        """Yield the single-distinction Φ-fold of each distinction, in order."""
        for distinction in self.distinctions:
            yield self.fold([distinction])
```

After the `CauseEffectStructure` class (after line 117), add `PhiFold`:

```python
@dataclass(frozen=True, eq=False)
class PhiFold(CauseEffectStructure):
    """A slice of a cause-effect structure: a set of seed distinctions and
    the relations incident to them.

    ``distinctions`` holds the seeds; ``relations`` holds every relation that
    binds at least one seed; ``sia`` and ``config`` come from the structure the
    fold was taken from, available as ``parent``. A fold is not a self-contained
    cause-effect structure — its relations may reference distinctions outside
    ``distinctions`` — so it is not accepted by ``plot_ces``/``project_ces``;
    use ``highlight_phi_fold`` to visualize it.
    """

    parent: "CauseEffectStructure" = field(kw_only=True)

    @property
    def sum_phi_relations_contribution(self):
        """Σ over incident relations of ``φ_r / |r|`` — the relations' share of
        the fold's contribution to the structure's Φ.
        """
        return self.relations.apportioned_sum_phi()

    @property
    def big_phi_contribution(self):
        """The fold's additive contribution to the structure's Φ (the paper's
        Φ_d): the seed distinctions' full φ plus each incident relation's φ
        apportioned across the distinctions it binds. Summing this over a
        structure's single-distinction folds recovers its ``big_phi``.
        """
        return self.sum_phi_distinctions + self.sum_phi_relations_contribution
```

**Import-cycle note:** the relation imports inside `fold()` are deliberately local — `pyphi/relations.py` imports model types, so a top-level `from pyphi.relations import ...` in `ces.py` may cycle. If a top-level import proves clean when you run `uv run python -c "import pyphi"`, you may hoist them; otherwise keep them local.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_phi_fold.py -k "fold or contribution or tile" -x -q`
Expected: PASS. The tiling test (`test_distinction_folds_tile_big_phi`) is the key invariant — Σ Φ_d == big_phi.

- [ ] **Step 5: Verify import is clean and commit**

```bash
uv run python -c "import pyphi; from pyphi.models.ces import PhiFold; print('ok')"
git add pyphi/models/ces.py test/test_phi_fold.py
git -c commit.gpgsign=false commit -m "Add PhiFold with big_phi_contribution and fold constructors"
```

---

## Task 4: `AnalyticalFoldRelations` (analytical parents)

**Files:**
- Modify: `pyphi/relations.py` (add after `AnalyticalRelations`, after line 404)
- Test: `test/test_phi_fold.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_phi_fold.py`:

```python
from pyphi.relations import AnalyticalFoldRelations
from pyphi.models.ces import CauseEffectStructure as _CES


@pytest.fixture(scope="module")
def xor_ces_analytical(xor_ces):
    # same distinctions/sia, but analytical relations
    return _CES(
        sia=xor_ces.sia,
        distinctions=xor_ces.distinctions,
        relations=AnalyticalRelations(xor_ces.distinctions),
    )


def test_analytical_fold_sum_matches_concrete_fold(xor_ces, xor_ces_analytical):
    for distinction in xor_ces.distinctions:
        mechanism = distinction.mechanism
        concrete_fold = xor_ces.fold([mechanism])
        analytical_fold = xor_ces_analytical.fold([mechanism])
        assert analytical_fold.relations.sum_phi() == pytest.approx(
            concrete_fold.relations.sum_phi()
        )
        assert analytical_fold.relations.num_relations() == (
            concrete_fold.relations.num_relations()
        )
        assert analytical_fold.relations.apportioned_sum_phi() == pytest.approx(
            concrete_fold.relations.apportioned_sum_phi()
        )


def test_analytical_fold_tiles_big_phi(xor_ces_analytical):
    total = sum(
        fold.big_phi_contribution
        for fold in xor_ces_analytical.distinction_folds()
    )
    assert total == pytest.approx(xor_ces_analytical.big_phi)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_phi_fold.py -k analytical_fold -x -q`
Expected: FAIL — `cannot import name 'AnalyticalFoldRelations'`.

- [ ] **Step 3: Implement `AnalyticalFoldRelations`**

In `pyphi/relations.py`, after the `AnalyticalRelations` class (after line 404), add:

```python
class AnalyticalFoldRelations(AnalyticalRelations):
    """Closed-form sums over the relations incident to a set of seed
    distinctions within a parent structure.

    Every analytical quantity is a sum over relations, and a relation either
    touches the seed set ``F`` or it does not, so the incident total is
    ``total(D) − total(D∖F)`` over two plain :class:`AnalyticalRelations`.
    Self-relations of ``D∖F`` cancel in the difference; self-relations of the
    seeds survive. Enumeration (iteration, faces) is not supported — use
    concrete relations for that.
    """

    def __init__(self, parent_distinctions, seeds):
        super().__init__(parent_distinctions)
        self._full = AnalyticalRelations(parent_distinctions)
        seed_mechanisms = {tuple(d.mechanism) for d in seeds}
        from pyphi.models.distinctions import ResolvedDistinctions

        complement = ResolvedDistinctions(
            d
            for d in parent_distinctions
            if tuple(d.mechanism) not in seed_mechanisms
        )
        self._complement = AnalyticalRelations(complement)

    def _sum_phi(self):
        return self._full.sum_phi() - self._complement.sum_phi()

    def _num_relations(self):
        return self._full.num_relations() - self._complement.num_relations()

    def _apportioned_sum_phi(self):
        return (
            self._full.apportioned_sum_phi()
            - self._complement.apportioned_sum_phi()
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_phi_fold.py -k analytical_fold -x -q`
Expected: PASS. `test_analytical_fold_sum_matches_concrete_fold` cross-validates the subtraction identity, the new helper, and the existing analytical formulas against brute-force concrete on every xor distinction fold.

- [ ] **Step 5: Commit**

```bash
git add pyphi/relations.py test/test_phi_fold.py
git -c commit.gpgsign=false commit -m "Add AnalyticalFoldRelations via the total(D) - total(D-F) identity"
```

---

## Task 5: Visualize guards + one-arg `highlight_phi_fold`

**Files:**
- Modify: `pyphi/visualize/projection/__init__.py` (`project_ces` `:226`)
- Modify: `pyphi/visualize/__init__.py` (`plot_ces` `:44`; `highlight_phi_fold` `:151`)
- Test: `test/test_visualize_simplicial_complex.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_visualize_simplicial_complex.py` (it already imports plotly/agg-free; confirm `from pyphi import examples` is present, add if not):

```python
import pytest

from pyphi import examples
from pyphi.visualize import highlight_phi_fold, plot_ces


def test_plot_ces_rejects_phi_fold():
    ces = examples.xor_system().ces()
    fold = ces.fold([ces.distinctions[0]])
    with pytest.raises(TypeError, match="highlight_phi_fold"):
        plot_ces(fold)


def test_highlight_phi_fold_one_argument():
    ces = examples.xor_system().ces()
    fold = ces.fold([ces.distinctions[0]])
    figure = highlight_phi_fold(fold)
    assert figure is not None
    assert len(figure.data) > 0


def test_highlight_phi_fold_two_argument_still_works():
    ces = examples.xor_system().ces()
    fold = ces.fold([ces.distinctions[0]])
    figure = highlight_phi_fold(ces, fold)
    assert figure is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_visualize_simplicial_complex.py -k "phi_fold or reject" -x -q`
Expected: FAIL — `plot_ces` renders instead of raising; `highlight_phi_fold(fold)` raises `TypeError: missing 'phi_fold'`.

- [ ] **Step 3a: Guard `project_ces`**

In `pyphi/visualize/projection/__init__.py`, at the top of `project_ces` (`:226`), before any projection work:

```python
def project_ces(ces, node_labels=None) -> CESProjection:
    from pyphi.models.ces import PhiFold

    if isinstance(ces, PhiFold):
        raise TypeError(
            "cannot project a PhiFold (its relations may reference distinctions "
            "outside the fold); use highlight_phi_fold to visualize a fold"
        )
    ...  # existing body unchanged
```

- [ ] **Step 3b: Guard `plot_ces`**

In `pyphi/visualize/__init__.py`, at the top of `plot_ces` (`:44`), before `project_ces` is called:

```python
    from pyphi.models.ces import PhiFold

    if isinstance(ces_, PhiFold):
        raise TypeError(
            "cannot plot a PhiFold directly; use highlight_phi_fold(fold) to "
            "render a fold against its parent structure"
        )
```

(Place this immediately after the docstring, before the `projection = project_ces(...)` line.)

- [ ] **Step 3c: One-arg `highlight_phi_fold`**

In `pyphi/visualize/__init__.py`, change `highlight_phi_fold` (`:151`) so the second positional is optional and a lone `PhiFold` supplies its own parent. Replace the signature and the first lines of the body:

```python
def highlight_phi_fold(
    ces_,
    phi_fold=None,
    *,
    theme=DEFAULT_THEME,
    node_labels=None,
    fig=None,
    geometry=None,
    show=None,
):
    """Plot a |CauseEffectStructure| dimmed, highlighting a phi-fold.

    Call with a single :class:`PhiFold` to highlight it against its own
    ``parent``, or with ``(ces_, phi_fold)`` to highlight any object with a
    ``distinctions`` attribute against an explicit structure.
    """
    from .render.simplicial_complex import render_simplicial_complex
    from pyphi.models.ces import PhiFold

    if phi_fold is None:
        if not isinstance(ces_, PhiFold):
            raise TypeError(
                "single-argument highlight_phi_fold requires a PhiFold; pass "
                "(ces, phi_fold) otherwise"
            )
        phi_fold = ces_
        ces_ = phi_fold.parent

    projection = project_ces(ces_, node_labels=node_labels)
    ...  # rest of the body unchanged
```

Note: `project_ces(ces_, ...)` here receives the *parent* (a `CauseEffectStructure`, not a `PhiFold`), so the new guard in 3a does not fire.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_visualize_simplicial_complex.py -k "phi_fold or reject" -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/__init__.py pyphi/visualize/projection/__init__.py test/test_visualize_simplicial_complex.py
git -c commit.gpgsign=false commit -m "Guard plot_ces/project_ces against PhiFold; add one-arg highlight_phi_fold"
```

---

## Task 6: Changelog fragment + full verification

**Files:**
- Create: `changelog.d/phi-fold.feature.md`

- [ ] **Step 1: Write the changelog fragment**

```bash
cat > changelog.d/phi-fold.feature.md <<'EOF'
Added `PhiFold`, the cause-effect-structure slice of one or more distinctions
with their incident relations. `CauseEffectStructure.fold(distinctions)` and
`.distinction_folds()` construct folds; `PhiFold.big_phi_contribution` gives the
fold's additive share of the structure's Φ (distinctions at full φ, relations
apportioned as φ_r/|r|), which tiles: summing it over a structure's
single-distinction folds recovers `big_phi`. Fold sums are computed in closed
form for analytical-relations structures via `AnalyticalFoldRelations`, with no
relation enumeration. `highlight_phi_fold(fold)` now accepts a lone fold and
renders it against its parent.
EOF
```

- [ ] **Step 2: Run the full doctest-inclusive suite**

Run: `uv run pytest test/test_phi_fold.py test/test_combinatorics.py test/test_visualize_simplicial_complex.py -q`
Expected: PASS.

Then the doctest sweep on the touched source modules (no path argument is the canonical full check, but it is slow; scope to the changed modules first):

Run: `uv run pytest --doctest-modules pyphi/models/ces.py pyphi/relations.py pyphi/combinatorics.py -q`
Expected: PASS (no doctests broken by the additions).

- [ ] **Step 3: Lint**

Run: `uv run ruff check pyphi/models/ces.py pyphi/relations.py pyphi/combinatorics.py pyphi/visualize/__init__.py pyphi/visualize/projection/__init__.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add changelog.d/phi-fold.feature.md
git -c commit.gpgsign=false commit -m "Add changelog fragment for PhiFold"
```

- [ ] **Step 5: Final full-suite check (commit boundary)**

Run the doctest-inclusive sweep with no path argument (per project testing rules) in the background and the fast lane in the foreground:

```bash
uv run pytest -q
```
Expected: green. If `examples.xor_system().ces()` makes the targeted files slow, run `uv run pytest test/ -m "not slow" -q` for fast feedback and the full `uv run pytest` once before declaring done.

---

## Self-review notes

- **Spec coverage:** `PhiFold` subclass + `parent` (Task 3); `big_phi_contribution`/`sum_phi_relations_contribution` (Task 3); `fold()`/`distinction_folds()` with object/tuple/unknown handling (Task 3); `Relations.apportioned_sum_phi()` across concrete/analytical/null (Task 2); `sum_of_minimum_over_size_among_subsets` (Task 1); `AnalyticalFoldRelations` subtraction identity (Task 4); relations-less and plot guards (Task 3 + Task 5); one-arg `highlight_phi_fold` (Task 5); tiling invariant + analytical≡concrete cross-validation (Tasks 3–4); changelog (Task 6). All spec sections map to a task.
- **Name consistency:** `big_phi_contribution`, `sum_phi_relations_contribution`, `apportioned_sum_phi`, `AnalyticalFoldRelations`, `sum_of_minimum_over_size_among_subsets`, `fold`, `distinction_folds`, `PhiFold.parent` used identically in every task.
- **Deferred to sub-project 3 (not here):** triggering coefficients, perception, the Eq (10)/(12)/(11) reconciliation. `big_phi_contribution` is the structural primitive those will build on.
- **One judgment call to watch:** the relation imports inside `CauseEffectStructure.fold()` are local to dodge an import cycle; if `import pyphi` stays clean with top-level imports, hoisting is fine (Task 3 Step 5 verifies import health).
