# Cause-Effect Structure Algebra Operations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the operations from
`docs/superpowers/specs/2026-07-07-ces-algebra-exploration.md` §10, with the
fold-contribution semantics generalized so that folds over *any* partition of
a structure's distinctions — multi-seed included — tile Φ exactly.

**Architecture:** All operations are selections, closures, measures, or frame
maps over existing result objects; no computation-path code changes. The fold
measure changes from count-once apportionment to share-weighting
(`φ_r · |r ∩ F| / |r|`), which is additive over disjoint seed sets. A new
`InducedSubstructure` view (relation-closed, unlike `PhiFold`) carries
`induce` and `meet`. Structure isomorphism and relabeling are value-level:
isomorphism compares canonical signatures; relabeling rebuilds each nested
result object through its public constructor.

**Tech Stack:** Python 3.13, numpy, pytest. No new dependencies.

## Global Constraints

- Run everything with `uv run` (e.g. `uv run pytest`, `uv run python`).
- Work in a git worktree under `.claude/worktrees/` (confirm branch name with
  the user at execution start; base on the current working branch).
- Float comparisons in tests use `pytest.approx` (default tolerance) — never
  `==` on φ values.
- Every user-facing change gets a changelog fragment in `changelog.d/`
  (`<name>.<type>.md`), committed with the task.
- Docstrings describe final state only — no migration narrative, no planning
  artifacts (no task numbers, no "generalized from", no design-alternative
  discussion).
- Do not use `git checkout -- <path>` for cleanup; other sessions may have
  unrelated working-tree changes — stage only files this plan touches.
- Never pass `--no-verify` to git. If pre-commit hooks fail, fix the failure.
- The final verification (Task 7) must run `uv run pytest` **with no path
  argument** at least once (bare paths skip the doctest sweep).

## Background for implementers (read once)

A `CauseEffectStructure` (`pyphi/models/ces.py`) is a frozen dataclass with
fields `sia`, `distinctions` (a `ResolvedDistinctions` sequence), `relations`
(a `Relations` — either `ConcreteRelations`, an enumerated frozenset of
`Relation` objects, or `AnalyticalRelations`, a non-iterable closed-form
view, or `NullRelations`), `config`, `provenance`. A `Relation` is a
frozenset of `Distinction` objects; its `phi` is computed lazily from its
own members only (`pyphi/relations.py:195`) — this locality is what makes
every operation below exact. `PhiFold` (`ces.py:320`) is the existing
"seeds + incident relations" view with a `parent` back-reference.

Quick way to get a real structure in tests: `examples.xor_system().ces()`
(4 distinctions, 15 relations — every subset of distinctions is related, so
any two distinctions share relations). `examples.grid3_system().ces()` gives
an asymmetric 3-node structure (7 distinctions, 39 relations).

---

### Task 1: `Distinctions.filter`

**Files:**
- Modify: `pyphi/models/distinctions.py` (add method to `Distinctions`, after `__getitem__` at line ~144)
- Test: `test/models/test_distinctions.py` (create)
- Create: `changelog.d/distinctions-filter.feature.md`

**Interfaces:**
- Produces: `Distinctions.filter(predicate: Callable[[Distinction], bool]) -> Distinctions`
  — returns the same runtime subtype as the receiver (`ResolvedDistinctions`
  stays `ResolvedDistinctions`). Used by later tasks' examples but not
  required by their code.

- [ ] **Step 1: Write the failing tests**

Create `test/models/test_distinctions.py`:

```python
"""Tests for predicate selection on distinction bags."""

import pytest

from pyphi import examples
from pyphi.models.distinctions import ResolvedDistinctions
from pyphi.models.distinctions import UnresolvedDistinctions


@pytest.fixture(scope="module")
def xor_ces():
    return examples.xor_system().ces()


def test_filter_selects_by_predicate(xor_ces):
    result = xor_ces.distinctions.filter(lambda d: len(d.mechanism) == 2)
    assert all(len(d.mechanism) == 2 for d in result)
    assert len(result) == sum(
        1 for d in xor_ces.distinctions if len(d.mechanism) == 2
    )


def test_filter_preserves_subtype(xor_ces):
    assert isinstance(xor_ces.distinctions, ResolvedDistinctions)
    result = xor_ces.distinctions.filter(lambda d: True)
    assert type(result) is type(xor_ces.distinctions)


def test_filter_on_unresolved_preserves_subtype(xor_ces):
    unresolved = UnresolvedDistinctions(xor_ces.distinctions)
    result = unresolved.filter(lambda d: True)
    assert type(result) is UnresolvedDistinctions


def test_filter_empty_result(xor_ces):
    result = xor_ces.distinctions.filter(lambda d: False)
    assert len(result) == 0
    assert type(result) is type(xor_ces.distinctions)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/models/test_distinctions.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'filter'`

- [ ] **Step 3: Implement**

In `pyphi/models/distinctions.py`, add to the `Distinctions` class body
(directly after `__getitem__`):

```python
    def filter(self, predicate) -> Distinctions:
        """Return the distinctions satisfying ``predicate``.

        Preserves the runtime subtype, so filtering a
        :class:`ResolvedDistinctions` yields a :class:`ResolvedDistinctions`.
        """
        return type(self)(d for d in self if predicate(d))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/models/test_distinctions.py -v`
Expected: 4 passed

- [ ] **Step 5: Changelog fragment and commit**

```bash
echo 'Added `Distinctions.filter(predicate)` — subtype-preserving predicate selection on distinction bags.' > changelog.d/distinctions-filter.feature.md
git add pyphi/models/distinctions.py test/models/test_distinctions.py changelog.d/distinctions-filter.feature.md
git commit -m "Add Distinctions.filter for predicate selection"
```

---

### Task 2: Share-weighted fold contribution (multi-seed folds tile Φ)

The current `PhiFold.sum_phi_relations_contribution` counts each incident
relation once at `φ_r/|r|`. That is correct for single-distinction folds but
subadditive for multi-seed folds: summing the contributions of the blocks of
a partition of the distinctions undercounts Φ whenever a relation binds seeds
from more than one block. The fix weights each incident relation by
`|relata(r) ∩ seeds| / |r|` — the seeds' share — which agrees with the old
value on singletons and makes any partition of the distinctions into folds
tile Φ exactly.

**Files:**
- Modify: `pyphi/models/ces.py:384-398` (`PhiFold.sum_phi_relations_contribution`, `big_phi_contribution` docstring)
- Modify: `pyphi/relations.py:495-525` (`AnalyticalFoldRelations`: store seeds, add `share_weighted_sum_phi`)
- Test: `test/models/test_phi_fold.py` (extend)
- Create: `changelog.d/fold-contribution-partition.change.md`

**Interfaces:**
- Consumes: `Distinctions.filter` not required; uses existing `fold`.
- Produces: `AnalyticalFoldRelations.share_weighted_sum_phi() -> float` —
  Σ over incident relations of `φ_r · |r ∩ F| / |r|`, computed in closed
  form. `PhiFold.sum_phi_relations_contribution` / `big_phi_contribution`
  keep their names; their multi-seed values change.

- [ ] **Step 1: Write the failing tests**

Append to `test/models/test_phi_fold.py`:

```python
def test_multiseed_fold_contribution_is_additive(xor_ces):
    # every pair of xor distinctions shares relations, so this discriminates
    # share-weighting from counting each incident relation once
    a, b = xor_ces.distinctions[0], xor_ces.distinctions[1]
    combined = xor_ces.fold([a, b]).big_phi_contribution
    separate = (
        xor_ces.fold([a]).big_phi_contribution
        + xor_ces.fold([b]).big_phi_contribution
    )
    assert combined == pytest.approx(separate)


def test_fold_partition_tiles_big_phi(xor_ces):
    ds = list(xor_ces.distinctions)
    for blocks in ([ds[:2], ds[2:]], [ds[:1], ds[1:]], [ds[:3], ds[3:]]):
        total = sum(xor_ces.fold(block).big_phi_contribution for block in blocks)
        assert total == pytest.approx(xor_ces.big_phi)


def test_multiseed_contribution_matches_manual_share_weighting(xor_ces):
    seeds = set(xor_ces.distinctions[:2])
    fold = xor_ces.fold(list(seeds))
    expected = sum(d.phi for d in seeds) + sum(
        r.phi * len(seeds & set(r)) / len(r)
        for r in xor_ces.relations
        if seeds & set(r)
    )
    assert fold.big_phi_contribution == pytest.approx(expected)


def test_analytical_multiseed_contribution_matches_concrete(
    xor_ces, xor_ces_analytical
):
    mechanisms = [d.mechanism for d in xor_ces.distinctions[:2]]
    concrete = xor_ces.fold(mechanisms).big_phi_contribution
    analytical = xor_ces_analytical.fold(mechanisms).big_phi_contribution
    assert analytical == pytest.approx(concrete)


def test_analytical_fold_partition_tiles_big_phi(xor_ces_analytical):
    ds = list(xor_ces_analytical.distinctions)
    total = sum(
        xor_ces_analytical.fold([d.mechanism for d in block]).big_phi_contribution
        for block in (ds[:2], ds[2:])
    )
    assert total == pytest.approx(xor_ces_analytical.big_phi)
```

Note: `xor_ces_analytical` is an existing module fixture in this file
(defined at line ~99); the new tests reuse it. Move the fixture definition
above the first test that uses it if pytest complains about ordering (it
should not — fixtures resolve by name regardless of position).

- [ ] **Step 2: Run tests to verify the multi-seed ones fail**

Run: `uv run pytest test/models/test_phi_fold.py -v`
Expected: the five new tests FAIL (values differ — current code counts shared
relations once); all pre-existing tests still PASS (singleton semantics are
unchanged by the current code).

- [ ] **Step 3: Implement the concrete path**

In `pyphi/models/ces.py`, replace `PhiFold.sum_phi_relations_contribution`
and the `big_phi_contribution` docstring (lines ~384-398):

```python
    @property
    def sum_phi_relations_contribution(self):
        """Σ over incident relations of ``φ_r · |r ∩ F| / |r|``, where ``F``
        is the set of seed distinctions — the seeds' share of each incident
        relation's φ.
        """
        from pyphi.relations import AnalyticalFoldRelations

        if isinstance(self.relations, AnalyticalFoldRelations):
            return self.relations.share_weighted_sum_phi()
        seeds = set(self.distinctions)
        return sum(
            relation.phi * len(seeds & set(relation)) / len(relation)
            for relation in self.relations
        )

    @property
    def big_phi_contribution(self):
        """The fold's additive contribution to the structure's Φ: the seed
        distinctions' full φ plus the seeds' share of each incident
        relation's φ (``φ_r · |r ∩ F| / |r|``). Summing this over the folds
        of any partition of the structure's distinctions recovers
        ``big_phi``.
        """
        return self.sum_phi_distinctions + self.sum_phi_relations_contribution
```

- [ ] **Step 4: Run the concrete tests**

Run: `uv run pytest test/models/test_phi_fold.py -v`
Expected: the three concrete tests PASS; the two analytical tests FAIL with
`AttributeError: 'AnalyticalFoldRelations' object has no attribute
'share_weighted_sum_phi'`.

- [ ] **Step 5: Implement the analytical path**

In `pyphi/relations.py`, modify `AnalyticalFoldRelations`:

In `__init__` (line ~507), after `super().__init__(parent_distinctions)`,
store the seeds and a cache slot:

```python
    def __init__(self, parent_distinctions, seeds):
        super().__init__(parent_distinctions)
        self._full = AnalyticalRelations(parent_distinctions)
        self._seeds = tuple(seeds)
        self._share_weighted_cached = None
        seed_mechanisms = {tuple(d.mechanism) for d in seeds}
        from pyphi.models.distinctions import ResolvedDistinctions

        complement = ResolvedDistinctions(
            d for d in parent_distinctions if tuple(d.mechanism) not in seed_mechanisms
        )
        self._complement = AnalyticalRelations(complement)
```

Then add the method to the class:

```python
    def share_weighted_sum_phi(self):
        """Σ over incident relations of ``φ_r · |r ∩ F| / |r|``, where ``F``
        is the seed set.

        Computed without enumeration: for a single seed ``d``, the incident
        apportioned total is ``total(D) − total(D∖{d})`` over two closed-form
        :class:`AnalyticalRelations` sums, and the share-weighted total over
        ``F`` is the sum of these single-seed incident totals (a relation of
        degree ``|r|`` binding ``k`` seeds is counted ``k`` times at
        ``φ_r/|r|``).
        """
        if self._share_weighted_cached is None:
            from pyphi.models.distinctions import ResolvedDistinctions

            total = self._full.apportioned_sum_phi()
            result = 0
            for seed in self._seeds:
                seed_mechanism = tuple(seed.mechanism)
                others = ResolvedDistinctions(
                    d
                    for d in self.distinctions
                    if tuple(d.mechanism) != seed_mechanism
                )
                result += total - AnalyticalRelations(others).apportioned_sum_phi()
            self._share_weighted_cached = result
        return self._share_weighted_cached
```

- [ ] **Step 6: Run all fold tests**

Run: `uv run pytest test/models/test_phi_fold.py -v`
Expected: all PASS, including the pre-existing singleton tests
(`test_big_phi_contribution_matches_manual`,
`test_distinction_folds_tile_big_phi`, `test_analytical_fold_tiles_big_phi`)
— for singletons `|r ∩ F| = 1` so the value is unchanged.

- [ ] **Step 7: Check for other callers of the changed properties**

Run: `uv run grep -rn "sum_phi_relations_contribution\|big_phi_contribution" pyphi/ test/ docs/examples/ --include="*.py"`
Expected: only `pyphi/models/ces.py`, `test/models/test_phi_fold.py`, and
possibly `pyphi/matching/perception.py` (`fold_perception`). If
`fold_perception` weights fold components, read it and confirm it uses
per-component `φ_c/|c|` directly rather than the fold aggregate — if it
calls `big_phi_contribution` or `sum_phi_relations_contribution` on
multi-seed folds, run `uv run pytest test/matching/ -v` and inspect any
failures before proceeding (matching tests pin paper values; a failure there
means the matching code needs the count-once quantity and should compute it
locally instead).

- [ ] **Step 8: Changelog fragment and commit**

```bash
cat > changelog.d/fold-contribution-partition.change.md <<'EOF'
`PhiFold.big_phi_contribution` now weights each incident relation by the
seeds' share `|r ∩ F| / |r|` instead of counting it once, so the fold
contributions of any partition of a structure's distinctions sum exactly to
`big_phi`. Single-distinction folds are unchanged.
EOF
git add pyphi/models/ces.py pyphi/relations.py test/models/test_phi_fold.py changelog.d/fold-contribution-partition.change.md
git commit -m "Make multi-seed fold contributions tile big phi exactly"
```

---

### Task 3: `StructureView` base, `relation_closed`, and `induce`

Adds the relation-closed view: `induce(distinctions)` returns the selected
distinctions plus exactly the relations whose relata are all selected
(exact by relation locality — a relation's φ depends only on its relata).
Refactors `PhiFold`'s `parent` field into a shared `StructureView` base and
replaces the projection layer's `isinstance(ces, PhiFold)` rejection with a
`relation_closed` check so induced substructures are projectable.

**Files:**
- Modify: `pyphi/models/ces.py` (add `StructureView`, `InducedSubstructure`, `relation_closed`, `induce`, extract `_resolve_members` from `fold`)
- Modify: `pyphi/visualize/projection/__init__.py:228-233` (rejection condition)
- Test: `test/models/test_ces_views.py` (create)
- Test: `test/visualize/test_visualize_projection.py` (extend)
- Create: `changelog.d/ces-induce.feature.md`

**Interfaces:**
- Produces:
  - `class StructureView(CauseEffectStructure)` with field `parent: CauseEffectStructure` (kw-only); `PhiFold(StructureView)`; `InducedSubstructure(StructureView)`.
  - `CauseEffectStructure.relation_closed -> bool` property: `True` on `CauseEffectStructure` and `InducedSubstructure`, `False` on `PhiFold`.
  - `CauseEffectStructure.induce(distinctions) -> InducedSubstructure` — accepts `Distinction` objects or mechanism index-tuples, like `fold`.
  - `CauseEffectStructure._resolve_members(items) -> list[Distinction]` — shared seed resolution (raises `ValueError` for unknown mechanisms).
- Consumes: nothing from earlier tasks.

- [ ] **Step 1: Write the failing tests**

Create `test/models/test_ces_views.py`:

```python
"""Tests for structure views: induced substructures and relation closure."""

import pytest

from pyphi import examples
from pyphi.models.ces import CauseEffectStructure
from pyphi.models.ces import InducedSubstructure
from pyphi.models.ces import PhiFold
from pyphi.models.ces import StructureView
from pyphi.models.distinctions import ResolvedDistinctions
from pyphi.relations import AnalyticalRelations
from pyphi.relations import ConcreteRelations
from pyphi.relations import concrete_relations


@pytest.fixture(scope="module")
def xor_ces():
    return examples.xor_system().ces()


@pytest.fixture(scope="module")
def xor_ces_analytical(xor_ces):
    return CauseEffectStructure(
        sia=xor_ces.sia,
        distinctions=xor_ces.distinctions,
        relations=AnalyticalRelations(xor_ces.distinctions),
    )


def test_relation_closed_flags(xor_ces):
    assert xor_ces.relation_closed is True
    fold = xor_ces.fold([xor_ces.distinctions[0]])
    assert fold.relation_closed is False
    induced = xor_ces.induce([xor_ces.distinctions[0]])
    assert induced.relation_closed is True


def test_view_hierarchy(xor_ces):
    fold = xor_ces.fold([xor_ces.distinctions[0]])
    induced = xor_ces.induce([xor_ces.distinctions[0]])
    assert isinstance(fold, StructureView)
    assert isinstance(induced, StructureView)
    assert fold.parent is xor_ces
    assert induced.parent is xor_ces


def test_induce_relations_are_the_contained_ones(xor_ces):
    members = list(xor_ces.distinctions)[:3]
    induced = xor_ces.induce(members)
    member_set = set(members)
    expected = {r for r in xor_ces.relations if member_set.issuperset(r)}
    assert set(induced.relations) == expected


def test_induce_matches_fresh_computation(xor_ces):
    # relation locality: filtering the parent's relations equals computing
    # relations over the subset from scratch
    members = ResolvedDistinctions(list(xor_ces.distinctions)[:3])
    induced = xor_ces.induce(members)
    fresh = ConcreteRelations(concrete_relations(members))
    assert frozenset(induced.relations) == frozenset(fresh)


def test_induce_accepts_mechanism_tuples(xor_ces):
    by_mech = xor_ces.induce([(0, 1)])
    assert [d.mechanism for d in by_mech.distinctions] == [(0, 1)]


def test_induce_unknown_mechanism_raises(xor_ces):
    with pytest.raises(ValueError, match="not in this cause-effect structure"):
        xor_ces.induce([(9,)])


def test_induce_composes(xor_ces):
    members = list(xor_ces.distinctions)
    inner = xor_ces.induce(members[:3]).induce(members[:2])
    direct = xor_ces.induce(members[:2])
    assert set(inner.distinctions) == set(direct.distinctions)
    assert frozenset(inner.relations) == frozenset(direct.relations)


def test_induce_all_is_whole_structure(xor_ces):
    induced = xor_ces.induce(list(xor_ces.distinctions))
    assert set(induced.distinctions) == set(xor_ces.distinctions)
    assert frozenset(induced.relations) == frozenset(xor_ces.relations)
    assert induced.big_phi == pytest.approx(xor_ces.big_phi)


def test_induce_analytical_aggregates_match_concrete(xor_ces, xor_ces_analytical):
    mechanisms = [d.mechanism for d in list(xor_ces.distinctions)[:3]]
    concrete = xor_ces.induce(mechanisms)
    analytical = xor_ces_analytical.induce(mechanisms)
    assert analytical.relations.sum_phi() == pytest.approx(
        concrete.relations.sum_phi()
    )
    assert analytical.relations.num_relations() == concrete.relations.num_relations()


def test_fold_still_works_after_refactor(xor_ces):
    # regression guard on the _resolve_members extraction
    seed = xor_ces.distinctions[0]
    fold = xor_ces.fold([seed])
    assert isinstance(fold, PhiFold)
    assert {r for r in xor_ces.relations if seed in r} == set(fold.relations)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/models/test_ces_views.py -v`
Expected: FAIL with `ImportError: cannot import name 'InducedSubstructure'`

- [ ] **Step 3: Implement in `pyphi/models/ces.py`**

Add the `relation_closed` property to `CauseEffectStructure` (after
`components`, ~line 94):

```python
    @property
    def relation_closed(self) -> bool:
        """Whether every relation's relata are members of ``distinctions``.

        True for complete structures and induced substructures; False for
        folds, whose incident relations may reference distinctions outside
        the seed set.
        """
        return True
```

Extract the seed-resolution logic from `fold` into a helper on
`CauseEffectStructure` (the body is the existing `fold` code from lines
~212-224, unchanged):

```python
    def _resolve_members(self, items) -> list:
        """Resolve an iterable of distinctions or mechanism index-tuples to
        this structure's own distinction objects, raising ``ValueError`` for
        mechanisms not in the structure."""
        from .distinction import Distinction

        by_mechanism = {tuple(d.mechanism): d for d in self.distinctions}
        members = []
        for item in items:
            mechanism = (
                tuple(item.mechanism)
                if isinstance(item, Distinction)
                else tuple(item)
            )
            if mechanism not in by_mechanism:
                raise ValueError(
                    f"mechanism {mechanism} not in this cause-effect structure"
                )
            members.append(by_mechanism[mechanism])
        return members
```

Rewrite `fold` to use it (delete the inlined resolution; keep everything
else identical):

```python
    def fold(self, distinctions) -> PhiFold:
        """Return the Φ-fold seeded by the given distinctions.

        ``distinctions`` is an iterable of :class:`Distinction` objects or
        mechanism index-tuples drawn from this structure. The fold contains
        those distinctions and every relation incident to at least one of
        them.
        """
        from pyphi.relations import AnalyticalRelations
        from pyphi.relations import ConcreteRelations
        from pyphi.relations import NullRelations

        seeds = self._resolve_members(distinctions)

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
                f"cannot fold a structure with {type(self.relations).__name__} relations"
            )
        return PhiFold(
            sia=self.sia,
            distinctions=ResolvedDistinctions(seeds),
            relations=incident,
            config=self.config,
            parent=self,
        )
```

Add `induce` after `fold`:

```python
    def induce(self, distinctions) -> InducedSubstructure:
        """Return the induced substructure on the given distinctions: those
        distinctions plus exactly the relations whose relata are all among
        them.

        ``distinctions`` is an iterable of :class:`Distinction` objects or
        mechanism index-tuples drawn from this structure. Because a
        relation's φ depends only on its relata, the induced relation set
        equals what computing relations over the subset from scratch would
        produce. The result is relation-closed (no dangling relata), so it
        can be displayed, aggregated, and projected as a self-contained
        object — but it is a view of this structure, not the cause-effect
        structure of any system.
        """
        from pyphi.relations import AnalyticalRelations
        from pyphi.relations import ConcreteRelations
        from pyphi.relations import NullRelations

        members = self._resolve_members(distinctions)
        member_set = set(members)
        bag = ResolvedDistinctions(members)
        if isinstance(self.relations, NullRelations):
            relations = self.relations
        elif isinstance(self.relations, ConcreteRelations):
            relations = ConcreteRelations(
                r for r in self.relations if member_set.issuperset(r)
            )
        elif isinstance(self.relations, AnalyticalRelations):
            relations = AnalyticalRelations(bag)
        else:
            raise TypeError(
                f"cannot induce a substructure of a structure with "
                f"{type(self.relations).__name__} relations"
            )
        return InducedSubstructure(
            sia=self.sia,
            distinctions=bag,
            relations=relations,
            config=self.config,
            parent=self,
        )
```

Restructure the view classes. Replace the current `PhiFold` class header
(`ces.py:320-333`) so `parent` moves to a shared base, and add
`InducedSubstructure`:

```python
@dataclass(frozen=True, eq=False, repr=False)
class StructureView(CauseEffectStructure):
    """A part of a cause-effect structure, carrying the structure it was
    taken from as ``parent``. Concrete views are :class:`PhiFold`
    (seeds + incident relations) and :class:`InducedSubstructure`
    (distinction subset + the relations contained in it)."""

    parent: CauseEffectStructure = field(kw_only=True)


@dataclass(frozen=True, eq=False, repr=False)
class InducedSubstructure(StructureView):
    """A relation-closed slice of a cause-effect structure: a subset of its
    distinctions together with exactly the relations whose relata all
    belong to the subset.

    Every relation endpoint is present (``relation_closed`` is True), so
    aggregation and projection treat it as self-contained. It is not the
    cause-effect structure of any system: its distinctions were computed
    and congruence-resolved in the parent structure's frame.
    """


@dataclass(frozen=True, eq=False, repr=False)
class PhiFold(StructureView):
    """A slice of a cause-effect structure: a set of seed distinctions and
    the relations incident to them.

    ``distinctions`` holds the seeds; ``relations`` holds every relation that
    binds at least one seed; ``sia`` and ``config`` come from the structure the
    fold was taken from, available as ``parent``. A fold is not a self-contained
    cause-effect structure — its relations may reference distinctions outside
    ``distinctions`` — so it is not accepted by ``plot_ces``/``project_ces``;
    use ``highlight_phi_fold`` to visualize it.
    """

    @property
    def relation_closed(self) -> bool:
        """False: incident relations may reference non-seed distinctions."""
        return False
```

(`PhiFold` keeps its existing `_describe`, `sum_phi_relations_contribution`,
and `big_phi_contribution` members below the header — only the class
signature and the `parent` field move.)

- [ ] **Step 4: Run the view tests**

Run: `uv run pytest test/models/test_ces_views.py test/models/test_phi_fold.py -v`
Expected: all PASS

- [ ] **Step 5: Update the projection rejection and its test**

In `pyphi/visualize/projection/__init__.py` (lines ~228-233), replace:

```python
    from pyphi.models.ces import PhiFold

    if isinstance(ces, PhiFold):
        raise ValueError(
            "cannot project a PhiFold (its relations may reference distinctions "
```

with:

```python
    if not getattr(ces, "relation_closed", True):
        raise ValueError(
            "cannot project a view that is not relation-closed (e.g. a "
            "PhiFold, whose relations may reference distinctions outside "
            "it); project the parent structure or an induced substructure, "
            "or use highlight_phi_fold"
        )
```

(Keep the remainder of the original error-message/return code as it is;
only the type test and message change. Check the original lines first and
preserve any trailing explanatory text of the message.)

Append to `test/visualize/test_visualize_projection.py`:

```python
def test_project_rejects_fold_accepts_induced():
    from pyphi import examples
    from pyphi.visualize.projection import project_ces

    ces = examples.xor_system().ces()
    fold = ces.fold([ces.distinctions[0]])
    with pytest.raises(ValueError, match="relation-closed"):
        project_ces(fold)
    induced = ces.induce(list(ces.distinctions)[:3])
    projection = project_ces(induced)
    assert len(projection.nodes) == 3
```

Note: check the actual attribute for node count on the returned projection —
read the `CESProjection` class in `pyphi/visualize/projection/__init__.py`
first; if the field is named differently (e.g. `distinction_nodes`), use
that name and assert its length is 3.

- [ ] **Step 6: Run projection tests**

Run: `uv run pytest test/visualize/test_visualize_projection.py -v`
Expected: all PASS

- [ ] **Step 7: Changelog fragment and commit**

```bash
cat > changelog.d/ces-induce.feature.md <<'EOF'
Added `CauseEffectStructure.induce(distinctions)`, returning an
`InducedSubstructure` view: the selected distinctions plus exactly the
relations contained among them. Views expose `relation_closed`, and
`project_ces` now accepts any relation-closed object (still rejecting
`PhiFold`).
EOF
git add pyphi/models/ces.py pyphi/visualize/projection/__init__.py test/models/test_ces_views.py test/visualize/test_visualize_projection.py changelog.d/ces-induce.feature.md
git commit -m "Add induced substructure views with relation_closed"
```

---

### Task 4: `meet` with a frame check

**Files:**
- Modify: `pyphi/models/ces.py` (add `_check_same_frame` and `meet` to `CauseEffectStructure`, after `induce`)
- Test: `test/models/test_ces_views.py` (extend)
- Create: `changelog.d/ces-meet.feature.md`

**Interfaces:**
- Consumes: `CauseEffectStructure.induce` (Task 3), `InducedSubstructure` (Task 3).
- Produces: `CauseEffectStructure.meet(other) -> InducedSubstructure`;
  `CauseEffectStructure._check_same_frame(other) -> None` (raises `ValueError`).

- [ ] **Step 1: Write the failing tests**

Append to `test/models/test_ces_views.py`:

```python
@pytest.fixture(scope="module")
def grid3_ces():
    return examples.grid3_system().ces()


def test_meet_with_itself_is_whole(xor_ces):
    met = xor_ces.meet(xor_ces)
    assert isinstance(met, InducedSubstructure)
    assert set(met.distinctions) == set(xor_ces.distinctions)
    assert met.big_phi == pytest.approx(xor_ces.big_phi)


def test_meet_of_induced_views(xor_ces):
    ds = list(xor_ces.distinctions)
    left = xor_ces.induce(ds[:3])
    right = xor_ces.induce(ds[1:])
    met = left.meet(right)
    assert set(met.distinctions) == set(ds[1:3])
    # R commutes with intersection of distinction sets
    expected = frozenset(left.relations) & frozenset(right.relations)
    assert frozenset(met.relations) == expected


def test_meet_is_commutative_on_aggregates(xor_ces):
    ds = list(xor_ces.distinctions)
    left, right = xor_ces.induce(ds[:3]), xor_ces.induce(ds[1:])
    a, b = left.meet(right), right.meet(left)
    assert set(a.distinctions) == set(b.distinctions)
    assert a.relations.sum_phi() == pytest.approx(b.relations.sum_phi())


def test_meet_requires_same_frame(xor_ces, grid3_ces):
    with pytest.raises(ValueError, match="not in the same frame"):
        xor_ces.meet(grid3_ces)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/models/test_ces_views.py -v -k meet`
Expected: FAIL with `AttributeError: ... no attribute 'meet'`

- [ ] **Step 3: Implement**

In `pyphi/models/ces.py`, add to `CauseEffectStructure` after `induce`:

```python
    def _check_same_frame(self, other: CauseEffectStructure) -> None:
        """Raise unless ``other`` is grounded in the same frame: the same
        candidate-system node indices and current state.

        Value-based distinction identity is only meaningful within one
        frame; combining structures across frames would silently produce
        empty results instead of surfacing the mismatch. The config
        snapshots are not compared.
        """
        from pyphi.substrate import _sia_node_indices

        mine = (
            _sia_node_indices(self.sia),
            getattr(self.sia, "current_state", None),
        )
        theirs = (
            _sia_node_indices(other.sia),
            getattr(other.sia, "current_state", None),
        )
        if mine != theirs:
            raise ValueError(
                "structures are not in the same frame: "
                f"(node_indices, state) {mine} != {theirs}"
            )

    def meet(self, other: CauseEffectStructure) -> InducedSubstructure:
        """The induced substructure on the distinctions common to both
        structures (value equality).

        Because a relation's φ depends only on its relata, the result's
        relation set equals the intersection of the two structures'
        relation sets. Requires both structures to be in the same frame;
        raises ``ValueError`` otherwise. The result is a view of ``self``.
        """
        self._check_same_frame(other)
        common = set(self.distinctions) & set(other.distinctions)
        return self.induce(d.mechanism for d in common)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/models/test_ces_views.py -v`
Expected: all PASS

- [ ] **Step 5: Changelog fragment and commit**

```bash
cat > changelog.d/ces-meet.feature.md <<'EOF'
Added `CauseEffectStructure.meet(other)`: the induced substructure on the
distinctions common to both structures. Structures must share a frame (same
candidate-system node indices and state); mismatches raise `ValueError`
instead of silently returning an empty result.
EOF
git add pyphi/models/ces.py test/models/test_ces_views.py changelog.d/ces-meet.feature.md
git commit -m "Add CauseEffectStructure.meet with frame check"
```

---

### Task 5: Structure signatures and isomorphism

Value-level canonical signature for a structure under an index mapping, and
exact isomorphism search over index bijections. The signature covers
mechanisms, mechanism states, purviews, specified states, and φ values
(rounded to the module's `_ROUND = 12` places) — not repertoires or
partitions. Search is brute-force over permutations with cheap invariant
pruning; substrates that afford Φ computation are small, per this module's
existing rationale.

**Files:**
- Modify: `pyphi/automorphism.py` (add `structure_signature`, `are_structures_isomorphic`)
- Modify: `test/example_substrates.py` (add `permuted_system` helper)
- Test: `test/test_automorphism.py` (extend)
- Create: `changelog.d/structure-isomorphism.feature.md`

**Interfaces:**
- Produces:
  - `structure_signature(ces, mapping: dict[int, int] | None = None) -> tuple` — canonical, order-independent value key; identity mapping when `mapping is None`.
  - `are_structures_isomorphic(ces1, ces2) -> bool`.
  - `test/example_substrates.py::permuted_system(system, perm) -> System` — the same dynamics with node `i` of the result holding node `perm[i]` of the input (used again by Task 6 tests).
- Consumes: nothing from earlier tasks (works on plain structures).

- [ ] **Step 1: Add the permuted-system test helper**

Append to `test/example_substrates.py`:

```python
def permuted_system(system, perm):
    """The same system with its units permuted: node ``i`` of the result is
    node ``perm[i]`` of the input. Structures computed from the result are
    relabelings of the input's under ``old -> perm.index(old)``.
    """
    import numpy as np

    from pyphi.substrate import Substrate
    from pyphi.system import System

    arr = np.asarray(system.substrate.tpm.to_array())
    n = len(system.node_indices)
    arr2 = np.transpose(arr, axes=(*perm, n, n + 1))[..., list(perm), :]
    cm2 = np.asarray(system.substrate.cm)[np.ix_(perm, perm)]
    state2 = tuple(system.state[p] for p in perm)
    return System(
        Substrate(arr2, cm=cm2), state=state2, node_indices=tuple(range(n))
    )
```

Note: this helper assumes the system spans its whole substrate with binary
units (the dense TPM array has shape `(2,)*n + (n, 2)`); the example systems
used in tests satisfy this.

- [ ] **Step 2: Write the failing tests**

Append to `test/test_automorphism.py`:

```python
def test_structure_signature_and_isomorphism():
    import pyphi
    from pyphi import examples
    from pyphi.automorphism import are_structures_isomorphic
    from pyphi.automorphism import structure_signature

    from .example_substrates import permuted_system

    with pyphi.config.override(progress_bars=False):
        system = examples.grid3_system()
        ces = system.ces()
        perm = (2, 0, 1)
        permuted_ces = permuted_system(system, perm).ces()

        # equal signatures under the inverse relabeling
        inverse = {old: perm.index(old) for old in range(3)}
        assert structure_signature(ces, inverse) == structure_signature(
            permuted_ces
        )

        # object equality is index-based and fails across the relabeling...
        assert ces != permuted_ces
        # ...but the structures are isomorphic
        assert are_structures_isomorphic(ces, permuted_ces)


def test_non_isomorphic_structures():
    import pyphi
    from pyphi import examples
    from pyphi.automorphism import are_structures_isomorphic

    with pyphi.config.override(progress_bars=False):
        xor_ces = examples.xor_system().ces()
        grid3_ces = examples.grid3_system().ces()
    assert not are_structures_isomorphic(xor_ces, grid3_ces)


def test_isomorphism_is_reflexive():
    import pyphi
    from pyphi import examples
    from pyphi.automorphism import are_structures_isomorphic

    with pyphi.config.override(progress_bars=False):
        ces = examples.xor_system().ces()
    assert are_structures_isomorphic(ces, ces)
```

Note: match the existing import style at the top of
`test/test_automorphism.py` (module-level imports there are fine to reuse;
the `from .example_substrates import permuted_system` may need to be
`from example_substrates import ...` depending on how that file is imported
by existing tests — check an existing usage and copy it).

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest test/test_automorphism.py -v -k structure`
Expected: FAIL with `ImportError: cannot import name 'structure_signature'`

- [ ] **Step 4: Implement in `pyphi/automorphism.py`**

Add at the end of the module:

```python
def _map_aligned(indices, aligned, mapping):
    """Map an index tuple through ``mapping`` and re-sort it ascending,
    reordering the position-aligned tuple ``aligned`` identically."""
    if aligned is None:
        return tuple(sorted(mapping[i] for i in indices)), None
    pairs = sorted(zip((mapping[i] for i in indices), aligned, strict=True))
    return tuple(i for i, _ in pairs), tuple(s for _, s in pairs)


def _distinction_record(distinction, mapping):
    from pyphi.direction import Direction

    mechanism, mechanism_state = _map_aligned(
        distinction.mechanism, distinction.mechanism_state, mapping
    )
    record = [mechanism, mechanism_state, round(float(distinction.phi), _ROUND)]
    for direction in Direction.both():
        mice = distinction.mice(direction)
        spec = mice.specified_state
        if spec is None:
            record.append((tuple(sorted(mapping[i] for i in mice.purview)),))
        else:
            purview, state = _map_aligned(spec.purview, spec.state, mapping)
            record.append((purview, state, round(float(mice.phi), _ROUND)))
    return tuple(record)


def _structure_node_indices(ces):
    from pyphi.substrate import _sia_node_indices

    indices = _sia_node_indices(ces.sia)
    if indices is None:
        raise ValueError("structure's SIA carries no node indices")
    return indices


def structure_signature(ces, mapping=None):
    """A canonical, order-independent value key for a cause-effect structure
    under a node-index mapping.

    Covers each distinction's mechanism, mechanism state, cause/effect
    purviews with their specified states, and φ values (rounded to
    ``_ROUND`` places), plus each relation's relata mechanisms and φ.
    Repertoires and partitions are not included. When the relation set is
    not enumerable (analytical), its rounded aggregates stand in for the
    per-relation records.
    """
    if mapping is None:
        mapping = {i: i for i in _structure_node_indices(ces)}
    distinction_records = tuple(
        sorted(_distinction_record(d, mapping) for d in ces.distinctions)
    )
    try:
        relation_records = tuple(
            sorted(
                (
                    tuple(
                        sorted(
                            tuple(sorted(mapping[i] for i in mechanism))
                            for mechanism in relation.mechanisms
                        )
                    ),
                    round(float(relation.phi), _ROUND),
                )
                for relation in ces.relations
            )
        )
    except TypeError:  # analytical relations are not enumerable
        relation_records = (
            round(float(ces.relations.sum_phi()), _ROUND),
            ces.relations.num_relations(),
        )
    return (distinction_records, relation_records)


def are_structures_isomorphic(ces1, ces2) -> bool:
    """Whether two cause-effect structures are equal up to a bijection of
    their node indices, at the resolution of :func:`structure_signature`.

    Exact search over index bijections, with cheap invariant pruning first.
    Factorial in the number of units; substrates on which Φ is computed are
    small, so this is tractable (same rationale as
    :func:`substrate_automorphisms`).
    """
    indices1 = _structure_node_indices(ces1)
    indices2 = _structure_node_indices(ces2)
    if len(indices1) != len(indices2):
        return False
    if len(ces1.distinctions) != len(ces2.distinctions):
        return False
    phis1 = sorted(round(float(d.phi), _ROUND) for d in ces1.distinctions)
    phis2 = sorted(round(float(d.phi), _ROUND) for d in ces2.distinctions)
    if phis1 != phis2:
        return False
    target = structure_signature(ces2)
    for permuted in permutations(indices2):
        mapping = dict(zip(indices1, permuted, strict=True))
        if structure_signature(ces1, mapping) == target:
            return True
    return False
```

(`_ROUND` and `permutations` are already imported/defined at module top.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest test/test_automorphism.py -v`
Expected: all PASS (existing substrate tests untouched)

- [ ] **Step 6: Changelog fragment and commit**

```bash
cat > changelog.d/structure-isomorphism.feature.md <<'EOF'
Added `pyphi.automorphism.structure_signature` (canonical value key for a
cause-effect structure under a node-index mapping) and
`are_structures_isomorphic` (exact structure equality up to unit
relabeling, at signature resolution: mechanisms, states, purviews, φ).
EOF
git add pyphi/automorphism.py test/test_automorphism.py test/example_substrates.py changelog.d/structure-isomorphism.feature.md
git commit -m "Add structure signatures and isomorphism up to relabeling"
```

---

### Task 6: `relabel`

Rebuilds a complete IIT 4.0 `CauseEffectStructure` through a node-index
bijection: every nested result object is reconstructed via its public
constructor with mapped indices, position-aligned tuples reordered, and
purview-shaped arrays transposed. Relations are rebuilt from the relabeled
distinctions (a relation's φ recomputes from its relata, so the values are
exact). Contract limits, documented in the module docstring: tie
back-references are dropped (each object keeps only itself as its tie);
IIT 3.0 SIAs and structure views are rejected; node labels are supplied by
the caller or dropped.

**Files:**
- Create: `pyphi/relabel.py`
- Modify: `pyphi/models/ces.py` (add `CauseEffectStructure.relabel` delegating method)
- Test: `test/test_relabel.py` (create)
- Create: `changelog.d/ces-relabel.feature.md`

**Interfaces:**
- Consumes: `StructureView` (Task 3, for the view guard);
  `structure_signature` and `permuted_system` (Task 5, in tests only).
- Produces: `pyphi.relabel.relabel_ces(ces, mapping, node_labels=None) -> CauseEffectStructure`
  and `CauseEffectStructure.relabel(mapping, node_labels=None)`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_relabel.py`:

```python
"""Tests for relabeling cause-effect structures through index bijections."""

import pytest

import pyphi
from pyphi import examples
from pyphi.automorphism import structure_signature

from example_substrates import permuted_system

PERM = (2, 0, 1)  # new index i holds old node PERM[i]


@pytest.fixture(scope="module")
def grid3_ces():
    with pyphi.config.override(progress_bars=False):
        return examples.grid3_system().ces()


@pytest.fixture(scope="module")
def permuted_ces():
    with pyphi.config.override(progress_bars=False):
        return permuted_system(examples.grid3_system(), PERM).ces()


@pytest.fixture(scope="module")
def relabeled(grid3_ces):
    mapping = {old: PERM.index(old) for old in range(3)}
    return grid3_ces.relabel(mapping)


def test_relabel_matches_recomputation(relabeled, permuted_ces):
    # relabeling the structure equals recomputing on the permuted system,
    # at signature resolution
    assert structure_signature(relabeled) == structure_signature(permuted_ces)


def test_relabel_preserves_aggregates(grid3_ces, relabeled):
    assert relabeled.big_phi == pytest.approx(grid3_ces.big_phi)
    assert relabeled.sum_phi_distinctions == pytest.approx(
        grid3_ces.sum_phi_distinctions
    )
    assert relabeled.relations.num_relations() == grid3_ces.relations.num_relations()
    assert float(relabeled.sia.phi) == pytest.approx(float(grid3_ces.sia.phi))


def test_relabel_round_trip(grid3_ces, relabeled):
    mapping = {old: PERM.index(old) for old in range(3)}
    inverse = {new: old for old, new in mapping.items()}
    back = relabeled.relabel(inverse)
    assert structure_signature(back) == structure_signature(grid3_ces)


def test_identity_relabel_is_signature_noop(grid3_ces):
    identity = {i: i for i in range(3)}
    assert structure_signature(grid3_ces.relabel(identity)) == structure_signature(
        grid3_ces
    )


def test_relabel_repr_does_not_crash(relabeled):
    assert repr(relabeled)
    assert repr(relabeled.sia)
    assert repr(relabeled.distinctions[0])


def test_relabel_rejects_views(grid3_ces):
    view = grid3_ces.induce(list(grid3_ces.distinctions)[:2])
    with pytest.raises(ValueError, match="parent structure"):
        view.relabel({i: i for i in range(3)})


def test_relabel_rejects_non_bijection(grid3_ces):
    with pytest.raises(ValueError, match="injective"):
        grid3_ces.relabel({0: 0, 1: 0, 2: 2})


def test_relabel_rejects_partial_mapping(grid3_ces):
    with pytest.raises(ValueError, match="cover"):
        grid3_ces.relabel({0: 1, 1: 0})
```

(Adjust the `example_substrates` import to match how `test/test_automorphism.py`
ended up importing it in Task 5.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_relabel.py -v`
Expected: FAIL with `AttributeError: ... no attribute 'relabel'`

- [ ] **Step 3: Create `pyphi/relabel.py`**

```python
"""Relabel result objects through a node-index bijection.

Relabeling rewrites every node index in a result object through a
bijective mapping, reordering position-aligned state tuples and
transposing purview-shaped arrays to match the re-sorted index order.
No φ value changes: relabeling is an isomorphism of the frame, not a
recomputation.

Contract:

- Only complete IIT 4.0 structures are supported. Structure views
  (folds, induced substructures) raise — relabel the parent and
  re-derive the view. IIT 3.0 SIAs raise ``NotImplementedError``.
- Tie back-references are dropped: each relabeled object records only
  itself as its tie. Tie *resolution* is unaffected (the resolved
  member is what gets relabeled).
- ``node_labels`` for the target coordinates may be passed; when
  omitted, labels are dropped.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from pyphi.models.ces import CauseEffectStructure
from pyphi.models.ces import StructureView
from pyphi.models.distinction import Distinction
from pyphi.models.distinctions import ResolvedDistinctions
from pyphi.models.partitions import NullCut
from pyphi.models.partitions import Part
from pyphi.models.ria import RepertoireIrreducibilityAnalysis
from pyphi.models.state_specification import StateSpecification
from pyphi.models.state_specification import SystemStateSpecification
from pyphi.relations import AnalyticalRelations
from pyphi.relations import ConcreteRelations
from pyphi.relations import NullRelations
from pyphi.relations import Relation


def _argsorted_map(indices, mapping):
    """Map an index tuple, returning it re-sorted ascending together with
    the positional order that re-sorts any aligned tuple identically."""
    mapped = [mapping[i] for i in indices]
    order = sorted(range(len(mapped)), key=lambda k: mapped[k])
    return tuple(mapped[k] for k in order), order


def _reorder(aligned, order):
    if aligned is None:
        return None
    return tuple(aligned[k] for k in order)


def _transpose(array, order):
    """Reorder the axes of a purview-shaped array to the re-sorted index
    order. Scalar (0-d) arrays and ``None`` pass through."""
    if array is None:
        return None
    array = np.asarray(array)
    if array.ndim == 0:
        return array
    if array.ndim == len(order):
        return np.transpose(array, axes=order)
    raise NotImplementedError(
        f"cannot relabel an array of ndim {array.ndim} over a purview of "
        f"size {len(order)}"
    )


def relabel_state_specification(spec, mapping):
    purview, order = _argsorted_map(spec.purview, mapping)
    return StateSpecification(
        direction=spec.direction,
        purview=purview,
        state=_reorder(spec.state, order),
        intrinsic_information=spec.intrinsic_information,
        repertoire=_transpose(spec.repertoire, order),
        unconstrained_repertoire=_transpose(spec.unconstrained_repertoire, order),
    )


def relabel_system_state(system_state, mapping):
    return SystemStateSpecification(
        cause=relabel_state_specification(system_state.cause, mapping),
        effect=relabel_state_specification(system_state.effect, mapping),
    )


def relabel_joint_partition(partition, mapping, node_labels=None):
    if partition is None:
        return None
    parts = tuple(
        Part(
            mechanism=tuple(sorted(mapping[i] for i in part.mechanism)),
            purview=tuple(sorted(mapping[i] for i in part.purview)),
            node_labels=node_labels,
        )
        for part in partition
    )
    return type(partition)(*parts, node_labels=node_labels)


def relabel_ria(ria, mapping, node_labels=None):
    mechanism, mechanism_order = _argsorted_map(ria.mechanism, mapping)
    purview, purview_order = _argsorted_map(ria.purview, mapping)
    specified_state = ria.specified_state
    return RepertoireIrreducibilityAnalysis(
        phi=ria.signed_phi,
        direction=ria.direction,
        mechanism=mechanism,
        purview=purview,
        partition=relabel_joint_partition(ria.partition, mapping, node_labels),
        repertoire=_transpose(ria.repertoire, purview_order),
        partitioned_repertoire=_transpose(
            ria.partitioned_repertoire, purview_order
        ),
        specified_state=(
            None
            if specified_state is None
            else relabel_state_specification(specified_state, mapping)
        ),
        mechanism_state=_reorder(ria.mechanism_state, mechanism_order),
        purview_state=_reorder(ria.purview_state, purview_order),
        node_labels=node_labels,
        selectivity=ria.selectivity,
        reasons=ria.reasons,
        signed_phi=ria.signed_phi,
    )


def relabel_mice(mice, mapping, node_labels=None):
    return type(mice)(relabel_ria(mice._ria, mapping, node_labels))


def relabel_distinction(distinction, mapping, node_labels=None):
    return Distinction(
        mechanism=tuple(sorted(mapping[i] for i in distinction.mechanism)),
        cause=relabel_mice(distinction.cause, mapping, node_labels),
        effect=relabel_mice(distinction.effect, mapping, node_labels),
    )


def _relabel_relations(relations, new_by_old, mapping):
    if isinstance(relations, NullRelations):
        return relations
    if isinstance(relations, ConcreteRelations):
        return ConcreteRelations(
            Relation(new_by_old[d] for d in relation) for relation in relations
        )
    if isinstance(relations, AnalyticalRelations):
        return AnalyticalRelations(ResolvedDistinctions(new_by_old.values()))
    raise TypeError(
        f"cannot relabel relations of type {type(relations).__name__}"
    )


def _relabel_system_partition(partition, mapping, node_labels=None):
    if partition is None:
        return None
    if isinstance(partition, NullCut):
        return NullCut(
            tuple(sorted(mapping[i] for i in partition.indices)), node_labels
        )
    if hasattr(partition, "relabel"):
        return partition.relabel(
            tuple(mapping[i] for i in partition.node_indices), node_labels
        )
    raise NotImplementedError(
        f"cannot relabel a system partition of type {type(partition).__name__}"
    )


def relabel_sia(sia, mapping, node_labels=None):
    from pyphi.formalism.iit4 import SystemIrreducibilityAnalysis

    if not isinstance(sia, SystemIrreducibilityAnalysis):
        raise NotImplementedError(
            "relabel supports IIT 4.0 system irreducibility analyses; got "
            f"{type(sia).__name__}"
        )
    node_indices, order = _argsorted_map(sia.node_indices, mapping)
    return dataclasses.replace(
        sia,
        phi=sia.signed_phi,
        normalized_phi=sia.signed_normalized_phi,
        signed_phi=sia.signed_phi,
        signed_normalized_phi=sia.signed_normalized_phi,
        partition=_relabel_system_partition(sia.partition, mapping, node_labels),
        cause=(
            None if sia.cause is None else relabel_ria(sia.cause, mapping, node_labels)
        ),
        effect=(
            None
            if sia.effect is None
            else relabel_ria(sia.effect, mapping, node_labels)
        ),
        system_state=(
            None
            if sia.system_state is None
            else relabel_system_state(sia.system_state, mapping)
        ),
        current_state=_reorder(sia.current_state, order),
        node_indices=node_indices,
        node_labels=node_labels,
    )


def relabel_ces(ces, mapping, node_labels=None) -> CauseEffectStructure:
    """Return ``ces`` rewritten through the node-index bijection ``mapping``.

    ``mapping`` must be injective and cover the structure's node indices.
    All φ values are preserved exactly; see the module docstring for the
    contract on ties, views, and labels.
    """
    from pyphi.substrate import _sia_node_indices

    if isinstance(ces, StructureView):
        raise ValueError(
            "cannot relabel a structure view; relabel the parent structure "
            "and re-derive the view"
        )
    mapping = dict(mapping)
    indices = _sia_node_indices(ces.sia)
    if indices is None:
        raise ValueError("structure's SIA carries no node indices")
    if not set(mapping) >= set(indices):
        raise ValueError(
            f"mapping must cover all node indices {tuple(indices)}"
        )
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("mapping must be injective")
    new_by_old = {
        d: relabel_distinction(d, mapping, node_labels) for d in ces.distinctions
    }
    return CauseEffectStructure(
        sia=relabel_sia(ces.sia, mapping, node_labels),
        distinctions=ResolvedDistinctions(new_by_old.values()),
        relations=_relabel_relations(ces.relations, new_by_old, mapping),
        config=ces.config,
    )
```

- [ ] **Step 4: Add the delegating method**

In `pyphi/models/ces.py`, add to `CauseEffectStructure` after `meet`:

```python
    def relabel(self, mapping, node_labels=None) -> CauseEffectStructure:
        """Return this structure rewritten through the node-index bijection
        ``mapping``. See :func:`pyphi.relabel.relabel_ces`."""
        from pyphi.relabel import relabel_ces

        return relabel_ces(self, mapping, node_labels=node_labels)
```

- [ ] **Step 5: Run tests, fix constructor mismatches**

Run: `uv run pytest test/test_relabel.py -v`
Expected: all PASS. Likely first-run failures and their fixes:

- `AttributeError` on an RIA accessor (e.g. `selectivity` or `reasons` not
  public): check `pyphi/models/ria.py` property list (lines ~238-310) and
  use the private attribute (`ria._selectivity`, `ria._reasons`) if a
  public property is missing.
- `dataclasses.replace` re-running `__post_init__` and double-clamping:
  the plan passes the signed values back as `phi`/`normalized_phi`, which
  reproduces the clamp idempotently — if a test still shows a changed φ,
  compare `sia.phi` before/after and adjust which field is passed.
- Import cycle from `pyphi/relabel.py` importing `pyphi.models.ces` at
  module scope: if it triggers, move the model imports into the functions
  (the codebase already uses deferred imports for cycles).

- [ ] **Step 6: Changelog fragment and commit**

```bash
cat > changelog.d/ces-relabel.feature.md <<'EOF'
Added `CauseEffectStructure.relabel(mapping)` (and `pyphi.relabel`):
rewrite a structure through a node-index bijection, reconstructing every
nested result object with mapped indices. φ values are preserved exactly;
tie back-references are dropped; IIT 3.0 SIAs and structure views are not
supported.
EOF
git add pyphi/relabel.py pyphi/models/ces.py test/test_relabel.py changelog.d/ces-relabel.feature.md
git commit -m "Add structure relabeling through node-index bijections"
```

---

### Task 7: Documentation close-out and full verification

**Files:**
- Modify: `docs/superpowers/specs/2026-07-07-ces-algebra-exploration.md` (§6.3, §9.3, §10)
- Modify: `ROADMAP.md` (Status Dashboard)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Update the exploration spec**

In `docs/superpowers/specs/2026-07-07-ces-algebra-exploration.md`:

- §6.3: after the recommendation paragraph, add a resolution line:

```markdown
**Resolution:** `big_phi_contribution` now computes the share-weighted
measure μ for all folds, so any partition of a structure's distinctions
into folds tiles Φ exactly; the count-once quantity is no longer exposed.
```

- §9.3: replace the open-problem paragraph body with a one-line pointer:

```markdown
Resolved: the compound-fold magnitude is the share-weighted measure μ
(see §6.3), implemented in `PhiFold.big_phi_contribution`.
```

- §10: mark items 1-6 as implemented, e.g. change the list intro to
  "Ordered by value per line of code, all small — **all implemented**:" and
  leave the item text unchanged.

- [ ] **Step 2: Add a ROADMAP dashboard row**

In `ROADMAP.md`'s Status Dashboard table, add a row following the existing
format (match column structure of neighboring rows):

```markdown
| CES algebra operations | ✅ landed | — | Operations over cause-effect structures as first-class objects: `Distinctions.filter`, `CauseEffectStructure.induce`/`meet` (relation-closed `InducedSubstructure` views with a same-frame check), share-weighted fold contributions (fold partitions tile Φ exactly, incl. multi-seed folds), structure signatures + `are_structures_isomorphic`, and `relabel` through index bijections. Design: `docs/superpowers/specs/2026-07-07-ces-algebra-exploration.md`. |
```

- [ ] **Step 3: Full test suite (no path argument — includes the doctest sweep)**

Run: `uv run pytest -x -q`
Expected: all pass. This is the complete verification recipe per project
convention; bare-path invocations skip doctests.

If any doctest or unrelated-looking test fails, diagnose before touching
anything — other sessions may have concurrent working-tree changes; only
fix failures traceable to this plan's commits.

- [ ] **Step 4: Pre-commit hooks over the changed files**

Run: `uv run pre-commit run --files $(git diff --name-only $(git merge-base HEAD <base-branch>) | tr '\n' ' ')` — substitute the branch this worktree was created from.
Expected: all hooks pass (ruff, pyright, file checks). Fix any findings and
amend or follow-up commit as appropriate.

- [ ] **Step 5: Commit the documentation updates**

```bash
git add docs/superpowers/specs/2026-07-07-ces-algebra-exploration.md ROADMAP.md
git commit -m "Record CES algebra operations in spec and roadmap"
```

---

## Self-review notes

- Spec §10 item coverage: filter (Task 1), induce (Task 3), μ/fold fix
  (Task 2, per the approved generalization), frame check (Task 4), relabel +
  isomorphism (Tasks 5-6), meet (Task 4). Spec updates (Task 7).
- Names used across tasks are consistent: `_resolve_members`,
  `relation_closed`, `StructureView`, `InducedSubstructure`,
  `share_weighted_sum_phi`, `structure_signature`, `permuted_system`,
  `relabel_ces`.
- Task 2 changes behavior of a shipped property; the matching module is the
  one other candidate consumer and Step 7 audits it explicitly.
- Task 6 is the highest-risk task (many constructors); its Step 5 lists the
  three expected failure modes and their fixes, and its end-to-end test
  (signature equality against a recomputed permuted system) catches any
  missed field.
